import asyncio
import os
import time
from dataclasses import dataclass
from typing import Optional
from fractions import Fraction

import av  # PyAV
import numpy as np

from livekit import rtc

try:
    import google.generativeai as genai  # type: ignore
    _HAS_GENAI = True
except Exception:
    _HAS_GENAI = False


@dataclass
class VideoFrameItem:
    pts_us: int  # Timestamp in microseconds from LiveKit
    frame: rtc.VideoFrame


@dataclass
class AudioFrameItem:
    pts_us: int  # Timestamp in microseconds (synchronized with video)
    frame: rtc.AudioFrame


class ParticipantVideoRecorder:
    def __init__(
        self,
        participant: rtc.RemoteParticipant,
        recordings_dir: str = "recordings",
        segment_seconds: int = 60,
        video_timebase_den: int = 1_000_000,
    ) -> None:
        self.participant = participant
        self.recordings_dir = recordings_dir
        self.segment_seconds = segment_seconds
        self.video_timebase_den = video_timebase_den

        os.makedirs(self.recordings_dir, exist_ok=True)

        self.video_track: Optional[rtc.VideoTrack] = None
        self.audio_track: Optional[rtc.Track] = None

        self.video_stream: Optional[rtc.VideoStream] = None
        self.audio_stream: Optional[rtc.AudioStream] = None

        self._writer_task: Optional[asyncio.Task] = None
        self._reader_tasks: set[asyncio.Task] = set()

        # Queues for frames
        self._video_q: asyncio.Queue[VideoFrameItem] = asyncio.Queue(maxsize=120)
        self._audio_q: asyncio.Queue[AudioFrameItem] = asyncio.Queue(maxsize=240)

        # Synchronization: use microseconds as common timebase
        self._segment_start_us: int = 0  # Segment start time in microseconds
        
        # Frame counters as fallback if timestamps are invalid
        self._video_frame_count: int = 0
        self._audio_sample_count: int = 0

        # Control
        self._stop_event = asyncio.Event()
        self._rotate_now: bool = False

        # Audio resampler for encoder (set when opening container if audio present)
        self._audio_resampler: Optional[av.audio.resampler.AudioResampler] = None

    async def start_if_ready(self) -> None:
        print(f"[RECORDER] start_if_ready called for {self.participant.identity or self.participant.sid}")
        if self._writer_task is not None:
            print("[RECORDER] Writer already running")
            return
        if self.video_track is None:
            print(f"[RECORDER] Missing tracks: video={'ok' if self.video_track else 'none'} audio={'ok' if self.audio_track else 'none'}")
            return

        # Create streams from tracks
        self.video_stream = rtc.VideoStream.from_track(track=self.video_track)
        if self.audio_track is not None:
            self.audio_stream = rtc.AudioStream.from_track(track=self.audio_track, sample_rate=48000, num_channels=1)

        # Launch readers
        self._reader_tasks.add(asyncio.create_task(self._video_reader(), name="video_reader"))
        if self.audio_stream is not None:
            self._reader_tasks.add(asyncio.create_task(self._audio_reader(), name="audio_reader"))

        # Launch writer
        print("[RECORDER] Starting mux writer loop")
        self._writer_task = asyncio.create_task(self._mux_writer(), name="mux_writer")

    async def stop(self) -> None:
        self._stop_event.set()
        # Close streams
        if self.video_stream is not None:
            await self.video_stream.aclose()
            self.video_stream = None
        if self.audio_stream is not None:
            await self.audio_stream.aclose()
            self.audio_stream = None
        # Cancel readers
        for t in list(self._reader_tasks):
            t.cancel()
        self._reader_tasks.clear()
        # Wait writer
        if self._writer_task is not None:
            try:
                await self._writer_task
            except asyncio.CancelledError:
                pass
            self._writer_task = None

    async def _video_reader(self) -> None:
        assert self.video_stream is not None
        async for event in self.video_stream:
            if self._stop_event.is_set():
                break
            pts_us = getattr(event, "timestamp_us", None)
            if pts_us is None:
                pts_us = int(time.time() * 1_000_000)
            await self._video_q.put(VideoFrameItem(pts_us=pts_us, frame=event.frame))

    async def _audio_reader(self) -> None:
        assert self.audio_stream is not None
        frame_count = 0
        print(f"[RECORDER] Audio reader started for participant {self.participant.identity or self.participant.sid}")
        try:
            async for event in self.audio_stream:
                if self._stop_event.is_set():
                    break
                af = event.frame
                # Get timestamp from event (microseconds) - same as video uses
                pts_us = getattr(event, "timestamp_us", None)
                if pts_us is None:
                    pts_us = int(time.time() * 1_000_000)
                
                # Derive channels and sample rate robustly
                channels = getattr(af, "num_channels", getattr(af, "channels", 1))
                sample_rate = getattr(af, "sample_rate", 48000)
                raw = getattr(af, "data", b"")
                # Each sample is int16 (2 bytes) per channel
                try:
                    samples_interleaved = len(raw) // 2
                    num_samples_total = max(0, samples_interleaved)
                    if channels > 0:
                        num_samples_per_channel = num_samples_total // channels
                    else:
                        channels = 1
                        num_samples_per_channel = num_samples_total
                except Exception:
                    channels = max(1, int(getattr(af, "num_channels", 1)))
                    num_samples_per_channel = int(getattr(af, "num_samples", getattr(af, "samples_per_channel", 0)))

                # Stash derived metadata on the frame for the writer
                try:
                    af._derived_channels = channels  # type: ignore[attr-defined]
                    af._derived_sample_rate = sample_rate  # type: ignore[attr-defined]
                    af._derived_num_samples = num_samples_per_channel  # type: ignore[attr-defined]
                except Exception:
                    pass

                await self._audio_q.put(AudioFrameItem(pts_us=pts_us, frame=af))
                frame_count += 1
                if frame_count % 100 == 0:  # Log every 100 frames
                    print(f"[RECORDER] Audio reader: collected {frame_count} frames (ch={channels}, sr={sample_rate}, samples/frame={num_samples_per_channel})")
        except asyncio.CancelledError:
            pass
        except Exception as e:
            print(f"[ERROR] Audio reader error: {e}")
            import traceback
            traceback.print_exc()
        finally:
            print(f"[RECORDER] Audio reader stopped. Total frames collected: {frame_count}")

    def _next_filepath(self, start_ts: float) -> str:
        ident = self.participant.identity or self.participant.sid
        timestr = time.strftime("%Y%m%d_%H%M%S", time.localtime(start_ts))
        filename = f"{ident}_{timestr}.mp4"
        # Use native OS path format
        return os.path.join(self.recordings_dir, filename)

    async def _mux_writer(self) -> None:
        segment_start = time.time()
        container, v_stream, a_stream, current_path, native_path = self._open_container(segment_start)
        print(f"[RECORDER] Opened new segment file: {native_path}")
        
        # Set segment start time in microseconds for synchronization
        self._segment_start_us = int(segment_start * 1_000_000)
        self._audio_frames_written = 0  # Track audio frames written to container
        self._video_frame_count = 0  # Reset frame counter
        self._audio_sample_count = 0  # Reset audio sample counter

        last_heartbeat = time.time()
        try:
            while not self._stop_event.is_set():
                # Periodic heartbeat log (every 10 seconds) to show we're alive
                now = time.time()
                if now - last_heartbeat >= 10:
                    print(f"[RECORDER] Heartbeat - writer loop active. Segment running for {now - segment_start:.1f}s")
                    last_heartbeat = now
                
                # Rotate segment
                segment_duration = now - segment_start
                if segment_duration >= self.segment_seconds or self._rotate_now:
                    try:
                        # Close current file and trigger analysis
                        print(f"[RECORDER] Closing segment after {segment_duration:.1f}s. Rotate trigger: time-based={segment_duration >= self.segment_seconds}, rotate_now={self._rotate_now}")
                        self._close_container(container)
                        
                        # Check if file exists and has content before analyzing
                        file_size = 0
                        if os.path.exists(native_path):
                            file_size = os.path.getsize(native_path)
                            print(f"[RECORDER] Segment file size: {file_size} bytes")
                        
                        # Only analyze completed segments with content (skip if rotation triggered by first audio attach and segment might be empty)
                        if not self._rotate_now and file_size > 1000:  # At least 1KB to have some content
                            print(f"[RECORDER] Triggering analysis for segment: {os.path.basename(native_path)}")
                            self._trigger_video_analysis(native_path)
                        elif self._rotate_now:
                            print(f"[RECORDER] Skipping analysis - rotation triggered by audio attachment (file might be empty)")
                        elif file_size <= 1000:
                            print(f"[RECORDER] Skipping analysis - file too small ({file_size} bytes), likely empty")
                        
                        # Open new container for next segment
                        segment_start = now
                        container, v_stream, a_stream, current_path, native_path = self._open_container(segment_start)
                        print(f"[RECORDER] Rotated segment. New file: {native_path}")
                        # Reset for new segment
                        self._segment_start_us = int(segment_start * 1_000_000)
                        self._audio_frames_written = 0
                        self._video_frame_count = 0
                        self._audio_sample_count = 0
                        self._audio_resampler = None
                        self._rotate_now = False
                    except Exception as e:
                        print(f"[ERROR] Error during segment rotation: {e}")
                        import traceback
                        traceback.print_exc()
                        # Try to continue with a new container
                        try:
                            segment_start = now
                            container, v_stream, a_stream, current_path, native_path = self._open_container(segment_start)
                            print(f"[RECORDER] Recovered - opened new container: {native_path}")
                            self._segment_start_us = int(segment_start * 1_000_000)
                            self._audio_frames_written = 0
                            self._video_frame_count = 0
                            self._audio_sample_count = 0
                            self._audio_resampler = None
                            self._rotate_now = False
                        except Exception as e2:
                            print(f"[ERROR] Failed to recover from rotation error: {e2}")
                            break  # Exit loop if we can't recover

                # Wait for either video or audio frame (with timeout to allow rotation check)
                # Only wait on audio queue if audio stream exists
                wait_tasks = [asyncio.create_task(self._video_q.get())]
                if self.audio_stream is not None:
                    wait_tasks.append(asyncio.create_task(self._audio_q.get()))
                
                try:
                    done, pending = await asyncio.wait(
                        wait_tasks,
                        return_when=asyncio.FIRST_COMPLETED,
                        timeout=0.5  # Check rotation every 0.5s even if no frames
                    )
                    
                    # Process completed tasks
                    for task in done:
                        try:
                            item = task.result()
                            if isinstance(item, VideoFrameItem):
                                self._write_video_frame(container, v_stream, item)
                            else:
                                self._write_audio_frame(container, a_stream, item)
                        except Exception as e:
                            print(f"[RECORDER] Error processing frame: {e}")
                            import traceback
                            traceback.print_exc()
                    
                    # Cancel pending tasks properly
                    for task in pending:
                        task.cancel()
                        try:
                            await task
                        except asyncio.CancelledError:
                            pass
                except asyncio.TimeoutError:
                    # No frames available, continue to check rotation
                    # Clean up any tasks we created
                    for task in wait_tasks:
                        if not task.done():
                            task.cancel()
                            try:
                                await task
                            except (asyncio.CancelledError, asyncio.InvalidStateError):
                                pass
                    continue
        except Exception as e:
            print(f"[ERROR] Fatal error in mux writer loop: {e}")
            import traceback
            traceback.print_exc()
        finally:
            print("[RECORDER] Writer loop exiting, closing container...")
            try:
                self._close_container(container)
            except Exception as e:
                print(f"[ERROR] Error closing container: {e}")

    def _open_container(self, start_ts: float):
        native_path = self._next_filepath(start_ts)
        # Ensure parent directory exists (use native path)
        os.makedirs(os.path.dirname(native_path), exist_ok=True)
        # Convert to absolute path with forward slashes for PyAV/FFmpeg
        abs_path = os.path.abspath(native_path).replace("\\", "/")
        container = av.open(abs_path, mode="w", format="mp4")

        # Video stream (libx264, yuv420p)
        v_stream = container.add_stream("libx264", rate=30)
        v_stream.pix_fmt = "yuv420p"
        # Use frame-based time_base (30 fps = 1/30)
        v_stream.time_base = Fraction(1, 30)

        # Audio stream (aac) if audio available
        a_stream = None
        has_audio = self.audio_stream is not None
        if has_audio:
            # Configure audio encoder (AAC). We'll feed it FLTP mono @ 48k via resampler.
            a_stream = container.add_stream("aac", rate=48000)
            try:
                # Some builds support setting time_base explicitly
                a_stream.time_base = Fraction(1, 48000)
            except Exception:
                pass
            # Create a resampler to convert incoming s16 mono to fltp mono @ 48k
            try:
                self._audio_resampler = av.AudioResampler(format="fltp", layout="mono", rate=48000)
                print(f"[RECORDER] Audio resampler initialized: fltp/mono/48000")
            except Exception as e:
                self._audio_resampler = None
                print(f"[RECORDER] Warning: failed to initialize audio resampler: {e}")
            print(f"[RECORDER] Container opened WITH audio stream for {os.path.basename(native_path)}")
        else:
            print(f"[RECORDER] Container opened WITHOUT audio stream for {os.path.basename(native_path)} (audio_stream is None)")

        return container, v_stream, a_stream, abs_path, native_path

    def _close_container(self, container: av.container.OutputContainer) -> None:
        try:
            # Check streams before closing
            video_streams = len([s for s in container.streams if s.type == 'video'])
            audio_streams = len([s for s in container.streams if s.type == 'audio'])
            print(f"[RECORDER] Closing container - Video streams: {video_streams}, Audio streams: {audio_streams}, Audio frames written: {getattr(self, '_audio_frames_written', 0)}")
            container.close()
        except Exception as e:
            print(f"[ERROR] Error closing container: {e}")

    def _write_video_frame(self, container: av.container.OutputContainer, v_stream: av.video.stream.VideoStream, item: VideoFrameItem) -> None:
        try:
            lk_frame = item.frame
            rgb_bytes = lk_frame.convert(rtc.VideoBufferType.RGB24).data
            arr = np.frombuffer(rgb_bytes, dtype=np.uint8).reshape((lk_frame.height, lk_frame.width, 3))

            av_frame = av.VideoFrame.from_ndarray(arr, format="rgb24")
            av_frame = av_frame.reformat(format="yuv420p")
            
            # Convert microseconds timestamp to stream timebase (30 fps = 1/30)
            # Use frame counter as fallback if timestamp is invalid
            try:
                relative_time_us = item.pts_us - self._segment_start_us
                if relative_time_us >= 0:
                    av_frame.pts = int((relative_time_us / 1_000_000) * 30)
                else:
                    # Timestamp before segment start - use frame counter
                    av_frame.pts = self._video_frame_count
            except Exception:
                # Fallback to frame counter
                av_frame.pts = self._video_frame_count
            
            # Ensure PTS is non-negative
            if av_frame.pts < 0:
                av_frame.pts = self._video_frame_count
            
            self._video_frame_count += 1
            
            try:
                for packet in v_stream.encode(av_frame):
                    if packet is None:
                        continue
                    # Validate packet PTS
                    if packet.pts is not None and packet.pts < 0:
                        print(f"[WARNING] Skipping video packet with negative PTS: {packet.pts}")
                        continue
                    container.mux(packet)
            except Exception as encode_error:
                print(f"[ERROR] Failed to encode/mux video packet: {encode_error}")
                import traceback
                traceback.print_exc()
        except Exception as e:
            print(f"Error writing video frame: {e}")
            import traceback
            traceback.print_exc()

    def _write_audio_frame(self, container: av.container.OutputContainer, a_stream: Optional[av.audio.stream.AudioStream], item: AudioFrameItem) -> None:
        try:
            if a_stream is None:
                print(f"[WARNING] Received audio frame but a_stream is None - audio not included in container")
                return
            af = item.frame
            raw = getattr(af, "data", b"")
            channels = getattr(af, "_derived_channels", getattr(af, "num_channels", getattr(af, "channels", 1)))
            sample_rate = getattr(af, "_derived_sample_rate", getattr(af, "sample_rate", 48000))
            num_samples = getattr(af, "_derived_num_samples", getattr(af, "num_samples", getattr(af, "samples_per_channel", 0)))
            if channels <= 0:
                channels = 1
            if num_samples <= 0 and len(raw) > 0:
                num_samples = (len(raw) // 2) // channels
            
            # Build ndarray - PyAV expects packed format (channels, num_samples)
            # LiveKit provides interleaved format: [L,R,L,R,...] or [S,S,S,...]
            try:
                arr = np.frombuffer(raw, dtype=np.int16)
                total_samples = len(arr)
                
                if channels > 1:
                    # Interleaved: reshape to (num_samples, channels) then transpose to (channels, num_samples)
                    num_samples = total_samples // channels
                    arr = arr.reshape((num_samples, channels)).T  # (channels, num_samples)
                    # Downmix to mono to match stream settings
                    arr = arr[:1, :]  # take first channel
                    channels = 1
                else:
                    # Mono: reshape to (1, num_samples) for packed format
                    num_samples = total_samples
                    arr = arr.reshape((1, num_samples))  # (1, num_samples)
                    
            except Exception as e:
                print(f"[RECORDER] Failed to reshape audio buffer (len={len(raw)}, ch={channels}): {e}")
                return

            layout = "mono"  # we encode mono consistently
            
            # Ensure sample rate matches stream (48kHz)
            target_sample_rate = 48000
            needs_resample = int(sample_rate) != target_sample_rate
            
            if needs_resample:
                # Need to resample - use resampler if available
                if self._audio_resampler is None:
                    print(f"[WARNING] Sample rate mismatch ({sample_rate} != {target_sample_rate}) but no resampler available")
                    return
            else:
                # If already at 48kHz, we still need to convert format (s16 -> fltp) for AAC
                # But let's try encoding s16 directly first - some AAC encoders accept it
                pass
            
            # Calculate samples in this frame
            num_samples = getattr(af, "_derived_num_samples", 0)
            if num_samples <= 0:
                num_samples = (len(raw) // 2) // channels if channels > 0 else len(raw) // 2
            
            in_frame = av.AudioFrame.from_ndarray(arr, format="s16", layout=layout)
            in_frame.sample_rate = int(sample_rate)
            # Use sample counter for PTS (monotonically increasing)
            # This ensures packets are always in order
            in_frame.pts = self._audio_sample_count
            in_frame.time_base = Fraction(1, int(sample_rate))
            
            # Increment counter AFTER setting PTS
            self._audio_sample_count += num_samples

            # Resample to AAC-preferred format (fltp mono 48k) if needed
            if needs_resample and self._audio_resampler is not None:
                try:
                    out_frames = self._audio_resampler.resample(in_frame)
                    if not out_frames:
                        print(f"[WARNING] Resampler returned no frames")
                        return
                except Exception as e:
                    print(f"[RECORDER] Resample failed: {e}")
                    import traceback
                    traceback.print_exc()
                    return
            else:
                # Use input frame directly if already at correct rate and format
                # Note: AAC prefers fltp, but s16 might work with some encoders
                out_frames = [in_frame]
                # If we have a resampler and want fltp format, use it even at same rate
                if self._audio_resampler is not None and in_frame.format.name != "fltp":
                    try:
                        # Resample to convert format (s16 -> fltp) at same rate
                        out_frames = self._audio_resampler.resample(in_frame)
                        if not out_frames:
                            out_frames = [in_frame]  # Fallback to original
                    except Exception:
                        out_frames = [in_frame]  # Fallback to original

            for out in out_frames:
                # Ensure output frame has correct properties
                # Convert input sample count (at input rate) to target rate (48kHz)
                # Use the base PTS before increment
                base_sample_count = self._audio_sample_count - num_samples
                audio_pts_samples = int((base_sample_count * target_sample_rate) / int(sample_rate))
                
                # Ensure non-negative
                if audio_pts_samples < 0:
                    audio_pts_samples = 0
                    
                out.pts = audio_pts_samples
                out.time_base = Fraction(1, target_sample_rate)
                out.sample_rate = target_sample_rate
                
                try:
                    for packet in a_stream.encode(out):
                        if packet is None:
                            continue
                        # Validate packet
                        if packet.pts is not None and packet.pts < 0:
                            print(f"[WARNING] Skipping packet with negative PTS: {packet.pts}")
                            continue
                        container.mux(packet)
                except Exception as encode_error:
                    print(f"[ERROR] Failed to encode/mux audio packet: {encode_error}")
                    import traceback
                    traceback.print_exc()
                    # Continue with next frame instead of breaking
                    continue
            self._audio_frames_written += 1
            if self._audio_frames_written % 500 == 0:  # Log every 500 frames
                print(f"[RECORDER] Written {self._audio_frames_written} audio frames to container")
        except Exception as e:
            print(f"[ERROR] Error writing audio frame: {e}")
            import traceback
            traceback.print_exc()

    def _trigger_video_analysis(self, path: str) -> None:
        print(f"[RECORDER] Triggering analysis for: {os.path.basename(path)}")
        task = asyncio.create_task(self._analyze_video_async(path))
        
        def log_task_result(task: asyncio.Task):
            try:
                task.result()  # This will raise if task had an exception
            except Exception as e:
                print(f"[ERROR] Analysis task failed silently: {e}")
                import traceback
                traceback.print_exc()
        
        task.add_done_callback(log_task_result)

    async def enable_audio(self, track: rtc.Track) -> None:
        """Attach audio mid-run and rotate to start muxing with audio on next segment."""
        try:
            self.audio_track = track
            self.audio_stream = rtc.AudioStream.from_track(track=self.audio_track, sample_rate=48000, num_channels=1)
            # Start the audio reader task if writer is already running
            if self._writer_task is not None:
                self._reader_tasks.add(asyncio.create_task(self._audio_reader(), name="audio_reader"))
                print("[RECORDER] Audio reader task started")
            # Request rotation so next segment includes audio
            self._rotate_now = True
            print("[RECORDER] Audio enabled; will rotate to include audio in next segment")
        except Exception as e:
            print(f"[RECORDER] Failed to enable audio: {e}")
            import traceback
            traceback.print_exc()

    async def _analyze_video_async(self, path: str) -> None:
        participant_id = self.participant.identity or self.participant.sid
        print(f"[ANALYZE] Starting analysis task for {os.path.basename(path)} (participant: {participant_id})")
        
        if not _HAS_GENAI:
            print(f"[ANALYZE] Skipping analysis (google-generativeai not installed) for {os.path.basename(path)}")
            return
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            print(f"[ANALYZE] Skipping analysis (GOOGLE_API_KEY not set) for {os.path.basename(path)}")
            return
        
        file = None
        try:
            # Normalize path to absolute path for better compatibility
            abs_path = os.path.abspath(path)
            print(f"[ANALYZE] Using absolute path: {abs_path}")
            
            genai.configure(api_key=api_key)
            model = genai.GenerativeModel("gemini-2.5-flash")
            
            # Wait for file to be fully closed and flushed
            await asyncio.sleep(2)  # Increased delay for safety
            
            if not os.path.exists(abs_path):
                print(f"[ERROR] File not found: {abs_path}")
                return
            
            file_size = os.path.getsize(abs_path)
            print(f"[ANALYZE] File size: {file_size} bytes")
            if file_size == 0:
                print(f"[ERROR] File is empty, skipping upload")
                return
            
            # Check if file has audio stream using PyAV
            try:
                probe_container = av.open(abs_path, mode="r")
                video_streams = [s for s in probe_container.streams if s.type == 'video']
                audio_streams = [s for s in probe_container.streams if s.type == 'audio']
                probe_container.close()
                print(f"[ANALYZE] File streams - Video: {len(video_streams)}, Audio: {len(audio_streams)}")
                if len(audio_streams) == 0:
                    print(f"[WARNING] ⚠️  Recorded file has NO audio stream - this may affect analysis")
                else:
                    print(f"[ANALYZE] ✓ Audio stream present in file")
            except Exception as probe_error:
                print(f"[WARNING] Could not probe file streams: {probe_error}")
            
            loop = asyncio.get_event_loop()
            print(f"[ANALYZE] Uploading {os.path.basename(path)} to Gemini...")
            
            # Upload file with explicit mime type for video
            file = await loop.run_in_executor(
                None, 
                lambda: genai.upload_file(abs_path, mime_type="video/mp4")
            )
            print(f"[ANALYZE] File uploaded, received file object: {file.name}")
            
            # Wait for file to be ACTIVE (upload complete and processing done)
            print(f"[ANALYZE] Waiting for file {file.name} to be ready...")
            max_wait = 120  # Increased timeout to 2 minutes for large videos
            wait_time = 0
            file_status = None
            
            while wait_time < max_wait:
                try:
                    file_status = await loop.run_in_executor(None, lambda: genai.get_file(file.name))
                    state_name = getattr(file_status.state, "name", str(file_status.state))
                    
                    if state_name == "ACTIVE":
                        print(f"[ANALYZE] File is ready for use (state: ACTIVE)")
                        break
                    elif state_name == "FAILED":
                        error_msg = getattr(file_status, "error", None)
                        print(f"[ERROR] File upload failed with state: {state_name}")
                        if error_msg:
                            print(f"[ERROR] Upload error details: {error_msg}")
                        return
                    
                    await asyncio.sleep(2)
                    wait_time += 2
                    if wait_time % 10 == 0:
                        print(f"[ANALYZE] Still waiting... ({wait_time}s elapsed, state: {state_name})")
                except Exception as status_error:
                    print(f"[WARNING] Error checking file status: {status_error}")
                    await asyncio.sleep(2)
                    wait_time += 2
            
            if wait_time >= max_wait or (file_status and getattr(file_status.state, "name", None) != "ACTIVE"):
                print(f"[ERROR] Timeout waiting for file to become ACTIVE (waited {wait_time}s)")
                return
            
            prompt = (
                "You are a proctoring assistant analyzing exam webcam footage with audio. "
                "Examine this video segment and determine if the participant is cheating. "
                "Return ONLY valid JSON with this exact structure:\n"
                '{"cheating": true/false, "audio_present": true/false, "reasons": ["reason1", "reason2"]}\n'
                "Look for: looking away from screen, multiple people visible, phone/device usage, "
                "suspicious hand movements, screen reflections, frequently going out of frame, "
                "talking to someone off-camera, reading from materials, suspicious audio cues."
            )
            print("[ANALYZE] Requesting analysis from Gemini...")
            resp = await loop.run_in_executor(None, lambda: model.generate_content([prompt, file]))
            
            text = getattr(resp, "text", "").strip()
            print(f"[DEBUG] Gemini raw response length: {len(text)} chars")
            if len(text) < 500:
                print(f"[DEBUG] Gemini response preview: {text[:500]}")
            
            cheating = False
            audio_present = False
            reasons = []
            import json as _json
            try:
                # Clean up response text
                text_clean = text
                if "```json" in text:
                    start = text.find("```json") + 7
                    end = text.find("```", start)
                    if end > start:
                        text_clean = text[start:end].strip()
                elif "```" in text:
                    start = text.find("```") + 3
                    end = text.find("```", start)
                    if end > start:
                        text_clean = text[start:end].strip()
                
                # Find JSON object
                start = text_clean.find("{")
                end = text_clean.rfind("}") + 1
                if start >= 0 and end > start:
                    text_clean = text_clean[start:end]
                
                data = _json.loads(text_clean)
                cheating = bool(data.get("cheating", False))
                audio_present = bool(data.get("audio_present", False))
                reasons = data.get("reasons", [])
                if not isinstance(reasons, list):
                    reasons = []
                print(f"[DEBUG] Parsed JSON successfully - cheating: {cheating}, audio_present: {audio_present}, reasons count: {len(reasons)}")
            except Exception as e:
                print(f"[WARNING] Failed to parse JSON response: {e}")
                print(f"[DEBUG] Attempted to parse: {text_clean[:200] if 'text_clean' in locals() else text[:200]}")
                # Fallback: check if response mentions cheating
                text_lower = text.lower()
                cheating = any(word in text_lower for word in ["cheating", "cheat", "suspicious", "violation"])
                reasons = [f"Parse error: {str(e)[:50]}. Response preview: {text[:200]}"]
            
            # Log results with clear formatting
            print("\n" + "="*70)
            if cheating:
                reasons_str = ", ".join(reasons) if reasons else "Unknown reasons"
                print(f"[ALERT] ⚠️  CHEATING DETECTED for participant '{participant_id}'")
                print(f"        File: {os.path.basename(path)}")
                print(f"        Reasons: {reasons_str}")
            else:
                print(f"[OK] ✓ No cheating detected for participant '{participant_id}'")
                print(f"        File: {os.path.basename(path)}")
                if reasons:
                    print(f"        Notes: {', '.join(reasons)}")
            print("="*70 + "\n")
            
            # Clean up uploaded file
            if file:
                try:
                    print(f"[ANALYZE] Cleaning up uploaded file on Gemini: {file.name}")
                    await loop.run_in_executor(None, lambda: genai.delete_file(file.name))
                    print(f"[ANALYZE] File cleanup successful")
                except Exception as cleanup_error:
                    print(f"[WARNING] Failed to cleanup uploaded file: {cleanup_error}")
                    
        except Exception as e:
            print(f"[ERROR] Analysis failed for {os.path.basename(path)}: {e}")
            import traceback
            traceback.print_exc()
            # Try to cleanup file even on error
            if file:
                try:
                    loop = asyncio.get_event_loop()
                    await loop.run_in_executor(None, lambda: genai.delete_file(file.name))
                except:
                    pass
