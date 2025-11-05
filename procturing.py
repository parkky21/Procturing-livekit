import asyncio
import os
import tempfile
import time
from dataclasses import dataclass
from typing import Optional
from fractions import Fraction
import json

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


class Procturing:
    """
    Procturing class that records participant video/audio segments,
    sends them directly to Gemini for cheating analysis,
    and emits alerts to frontend via LiveKit data channels.
    Files are stored temporarily and deleted after analysis.
    """
    
    def __init__(
        self,
        participant: rtc.RemoteParticipant,
        room: rtc.Room,
        segment_seconds: int = 60,
        video_timebase_den: int = 1_000_000,
    ) -> None:
        self.participant = participant
        self.room = room
        self.segment_seconds = segment_seconds
        self.video_timebase_den = video_timebase_den

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
        
        # Track temp files for cleanup
        self._current_temp_file: Optional[str] = None

    async def start_if_ready(self) -> None:
        print(f"[PROCTORING] start_if_ready called for {self.participant.identity or self.participant.sid}")
        if self._writer_task is not None:
            print("[PROCTORING] Writer already running")
            return
        if self.video_track is None:
            print(f"[PROCTORING] Missing tracks: video={'ok' if self.video_track else 'none'} audio={'ok' if self.audio_track else 'none'}")
            return

        participant_id = self.participant.identity or self.participant.sid
        print(f"[PROCTORING] Starting proctoring for {participant_id}")
        print(f"[PROCTORING] Video track: {self.video_track.sid if self.video_track else 'None'}")
        print(f"[PROCTORING] Audio track: {self.audio_track.sid if self.audio_track else 'None'}")
        print(f"[PROCTORING] About to create streams and tasks...")
        
        try:
            print(f"[PROCTORING] Creating video stream from track...")
            self.video_stream = rtc.VideoStream.from_track(track=self.video_track)
            print(f"[PROCTORING] ✓ Video stream created successfully")
            
            # Create audio stream if audio track exists (might have arrived before video)
            if self.audio_track is not None:
                print(f"[PROCTORING] Audio track exists, creating audio stream...")
                self.audio_stream = rtc.AudioStream.from_track(track=self.audio_track, sample_rate=48000, num_channels=1)
                print(f"[PROCTORING] ✓ Audio stream created successfully")
            
            print(f"[PROCTORING] Creating video reader task...")
            video_task = asyncio.create_task(self._video_reader(), name="video_reader")
            self._reader_tasks.add(video_task)
            task_name = getattr(video_task, 'get_name', lambda: 'video_reader')()
            print(f"[PROCTORING] ✓ Video reader task created (task: {task_name})")

            if self.audio_stream is not None:
                print(f"[PROCTORING] Creating audio reader task...")
                audio_task = asyncio.create_task(self._audio_reader(), name="audio_reader")
                self._reader_tasks.add(audio_task)
                audio_task_name = getattr(audio_task, 'get_name', lambda: 'audio_reader')()
                print(f"[PROCTORING] ✓ Audio reader task created (task: {audio_task_name})")
            else:
                print(f"[PROCTORING] No audio stream available, will work with video only")

            print(f"[PROCTORING] Creating writer task...")
            self._writer_task = asyncio.create_task(self._mux_writer(), name="mux_writer")
            writer_task_name = getattr(self._writer_task, 'get_name', lambda: 'mux_writer')()
            print(f"[PROCTORING] ✓ Proctoring started for {participant_id} - Writer task created (task: {writer_task_name})")
            print(f"[PROCTORING] All tasks scheduled - Waiting for async execution...")
        except Exception as e:
            print(f"[ERROR] Failed to start proctoring: {e}")
            import traceback
            traceback.print_exc()
            # Don't raise - let it continue, but log the error
            print(f"[ERROR] Continuing despite error...")

    async def stop(self) -> None:
        participant_id = self.participant.identity or self.participant.sid
        print(f"[PROCTORING] 🛑 Stopping proctoring for {participant_id}")
        if self._writer_task:
            task_name = getattr(self._writer_task, 'get_name', lambda: 'mux_writer')()
            print(f"[PROCTORING] Writer task state: {task_name} - Done: {self._writer_task.done()}")
        else:
            print(f"[PROCTORING] Writer task: None")
        print(f"[PROCTORING] Reader tasks count: {len(self._reader_tasks)}")
        self._stop_event.set()
        if self._writer_task:
            self._writer_task.cancel()
            try:
                await self._writer_task
            except asyncio.CancelledError:
                print(f"[PROCTORING] Writer task cancelled")
            except Exception as e:
                print(f"[PROCTORING] Error awaiting writer task: {e}")
        for task in self._reader_tasks.copy():
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
            except Exception as e:
                print(f"[PROCTORING] Error awaiting reader task: {e}")
        self._reader_tasks.clear()
        print(f"[PROCTORING] ✓ Proctoring stopped for {participant_id}")

    async def _video_reader(self) -> None:
        frame_count = 0
        try:
            print(f"[PROCTORING] Video reader started for {self.participant.identity or self.participant.sid}")
            async for event in self.video_stream:
                if self._stop_event.is_set():
                    break
                frame_count += 1
                # Extract timestamp from event
                timestamp_us = getattr(event, "timestamp_us", None)
                if timestamp_us is None:
                    timestamp_us = int(time.time() * 1_000_000)
                
                # Access the actual VideoFrame from the event
                frame = event.frame
                try:
                    self._video_q.put_nowait(VideoFrameItem(pts_us=timestamp_us, frame=frame))
                    # Log first few frames
                    if frame_count <= 5:
                        print(f"[PROCTORING] Video frame {frame_count} received - Queue size: {self._video_q.qsize()}")
                except asyncio.QueueFull:
                    print("[PROCTORING] ⚠️ Video queue full, dropping frame")
        except asyncio.CancelledError:
            print(f"[PROCTORING] Video reader cancelled (processed {frame_count} frames)")
            pass
        except Exception as e:
            print(f"[ERROR] Video reader error: {e}")
            import traceback
            traceback.print_exc()
        finally:
            print(f"[PROCTORING] Video reader stopped (total frames: {frame_count})")

    async def _audio_reader(self) -> None:
        frame_count = 0
        try:
            print(f"[PROCTORING] Audio reader started for {self.participant.identity or self.participant.sid}")
            async for event in self.audio_stream:
                if self._stop_event.is_set():
                    break
                frame_count += 1
                # Extract timestamp from event
                timestamp_us = getattr(event, "timestamp_us", None)
                if timestamp_us is None:
                    timestamp_us = int(time.time() * 1_000_000)
                
                # Access the actual AudioFrame from the event
                af = event.frame
                # Derive audio properties robustly
                raw = getattr(af, "data", b"")
                if len(raw) == 0:
                    continue
                channels = getattr(af, "_derived_channels", getattr(af, "num_channels", getattr(af, "channels", 1)))
                sample_rate = getattr(af, "_derived_sample_rate", getattr(af, "sample_rate", 48000))
                num_samples = (len(raw) // 2) // channels if channels > 0 else len(raw) // 2
                af._derived_channels = channels
                af._derived_sample_rate = sample_rate
                af._derived_num_samples = num_samples
                try:
                    self._audio_q.put_nowait(AudioFrameItem(pts_us=timestamp_us, frame=af))
                    # Log first few frames
                    if frame_count <= 5:
                        print(f"[PROCTORING] Audio frame {frame_count} received - Queue size: {self._audio_q.qsize()}")
                except asyncio.QueueFull:
                    print("[PROCTORING] ⚠️ Audio queue full, dropping frame")
        except asyncio.CancelledError:
            print(f"[PROCTORING] Audio reader cancelled (processed {frame_count} frames)")
            pass
        except Exception as e:
            print(f"[ERROR] Audio reader error: {e}")
            import traceback
            traceback.print_exc()
        finally:
            print(f"[PROCTORING] Audio reader stopped (total frames: {frame_count})")

    async def _mux_writer(self) -> None:
        print(f"[PROCTORING] 🚀 Writer loop STARTING for {self.participant.identity or self.participant.sid}")
        segment_start = time.time()
        try:
            print(f"[PROCTORING] Opening container...")
            container, v_stream, a_stream, temp_path = self._open_container(segment_start)
            print(f"[PROCTORING] Opened new temporary segment file: {os.path.basename(temp_path)}")
        except Exception as e:
            print(f"[ERROR] Failed to open container: {e}")
            import traceback
            traceback.print_exc()
            return
        
        self._current_temp_file = temp_path
        
        # Set segment start time in microseconds for synchronization
        self._segment_start_us = int(segment_start * 1_000_000)
        self._audio_frames_written = 0
        self._video_frame_count = 0
        self._audio_sample_count = 0

        last_heartbeat = time.time()
        last_frame_log = time.time()
        print(f"[PROCTORING] 💓 Writer loop started - Waiting for frames...")
        # Log immediate heartbeat to confirm loop is running
        print(f"[PROCTORING] 💓 Initial heartbeat - Video queue: {self._video_q.qsize()}, Audio queue: {self._audio_q.qsize()}")
        try:
            while not self._stop_event.is_set():
                # Periodic heartbeat log (every 5 seconds - more frequent)
                now = time.time()
                if now - last_heartbeat >= 5:
                    elapsed_total = now - segment_start
                    print(f"[PROCTORING] 💓 Heartbeat - Video queue: {self._video_q.qsize()}, Audio queue: {self._audio_q.qsize()}, Segment elapsed: {elapsed_total:.1f}s")
                    last_heartbeat = now

                # Check if we need to rotate segments
                elapsed = time.time() - segment_start
                if self._rotate_now or elapsed >= self.segment_seconds:
                    print(f"[PROCTORING] Rotating segment (elapsed: {elapsed:.1f}s, rotate_now: {self._rotate_now})")
                    self._rotate_now = False
                    
                    # Close current container
                    self._close_container(container)
                    
                    # Trigger analysis on the completed segment
                    if temp_path and os.path.exists(temp_path):
                        asyncio.create_task(self._analyze_and_cleanup(temp_path))
                    
                    # Break if stopping, otherwise start new segment
                    if self._stop_event.is_set():
                        break
                    
                    # Start new segment
                    segment_start = time.time()
                    container, v_stream, a_stream, temp_path = self._open_container(segment_start)
                    self._current_temp_file = temp_path
                    self._segment_start_us = int(segment_start * 1_000_000)
                    self._audio_frames_written = 0
                    self._video_frame_count = 0
                    self._audio_sample_count = 0
                    print(f"[PROCTORING] Opened new temporary segment file: {os.path.basename(temp_path)}")

                # Wait for frames with timeout - use asyncio.wait() like backup.py
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
                                # Log frame receipt periodically (every 30 seconds)
                                if time.time() - last_frame_log >= 30:
                                    print(f"[PROCTORING] ✓ Receiving video frames - Frames written: {self._video_frame_count}")
                                    last_frame_log = time.time()
                            else:
                                self._write_audio_frame(container, a_stream, item)
                        except Exception as e:
                            print(f"[PROCTORING] Error processing frame: {e}")
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
            print("[PROCTORING] Writer loop exiting, closing container...")
            try:
                if 'container' in locals() and container:
                    self._close_container(container)
                # Cleanup final segment if exists
                if 'temp_path' in locals() and temp_path and os.path.exists(temp_path):
                    asyncio.create_task(self._analyze_and_cleanup(temp_path))
            except Exception as e:
                print(f"[ERROR] Error closing container: {e}")
        print(f"[PROCTORING] 🛑 Writer loop STOPPED for {self.participant.identity or self.participant.sid}")

    def _open_container(self, start_ts: float):
        # Create temporary file
        temp_fd, temp_path = tempfile.mkstemp(suffix='.mp4', prefix='proctoring_', dir=None)
        os.close(temp_fd)  # Close file descriptor, we'll use the path
        
        # Convert to absolute path with forward slashes for PyAV/FFmpeg
        abs_path = os.path.abspath(temp_path).replace("\\", "/")
        container = av.open(abs_path, mode="w", format="mp4")

        # Video stream (libx264, yuv420p)
        v_stream = container.add_stream("libx264", rate=30)
        v_stream.pix_fmt = "yuv420p"
        v_stream.time_base = Fraction(1, 30)

        # Audio stream (aac) if audio available
        a_stream = None
        has_audio = self.audio_stream is not None
        if has_audio:
            a_stream = container.add_stream("aac", rate=48000)
            try:
                a_stream.time_base = Fraction(1, 48000)
            except Exception:
                pass
            try:
                self._audio_resampler = av.AudioResampler(format="fltp", layout="mono", rate=48000)
                print(f"[PROCTORING] Audio resampler initialized: fltp/mono/48000")
            except Exception as e:
                self._audio_resampler = None
                print(f"[PROCTORING] Warning: failed to initialize audio resampler: {e}")
            print(f"[PROCTORING] Container opened WITH audio stream")
        else:
            print(f"[PROCTORING] Container opened WITHOUT audio stream")

        return container, v_stream, a_stream, abs_path

    def _close_container(self, container: av.container.OutputContainer) -> None:
        try:
            video_streams = len([s for s in container.streams if s.type == 'video'])
            audio_streams = len([s for s in container.streams if s.type == 'audio'])
            print(f"[PROCTORING] Closing container - Video streams: {video_streams}, Audio streams: {audio_streams}, Audio frames written: {getattr(self, '_audio_frames_written', 0)}")
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
            
            # Use simple frame counter for PTS (monotonically increasing)
            av_frame.pts = self._video_frame_count
            self._video_frame_count += 1
            
            try:
                for packet in v_stream.encode(av_frame):
                    if packet is None:
                        continue
                    if packet.pts is not None and packet.pts < 0:
                        print(f"[WARNING] Skipping video packet with negative PTS: {packet.pts}")
                        continue
                    container.mux(packet)
            except Exception as e:
                print(f"[ERROR] Failed to encode/mux video packet: {e}")
        except Exception as e:
            print(f"Error writing video frame: {e}")

    def _write_audio_frame(self, container: av.container.OutputContainer, a_stream: Optional[av.audio.stream.AudioStream], item: AudioFrameItem) -> None:
        try:
            if a_stream is None:
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
            
            try:
                arr = np.frombuffer(raw, dtype=np.int16)
                total_samples = len(arr)
                
                if channels > 1:
                    num_samples = total_samples // channels
                    arr = arr.reshape((num_samples, channels)).T
                    arr = arr[:1, :]  # Downmix to mono
                    channels = 1
                else:
                    num_samples = total_samples
                    arr = arr.reshape((1, num_samples))
                    
            except Exception as e:
                print(f"[PROCTORING] Failed to reshape audio buffer: {e}")
                return

            layout = "mono"
            target_sample_rate = 48000
            needs_resample = int(sample_rate) != target_sample_rate
            
            if needs_resample:
                if self._audio_resampler is None:
                    print(f"[WARNING] Sample rate mismatch but no resampler available")
                    return
            
            num_samples = getattr(af, "_derived_num_samples", 0)
            if num_samples <= 0:
                num_samples = (len(raw) // 2) // channels if channels > 0 else len(raw) // 2
            
            in_frame = av.AudioFrame.from_ndarray(arr, format="s16", layout=layout)
            in_frame.sample_rate = int(sample_rate)
            in_frame.pts = self._audio_sample_count
            in_frame.time_base = Fraction(1, int(sample_rate))
            
            self._audio_sample_count += num_samples

            if needs_resample and self._audio_resampler is not None:
                try:
                    out_frames = self._audio_resampler.resample(in_frame)
                    if not out_frames:
                        return
                except Exception as e:
                    print(f"[PROCTORING] Resample failed: {e}")
                    return
            else:
                out_frames = [in_frame]
                if self._audio_resampler is not None and in_frame.format.name != "fltp":
                    try:
                        out_frames = self._audio_resampler.resample(in_frame)
                        if not out_frames:
                            out_frames = [in_frame]
                    except Exception:
                        out_frames = [in_frame]

            for out in out_frames:
                base_sample_count = self._audio_sample_count - num_samples
                audio_pts_samples = int((base_sample_count * target_sample_rate) / int(sample_rate))
                
                if audio_pts_samples < 0:
                    audio_pts_samples = 0
                    
                out.pts = audio_pts_samples
                out.time_base = Fraction(1, target_sample_rate)
                out.sample_rate = target_sample_rate
                
                try:
                    for packet in a_stream.encode(out):
                        if packet is None:
                            continue
                        if packet.pts is not None and packet.pts < 0:
                            print(f"[WARNING] Skipping packet with negative PTS: {packet.pts}")
                            continue
                        container.mux(packet)
                        self._audio_frames_written += 1
                except Exception as encode_error:
                    print(f"[ERROR] Failed to encode/mux audio packet: {encode_error}")
                    continue
        except Exception as e:
            print(f"[ERROR] Error writing audio frame: {e}")

    async def _analyze_and_cleanup(self, temp_path: str) -> None:
        """
        Analyze video segment with Gemini and clean up temporary file.
        Emits cheating alerts to frontend via LiveKit data channel.
        """
        participant_id = self.participant.identity or self.participant.sid
        print(f"[PROCTORING] Starting analysis for {os.path.basename(temp_path)} (participant: {participant_id})")
        
        if not _HAS_GENAI:
            print(f"[PROCTORING] Skipping analysis (google-generativeai not installed)")
            self._delete_temp_file(temp_path)
            return
        
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            print(f"[PROCTORING] Skipping analysis (GOOGLE_API_KEY not set)")
            self._delete_temp_file(temp_path)
            return
        
        file = None
        try:
            abs_path = os.path.abspath(temp_path)
            print(f"[PROCTORING] Analyzing file: {os.path.basename(abs_path)}")
            
            genai.configure(api_key=api_key)
            model = genai.GenerativeModel("gemini-2.5-flash")
            
            await asyncio.sleep(2)  # Wait for file to be fully closed
            
            if not os.path.exists(abs_path):
                print(f"[ERROR] File not found: {abs_path}")
                return
            
            file_size = os.path.getsize(abs_path)
            if file_size == 0:
                print(f"[ERROR] File is empty, skipping")
                self._delete_temp_file(temp_path)
                return
            
            loop = asyncio.get_event_loop()
            print(f"[PROCTORING] Uploading to Gemini...")
            
            file = await loop.run_in_executor(
                None, 
                lambda: genai.upload_file(abs_path, mime_type="video/mp4")
            )
            print(f"[PROCTORING] File uploaded: {file.name}")
            
            # Wait for file to be ACTIVE
            max_wait = 120
            wait_time = 0
            file_status = None
            
            while wait_time < max_wait:
                try:
                    file_status = await loop.run_in_executor(None, lambda: genai.get_file(file.name))
                    state_name = getattr(file_status.state, "name", str(file_status.state))
                    
                    if state_name == "ACTIVE":
                        print(f"[PROCTORING] File is ready (state: ACTIVE)")
                        break
                    elif state_name == "FAILED":
                        print(f"[ERROR] File upload failed: {state_name}")
                        self._delete_temp_file(temp_path)
                        return
                    
                    await asyncio.sleep(2)
                    wait_time += 2
                    if wait_time % 10 == 0:
                        print(f"[PROCTORING] Still waiting... ({wait_time}s)")
                except Exception as status_error:
                    print(f"[WARNING] Error checking file status: {status_error}")
                    await asyncio.sleep(2)
                    wait_time += 2
            
            if wait_time >= max_wait or (file_status and getattr(file_status.state, "name", None) != "ACTIVE"):
                print(f"[ERROR] Timeout waiting for file to become ACTIVE")
                self._delete_temp_file(temp_path)
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
            
            print("[PROCTORING] Requesting analysis from Gemini...")
            resp = await loop.run_in_executor(None, lambda: model.generate_content([prompt, file]))
            
            text = getattr(resp, "text", "").strip()
            
            cheating = False
            audio_present = False
            reasons = []
            
            try:
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
                
                start = text_clean.find("{")
                end = text_clean.rfind("}") + 1
                if start >= 0 and end > start:
                    text_clean = text_clean[start:end]
                
                data = json.loads(text_clean)
                cheating = bool(data.get("cheating", False))
                audio_present = bool(data.get("audio_present", False))
                reasons = data.get("reasons", [])
                if not isinstance(reasons, list):
                    reasons = []
            except Exception as e:
                print(f"[WARNING] Failed to parse JSON response: {e}")
                text_lower = text.lower()
                cheating = any(word in text_lower for word in ["cheating", "cheat", "suspicious", "violation"])
                reasons = [f"Parse error: {str(e)[:50]}"]
            
            # Log results
            print("\n" + "="*70)
            if cheating:
                reasons_str = ", ".join(reasons) if reasons else "Unknown reasons"
                print(f"[ALERT] ⚠️  CHEATING DETECTED for participant '{participant_id}'")
                print(f"        Reasons: {reasons_str}")
            else:
                print(f"[OK] ✓ No cheating detected for participant '{participant_id}'")
                if reasons:
                    print(f"        Notes: {', '.join(reasons)}")
            print("="*70 + "\n")
            
            # Emit message to frontend via LiveKit data channel
            await self._emit_proctoring_result(cheating, reasons, audio_present, participant_id)
            
            # Clean up uploaded file on Gemini
            if file:
                try:
                    await loop.run_in_executor(None, lambda: genai.delete_file(file.name))
                except Exception as cleanup_error:
                    print(f"[WARNING] Failed to cleanup uploaded file: {cleanup_error}")
            
            # Delete temporary file
            self._delete_temp_file(temp_path)
                    
        except Exception as e:
            print(f"[ERROR] Analysis failed: {e}")
            import traceback
            traceback.print_exc()
            self._delete_temp_file(temp_path)

    async def _emit_proctoring_result(self, cheating: bool, reasons: list, audio_present: bool, participant_id: str) -> None:
        """
        Emit proctoring result to frontend via LiveKit data channel.
        Uses room.local_participant.publish_data() to send JSON messages.
        Frontend can subscribe to data messages with topic "proctoring".
        """
        try:
            # Prepare message payload
            message = {
                "type": "proctoring_result",
                "participant_id": participant_id,
                "cheating": cheating,
                "audio_present": audio_present,
                "reasons": reasons,
                "timestamp": time.time()
            }
            
            # Convert to JSON string (publish_data accepts both str and bytes)
            message_json = json.dumps(message)
            
            # Publish data to room via local_participant
            local_participant = self.room.local_participant
            if local_participant:
                # publish_data accepts Union[bytes, str] and converts str to utf-8
                await local_participant.publish_data(
                    message_json,
                    topic="proctoring",  # Frontend can filter by topic
                    reliable=True  # Ensure delivery
                )
                print(f"[PROCTORING] ✓ Emitted proctoring result to frontend: cheating={cheating}, participant={participant_id}")
            else:
                print(f"[WARNING] local_participant is None, cannot emit proctoring result")
        except Exception as e:
            print(f"[ERROR] Failed to emit proctoring result: {e}")
            import traceback
            traceback.print_exc()

    def _delete_temp_file(self, temp_path: str) -> None:
        """Delete temporary file safely."""
        try:
            if temp_path and os.path.exists(temp_path):
                os.remove(temp_path)
                print(f"[PROCTORING] Deleted temporary file: {os.path.basename(temp_path)}")
        except Exception as e:
            print(f"[WARNING] Failed to delete temporary file {temp_path}: {e}")

    async def enable_audio(self, track: rtc.Track) -> None:
        """Attach audio mid-run and rotate to start muxing with audio on next segment."""
        try:
            self.audio_track = track
            self.audio_stream = rtc.AudioStream.from_track(track=self.audio_track, sample_rate=48000, num_channels=1)
            if self._writer_task is not None:
                self._reader_tasks.add(asyncio.create_task(self._audio_reader(), name="audio_reader"))
                print("[PROCTORING] Audio reader task started")
            self._rotate_now = True
            print("[PROCTORING] Audio enabled; will rotate to include audio in next segment")
        except Exception as e:
            print(f"[PROCTORING] Failed to enable audio: {e}")
            import traceback
            traceback.print_exc()

