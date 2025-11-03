import asyncio
import os
import time
from dataclasses import dataclass
from typing import Optional, List
import numpy as np
import cv2

from livekit import rtc

try:
    import google.generativeai as genai  # type: ignore
    _HAS_GENAI = True
except Exception:
    _HAS_GENAI = False


@dataclass
class FrameBuffer:
    """Buffer to collect frames for analysis"""
    frames: List[rtc.VideoFrame]
    start_time: float
    participant_id: str


class ParticipantRecorder:
    def __init__(
        self,
        participant: rtc.RemoteParticipant,
        segment_seconds: int = 60,
        frames_per_analysis: int = 10,  # Sample N frames per minute
    ) -> None:
        self.participant = participant
        self.segment_seconds = segment_seconds
        self.frames_per_analysis = frames_per_analysis

        self.video_track: Optional[rtc.VideoTrack] = None
        self.video_stream: Optional[rtc.VideoStream] = None

        self._reader_task: Optional[asyncio.Task] = None
        self._analyzer_task: Optional[asyncio.Task] = None

        # Buffer for collecting frames
        self._frame_buffer: List[rtc.VideoFrame] = []
        self._buffer_start_time: float = 0

        # Control
        self._stop_event = asyncio.Event()

    async def start_if_ready(self) -> None:
        if self._reader_task is not None:
            return
        if self.video_track is None:
            return

        # Create video stream from track
        self.video_stream = rtc.VideoStream.from_track(track=self.video_track)

        # Launch video reader
        self._reader_task = asyncio.create_task(self._video_reader(), name="video_reader")
        
        # Launch analyzer
        self._analyzer_task = asyncio.create_task(self._analyzer_loop(), name="analyzer_loop")

    async def stop(self) -> None:
        self._stop_event.set()
        # Close stream
        if self.video_stream is not None:
            await self.video_stream.aclose()
            self.video_stream = None
        # Cancel tasks
        if self._reader_task:
            self._reader_task.cancel()
        if self._analyzer_task:
            self._analyzer_task.cancel()
        self._reader_task = None
        self._analyzer_task = None

    async def _video_reader(self) -> None:
        """Collect video frames into buffer"""
        assert self.video_stream is not None
        async for event in self.video_stream:
            if self._stop_event.is_set():
                break
            self._frame_buffer.append(event.frame)
            
            # Start timer on first frame
            if len(self._frame_buffer) == 1:
                self._buffer_start_time = time.time()

    async def _analyzer_loop(self) -> None:
        """Analyze frames every segment_seconds interval"""
        while not self._stop_event.is_set():
            await asyncio.sleep(self.segment_seconds)
            
            if not self._frame_buffer:
                continue
            
            # Get sample frames for analysis
            buffer_len = len(self._frame_buffer)
            if buffer_len == 0:
                continue
                
            # Sample frames evenly across the buffer
            indices = np.linspace(0, buffer_len - 1, min(self.frames_per_analysis, buffer_len), dtype=int)
            sample_frames = [self._frame_buffer[i] for i in indices]
            
            # Clear buffer for next interval
            self._frame_buffer.clear()
            
            # Analyze asynchronously
            participant_id = self.participant.identity or self.participant.sid
            asyncio.create_task(self._analyze_frames(sample_frames, participant_id))

    async def _analyze_frames(self, frames: List[rtc.VideoFrame], participant_id: str) -> None:
        """Send frames to Gemini for proctoring analysis"""
        if not _HAS_GENAI:
            print(f"[recording] Skipping analysis (google-generativeai not installed) for {participant_id}")
            return
        
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            print(f"[recording] Skipping analysis (GOOGLE_API_KEY not set) for {participant_id}")
            return
        
        try:
            genai.configure(api_key=api_key)
            model = genai.GenerativeModel("gemini-2.5-flash")
            
            # Convert frames to images
            images = []
            for frame in frames:
                img = self._frame_to_image(frame)
                if img is not None:
                    # Wrap raw JPEG bytes in the format expected by the SDK
                    images.append({"mime_type": "image/jpeg", "data": img})
            
            if not images:
                print(f"[recording] No valid frames to analyze for {participant_id}")
                return
            
            prompt = (
                "You are a proctoring assistant analyzing exam webcam footage. "
                "Examine these frames and determine if the participant is cheating. "
                "Return ONLY valid JSON with this exact structure:\n"
                '{"cheating": true/false, "reasons": ["reason1", "reason2"]}\n'
                "Look for: looking away from screen, multiple people visible, phone/device usage, "
                "suspicious hand movements, screen reflections, frequently going out of frame, "
                "talking to someone off-camera, reading from materials."
            )
            
            # Run in executor to avoid blocking
            loop = asyncio.get_event_loop()
            content = [prompt] + images
            resp = await loop.run_in_executor(None, lambda: model.generate_content(content))
            
            text = getattr(resp, "text", "").strip()
            
            # Parse response
            cheating = False
            reasons = []
            
            # Try to extract JSON from response
            import json as _json
            try:
                # Find JSON in response (might have markdown code blocks)
                if "```" in text:
                    # Extract JSON from code block
                    start = text.find("{")
                    end = text.rfind("}") + 1
                    if start >= 0 and end > start:
                        text = text[start:end]
                
                data = _json.loads(text)
                cheating = bool(data.get("cheating", False))
                reasons = data.get("reasons", [])
                if not isinstance(reasons, list):
                    reasons = []
            except Exception as e:
                # Fallback: check if response mentions cheating
                text_lower = text.lower()
                cheating = any(word in text_lower for word in ["cheating", "cheat", "suspicious", "violation"])
                reasons = [f"Parse error: {str(e)[:50]}. Response: {text[:200]}"]
            
            # Log results
            if cheating:
                reasons_str = ", ".join(reasons) if reasons else "Unknown reasons"
                print(f"[ALERT] ⚠️  CHEATING DETECTED for participant '{participant_id}'")
                print(f"        Reasons: {reasons_str}")
            else:
                print(f"[OK] ✓ No cheating detected for participant '{participant_id}'")
                
        except Exception as e:
            print(f"[ERROR] Analysis failed for {participant_id}: {e}")

    def _frame_to_image(self, frame: rtc.VideoFrame) -> Optional[bytes]:
        """Convert LiveKit VideoFrame to JPEG bytes for Gemini"""
        try:
            # Convert frame to numpy array (BGR format)
            data = frame.data
            buffer_type = frame.type
            
            if buffer_type == rtc.VideoBufferType.RGBA:
                rgba = np.frombuffer(data, dtype=np.uint8).reshape((frame.height, frame.width, 4))
                bgr = cv2.cvtColor(rgba, cv2.COLOR_RGBA2BGR)
            elif buffer_type == rtc.VideoBufferType.RGB24:
                rgb = np.frombuffer(data, dtype=np.uint8).reshape((frame.height, frame.width, 3))
                bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            elif buffer_type == rtc.VideoBufferType.I420:
                # Handle I420
                data_size = len(data)
                expected_pixels = frame.width * frame.height
                if data_size >= int(expected_pixels * 1.5):
                    yuv_array = np.frombuffer(data, dtype=np.uint8)
                    height, width = frame.height, frame.width
                    y = yuv_array[:width * height].reshape((height, width))
                    uv_size = (width // 2) * (height // 2)
                    u = yuv_array[width * height:width * height + uv_size].reshape((height // 2, width // 2))
                    v = yuv_array[width * height + uv_size:].reshape((height // 2, width // 2))
                    u_full = cv2.resize(u, (width, height), interpolation=cv2.INTER_LINEAR)
                    v_full = cv2.resize(v, (width, height), interpolation=cv2.INTER_LINEAR)
                    yuv_full = cv2.merge([y, u_full, v_full])
                    bgr = cv2.cvtColor(yuv_full, cv2.COLOR_YUV2BGR)
                else:
                    y = np.frombuffer(data[:expected_pixels], dtype=np.uint8).reshape((frame.height, frame.width))
                    bgr = cv2.cvtColor(y, cv2.COLOR_GRAY2BGR)
            else:
                # Try to convert using built-in method if available
                try:
                    rgb_frame = frame.convert(rtc.VideoBufferType.RGB24)
                    rgb = np.frombuffer(rgb_frame.data, dtype=np.uint8).reshape((frame.height, frame.width, 3))
                    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
                except:
                    return None
            
            # Encode as JPEG
            _, jpeg_bytes = cv2.imencode('.jpg', bgr, [cv2.IMWRITE_JPEG_QUALITY, 85])
            return jpeg_bytes.tobytes()
            
        except Exception as e:
            print(f"Error converting frame to image: {e}")
            return None
