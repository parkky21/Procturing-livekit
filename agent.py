import asyncio
from dotenv import load_dotenv
load_dotenv()
import os
from livekit import agents
from livekit.agents.voice import AgentSession, Agent
from livekit.plugins import google, deepgram, silero, openai
from livekit import rtc
import cv2
import numpy as np

async def entrypoint(ctx: agents.JobContext):
    await ctx.connect()

    # Video setup
    WIDTH = 640
    HEIGHT = 480

    video_source = rtc.VideoSource(WIDTH, HEIGHT)
    video_track = rtc.LocalVideoTrack.create_video_track("example-track", video_source)
    video_options = rtc.TrackPublishOptions(
        source=rtc.TrackSource.SOURCE_CAMERA,
        simulcast=True,
        video_encoding=rtc.VideoEncoding(
            max_framerate=30,
            max_bitrate=3_000_000,
        ),
        video_codec=rtc.VideoCodec.H264,
    )
    await ctx.agent.publish_track(video_track, video_options)

    # Note: Audio for agent TTS is automatically handled by AgentSession
    # We don't need to manually publish an audio track for the agent
    
    COLOR = [255, 255, 0, 0]  # Red color
    frame_viewers = {}
    
    def convert_frame_to_numpy(frame: rtc.VideoFrame) -> np.ndarray:
        """Convert LiveKit VideoFrame to numpy array for OpenCV"""
        data = frame.data
        buffer_type = frame.type
        data_size = len(data)
        expected_pixels = frame.width * frame.height
        
        try:
            if buffer_type == rtc.VideoBufferType.RGBA:
                rgba = np.frombuffer(data, dtype=np.uint8).reshape((frame.height, frame.width, 4))
                return cv2.cvtColor(rgba, cv2.COLOR_RGBA2BGR)
            elif buffer_type == rtc.VideoBufferType.RGB24:
                rgb = np.frombuffer(data, dtype=np.uint8).reshape((frame.height, frame.width, 3))
                return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            elif buffer_type == rtc.VideoBufferType.ABGR:
                abgr = np.frombuffer(data, dtype=np.uint8).reshape((frame.height, frame.width, 4))
                return abgr[:, :, [2, 1, 0]]  # ABGR -> BGR
            elif buffer_type == rtc.VideoBufferType.ARGB:
                argb = np.frombuffer(data, dtype=np.uint8).reshape((frame.height, frame.width, 4))
                return argb[:, :, [3, 2, 1]]  # ARGB -> BGR
            elif buffer_type == rtc.VideoBufferType.BGRA:
                bgra = np.frombuffer(data, dtype=np.uint8).reshape((frame.height, frame.width, 4))
                return bgra[:, :, [0, 1, 2]]  # BGRA -> BGR
            elif buffer_type == rtc.VideoBufferType.I420:
                # I420: Y plane full size, U and V planes at half resolution
                # OpenCV expects I420 data directly - it's already in the right format
                # But we need to ensure the size matches
                if data_size == int(expected_pixels * 1.5):
                    # Use OpenCV's direct I420 to BGR conversion
                    yuv_array = np.frombuffer(data, dtype=np.uint8)
                    # Create temporary I420 image and convert
                    height = frame.height
                    width = frame.width
                    y = yuv_array[:width * height].reshape((height, width))
                    uv_size = (width // 2) * (height // 2)
                    u = yuv_array[width * height:width * height + uv_size].reshape((height // 2, width // 2))
                    v = yuv_array[width * height + uv_size:].reshape((height // 2, width // 2))
                    # Upsample U and V to full resolution
                    u_full = cv2.resize(u, (width, height), interpolation=cv2.INTER_LINEAR)
                    v_full = cv2.resize(v, (width, height), interpolation=cv2.INTER_LINEAR)
                    # Merge and convert
                    yuv_full = cv2.merge([y, u_full, v_full])
                    return cv2.cvtColor(yuv_full, cv2.COLOR_YUV2BGR)
                else:
                    # Fallback: just use Y plane as grayscale
                    y = np.frombuffer(data[:expected_pixels], dtype=np.uint8).reshape((frame.height, frame.width))
                    return cv2.cvtColor(y, cv2.COLOR_GRAY2BGR)
            else:
                # Fallback: calculate channels from data size
                bytes_per_pixel = data_size / expected_pixels if expected_pixels > 0 else 0
                
                if abs(bytes_per_pixel - 1.5) < 0.1:  # I420 format (YUV 4:2:0)
                    # Handle as I420
                    yuv_array = np.frombuffer(data, dtype=np.uint8)
                    height, width = frame.height, frame.width
                    y = yuv_array[:width * height].reshape((height, width))
                    uv_size = (width // 2) * (height // 2)
                    if len(yuv_array) >= width * height + uv_size * 2:
                        u = yuv_array[width * height:width * height + uv_size].reshape((height // 2, width // 2))
                        v = yuv_array[width * height + uv_size:width * height + uv_size * 2].reshape((height // 2, width // 2))
                        u_full = cv2.resize(u, (width, height), interpolation=cv2.INTER_LINEAR)
                        v_full = cv2.resize(v, (width, height), interpolation=cv2.INTER_LINEAR)
                        yuv_full = cv2.merge([y, u_full, v_full])
                        return cv2.cvtColor(yuv_full, cv2.COLOR_YUV2BGR)
                    else:
                        return cv2.cvtColor(y, cv2.COLOR_GRAY2BGR)
                elif bytes_per_pixel == 4:
                    arr = np.frombuffer(data, dtype=np.uint8).reshape((frame.height, frame.width, 4))
                    return cv2.cvtColor(arr, cv2.COLOR_RGBA2BGR)
                elif bytes_per_pixel == 3:
                    arr = np.frombuffer(data, dtype=np.uint8).reshape((frame.height, frame.width, 3))
                    return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
                elif bytes_per_pixel == 1 or bytes_per_pixel == 0:
                    # Grayscale
                    y = np.frombuffer(data[:expected_pixels], dtype=np.uint8).reshape((frame.height, frame.width))
                    return cv2.cvtColor(y, cv2.COLOR_GRAY2BGR)
                else:
                    # Unknown format, try to extract first expected_pixels as grayscale
                    return cv2.cvtColor(
                        np.frombuffer(data[:expected_pixels], dtype=np.uint8).reshape((frame.height, frame.width)),
                        cv2.COLOR_GRAY2BGR
                    )
        except Exception:
            # Final fallback: return blank frame
            return np.zeros((frame.height, frame.width, 3), dtype=np.uint8)

    async def _draw_color():
        """Generate and publish video frames"""
        argb_frame = bytearray(WIDTH * HEIGHT * 4)
        while True:
            await asyncio.sleep(0.1)  # 10 fps
            argb_frame[:] = COLOR * WIDTH * HEIGHT
            frame = rtc.VideoFrame(WIDTH, HEIGHT, rtc.VideoBufferType.RGBA, argb_frame)
            video_source.capture_frame(frame)

    asyncio.create_task(_draw_color())
    
    async def display_video_frames(participant: rtc.RemoteParticipant, track: rtc.VideoTrack):
        """Display video frames from a remote participant"""
        participant_id = participant.identity or participant.sid
        window_name = f"Video - {participant_id}"
        
        stream = rtc.VideoStream.from_track(track=track)
        frame_viewers[participant_id] = {"window_name": window_name, "stream": stream}
        
        async for event in stream:
            try:
                frame_np = convert_frame_to_numpy(event.frame)
                cv2.imshow(window_name, frame_np)
                cv2.waitKey(1)
            except Exception as e:
                print(f"Error displaying frame for {participant_id}: {e}")
    
    def handle_track_subscribed(track: rtc.Track, publication: rtc.TrackPublication, participant: rtc.RemoteParticipant):
        """Handle new track subscription"""
        if track.kind == rtc.TrackKind.KIND_VIDEO and publication.source == rtc.TrackSource.SOURCE_CAMERA:
            asyncio.create_task(display_video_frames(participant, track))
        elif track.kind == rtc.TrackKind.KIND_AUDIO and publication.source == rtc.TrackSource.SOURCE_MICROPHONE:
            print(f"Audio track subscribed for participant: {participant.identity or participant.sid}")
            # Audio tracks are automatically handled by LiveKit SDK - no need for manual processing
    
    def on_participant_connected(participant: rtc.RemoteParticipant):
        """Handle new participant connection"""
        for publication in participant.track_publications.values():
            # Subscribe to video tracks
            if publication.kind == rtc.TrackKind.KIND_VIDEO and publication.source == rtc.TrackSource.SOURCE_CAMERA:
                if publication.subscribed and publication.track:
                    asyncio.create_task(display_video_frames(participant, publication.track))
                else:
                    asyncio.create_task(publication.set_subscribed(True))
            # Subscribe to audio tracks so we can hear participants
            elif publication.kind == rtc.TrackKind.KIND_AUDIO and publication.source == rtc.TrackSource.SOURCE_MICROPHONE:
                if not publication.subscribed:
                    print(f"Subscribing to audio track for participant: {participant.identity or participant.sid}")
                    asyncio.create_task(publication.set_subscribed(True))
    
    def on_participant_disconnected(participant: rtc.RemoteParticipant):
        """Handle participant disconnection"""
        participant_id = participant.identity or participant.sid
        if participant_id in frame_viewers:
            viewer_info = frame_viewers[participant_id]
            cv2.destroyWindow(viewer_info["window_name"])
            asyncio.create_task(viewer_info["stream"].aclose())
            del frame_viewers[participant_id]
    
    ctx.room.on("participant_connected", on_participant_connected)
    ctx.room.on("participant_disconnected", on_participant_disconnected)
    ctx.room.on("track_subscribed", handle_track_subscribed)
    
    for participant in ctx.room.remote_participants.values():
        on_participant_connected(participant)
    
    agent = Agent(
        instructions="""
        Your friedrich Nietzeshe, a german philosopher
        You speak in most brutal way possible
        You are having a conservation with a human
        Speak in the way how Nietzeshe would speak
        Your speaking to genz kid and help him become ubermensch the strongest version of himself
        """,
        vad=silero.VAD.load(),
        stt=deepgram.STT(model="nova-2-general", language="en"),
        llm=openai.LLM(model="gpt-4o-mini"),
        tts=google.TTS(
            voice_name="en-IN-Chirp3-HD-Charon",
            language="en-IN",
            credentials_file="G_cred.json"
        ),
    )

    session = AgentSession()
    try:
        await session.start(agent=agent, room=ctx.room)
        await session.generate_reply()
    finally:
        cv2.destroyAllWindows()

if __name__ == "__main__":
    agents.cli.run_app(agents.WorkerOptions(entrypoint_fnc=entrypoint))
