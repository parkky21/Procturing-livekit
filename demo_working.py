import asyncio
import uuid
from dotenv import load_dotenv
load_dotenv()
import os
import requests
from livekit import agents ,api
from livekit.agents.voice import AgentSession, Agent
from livekit.plugins import google, deepgram, silero, openai
from livekit import rtc
from livekit.agents import JobContext
import cv2
import numpy as np
LIVEKIT_API_KEY = os.getenv("LIVEKIT_API_KEY")
LIVEKIT_API_SECRET = os.getenv("LIVEKIT_API_SECRET")
LIVEKIT_WS_URL = os.getenv("LIVEKIT_WS_URL", "wss://alice-sjgt0raw.livekit.cloud")

async def entrypoint(ctx: agents.JobContext):
    await ctx.connect()

    WIDTH = 640
    HEIGHT = 480

    source = rtc.VideoSource(WIDTH, HEIGHT)
    track = rtc.LocalVideoTrack.create_video_track("example-track", source)
    options = rtc.TrackPublishOptions(
        # since the agent is a participant, our video I/O is its "camera"
        source=rtc.TrackSource.SOURCE_CAMERA,
        simulcast=True,
        # when modifying encoding options, max_framerate and max_bitrate must both be set
        video_encoding=rtc.VideoEncoding(
            max_framerate=30,
            max_bitrate=3_000_000,
        ),
        video_codec=rtc.VideoCodec.H264,
    )
    publication = await ctx.agent.publish_track(track, options)

    # this color is encoded as ARGB. when passed to VideoFrame it gets re-encoded.
    COLOR = [255, 255, 0, 0]; # FFFF0000 RED
    
    # Optional: Display agent's own video frames
    SHOW_AGENT_VIDEO = False # Set to False to disable agent video preview
    agent_window_name = "Agent Video Output"
    
    # Dictionary to store video frame viewers for each participant
    frame_viewers = {}
    
    def convert_frame_to_numpy(frame: rtc.VideoFrame) -> np.ndarray:
        """Convert LiveKit VideoFrame to numpy array for OpenCV"""
        try:
            # Try to access frame data - frame.data returns memoryview
            if hasattr(frame, 'data') and frame.data is not None:
                # frame.data is a memoryview, np.frombuffer can handle it directly
                data = frame.data
            elif hasattr(frame, 'buffer'):
                data = bytes(frame.buffer)
            else:
                # Fallback: try to convert using to_ndarray if available
                if hasattr(frame, 'to_ndarray'):
                    return frame.to_ndarray(format=rtc.VideoBufferType.RGB24)
                raise ValueError("Cannot access frame data")
            
            # Convert based on frame buffer type - use 'type' property, not 'buffer_type'
            buffer_type = frame.type if hasattr(frame, 'type') else rtc.VideoBufferType.RGBA
            
            if buffer_type == rtc.VideoBufferType.RGBA:
                # RGBA format: convert to BGR for OpenCV
                rgba = np.frombuffer(data, dtype=np.uint8)
                rgba = rgba.reshape((frame.height, frame.width, 4))
                bgr = cv2.cvtColor(rgba, cv2.COLOR_RGBA2BGR)
                return bgr
            elif buffer_type == rtc.VideoBufferType.RGB24:
                # RGB24 format: convert to BGR
                rgb = np.frombuffer(data, dtype=np.uint8)
                rgb = rgb.reshape((frame.height, frame.width, 3))
                bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
                return bgr
            elif buffer_type == rtc.VideoBufferType.ABGR:
                # ABGR format: convert to BGR
                abgr = np.frombuffer(data, dtype=np.uint8)
                abgr = abgr.reshape((frame.height, frame.width, 4))
                # ABGR is Alpha, Blue, Green, Red -> we want BGR, so reverse and drop alpha
                bgr = abgr[:, :, [2, 1, 0]]
                return bgr
            elif buffer_type == rtc.VideoBufferType.ARGB:
                # ARGB format: convert to BGR
                argb = np.frombuffer(data, dtype=np.uint8)
                argb = argb.reshape((frame.height, frame.width, 4))
                # ARGB is Alpha, Red, Green, Blue -> we want BGR
                bgr = argb[:, :, [3, 2, 1]]
                return bgr
            elif buffer_type == rtc.VideoBufferType.BGRA:
                # BGRA format: convert to BGR (just drop alpha channel)
                bgra = np.frombuffer(data, dtype=np.uint8)
                bgra = bgra.reshape((frame.height, frame.width, 4))
                bgr = bgra[:, :, [0, 1, 2]]  # Take B, G, R channels
                return bgr
            elif buffer_type == rtc.VideoBufferType.I420:
                # I420 (YUV) format: convert to BGR
                yuv = np.frombuffer(data, dtype=np.uint8)
                y = yuv[:frame.width * frame.height].reshape((frame.height, frame.width))
                u = yuv[frame.width * frame.height:frame.width * frame.height * 5 // 4].reshape((frame.height // 2, frame.width // 2))
                v = yuv[frame.width * frame.height * 5 // 4:].reshape((frame.height // 2, frame.width // 2))
                # Resize U and V to match Y dimensions for cv2
                u_resized = cv2.resize(u, (frame.width, frame.height))
                v_resized = cv2.resize(v, (frame.width, frame.height))
                yuv_full = np.dstack([y, u_resized, v_resized])
                bgr = cv2.cvtColor(yuv_full, cv2.COLOR_YUV2BGR)
                return bgr
            else:
                # For other formats, try to handle as RGB
                rgb = np.frombuffer(data, dtype=np.uint8)
                rgb = rgb.reshape((frame.height, frame.width, -1))
                if rgb.shape[2] == 4:
                    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGBA2BGR)
                elif rgb.shape[2] == 3:
                    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
                else:
                    bgr = rgb
                return bgr
        except Exception as e:
            print(f"Error converting frame: {e}")
            # Return a blank frame as fallback
            return np.zeros((frame.height, frame.width, 3), dtype=np.uint8)

    async def _draw_color():
        argb_frame = bytearray(WIDTH * HEIGHT * 4)
        while True:
            await asyncio.sleep(0.1) # 10 fps
            argb_frame[:] = COLOR * WIDTH * HEIGHT
            frame = rtc.VideoFrame(WIDTH, HEIGHT, rtc.VideoBufferType.RGBA, argb_frame)

            # send this frame to the track
            source.capture_frame(frame)
            
            # Display agent's own video frames if enabled
            if SHOW_AGENT_VIDEO:
                try:
                    frame_np = convert_frame_to_numpy(frame)
                    cv2.imshow(agent_window_name, frame_np)
                    cv2.waitKey(1)
                except Exception as e:
                    pass  # Silently ignore errors during initialization

    asyncio.create_task(_draw_color())
    
    async def display_video_frames(participant: rtc.RemoteParticipant, track: rtc.VideoTrack):
        """Display video frames from a remote participant"""
        participant_id = participant.identity or participant.sid
        window_name = f"Video - {participant_id}"
        
        try:
            # Create a VideoStream from the track to receive frames
            stream = rtc.VideoStream.from_track(track=track)
            
            # Keep track of the stream for cleanup
            frame_viewers[participant_id] = {
                "window_name": window_name,
                "track": track,
                "participant": participant,
                "stream": stream
            }
            
            print(f"Started displaying video for participant: {participant_id}")
            
            # Process frames from the stream
            async for event in stream:
                try:
                    frame = event.frame
                    # Convert frame to numpy array
                    frame_np = convert_frame_to_numpy(frame)
                    
                    # Display frame using OpenCV
                    cv2.imshow(window_name, frame_np)
                    cv2.waitKey(1)  # Non-blocking wait for key press
                except Exception as e:
                    print(f"Error displaying frame for {participant_id}: {e}")
                    
        except Exception as e:
            print(f"Error setting up video stream for {participant_id}: {e}")
            # Clean up on error
            if participant_id in frame_viewers:
                del frame_viewers[participant_id]
    
    def handle_track_subscribed(track: rtc.Track, publication: rtc.TrackPublication, participant: rtc.RemoteParticipant):
        """Handle new track subscription"""
        if track.kind == rtc.TrackKind.KIND_VIDEO:
            print(f"Video track subscribed: {track.sid}, source: {publication.source}")
            # Only display camera video (not screen share)
            if publication.source == rtc.TrackSource.SOURCE_CAMERA:
                asyncio.create_task(display_video_frames(participant, track))
    
    def on_participant_connected(participant: rtc.RemoteParticipant):
        """Handle new participant connection"""
        participant_id = participant.identity or participant.sid
        print(f"Participant connected: {participant_id}")
        
        # Subscribe to existing video tracks from camera
        for publication in participant.track_publications.values():
            if publication.kind == rtc.TrackKind.KIND_VIDEO and publication.source == rtc.TrackSource.SOURCE_CAMERA:
                if publication.subscribed and publication.track:
                    print(f"Found existing video track: {publication.sid}")
                    asyncio.create_task(display_video_frames(participant, publication.track))
                else:
                    # Subscribe to the track if not already subscribed
                    print(f"Subscribing to video track: {publication.sid}")
                    asyncio.create_task(publication.set_subscribed(True))
    
    def on_participant_disconnected(participant: rtc.RemoteParticipant):
        """Handle participant disconnection"""
        participant_id = participant.identity or participant.sid
        print(f"Participant disconnected: {participant_id}")
        
        if participant_id in frame_viewers:
            viewer_info = frame_viewers[participant_id]
            window_name = viewer_info["window_name"]
            try:
                cv2.destroyWindow(window_name)
            except:
                pass
            # Close the video stream
            if "stream" in viewer_info:
                try:
                    asyncio.create_task(viewer_info["stream"].aclose())
                except:
                    pass
            del frame_viewers[participant_id]
    
    # Set up room event handlers
    if hasattr(ctx.room, 'on'):
        ctx.room.on("participant_connected", on_participant_connected)
        ctx.room.on("participant_disconnected", on_participant_disconnected)
        ctx.room.on("track_subscribed", handle_track_subscribed)
    
    # Subscribe to existing participants
    for participant in ctx.room.remote_participants.values():
        on_participant_connected(participant)
    
    agent = Agent(
        instructions=
        """
        Your friedrich Nietzeshe, a german philosopher
        You speak in most brutal way possible
        You are having a conservation with a human
        Speak in the way how Nietzeshe would speak
        Your speaking to genz kid and help him become ubermensch the strongest version of himself
        """,
        vad=silero.VAD.load(),
        stt=deepgram.STT(  # Whisper model
        model="nova-2-general",
        language="en",
        ),
        llm=openai.LLM(model="gpt-4o-mini"),
        tts=google.TTS(
            voice_name="en-IN-Chirp3-HD-Charon",
            language="en-IN",
            credentials_file="G_cred.json"
        ),
    )

    session = AgentSession()
    try:
        await session.start(
            agent=agent, 
            room=ctx.room,
            # room_input_options=room_io.RoomInputOptions(
            #     # noise_cancellation=noise_cancellation.BVC(),
            # ),
        )
        await session.generate_reply()
    finally:
        # Cleanup: Close all OpenCV windows
        cv2.destroyAllWindows()
        print("Video viewers closed")

if __name__ == "__main__":
    agents.cli.run_app(agents.WorkerOptions(entrypoint_fnc=entrypoint))
