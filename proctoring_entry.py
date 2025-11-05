import asyncio
import os
from typing import Dict
from dotenv import load_dotenv
from livekit import agents, rtc
from livekit.agents.voice import AgentSession, Agent
from livekit.plugins import google, deepgram, silero, openai
from procturing import Procturing

load_dotenv()

async def entrypoint(ctx: agents.JobContext) -> None:
    """
    Entrypoint for proctoring agent.
    Monitors participant video/audio, sends segments to Gemini for analysis,
    and emits cheating alerts to frontend via LiveKit data channels.
    """
    await ctx.connect()

    segment_seconds = int(os.getenv("PROCTORING_SEGMENT_SECONDS", "60"))
    print(f"[ENTRY] proctoring_entry started. segment_seconds={segment_seconds}")

    proctoring_instances: Dict[str, Procturing] = {}

    def get_id(p: rtc.RemoteParticipant) -> str:
        return p.identity or p.sid

    async def ensure_proctoring(p: rtc.RemoteParticipant) -> Procturing:
        pid = get_id(p)
        if pid not in proctoring_instances:
            proctoring_instances[pid] = Procturing(
                participant=p,
                room=ctx.room,
                segment_seconds=segment_seconds,
            )
            print(f"[PROCTORING] Created proctoring instance for participant={pid}")
        return proctoring_instances[pid]

    def on_participant_connected(p: rtc.RemoteParticipant) -> None:
        print(f"[ROOM] participant_connected: {get_id(p)}")
        # Attempt to attach existing publications
        for pub in p.track_publications.values():
            _maybe_attach(pub, p)

    def on_participant_disconnected(p: rtc.RemoteParticipant) -> None:
        print(f"[ROOM] participant_disconnected: {get_id(p)}")
        pid = get_id(p)
        proc = proctoring_instances.pop(pid, None)
        if proc:
            asyncio.create_task(proc.stop())

    def on_track_subscribed(track: rtc.Track, pub: rtc.TrackPublication, p: rtc.RemoteParticipant) -> None:
        print(f"[ROOM] track_subscribed: kind={pub.kind} source={pub.source} participant={get_id(p)}")
        _maybe_attach(pub, p)

    def _maybe_attach(pub: rtc.TrackPublication, p: rtc.RemoteParticipant) -> None:
        """
        Attach video/audio tracks to proctoring instance.
        Video is required, audio is optional.
        """
        # Video is required, audio is optional
        if pub.kind == rtc.TrackKind.KIND_VIDEO and pub.source == rtc.TrackSource.SOURCE_CAMERA and pub.track:
            async def set_video() -> None:
                proc = await ensure_proctoring(p)
                proc.video_track = pub.track  # type: ignore
                # If audio track already exists, attach it now
                if proc.audio_track is not None:
                    print(f"[PROCTORING] Video arrived, audio already exists, enabling audio...")
                    await proc.enable_audio(proc.audio_track)
                await proc.start_if_ready()

            asyncio.create_task(set_video())
        elif pub.kind == rtc.TrackKind.KIND_AUDIO and pub.source == rtc.TrackSource.SOURCE_MICROPHONE and pub.track:
            async def set_audio() -> None:
                proc = await ensure_proctoring(p)
                print(f"[PROCTORING] Audio track received for {get_id(p)} - track: {pub.track.sid}")
                if proc.video_track is not None:
                    print(f"[PROCTORING] Video track exists, calling enable_audio()...")
                    await proc.enable_audio(pub.track)  # type: ignore
                    print(f"[PROCTORING] enable_audio() completed")
                else:
                    print(f"[PROCTORING] Audio arrived before video for {get_id(p)}; waiting for video")
                    # Store audio track for later
                    proc.audio_track = pub.track  # type: ignore

            asyncio.create_task(set_audio())

    # Wire room events
    ctx.room.on("participant_connected", on_participant_connected)
    ctx.room.on("participant_disconnected", on_participant_disconnected)
    ctx.room.on("track_subscribed", on_track_subscribed)

    # Optional: Start agent session for voice interaction
    # Uncomment if you want the agent to also have voice capabilities

    # agent = Agent(
    #     instructions=
    #     """
    #     Your friedrich Nietzeshe, a german philosopher
    #     You speak in most brutal way possible
    #     You are having a conservation with a human
    #     Speak in the way how Nietzeshe would speak
    #     Your speaking to genz kid and help him become ubermensch the strongest version of himself
    #     """,
    #     vad=silero.VAD.load(),
    #     stt=deepgram.STT(model="nova-2-general", language="en"),
    #     llm=openai.LLM(model="gpt-4o-mini"),
    #     tts=google.TTS(
    #         voice_name="en-IN-Chirp3-HD-Charon",
    #         language="en-IN",
    #         credentials_file="G_cred.json"
    #     ),
    # )
    # session = AgentSession()
    # try:
    #     await session.start(agent=agent, room=ctx.room)
    #     await session.generate_reply()
    # finally:
    #     pass

    
    # Attach existing remote participants
    for p in ctx.room.remote_participants.values():
        on_participant_connected(p)

    # Keep running
    print("[ENTRY] Proctoring agent is running. Waiting for participants...")
    while True:
        await asyncio.sleep(3600)


if __name__ == "__main__":
    LIVEKIT_API_KEY = os.getenv("LIVEKIT_API_KEY")
    LIVEKIT_API_SECRET = os.getenv("LIVEKIT_API_SECRET")
    LIVEKIT_WS_URL = os.getenv("LIVEKIT_WS_URL") or os.getenv("LIVEKIT_URL")

    # Only pass explicit credentials if all are provided; otherwise let the SDK read env vars
    if LIVEKIT_API_KEY and LIVEKIT_API_SECRET and LIVEKIT_WS_URL:
        opts = agents.WorkerOptions(
            entrypoint_fnc=entrypoint,
            ws_url=LIVEKIT_WS_URL,
            api_key=LIVEKIT_API_KEY,
            api_secret=LIVEKIT_API_SECRET,
        )
    else:
        opts = agents.WorkerOptions(entrypoint_fnc=entrypoint)

    agents.cli.run_app(opts)

