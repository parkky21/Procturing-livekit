import asyncio
import os
from typing import Dict
from dotenv import load_dotenv
from livekit import agents, rtc
from livekit.agents.voice import AgentSession, Agent
from livekit.plugins import google, deepgram, silero, openai

from recording import ParticipantRecorder

load_dotenv()

async def entrypoint(ctx: agents.JobContext) -> None:
    await ctx.connect()

    segment_seconds = int(os.getenv("RECORD_SEGMENT_SECONDS", "60"))

    recorders: Dict[str, ParticipantRecorder] = {}

    def get_id(p: rtc.RemoteParticipant) -> str:
        return p.identity or p.sid

    async def ensure_recorder(p: rtc.RemoteParticipant) -> ParticipantRecorder:
        pid = get_id(p)
        if pid not in recorders:
            recorders[pid] = ParticipantRecorder(
                participant=p,
                segment_seconds=segment_seconds,
            )
        return recorders[pid]

    def on_participant_connected(p: rtc.RemoteParticipant) -> None:
        # Attempt to attach existing publications
        for pub in p.track_publications.values():
            _maybe_attach(pub, p)

    def on_participant_disconnected(p: rtc.RemoteParticipant) -> None:
        pid = get_id(p)
        rec = recorders.pop(pid, None)
        if rec:
            asyncio.create_task(rec.stop())

    def on_track_subscribed(track: rtc.Track, pub: rtc.TrackPublication, p: rtc.RemoteParticipant) -> None:
        _maybe_attach(pub, p)

    def _maybe_attach(pub: rtc.TrackPublication, p: rtc.RemoteParticipant) -> None:
        # Only need video for proctoring analysis
        if pub.kind == rtc.TrackKind.KIND_VIDEO and pub.source == rtc.TrackSource.SOURCE_CAMERA and pub.track:
            async def set_video() -> None:
                rec = await ensure_recorder(p)
                rec.video_track = pub.track  # type: ignore
                await rec.start_if_ready()

            asyncio.create_task(set_video())

    # Wire events
    ctx.room.on("participant_connected", on_participant_connected)
    ctx.room.on("participant_disconnected", on_participant_disconnected)
    ctx.room.on("track_subscribed", on_track_subscribed)

    # Start agent session (optional; keep if you want the agent active while recording)
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
        pass
    # Attach existing remote participants
    for p in ctx.room.remote_participants.values():
        on_participant_connected(p)

    # Keep running
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


