from dotenv import load_dotenv
import requests
load_dotenv()
from datetime import datetime
import json
import os
import uuid
import asyncio
from livekit import agents 
from livekit import agents ,api
from livekit.agents import RoomInputOptions
from livekit.agents.voice import AgentSession, Agent
from livekit.plugins import google, deepgram, silero, openai
from exam import evaluate_candidate
from text import jd, evaluation_template, interview_template, resume
from urllib.parse import quote_plus

LIVEKIT_API_KEY = os.getenv("LIVEKIT_API_KEY")
LIVEKIT_API_SECRET = os.getenv("LIVEKIT_API_SECRET")
LIVEKIT_WS_URL = os.getenv("LIVEKIT_WS_URL", "wss://alice-sjgt0raw.livekit.cloud")
MEET_HOST = "https://meet.livekit.io/custom"
TINYURL_API_KEY= os.getenv("TINYURL_API_KEY")

async def entrypoint(ctx: agents.JobContext):

    async def write_transcript():
        current_date = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Ensure directory exists
        save_dir = "transcriptions"   # relative to project root
        os.makedirs(save_dir, exist_ok=True)

        filename = os.path.join(save_dir, f"transcript_{ctx.room.name}_{current_date}.json")

        with open(filename, 'w') as f:
            json.dump(session.history.to_dict(), f, indent=2)

        print(f"Transcript for {ctx.room.name} saved to {filename}")
        # result = evaluate_candidate(session.history.to_dict(),evaluation_template=evaluation_template,jd_text=jd,resume_text=resume ,save_dir="evaluations")
        # print("Evaluation Result:")
        # print("-" * 20)
        # print(json.dumps(result, indent=2))

    ctx.add_shutdown_callback(write_transcript)
    await ctx.connect()

    agent = Agent(
        instructions=
        """
        YOur an interview agent with acess to user cameras
        Prompt the user to maintain straight posture and face the camera
        if you find the user cheating 3 times terminate the interview by calling the function terminate()
        Ask the user to introduce themselves and their background
        Ask the user to explain their resume

        """,
        vad=silero.VAD.load(),
        # stt=deepgram.STT(  # Whisper model
        # model="nova-2-general",
        # language="en",
        # ),
        # llm=openai.LLM(model="gpt-4o-mini"),
        # tts=google.TTS(
        #     voice_name="en-IN-Chirp3-HD-Charon",
        #     language="en-IN",
        #     credentials_file="ciplaxalgovox-23eb479f0a6f.json"
        # ),
        llm=google.realtime.RealtimeModel(
                voice="Puck",
                temperature=0.8,
            ),
        # tools=[query_info],
    )

    session = AgentSession()
    await session.start(
        agent=agent, 
        room=ctx.room,
        room_input_options=RoomInputOptions(
            audio_enabled=True,
            video_enabled=False,
            # noise_cancellation=noise_cancellation.BVC(),
        ),
    )

    # async def _time_warning():
    #     try:
    #         await asyncio.sleep(40)  # wait 1 minute
    #         # Try to prompt participants to wrap up. This attempts to let the agent speak a reminder.
    #         try:
    #             print("Time warning: please wrap up — time is limited.")
    #             # await session.generate_reply(
    #             #     instructions="Time is running out, please conclude soon."
    #             #     )  # generate a normal reply (agent will speak)
    #             # asyncio.sleep(2)  # brief pause
    #             await session.say("Time is running out, please conclude soon.")  # direct TTS
    #         except TypeError:
    #             # If generate_reply requires context and fails, log and fallback to printing
    #             print("Time warning: please wrap up — time is limited.")
    #         except Exception as e:
    #             print("Failed to generate time warning via agent:", e)
    #             print("Time warning: please wrap up — time is limited.")
    #     except asyncio.CancelledError:
    #         return

    # # start background timer that will prompt after 1 minute
    # _warning_task = asyncio.create_task(_time_warning())

    # ensure the warning task is cancelled on shutdown
    # def _cancel_warning():
    #     _warning_task.cancel()
    # ctx.add_shutdown_callback(_cancel_warning)

    # continue normal flow
    await session.generate_reply()

# def get_tiny_url(long_url: str) -> str:
#     """
#     Uses the TinyURL API v2 to shorten a URL with an API key.
#     """
#     if not TINYURL_API_KEY:
    #     print("Warning: TINYURL_API_KEY not found in .env file. Returning long URL.")
    #     return long_url

    # api_url = "https://api.tinyurl.com/create"
    # headers = {
    #     "Authorization": f"Bearer {TINYURL_API_KEY}",
    #     "Content-Type": "application/json",
    # }
    # payload = {"url": long_url}

    # try:
    #     response = requests.post(api_url, headers=headers, json=payload)
    #     response.raise_for_status()  # Check for HTTP errors (e.g., 401 Unauthorized)
    #     data = response.json()
    #     # Safely access the nested tiny_url key
    #     return data.get("data", {}).get("tiny_url", long_url)
    # except requests.exceptions.RequestException as e:
    #     print(f"Error creating short URL with TinyURL: {e}")
    #     return long_url # Return the original URL as a fallback


if __name__ == "__main__":

    # room = f"room-{uuid.uuid4().hex[:8]}"
    # user_identity = f"user-{uuid.uuid4().hex[:6]}"
    # agent_identity = f"agent-{uuid.uuid4().hex[:6]}"
    # token = generate_token(identity=user_identity, name="Guest User", room=room)
    # meet_link = (
    #     f"{MEET_HOST}?liveKitUrl={quote_plus(LIVEKIT_WS_URL)}&token={quote_plus(token)}"
    # )
    # tiny_url = get_tiny_url(meet_link)
    # print("🎯 Share this link to join the meet:")
    # print(tiny_url)

    agents.cli.run_app(agents.WorkerOptions(entrypoint_fnc=entrypoint))
