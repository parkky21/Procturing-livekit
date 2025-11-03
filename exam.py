import json
import re
from openai import OpenAI
import os
from datetime import datetime
client = OpenAI()  # make sure OPENAI_API_KEY is set in env
MODEL = "gpt-4o-mini"


def safe_parse_json(text: str):
    try:
        return json.loads(text)
    except Exception:
        # best-effort JSON extraction
        m = re.search(r"\{(?:[^{}]|(?R))*\}", text, re.DOTALL)
        if m:
            try:
                return json.loads(m.group())
            except Exception:
                pass
        return None


def evaluate_candidate(
        transcript: dict,
        evaluation_template: str,
        jd_text: str, 
        resume_text: str,
        save_dir: str = "evaluations"
        ) -> dict:
    
    system_prompt = evaluation_template
    user_prompt = (
        "Transcript JSON:\n"
        f"{json.dumps(transcript, indent=2)}\n\n"
        "Job Description:\n"
        f"{jd_text}\n\n"
        "Candidate resume/details:\n"
        f"{resume_text}\n\n"
        "Now return the evaluation JSON exactly as required by the system prompt."
    )


    # call the new client
    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.0,
        max_tokens=800,
    )

    # ---- robust extraction of assistant text ----
    text = ""
    try:
        # response.choices is a list-like of Choice objects
        if getattr(response, "choices", None):
            choice0 = response.choices[0]

            # In the new SDK, choice0.message is a ChatCompletionMessage model
            msg = getattr(choice0, "message", None) or getattr(choice0, "message", None)

            # Try common attribute names in order of likelihood
            if msg is None:
                # sometimes the SDK shape differs; try str(choice0)
                text = str(choice0)
            else:
                # If msg has .content attribute (most likely), use it
                content = getattr(msg, "content", None)
                if content is None:
                    # If content missing, try converting msg to dict then reading 'content'
                    try:
                        msg_dict = msg.dict()
                        content = msg_dict.get("content")
                    except Exception:
                        content = None

                if isinstance(content, str):
                    text = content
                else:
                    # final fallback: stringify the msg object
                    text = str(msg)
        else:
            # Unexpected shape; stringify full response
            text = str(response)
    except Exception as e:
        # Defensive: if anything fails, stringify response for debugging
        try:
            text = json.dumps(response, default=str)[:4000]
        except Exception:
            text = f"<failed to stringify response: {e}>"

    # Optional: log first ~1200 chars for debugging (you can remove in prod)
    # print("DEBUG model output:", text[:1200])

    parsed = safe_parse_json(text)
    if not parsed:
        # second attempt: if the model returned plain text with JSON after some lines,
        # try to extract JSON block using regex (safe_parse_json already does), else fallback
        parsed = {
            "score": None,
            "label": "Not suitable",
            "matched_keywords": [],
            "strengths": [],
            "weaknesses": [],
            "notes": "Failed to parse model output. Raw (truncated): " + text[:800],
        }

        # ---- Save to file ----
    os.makedirs(save_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = os.path.join(save_dir, f"evaluation_{timestamp}.txt")

    with open(filename, "w", encoding="utf-8") as f:
        f.write(json.dumps(parsed, indent=2))

    print(f"✅ Evaluation saved to {filename}")

    return parsed
