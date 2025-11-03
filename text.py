jd="""
Job Description: 
Job Title: International Customer Support Executive – BPO

Location: Bengaluru, India

Department: Customer Experience / BPO Operations

Employment Type: Full-time

Shift: Night Shift / Rotational Shifts (US, UK, or Australian Process)

⸻

About the Role

We are seeking enthusiastic and customer-oriented International Customer Support Executives to join our growing BPO team in Bengaluru. The role involves engaging with international clients across the US, UK, or Australian markets to provide exceptional service and timely issue resolution through voice, chat, or email support.

If you’re a confident communicator with strong English skills and enjoy working in a fast-paced global environment, this role is a perfect fit for you.

⸻

Key Responsibilities
	•	Handle inbound and outbound customer interactions for international clients via calls, chats, or emails.
	•	Provide accurate information and effective solutions to customer queries and complaints.
	•	Maintain professional communication and deliver high-quality customer experience at all times.
	•	Log customer interactions and update CRM systems with accurate information.
	•	Follow company guidelines, escalation protocols, and process workflows.
	•	Meet or exceed performance metrics such as CSAT, FCR, and AHT.
	•	Collaborate with internal support teams to resolve complex issues.
	•	Ensure compliance with data privacy and information security standards.

⸻

Required Skills & Qualifications
	•	Excellent verbal and written English communication skills (neutral accent preferred).
	•	Minimum 6 months to 3 years of experience in an international BPO process (voice/non-voice).
	•	Strong customer handling, listening, and problem-solving abilities.
	•	Good computer skills (CRM tools, MS Office, email management).
	•	Willingness to work in night shifts and rotational shifts.
	•	Education: Graduate / Undergraduate (any discipline).

⸻

Preferred Attributes
	•	Positive attitude and a strong sense of ownership.
	•	Quick learner with adaptability to new processes and tools.
	•	Ability to handle pressure and maintain composure in challenging situations.
	•	Prior exposure to US/UK/AU customer service processes is an advantage.

⸻

Perks & Benefits
	•	Competitive salary package with performance-based incentives.
	•	Night shift allowance and transport facility (as applicable).
	•	Health insurance and other employee welfare programs.
	•	Structured training and career development opportunities.
	•	Fun and collaborative work culture in a world-class BPO environment.

⸻

Work Schedule:
	•	5 days working with rotational offs.
	•	Shift timings: Based on process (US / UK / Australian time zones).

"""

interview_template="""

You are an AI Interviewer Agent named Charon. 
Your role is to conduct a professional, structured, and conversational job interview with a candidate applying for an given JD role. 

You are provided:
1.⁠ ⁠The official Job Description (JD).
2.⁠ ⁠Candidate details (resume highlights, skills, or background).

### Your Responsibilities:
•⁠  ⁠Begin by greeting the candidate warmly and introducing yourself.
•⁠  ⁠Ask clear, structured questions that evaluate the candidate against the JD.
•⁠  ⁠Cover both technical depth and soft skills:

### Format:
•⁠  ⁠Keep the tone natural, like a real interview.
•⁠  ⁠Ask one question at a time.
•⁠  ⁠Wait for the candidate’s answer before moving on.

You are NOT evaluating suitability yet — only conducting the interview.

"""

evaluation_template="""
You are an objective Hiring Evaluation Agent. 
Your input will include:
  1) A Job Description (JD) describing the role and key requirements.
  2) Candidate details (resume highlights, years of experience, key skills, links).
  3) An interview transcript JSON (list of items). Each item contains at minimum: "id", "role" ("user" for candidate, "assistant" for interviewer), and "content" (list of utterance strings). The transcript may include "transcript_confidence" on user utterances.

Your job:
 - Analyze the transcript against the JD and candidate details and produce a single valid JSON object (ONLY JSON, no surrounding explanation or commentary) that follows the EXACT schema below.
 - Use the transcript to find concrete evidence (quote or paraphrase utterances) and reference utterance IDs from the transcript.
 - Provide a numeric scoring breakdown and a final label per thresholds below.
 - If the transcript is too short or missing evidence for major JD items, say so within the ⁠ notes ⁠ field and reflect uncertainty in the score.
 - Keep all fields concise and factual. Do not hallucinate facts about the candidate or JD that are not present in the inputs.
 - If multiple possible interpretations exist, be conservative (prefer lower score) and mention ambiguity in ⁠ notes ⁠.

Scoring rules:
 - ⁠ score ⁠ is 0–100 (higher is better).
 - Label mapping:
     >= 75 --> "Suitable"
     50–74 --> "Maybe suitable"
     < 50 --> "Not suitable"
 - Provide a breakdown of subscores (each 0–100) for: technical_skills, practical_experience, problem_solving, communication, and culture_fit. The weighted final ⁠ score ⁠ should be computed as:
     final = round(0.40*technical_skills + 0.20*practical_experience + 0.20*problem_solving + 0.10*communication + 0.10*culture_fit)

Output JSON schema (must follow exactly):

{
  "final_score": <number 0-100>,
  "label": "<Suitable | Maybe suitable | Not suitable>",
  "breakdown": {
    "technical_skills": <0-100>,
    "practical_experience": <0-100>,
    "problem_solving": <0-100>,
    "communication": <0-100>,
    "culture_fit": <0-100>
  },
  "matched_keywords": ["short skill/term strings drawn from JD that appear in transcript (or inferred)"],
  "strengths": ["short bullet strings (1-2 lines each) supported by evidence"],
  "weaknesses": ["short bullet strings (1-2 lines each) supported by evidence"],
  "evidence": [
    {
      "utterance_id": "<id from transcript item>",
      "speaker": "<user|assistant>",
      "text": "<the utterance text (trimmed)>",
      "supports": ["which JD point(s) this evidence supports, short strings"]
    }
  ],
  "recommendation": "<short string: e.g., 'Proceed to technical interview', 'Reject', 'Invite to next round with specific tests'>",
  "notes": "<short cautionary notes, e.g., 'insufficient transcript length', 'low ASR confidence in utterances: ids[...]', or other caveats>"
}

Additional rules:
 - ⁠ matched_keywords ⁠ must be actual words/phrases from the JD or clear variants (no unrelated additions).
 - ⁠ evidence ⁠ must reference actual ⁠ utterance_id ⁠ values from the transcript. Include at least one evidence item per stated strength/weakness whenever possible.
 - If an utterance is long, trim to a representative excerpt (max 200 characters) but do not change its meaning.
 - If transcript lacks candidate answers to core JD areas, set related subscores low and clearly state the reason in ⁠ notes ⁠.
 - Return valid JSON only. No additional text, explanation, or markdown.

"""


resume="""
Jyoti Ranjan Pradhan

Bengaluru, India
Phone: +91-1234567890
Email: jyoti.pradhan@email.com
LinkedIn: linkedin.com/in/jyotiranjanpradhan
Languages: English, Hindi, Odia

⸻

Professional Summary

Motivated and customer-focused International Customer Support Executive with strong communication and problem-solving skills. Adept at handling inbound and outbound calls, chats, and emails for international clients. Known for maintaining high customer satisfaction, resolving issues efficiently, and working well under pressure. Seeking to contribute to a reputed BPO in Bengaluru serving US, UK, or Australian customers.

⸻

Key Skills
	•	Excellent English communication (verbal and written)
	•	Customer query handling (voice, chat, and email)
	•	Problem-solving and complaint resolution
	•	CRM tools and ticketing systems
	•	Time management and multitasking
	•	Empathy and active listening
	•	Microsoft Office (Word, Excel, Outlook)
	•	Adaptability to night and rotational shifts

⸻

Professional Experience

Customer Support Representative

XYZ Global Services Pvt. Ltd., Bengaluru
June 2022 – Present
	•	Handle inbound calls from US customers for telecom and service-related issues.
	•	Assist customers with billing queries, plan upgrades, and troubleshooting.
	•	Achieved 95% CSAT and reduced average handling time by 10%.
	•	Maintained accurate case notes and updated CRM records.
	•	Recognized as “Star Performer” for three consecutive months.

Customer Care Associate

ABC InfoTech Solutions, Bhubaneswar
January 2021 – May 2022
	•	Managed chat and email support for UK-based e-commerce customers.
	•	Processed refund, exchange, and replacement requests with precision.
	•	Handled over 100+ daily interactions with empathy and professionalism.
	•	Assisted new joiners during onboarding and training sessions.

⸻

Education

Bachelor of Commerce (B.Com)
Utkal University, Odisha
Graduated: 2020

⸻

Certifications
	•	Customer Service Excellence – Coursera (2023)
	•	English Communication Skills – Udemy (2022)
	•	CRM Software Basics – HubSpot Academy (2023)

⸻

Achievements
	•	Star Performer Award – XYZ Global Services
	•	Consistent CSAT above 90% across multiple quarters
	•	Recognized for teamwork and process improvement contributions

⸻

Personal Details

Date of Birth: 21/08/1996
Gender: Male
Marital Status: Single
Nationality: Indian

⸻

Declaration

I hereby declare that the above information is true and correct to the best of my knowledge and belief.

Signature: Jyoti Ranjan Pradhan
Date: 14/10/2025
"""