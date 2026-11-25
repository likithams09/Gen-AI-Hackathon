import json
import os
import re
import time
from typing import List, Literal, Optional

from dotenv import load_dotenv
from google import genai
from google.genai import errors as genai_errors
from google.genai import types
from pydantic import BaseModel, Field

load_dotenv()

MODEL_NAME = "gemini-3.8-flash"
FALLBACK_MODELS = ["gemini-3.5-flash-lite", "gemini-3.8-flash", "gemini-2.0-flash"]
REQUEST_TIMEOUT_MS = 15000
RETRY_DELAY_SECONDS = 2
MAX_RESUME_CHARS = 5000


class GenAIConfigError(RuntimeError):
    """Raised when Gemini API configuration is missing or invalid."""


class RewrittenBullet(BaseModel):
    before: str
    after: str
    rationale: str


class ResumeAnalysisSchema(BaseModel):
    strengths: List[str]
    weaknesses: List[str]
    missing_impact_metrics: List[str]
    rewritten_bullets: List[RewrittenBullet] = Field(min_length=3, max_length=3)


class PrioritizedSkill(BaseModel):
    skill: str
    priority: Literal["High", "Medium", "Low"]
    prerequisites: List[str]
    reason: str


class RoadmapWeek(BaseModel):
    week: str
    focus_topic: str
    action_items: List[str]
    recommended_resource_type: str


class CapstoneProject(BaseModel):
    title: str
    description: str
    tech_stack: List[str]
    acceptance_criteria: List[str] = Field(min_length=3, max_length=3)


class SkillGapRoadmapSchema(BaseModel):
    role_fit_summary: str
    prioritized_missing_skills: List[PrioritizedSkill]
    four_week_roadmap: List[RoadmapWeek] = Field(min_length=4, max_length=4)
    capstone_mini_project: CapstoneProject


class CareerPath(BaseModel):
    role_title: str
    match_percentage: int
    why_it_fits: str
    next_step: str


class CareerRecommendationsSchema(BaseModel):
    career_paths: List[CareerPath] = Field(min_length=3, max_length=3)


class QuizQuestion(BaseModel):
    question: str
    options: List[str] = Field(min_length=4, max_length=4)
    correct_answer: str
    explanation: str


class SkillQuizSchema(BaseModel):
    questions: List[QuizQuestion] = Field(min_length=3, max_length=3)


class TailoredKeywordsSchema(BaseModel):
    priority_keywords: List[str]
    tailoring_tips: List[str]


class OrphanProofItem(BaseModel):
    orphan_skill: str
    target_project_or_role: str
    suggested_bullet: str
    verification_tip: str


class OrphanProofSchema(BaseModel):
    suggestions: List[OrphanProofItem]


class SkillROIItem(BaseModel):
    skill: str
    estimated_hours: int = Field(ge=1, le=200)
    impact_weight: int = Field(ge=1, le=10)
    why_it_matters: str


class SkillROISchema(BaseModel):
    items: List[SkillROIItem]


class RedFlagItem(BaseModel):
    category: str
    issue: str
    exact_quote_or_gap: str
    fix: str


class TrapQuestionItem(BaseModel):
    question: str
    what_interviewer_is_testing: str
    how_to_defend: str


class BrutalRecruiterSchema(BaseModel):
    overall_verdict: Literal["Strong Pass", "Borderline", "High Rejection Risk"]
    six_second_impression: str
    red_flags: List[RedFlagItem]
    trap_interview_questions: List[TrapQuestionItem] = Field(min_length=3, max_length=3)


class TransferableBridgeItem(BaseModel):
    existing_project_or_skill: str
    target_role_equivalent: str
    how_to_reframe_in_interview: str


class PivotBridgeSchema(BaseModel):
    pivot_feasibility_summary: str
    transferable_bridges: List[TransferableBridgeItem] = Field(min_length=3, max_length=4)
    reframed_pitch: str


def resolve_api_key(api_key: Optional[str] = None) -> str:
    """Resolve Gemini API key from explicit override or environment."""
    load_dotenv(override=False)
    key = (api_key or os.getenv("GEMINI_API_KEY") or "").strip()
    if not key or key == "your_api_key_here":
        raise GenAIConfigError(
            "GEMINI_API_KEY is missing. Add it to your .env file or enter it in the sidebar."
        )
    return key


def _get_client(api_key: Optional[str] = None) -> genai.Client:
    return genai.Client(
        api_key=resolve_api_key(api_key),
        http_options=types.HttpOptions(timeout=REQUEST_TIMEOUT_MS),
    )


def _parse_json_response(raw_text: str) -> dict:
    if not raw_text:
        raise RuntimeError("Gemini returned an empty response.")
    cleaned = raw_text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned)
    return json.loads(cleaned)


def _is_retryable_status_error(exc: Exception) -> bool:
    """Return True when Gemini responds with 503 (UNAVAILABLE) or 429 (RESOURCE_EXHAUSTED)."""
    if isinstance(exc, genai_errors.APIError):
        if getattr(exc, "code", None) in (429, 503):
            return True
        status = str(getattr(exc, "status", "") or "").upper()
        if status in {"UNAVAILABLE", "RESOURCE_EXHAUSTED"}:
            return True

    msg = str(exc).upper()
    return any(
        token in msg
        for token in ("503", "UNAVAILABLE", "429", "RESOURCE_EXHAUSTED")
    )


def _build_generation_config(
    model_name: str, response_schema: type[BaseModel]
) -> types.GenerateContentConfig:
    config_kwargs = {
        "response_mime_type": "application/json",
        "response_schema": response_schema,
        "temperature": 0.3,
    }
    if model_name != "gemini-2.0-flash":
        config_kwargs["thinking_config"] = types.ThinkingConfig(thinking_level="LOW")
    return types.GenerateContentConfig(**config_kwargs)


def _generate_with_fallback(
    prompt: str,
    response_schema: type[BaseModel],
    api_key: Optional[str] = None,
) -> dict:
    """Try FALLBACK_MODELS in order with a 15s timeout and 2s backoff on 503/429."""
    client = _get_client(api_key)
    last_exc: Optional[Exception] = None

    for idx, model_name in enumerate(FALLBACK_MODELS):
        try:
            response = client.models.generate_content(
                model=model_name,
                contents=prompt,
                config=_build_generation_config(model_name, response_schema),
            )
            return _parse_json_response(response.text)
        except GenAIConfigError:
            raise
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"Failed to parse structured JSON from Gemini: {exc}") from exc
        except Exception as exc:
            last_exc = exc
            if _is_retryable_status_error(exc) and idx < len(FALLBACK_MODELS) - 1:
                time.sleep(RETRY_DELAY_SECONDS)
                continue
            raise RuntimeError(
                f"Gemini API request failed on model '{model_name}': {exc}"
            ) from exc

    raise RuntimeError(f"All fallback Gemini models failed: {last_exc}")


def _generate_structured(prompt: str, schema: type[BaseModel], api_key: Optional[str] = None) -> dict:
    return _generate_with_fallback(prompt, schema, api_key=api_key)


def analyze_resume_with_ai(resume_text: str, api_key: Optional[str] = None) -> dict:
    """Critique resume quality and generate 3 Before/After ATS-optimized bullet rewrites."""
    if not resume_text or not resume_text.strip():
        raise ValueError("Resume text is required for AI analysis.")

    prompt = f"""You are an executive technical recruiter and ATS optimization specialist.
Analyze the following resume text carefully.

Return a JSON object with:
1. "strengths": 3-5 concrete strengths grounded in the resume content.
2. "weaknesses": 3-5 specific weaknesses or gaps in phrasing, clarity, or structure.
3. "missing_impact_metrics": 3-4 places where quantifiable metrics (%, $, scale, latency, throughput) are missing and should be added.
4. "rewritten_bullets": Exactly 3 objects, each with:
   - "before": an actual weak/passive bullet or sentence extracted or closely paraphrased from the resume
   - "after": an ATS-optimized, action-verb-led rewrite with realistic placeholder metrics where needed (e.g., [X%])
   - "rationale": a brief explanation of why the rewrite improves ATS ranking and recruiter readability.

Do not use emojis anywhere.

Resume:
\"\"\"{resume_text[:MAX_RESUME_CHARS]}\"\"\"
"""
    return _generate_with_fallback(prompt, ResumeAnalysisSchema, api_key=api_key)


def semantic_skill_gap_and_roadmap(
    resume_text: str,
    target_role: str,
    missing_skills: Optional[List[str]] = None,
    api_key: Optional[str] = None,
) -> dict:
    """Analyze semantic skill gaps for any target role and produce a 4-week roadmap + capstone."""
    if not resume_text or not resume_text.strip():
        raise ValueError("Resume text is required for skill gap analysis.")
    if not target_role or not target_role.strip():
        raise ValueError("Target role is required.")

    missing_str = ", ".join(missing_skills) if missing_skills else "None pre-identified; infer from role vs. resume."

    prompt = f"""You are a senior career architect and technical mentor.
Compare the candidate's resume against the target role "{target_role.strip()}".
Pre-identified missing skills (if any): {missing_str}

Return a JSON object with:
1. "role_fit_summary": A concise 2-3 sentence assessment of current readiness for "{target_role.strip()}".
2. "prioritized_missing_skills": 4-6 missing or under-demonstrated skills for this role, each with:
   - "skill": skill name
   - "priority": "High", "Medium", or "Low"
   - "prerequisites": list of prerequisite concepts/skills
   - "reason": why it matters for this role
3. "four_week_roadmap": Exactly 4 objects covering "Week 1", "Week 2", "Week 3", and "Week 4", each with:
   - "week": e.g. "Week 1"
   - "focus_topic": primary learning theme
   - "action_items": 3-4 concrete deliverables/tasks for that week
   - "recommended_resource_type": e.g. "Official Documentation & Hands-on Lab", "Project Build & Code Review"
4. "capstone_mini_project": A concrete portfolio project bridging the skill gap, with:
   - "title": project name
   - "description": 2-3 sentence scope
   - "tech_stack": list of 4-6 technologies
   - "acceptance_criteria": exactly 3 measurable completion criteria.

Do not use emojis anywhere.

Resume:
\"\"\"{resume_text[:MAX_RESUME_CHARS]}\"\"\"
"""
    return _generate_with_fallback(prompt, SkillGapRoadmapSchema, api_key=api_key)


def recommend_careers_with_ai(
    resume_text: str,
    extracted_skills: Optional[List[str]] = None,
    api_key: Optional[str] = None,
) -> list[dict]:
    """Recommend 3 personalized career paths grounded in resume experience and skills."""
    if not resume_text or not resume_text.strip():
        raise ValueError("Resume text is required for career recommendations.")

    skills_str = ", ".join(extracted_skills) if extracted_skills else "Extract directly from resume."

    prompt = f"""You are a career strategist specializing in technology and analytical roles.
Evaluate the candidate's resume and detected skills ({skills_str}).

Return a JSON object containing "career_paths" with exactly 3 personalized career paths.
Each career path must contain:
- "role_title": specific job title
- "match_percentage": integer between 0 and 100 reflecting realistic current fit
- "why_it_fits": 2 concise sentences tying the recommendation directly to evidence in the resume
- "next_step": 1 concrete action the candidate should take in the next 14 days to become competitive.

Do not use emojis anywhere.

Resume:
\"\"\"{resume_text[:MAX_RESUME_CHARS]}\"\"\"
"""
    data = _generate_with_fallback(prompt, CareerRecommendationsSchema, api_key=api_key)
    return data.get("career_paths", [])


def generate_skill_verification_quiz(
    claimed_skills: List[str],
    api_key: Optional[str] = None,
) -> list[dict]:
    """Generate 3 short technical MCQs to verify skills claimed on the candidate's resume."""
    skills = [s.strip() for s in (claimed_skills or []) if s and s.strip()]
    if not skills:
        raise ValueError("At least one claimed skill is required to generate a verification quiz.")

    selected_skills = ", ".join(skills[:10])

    prompt = f"""You are a technical interviewer designing a fast diagnostic screening quiz.
Based on these skills claimed on a candidate's resume: {selected_skills}

Generate a JSON object with "questions" containing exactly 3 practical, unambiguous technical multiple-choice questions.
Each question object must contain:
- "question": concise technical question testing real applied understanding (not trivia)
- "options": list of exactly 4 distinct answer choices (plain text without A/B/C/D prefixes)
- "correct_answer": exact string matching one of the 4 items in "options"
- "explanation": 1-2 sentence explanation of why the correct answer is right.

Do not use emojis anywhere.
"""
    data = _generate_with_fallback(prompt, SkillQuizSchema, api_key=api_key)
    return data.get("questions", [])


def generate_tailored_keywords(
    resume_text: str,
    job_description: str,
    missing_skills: Optional[List[str]] = None,
    api_key: Optional[str] = None,
) -> dict:
    """Suggest AI-tailored ATS keywords and positioning tips for a specific job description."""
    if not resume_text or not job_description:
        raise ValueError("Both resume text and job description are required.")

    missing_str = ", ".join(missing_skills) if missing_skills else "None"

    prompt = f"""You are an ATS optimization specialist.
Compare the candidate's resume with the target job description.
Regex-missing technical skills: {missing_str}

Return a JSON object with:
- "priority_keywords": 6-10 exact domain phrases, tools, or action keywords from the job description that the candidate should truthfully weave into their resume if applicable.
- "tailoring_tips": 3-4 concise, actionable suggestions for tailoring the resume specifically to this job description without fabricating experience.

Do not use emojis anywhere.

Job Description:
\"\"\"{job_description[:MAX_RESUME_CHARS]}\"\"\"

Resume:
\"\"\"{resume_text[:MAX_RESUME_CHARS]}\"\"\"
"""
    return _generate_with_fallback(prompt, TailoredKeywordsSchema, api_key=api_key)


def generate_orphan_proof_bullets(
    resume_text: str,
    orphan_skills: List[str],
    api_key: Optional[str] = None,
) -> list[dict]:
    """Suggest truthful resume bullets to substantiate orphan skills in existing projects/experience."""
    skills = [s.strip() for s in (orphan_skills or []) if s and s.strip()]
    if not resume_text or not resume_text.strip():
        raise ValueError("Resume text is required.")
    if not skills:
        return []

    skills_str = ", ".join(skills[:12])
    prompt = f"""You are a technical resume coach.
The candidate lists these skills on their resume without mentioning them in any project or work experience bullet: {skills_str}

Review the candidate's existing projects, coursework, or experience in the resume below, and suggest concrete ways to demonstrate each orphan skill truthfully (or note what mini-deliverable to add to an existing project).

Return a JSON object with "suggestions", where each item contains:
- "orphan_skill": the skill name
- "target_project_or_role": which project, internship, or section in the resume to attach proof to (or "New Portfolio Extension" if none fits)
- "suggested_bullet": an ATS-friendly bullet starting with a strong action verb demonstrating that skill in context
- "verification_tip": 1 concise tip on what artifact or metric makes this claim credible to a hiring manager.

Do not use emojis anywhere.

Resume:
\"\"\"{resume_text[:MAX_RESUME_CHARS]}\"\"\"
"""
    data = _generate_with_fallback(prompt, OrphanProofSchema, api_key=api_key)
    return data.get("suggestions", [])


def fallback_skill_roi(missing_skills: List[str]) -> list[dict]:
    """Deterministic offline fallback ROI estimates when Gemini is skipped or unavailable."""
    skills = [s.strip() for s in (missing_skills or []) if s and s.strip()]
    return [
        {
            "skill": skill,
            "estimated_hours": 15,
            "impact_weight": 5,
            "why_it_matters": "Baseline estimate: foundational competency for the target role.",
        }
        for skill in skills
    ]


def estimate_skill_roi(
    missing_skills: List[str],
    target_role: str,
    api_key: Optional[str] = None,
) -> list[dict]:
    """Estimate learning hours and role impact weight (1-10) for each missing skill."""
    skills = [s.strip() for s in (missing_skills or []) if s and s.strip()]
    if not skills:
        return []

    role_label = (target_role or "Target Technical Role").strip()
    skills_str = ", ".join(skills[:15])

    try:
        resolve_api_key(api_key)
    except GenAIConfigError:
        return fallback_skill_roi(skills)

    prompt = f"""You are a technical learning strategist.
For a candidate targeting the role "{role_label}", estimate realistic job-ready learning effort and hiring impact for each missing skill:
{skills_str}

Return a JSON object with "items" containing one object per listed skill:
- "skill": exact skill name from the list above
- "estimated_hours": realistic integer hours (between 5 and 80) to reach interview/job-ready proficiency
- "impact_weight": integer from 1 (nice-to-have) to 10 (critical hiring requirement for "{role_label}")
- "why_it_matters": 1 concise sentence explaining its impact for "{role_label}".

Do not use emojis anywhere.
"""
    try:
        data = _generate_with_fallback(prompt, SkillROISchema, api_key=api_key)
        items = data.get("items", [])
        if not items:
            return fallback_skill_roi(skills)
        # Ensure clean integer bounds
        normalized = []
        for item in items:
            normalized.append(
                {
                    "skill": item.get("skill", ""),
                    "estimated_hours": max(1, int(item.get("estimated_hours", 15))),
                    "impact_weight": min(10, max(1, int(item.get("impact_weight", 5)))),
                    "why_it_matters": item.get("why_it_matters", ""),
                }
            )
        return normalized
    except Exception:
        return fallback_skill_roi(skills)


def run_brutal_recruiter_audit(
    resume_text: str,
    target_role: str = "",
    api_key: Optional[str] = None,
) -> dict:
    """Perform a skeptical hiring-manager audit surfacing red flags and 3 trap interview questions."""
    if not resume_text or not resume_text.strip():
        raise ValueError("Resume text is required for the recruiter audit.")

    role_clause = (
        f'Target Role: "{target_role.strip()}". Evaluate role alignment strictly against this target.'
        if target_role and target_role.strip()
        else "No target role specified; evaluate general industry engineering/analytical hiring bar."
    )

    prompt = f"""You are a skeptical staff hiring manager and senior technical recruiter conducting a strict 6-second resume screen.
{role_clause}

Hunt aggressively for:
- Buzzword inflation or skills listed with zero project/experience proof
- Unquantified, vague claims ("worked on", "helped with", "responsible for")
- Formatting, density, chronology, or structural red flags
- Role mismatch or shallow project depth

Return a JSON object with:
1. "overall_verdict": strictly one of "Strong Pass", "Borderline", or "High Rejection Risk".
2. "six_second_impression": blunt, realistic 2-sentence reaction after a 6-second skim.
3. "red_flags": 3-6 specific red flag objects, each with:
   - "category": e.g. "Buzzword Inflation", "Vague Claim", "Missing Impact", "Role Mismatch", "Density/Structure"
   - "issue": clear statement of the flaw
   - "exact_quote_or_gap": exact phrase from the resume or specific omission that triggered the flag
   - "fix": concrete, actionable remediation
4. "trap_interview_questions": exactly 3 tough grilling questions an interviewer would ask to expose weak spots on this exact resume, each with:
   - "question": direct, pointed interview question
   - "what_interviewer_is_testing": the hidden skepticism or claim being stress-tested
   - "how_to_defend": concrete strategy and evidence the candidate should prepare to answer convincingly.

Do not use emojis anywhere.

Resume:
\"\"\"{resume_text[:MAX_RESUME_CHARS]}\"\"\"
"""
    return _generate_with_fallback(prompt, BrutalRecruiterSchema, api_key=api_key)


def generate_pivot_bridge(
    resume_text: str,
    extracted_skills: Optional[List[str]],
    target_role: str,
    api_key: Optional[str] = None,
) -> dict:
    """Map existing skills and projects into transferable bridges for any target pivot role."""
    if not resume_text or not resume_text.strip():
        raise ValueError("Resume text is required to build a pivot bridge.")
    if not target_role or not target_role.strip():
        raise ValueError("Target pivot role is required.")

    skills_str = ", ".join(extracted_skills) if extracted_skills else "Infer directly from resume."

    prompt = f"""You are an executive career transition strategist.
The candidate wants to pivot into the target role: "{target_role.strip()}".
Detected skills on resume: {skills_str}

Analyze the candidate's existing projects, coursework, and skills and translate them into credible transferable competencies for "{target_role.strip()}".

Return a JSON object with:
1. "pivot_feasibility_summary": 2-3 sentence honest assessment of the pivot path, strengths to leverage, and main framing gap.
2. "transferable_bridges": 3 to 4 mapping objects, each with:
   - "existing_project_or_skill": specific project, domain experience, or skill from the resume
   - "target_role_equivalent": the corresponding competency or workflow expected in "{target_role.strip()}"
   - "how_to_reframe_in_interview": exact framing language showing how to present this experience in interviews
3. "reframed_pitch": a polished, natural 3-sentence "Tell me about yourself" pivot pitch tailored to "{target_role.strip()}".

Do not use emojis anywhere.

Resume:
\"\"\"{resume_text[:MAX_RESUME_CHARS]}\"\"\"
"""
    return _generate_with_fallback(prompt, PivotBridgeSchema, api_key=api_key)
