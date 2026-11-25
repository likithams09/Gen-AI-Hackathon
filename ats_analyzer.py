import re

SECTION_WEIGHTS = {
    "Education": (10, r"\b(?:education|academic\s+background|academics)\b"),
    "Skills": (10, r"\b(?:skills|technical\s+skills|core\s+competencies)\b"),
    "Experience": (12, r"\b(?:experience|work\s+experience|employment|career\s+history)\b"),
    "Project": (10, r"\bprojects?\b"),
    "Certification": (6, r"\bcertifications?\b"),
    "Achievement": (6, r"\b(?:achievements?|awards?|honors?)\b"),
    "Internship": (6, r"\binternships?\b"),
}

ACTION_VERBS = [
    "achieved", "architected", "automated", "built", "collaborated",
    "consolidated", "created", "decreased", "delivered", "deployed",
    "designed", "developed", "engineered", "established", "evaluated",
    "executed", "expanded", "generated", "implemented", "improved",
    "increased", "initiated", "integrated", "launched", "led",
    "managed", "mentored", "migrated", "optimized", "orchestrated",
    "reduced", "refactored", "researched", "resolved", "scaled",
    "spearheaded", "streamlined", "tested", "trained",
]

METRIC_PATTERN = re.compile(
    r"(?:"
    r"\b\d+(?:\.\d+)?\s*%"
    r"|[\$\u20b9\u20ac\u00a3]\s*\d+(?:[.,]\d+)?(?:\s*[kmbKMB])?"
    r"|\b\d+(?:\.\d+)?x\b"
    r"|\b\d{1,3}(?:,\d{3})+\b"
    r"|\b\d+\+?\s*(?:users|customers|clients|projects|models|apis|endpoints|requests|records|rows|ms|seconds|hours|days|months|years|engineers|members)\b"
    r")",
    re.IGNORECASE,
)


def analyze_ats_detailed(resume_text: str) -> dict:
    """Evaluate ATS readiness via section detection, quantifiable metrics, and action verbs."""
    if not resume_text or not resume_text.strip():
        return {
            "score": 0,
            "section_score": 0,
            "metrics_score": 0,
            "action_verb_score": 0,
            "found_sections": [],
            "missing_sections": list(SECTION_WEIGHTS.keys()),
            "metrics_found": [],
            "action_verbs_found": [],
            "word_count": 0,
            "suggestions": ["Provide resume text to run ATS analysis."],
        }

    found_sections = []
    missing_sections = []
    section_score = 0

    for section_name, (weight, pattern) in SECTION_WEIGHTS.items():
        if re.search(pattern, resume_text, re.IGNORECASE):
            section_score += weight
            found_sections.append(section_name)
        else:
            missing_sections.append(section_name)

    metrics_found = [m.group(0).strip() for m in METRIC_PATTERN.finditer(resume_text)]
    metric_count = len(metrics_found)
    if metric_count >= 5:
        metrics_score = 20
    elif metric_count >= 3:
        metrics_score = 14
    elif metric_count >= 1:
        metrics_score = 8
    else:
        metrics_score = 0

    action_verbs_found = sorted(
        verb
        for verb in ACTION_VERBS
        if re.search(rf"\b{re.escape(verb)}\b", resume_text, re.IGNORECASE)
    )
    action_verb_score = min(round((len(action_verbs_found) / 6) * 20), 20)

    total_score = min(section_score + metrics_score + action_verb_score, 100)
    word_count = len(resume_text.split())

    suggestions = []
    for section in missing_sections:
        suggestions.append(f"Add a clearly labeled {section} section for stronger ATS parsing.")

    if metric_count < 3:
        suggestions.append(
            "Add quantifiable impact metrics (percentages, dollar figures, latency reductions, or user scale) to your experience bullets."
        )
    if len(action_verbs_found) < 5:
        suggestions.append(
            "Begin bullet points with strong action verbs (e.g., Engineered, Optimized, Automated, Delivered)."
        )
    if word_count < 150:
        suggestions.append(
            "Resume content is sparse (under 150 words); expand project descriptions and technical contributions."
        )
    elif word_count > 1200:
        suggestions.append(
            "Resume exceeds 1,200 words; consider tightening phrasing to maintain concise, high-impact bullets."
        )

    return {
        "score": total_score,
        "section_score": section_score,
        "metrics_score": metrics_score,
        "action_verb_score": action_verb_score,
        "found_sections": found_sections,
        "missing_sections": missing_sections,
        "metrics_found": metrics_found,
        "action_verbs_found": action_verbs_found,
        "word_count": word_count,
        "suggestions": suggestions,
    }


def calculate_ats_score(resume_text: str):
    """Return (score, found_sections, missing_sections, suggestions) for ATS compatibility."""
    details = analyze_ats_detailed(resume_text)
    return (
        details["score"],
        details["found_sections"],
        details["missing_sections"],
        details["suggestions"],
    )
