import re

try:
    from modules.skill_extractor import SKILL_TAXONOMY, build_skill_pattern
except ImportError:
    from skill_extractor import SKILL_TAXONOMY, build_skill_pattern

BASE_SKILLS = [
    "python", "sql", "power bi", "excel", "machine learning", "deep learning",
    "data analysis", "statistics", "tableau", "java", "c++", "c", "c#",
    "javascript", "typescript", "html", "css", "react", "nodejs", "node.js",
    "mongodb", "mysql", "postgresql", "aws", "azure", "gcp", "cloud", "docker",
    "kubernetes", "git", "github", "nlp", "pandas", "numpy", "scikit-learn",
    "tensorflow", "pytorch", "flask", "django", "fastapi", "rest api",
    "generative ai", "llm", "langchain", "ci/cd", "linux",
]

SKILLS = sorted(
    set(BASE_SKILLS).union(
        skill.lower()
        for category_skills in SKILL_TAXONOMY.values()
        for skill in category_skills
    )
)


def _skill_in_text(text: str, skill: str) -> bool:
    """Check skill presence using boundary-aware regex matching."""
    if re.match(r"^[a-z0-9_ ]+$", skill) and skill not in {"c"}:
        pattern = rf"\b{re.escape(skill)}\b"
        return bool(re.search(pattern, text, re.IGNORECASE))
    return bool(build_skill_pattern(skill).search(text))


def calculate_job_match(resume_text: str, job_description: str):
    """Compute true percentage skill match between a resume and job description."""
    if not resume_text or not job_description:
        return 0, [], []

    resume_lower = resume_text.lower()
    jd_lower = job_description.lower()

    resume_skills = {skill for skill in SKILLS if _skill_in_text(resume_lower, skill)}
    jd_skills = {skill for skill in SKILLS if _skill_in_text(jd_lower, skill)}

    if not jd_skills:
        return 0, [], []

    matched = sorted(resume_skills.intersection(jd_skills))
    missing = sorted(jd_skills - resume_skills)
    score = round((len(matched) / len(jd_skills)) * 100)

    return score, matched, missing
