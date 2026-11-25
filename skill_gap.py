try:
    from modules.skill_extractor import build_skill_pattern
except ImportError:
    from skill_extractor import build_skill_pattern

ROLE_SKILLS = {
    "INFORMATION-TECHNOLOGY": [
        "python", "sql", "git", "rest api", "data structures", "problem solving",
    ],
    "AI-ML-ENGINEER": [
        "python", "machine learning", "deep learning", "pytorch",
        "scikit-learn", "llm", "rag", "docker", "git",
    ],
    "DATA-SCIENTIST": [
        "python", "sql", "statistics", "pandas", "numpy",
        "machine learning", "data visualization", "scikit-learn",
    ],
    "FULL-STACK-DEVELOPER": [
        "javascript", "typescript", "react", "node.js", "python",
        "sql", "rest api", "git", "docker",
    ],
    "CLOUD-DEVOPS-ENGINEER": [
        "aws", "docker", "kubernetes", "terraform", "ci/cd",
        "linux", "python", "git",
    ],
    "HR": [
        "recruitment", "communication", "employee relations", "hrms", "payroll",
    ],
    "FINANCE": [
        "financial analysis", "excel", "accounting", "budgeting", "forecasting",
    ],
    "ACCOUNTANT": [
        "tally", "gst", "bookkeeping", "accounting", "taxation",
    ],
    "BANKING": [
        "risk management", "customer service", "banking operations",
        "financial analysis", "excel",
    ],
    "HEALTHCARE": [
        "patient care", "medical records", "clinical support",
        "healthcare compliance",
    ],
    "SALES": [
        "lead generation", "communication", "negotiation", "crm", "sales strategy",
    ],
    "BUSINESS-DEVELOPMENT": [
        "market research", "communication", "sales strategy",
        "lead generation", "client management",
    ],
    "DIGITAL-MEDIA": [
        "content creation", "seo", "social media", "video editing", "marketing",
    ],
    "TEACHER": [
        "classroom management", "lesson planning", "communication",
        "assessment", "student engagement",
    ],
    "ENGINEERING": [
        "problem solving", "technical design", "project management", "analysis",
    ],
}


def analyze_skill_gap(resume_text: str, target_role: str):
    """Compare resume text against required skills for a preset or custom role."""
    if not resume_text or not target_role:
        return [], []

    normalized_role = target_role.strip().upper().replace(" ", "-")
    required_skills = ROLE_SKILLS.get(target_role) or ROLE_SKILLS.get(normalized_role, [])

    if not required_skills:
        return [], []

    existing_skills = []
    missing_skills = []

    for skill in required_skills:
        if build_skill_pattern(skill).search(resume_text):
            existing_skills.append(skill)
        else:
            missing_skills.append(skill)

    return existing_skills, missing_skills
