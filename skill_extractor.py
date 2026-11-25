import os
import re
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

SKILLS_CSV_PATH = os.path.join(
    BASE_DIR,
    "data",
    "resume_dataset",
    "resume_dataset",
    "structured_dataset",
    "structured_resume_dataset",
    "06_skills.csv",
)

SKILL_TAXONOMY = {
    "Software": [
        "python", "java", "c", "c++", "c#", "javascript", "typescript",
        "go", "rust", "php", "ruby", "kotlin", "swift", "html", "css",
        "react", "angular", "vue", "next.js", "nodejs", "node.js",
        "express", "django", "flask", "fastapi", "spring boot",
        "rest api", "graphql", "microservices", "data structures",
        "algorithms", "object oriented programming", "oop", "git",
        "github", "unit testing", "pytest", "selenium", "sql",
        "postgresql", "mysql", "sqlite", "mongodb", "redis",
    ],
    "Data/AI": [
        "machine learning", "deep learning", "nlp", "natural language processing",
        "computer vision", "generative ai", "llm", "llms", "prompt engineering",
        "langchain", "rag", "data analysis", "data science", "statistics",
        "pandas", "numpy", "scikit-learn", "tensorflow", "pytorch", "keras",
        "power bi", "tableau", "excel", "apache spark", "hadoop", "etl",
        "data visualization", "feature engineering", "hugging face",
        "streamlit", "matplotlib", "seaborn", "a/b testing",
    ],
    "Cloud/DevOps": [
        "aws", "azure", "gcp", "google cloud", "docker", "kubernetes",
        "terraform", "ansible", "jenkins", "github actions", "ci/cd",
        "linux", "bash", "shell scripting", "nginx", "prometheus",
        "grafana", "serverless", "cloud", "devops", "mlops",
    ],
    "Core Engineering": [
        "autocad", "solidworks", "matlab", "ansys", "catia",
        "embedded systems", "microcontrollers", "arduino", "raspberry pi",
        "iot", "vlsi", "verilog", "vhdl", "plc", "scada", "cad", "cam",
        "manufacturing", "quality control", "six sigma", "lean manufacturing",
        "thermodynamics", "signal processing", "pcb design", "robotics",
        "electronics", "networking",
    ],
    "Business/Finance": [
        "financial analysis", "financial modeling", "accounting",
        "bookkeeping", "budgeting", "forecasting", "auditing", "taxation",
        "gst", "tally", "risk management", "banking operations", "crm",
        "salesforce", "project management", "agile", "scrum",
        "product management", "market research", "lead generation",
        "negotiation", "communication", "leadership", "stakeholder management",
        "client management", "strategic planning", "seo", "content creation",
        "marketing", "problem solving",
    ],
}


def _load_skill_database() -> list[str]:
    """Load skills from CSV if available; otherwise fall back to built-in taxonomy."""
    builtin_skills = {
        skill.lower()
        for category_skills in SKILL_TAXONOMY.values()
        for skill in category_skills
    }

    if os.path.exists(SKILLS_CSV_PATH):
        try:
            skills_df = pd.read_csv(SKILLS_CSV_PATH)
            if "skill" in skills_df.columns:
                csv_skills = (
                    skills_df["skill"]
                    .dropna()
                    .astype(str)
                    .str.strip()
                    .str.lower()
                    .tolist()
                )
                builtin_skills.update(s for s in csv_skills if s)
        except Exception:
            pass

    return sorted(builtin_skills)


skill_database = _load_skill_database()


def build_skill_pattern(skill: str) -> re.Pattern:
    """Compile a boundary-aware regex pattern that avoids false substring matches."""
    escaped = re.escape(skill.strip().lower())
    return re.compile(rf"(?<![a-z0-9_]){escaped}(?![a-z0-9_+#])", re.IGNORECASE)


_SKILL_PATTERNS = {skill: build_skill_pattern(skill) for skill in skill_database}


def extract_skills(resume_text: str) -> list[str]:
    """Extract normalized skills from resume text using regex boundary matching."""
    if not resume_text or not resume_text.strip():
        return []

    detected_skills = [
        skill
        for skill, pattern in _SKILL_PATTERNS.items()
        if pattern.search(resume_text)
    ]
    return sorted(set(detected_skills))


def extract_skills_by_category(resume_text: str) -> dict[str, list[str]]:
    """Group detected built-in skills by domain category."""
    if not resume_text or not resume_text.strip():
        return {}

    categorized = {}
    for category, skills in SKILL_TAXONOMY.items():
        matched = [
            skill
            for skill in skills
            if build_skill_pattern(skill).search(resume_text)
        ]
        if matched:
            categorized[category] = sorted(set(matched))
    return categorized
