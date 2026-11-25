CAREER_MAP = {
    "CSE": {
        "roles": [
            "Software Developer",
            "Backend Developer",
            "Data Analyst",
            "ML Engineer",
        ],
        "skills": [
            "Python",
            "SQL",
            "Git",
            "Data Structures",
            "Problem Solving",
        ],
    },
    "IT": {
        "roles": [
            "Software Engineer",
            "Cloud Engineer",
            "Backend Developer",
            "System Analyst",
        ],
        "skills": [
            "Python",
            "Networking",
            "Cloud",
            "Linux",
            "SQL",
        ],
    },
    "ECE": {
        "roles": [
            "Embedded Engineer",
            "IoT Engineer",
            "Software Developer",
            "Data Analyst",
        ],
        "skills": [
            "C",
            "Python",
            "Microcontrollers",
            "IoT",
            "Electronics",
        ],
    },
    "MECH": {
        "roles": [
            "Design Engineer",
            "Production Engineer",
            "Quality Engineer",
            "CAD Engineer",
        ],
        "skills": [
            "AutoCAD",
            "SolidWorks",
            "Manufacturing",
            "Quality Control",
        ],
    },
    "AI/ML": {
        "roles": [
            "Machine Learning Engineer",
            "Applied AI Engineer",
            "Data Scientist",
            "NLP Engineer",
        ],
        "skills": [
            "Python",
            "PyTorch",
            "Scikit-Learn",
            "SQL",
            "LLMs & RAG",
        ],
    },
}

ALIASES = {
    "CS": "CSE",
    "COMPUTER SCIENCE": "CSE",
    "INFORMATION TECHNOLOGY": "IT",
    "ELECTRONICS": "ECE",
    "MECHANICAL": "MECH",
    "AI": "AI/ML",
    "ML": "AI/ML",
    "AIDS": "AI/ML",
    "DATA SCIENCE": "AI/ML",
}


def recommend_career(specialization: str):
    """Map an academic specialization to baseline roles and foundational skills."""
    key = (specialization or "").strip().upper()
    key = ALIASES.get(key, key)

    if key in CAREER_MAP:
        return (
            CAREER_MAP[key]["roles"],
            CAREER_MAP[key]["skills"],
        )

    return (
        ["Software Developer", "Data Analyst", "Technical Consultant"],
        ["Python", "SQL", "Git", "Problem Solving"],
    )
