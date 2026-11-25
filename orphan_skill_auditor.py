import re

try:
    from modules.skill_extractor import build_skill_pattern
except ImportError:
    from skill_extractor import build_skill_pattern

SKILLS_HEADER_RE = re.compile(
    r"^\s*(?:technical\s+skills|core\s+competencies|skills\s*(?:&|and)\s*tools|key\s+skills|skills)\s*[:\-|]?\s*(.*)$",
    re.IGNORECASE,
)

EVIDENCE_HEADER_RE = re.compile(
    r"^\s*(?:work\s+experience|professional\s+experience|experience|projects|academic\s+projects|personal\s+projects|key\s+projects|internships?|research(?:\s+experience)?|employment(?:\s+history)?)\s*[:\-|]?\s*(.*)$",
    re.IGNORECASE,
)

OTHER_HEADER_RE = re.compile(
    r"^\s*(?:education|academic\s+background|summary|professional\s+summary|profile|objective|career\s+objective|certifications?|achievements?|awards?|honors?|publications?|languages?|interests?|hobbies|references?)\s*[:\-|]?\s*(.*)$",
    re.IGNORECASE,
)


def _skill_in_evidence(evidence_text: str, skill: str) -> re.Match | None:
    """Match skill in evidence text using word-boundary regex."""
    normalized = skill.strip()
    if not normalized:
        return None
    if re.match(r"^[a-z0-9_ ]+$", normalized, re.IGNORECASE) and normalized.lower() != "c":
        return re.search(rf"\b{re.escape(normalized)}\b", evidence_text, re.IGNORECASE)
    return build_skill_pattern(normalized).search(evidence_text)


def _extract_snippet(text: str, match: re.Match, max_len: int = 140) -> str:
    """Extract the line or concise window around a regex match."""
    line_start = text.rfind("\n", 0, match.start()) + 1
    line_end = text.find("\n", match.end())
    if line_end == -1:
        line_end = len(text)
    line = text[line_start:line_end].strip()
    if len(line) <= max_len:
        return line

    start = max(0, match.start() - 50)
    end = min(len(text), match.end() + 70)
    snippet = text[start:end].replace("\n", " ").strip()
    if start > 0:
        snippet = "..." + snippet
    if end < len(text):
        snippet = snippet + "..."
    return snippet


def split_resume_zones(resume_text: str) -> tuple[str, str, bool, bool]:
    """Split resume text into skills_text and evidence_text zones."""
    lines = resume_text.splitlines()
    skills_lines: list[str] = []
    evidence_lines: list[str] = []
    non_skills_lines: list[str] = []

    current_zone = "other"
    skills_header_found = False
    evidence_header_found = False

    for line in lines:
        stripped = line.strip()
        # Only treat reasonably short lines as potential section headers
        is_candidate_header = len(stripped) <= 65 or ":" in stripped[:35]

        if is_candidate_header:
            skills_match = SKILLS_HEADER_RE.match(stripped)
            if skills_match:
                skills_header_found = True
                inline_rest = skills_match.group(1).strip()
                if inline_rest:
                    skills_lines.append(inline_rest)
                    current_zone = "skills_inline"
                else:
                    current_zone = "skills"
                continue

            evidence_match = EVIDENCE_HEADER_RE.match(stripped)
            if evidence_match:
                evidence_header_found = True
                current_zone = "evidence"
                inline_rest = evidence_match.group(1).strip()
                if inline_rest:
                    evidence_lines.append(inline_rest)
                    non_skills_lines.append(inline_rest)
                continue

            if OTHER_HEADER_RE.match(stripped):
                current_zone = "other"
                non_skills_lines.append(line)
                continue

        if current_zone == "skills_inline":
            # Inline skill lines end once prose/sentence content begins
            if len(stripped) > 75 or stripped.endswith("."):
                current_zone = "other"
            else:
                skills_lines.append(line)
                continue

        if current_zone == "skills":
            skills_lines.append(line)
        elif current_zone == "evidence":
            evidence_lines.append(line)
            non_skills_lines.append(line)
        else:
            non_skills_lines.append(line)

    skills_text = "\n".join(skills_lines).strip()
    evidence_text = "\n".join(evidence_lines).strip()

    if not evidence_text:
        fallback_text = "\n".join(non_skills_lines).strip()
        if fallback_text:
            evidence_text = fallback_text
        else:
            evidence_text = "\n".join(
                l for l in lines if not SKILLS_HEADER_RE.match(l.strip())
            ).strip()

    return skills_text, evidence_text, skills_header_found, evidence_header_found


def audit_orphan_skills(resume_text: str, extracted_skills: list[str]) -> dict:
    """Audit extracted skills to distinguish project/experience-backed skills from orphan skills."""
    skills = sorted({s.strip() for s in (extracted_skills or []) if s and s.strip()})
    if not resume_text or not resume_text.strip() or not skills:
        return {
            "proof_ratio": 0,
            "verified_skills": [],
            "orphan_skills": skills,
            "skills_section_found": False,
            "evidence_section_found": False,
        }

    _, evidence_text, skills_found, evidence_found = split_resume_zones(resume_text)

    verified_skills = []
    orphan_skills = []

    for skill in skills:
        match = _skill_in_evidence(evidence_text, skill)
        if match:
            verified_skills.append(
                {
                    "skill": skill,
                    "snippet": _extract_snippet(evidence_text, match),
                }
            )
        else:
            orphan_skills.append(skill)

    proof_ratio = round((len(verified_skills) / max(len(skills), 1)) * 100)

    return {
        "proof_ratio": proof_ratio,
        "verified_skills": verified_skills,
        "orphan_skills": orphan_skills,
        "skills_section_found": skills_found,
        "evidence_section_found": evidence_found,
    }
