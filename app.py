import html
import os

import pandas as pd
import streamlit as st
from dotenv import load_dotenv

from modules.ats_analyzer import analyze_ats_detailed
from modules.career_recommender import recommend_career
from modules.genai_engine import (
    analyze_resume_with_ai,
    estimate_skill_roi,
    fallback_skill_roi,
    generate_orphan_proof_bullets,
    generate_pivot_bridge,
    generate_skill_verification_quiz,
    generate_tailored_keywords,
    recommend_careers_with_ai,
    resolve_api_key,
    run_brutal_recruiter_audit,
    semantic_skill_gap_and_roadmap,
)
from modules.job_matcher import calculate_job_match
from modules.orphan_skill_auditor import audit_orphan_skills
from modules.resume_parser import extract_resume_text
from modules.skill_extractor import extract_skills, extract_skills_by_category
from modules.skill_gap import ROLE_SKILLS, analyze_skill_gap

load_dotenv()

PAGES = [
    "Dashboard",
    "Resume & ATS Analysis",
    "Job Matching",
    "Skill Gap & 4-Week Roadmap",
    "Career Recommendations",
    "Skill Verification Quiz",
]

st.set_page_config(
    page_title="CareerIQ | AI Career & Skill Gap Platform",
    layout="wide",
    initial_sidebar_state="expanded",
)

try:
    with open("assets/style.css", "r", encoding="utf-8") as css_file:
        st.markdown(f"<style>{css_file.read()}</style>", unsafe_allow_html=True)
except FileNotFoundError:
    pass


def _init_session_state():
    defaults = {
        "nav_page": "Dashboard",
        "resume_text": "",
        "extracted_skills": [],
        "skills_by_category": {},
        "resume_source": "",
        "resume_signature": "",
        "api_key_override": "",
        "ai_resume_analysis": None,
        "orphan_proof_suggestions": None,
        "brutal_audit_result": None,
        "job_match_result": None,
        "job_keywords_result": None,
        "roadmap_result": None,
        "roi_sim_data": None,
        "ai_career_paths": None,
        "pivot_bridge_result": None,
        "quiz_questions": None,
        "quiz_submitted": False,
        "quiz_user_answers": {},
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def _clear_ai_caches():
    st.session_state["ai_resume_analysis"] = None
    st.session_state["orphan_proof_suggestions"] = None
    st.session_state["brutal_audit_result"] = None
    st.session_state["job_match_result"] = None
    st.session_state["job_keywords_result"] = None
    st.session_state["roadmap_result"] = None
    st.session_state["roi_sim_data"] = None
    st.session_state["ai_career_paths"] = None
    st.session_state["pivot_bridge_result"] = None
    st.session_state["quiz_questions"] = None
    st.session_state["quiz_submitted"] = False
    st.session_state["quiz_user_answers"] = {}


def _set_resume_data(text: str, source_label: str, signature: str):
    if st.session_state.get("resume_signature") != signature:
        st.session_state["resume_text"] = text
        st.session_state["extracted_skills"] = extract_skills(text)
        st.session_state["skills_by_category"] = extract_skills_by_category(text)
        st.session_state["resume_source"] = source_label
        st.session_state["resume_signature"] = signature
        _clear_ai_caches()


def _navigate_to(page_name: str):
    st.session_state["nav_page"] = page_name


def _get_active_api_key() -> str | None:
    override = st.session_state.get("api_key_override", "").strip()
    if override:
        return override
    env_key = (os.getenv("GEMINI_API_KEY") or "").strip()
    if env_key and env_key != "your_api_key_here":
        return env_key
    return None


def _render_skill_pills(skills: list[str], variant: str = "default"):
    if not skills:
        st.caption("None detected.")
        return
    cls = "skill-tag"
    if variant == "matched":
        cls = "skill-tag skill-tag-matched"
    elif variant == "missing":
        cls = "skill-tag skill-tag-missing"
    pills_html = "".join(
        f'<span class="{cls}">{html.escape(skill)}</span>' for skill in skills
    )
    st.markdown(pills_html, unsafe_allow_html=True)


def _require_resume() -> bool:
    if not st.session_state.get("resume_text"):
        st.info(
            "Upload a PDF resume or paste resume text in the sidebar to activate this module."
        )
        return False
    return True


_init_session_state()

# SIDEBAR
with st.sidebar:
    st.title("CareerIQ")
    st.caption("Hybrid ML + Generative AI Career Platform")
    st.markdown("---")

    st.subheader("Navigation")
    page = st.radio(
        "Select Workspace",
        PAGES,
        key="nav_page",
        label_visibility="collapsed",
    )

    st.markdown("---")
    st.subheader("Global Resume Input")
    input_mode = st.radio(
        "Resume Source",
        ["Upload PDF", "Paste Raw Text"],
        horizontal=True,
    )

    if input_mode == "Upload PDF":
        uploaded_file = st.file_uploader(
            "Upload Resume (PDF)",
            type=["pdf"],
            help="Upload once to power ATS scoring, job matching, roadmaps, and quizzes across all pages.",
        )
        if uploaded_file is not None:
            sig = f"pdf:{uploaded_file.name}:{uploaded_file.size}"
            if st.session_state.get("resume_signature") != sig:
                try:
                    parsed_text = extract_resume_text(uploaded_file)
                    if not parsed_text:
                        st.error("No extractable text found in the uploaded PDF.")
                    else:
                        _set_resume_data(parsed_text, f"PDF ({uploaded_file.name})", sig)
                except ValueError as exc:
                    st.error(str(exc))
    else:
        pasted_text = st.text_area(
            "Paste Resume Text",
            height=150,
            placeholder="Paste resume content here for quick analysis...",
        )
        if st.button("Load Pasted Resume", use_container_width=True):
            cleaned = pasted_text.strip()
            if not cleaned:
                st.warning("Please paste resume text before loading.")
            else:
                sig = f"text:{hash(cleaned)}"
                _set_resume_data(cleaned, "Pasted Text", sig)

    if st.session_state.get("resume_text"):
        word_count = len(st.session_state["resume_text"].split())
        skill_count = len(st.session_state["extracted_skills"])
        st.success(
            f"Active: {st.session_state['resume_source']} | {word_count} words | {skill_count} skills"
        )
        if st.button("Clear Active Resume", use_container_width=True):
            st.session_state["resume_text"] = ""
            st.session_state["extracted_skills"] = []
            st.session_state["skills_by_category"] = {}
            st.session_state["resume_source"] = ""
            st.session_state["resume_signature"] = ""
            _clear_ai_caches()
            st.rerun()

    st.markdown("---")
    st.subheader("Gemini API Configuration")
    api_override = st.text_input(
        "GEMINI_API_KEY (Optional Override)",
        type="password",
        value=st.session_state["api_key_override"],
        placeholder="Paste key to override .env",
    )
    st.session_state["api_key_override"] = api_override

    active_key = _get_active_api_key()
    if st.session_state["api_key_override"].strip():
        st.caption("API Status: Configured (Sidebar Override)")
    elif active_key:
        st.caption("API Status: Configured (.env)")
    else:
        st.caption("API Status: Not Configured (AI features require key)")


# MAIN HEADER
st.markdown(
    """
    <div class="careeriq-header">
        <div class="careeriq-title">CareerIQ</div>
        <div class="careeriq-subtitle">
            Deterministic ATS & Skill Analytics paired with Gemini 3.8 Flash Career Intelligence
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)


# PAGE 1: DASHBOARD
if page == "Dashboard":
    st.header("Executive Dashboard")

    if not st.session_state["resume_text"]:
        st.info(
            "No active resume loaded. Use the Global Resume Input panel in the sidebar to upload a PDF or paste text."
        )
        col_a, col_b, col_c = st.columns(3)
        with col_a:
            st.markdown(
                """
                <div class="iq-card">
                    <h4>ATS & AI Critique</h4>
                    <p>Evaluate section completeness, quantifiable metrics, action verbs, and receive Gemini-rewritten bullets.</p>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with col_b:
            st.markdown(
                """
                <div class="iq-card">
                    <h4>Job Matching & Skill Gap</h4>
                    <p>Measure regex-verified JD skill alignment and generate a personalized 4-week roadmap with a capstone project.</p>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with col_c:
            st.markdown(
                """
                <div class="iq-card">
                    <h4>Career Fit & Verification</h4>
                    <p>Discover AI-ranked career paths and validate claimed skills with a 3-question diagnostic quiz.</p>
                </div>
                """,
                unsafe_allow_html=True,
            )
    else:
        resume_text = st.session_state["resume_text"]
        extracted_skills = st.session_state["extracted_skills"]
        skills_by_cat = st.session_state["skills_by_category"]
        ats_details = analyze_ats_detailed(resume_text)

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Word Count", ats_details["word_count"])
        m2.metric("Detected Skills", len(extracted_skills))
        m3.metric("ATS Readiness Score", f"{ats_details['score']}%")
        m4.metric(
            "Sections Found",
            f"{len(ats_details['found_sections'])} / {len(ats_details['found_sections']) + len(ats_details['missing_sections'])}",
        )

        st.markdown("---")
        left_col, right_col = st.columns([3, 2])

        with left_col:
            st.subheader("Extracted Skill Inventory")
            if skills_by_cat:
                for category, cat_skills in skills_by_cat.items():
                    st.markdown(f"**{category}**")
                    _render_skill_pills(cat_skills)
            else:
                _render_skill_pills(extracted_skills)

        with right_col:
            st.subheader("Quick Navigation")
            st.button(
                "Open Resume & ATS Analysis",
                on_click=_navigate_to,
                args=("Resume & ATS Analysis",),
                use_container_width=True,
            )
            st.button(
                "Open Job Matching",
                on_click=_navigate_to,
                args=("Job Matching",),
                use_container_width=True,
            )
            st.button(
                "Open Skill Gap & 4-Week Roadmap",
                on_click=_navigate_to,
                args=("Skill Gap & 4-Week Roadmap",),
                use_container_width=True,
            )
            st.button(
                "Open Career Recommendations",
                on_click=_navigate_to,
                args=("Career Recommendations",),
                use_container_width=True,
            )
            st.button(
                "Open Skill Verification Quiz",
                on_click=_navigate_to,
                args=("Skill Verification Quiz",),
                use_container_width=True,
            )

        st.markdown("---")
        with st.expander("Preview Parsed Resume Text"):
            st.text(resume_text[:4000] + ("..." if len(resume_text) > 4000 else ""))


# PAGE 2: RESUME & ATS ANALYSIS
elif page == "Resume & ATS Analysis":
    st.header("Resume & ATS Analysis")

    if _require_resume():
        resume_text = st.session_state["resume_text"]
        extracted_skills = st.session_state["extracted_skills"]

        tab_ats, tab_orphan, tab_brutal = st.tabs(
            ["ATS & AI Critique", "Orphan Skill Auditor", "Brutal Recruiter Audit"]
        )

        with tab_ats:
            ats = analyze_ats_detailed(resume_text)

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Overall ATS Score", f"{ats['score']}%")
            c2.metric("Section Score", f"{ats['section_score']} / 60")
            c3.metric("Metrics Score", f"{ats['metrics_score']} / 20")
            c4.metric("Action Verbs Score", f"{ats['action_verb_score']} / 20")

            st.markdown("---")
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("Detected Sections")
                if ats["found_sections"]:
                    for sec in ats["found_sections"]:
                        st.write(f"- {sec}")
                else:
                    st.caption("No standard resume sections detected.")

                st.subheader("Quantifiable Metrics Detected")
                if ats["metrics_found"]:
                    _render_skill_pills(ats["metrics_found"][:15], variant="matched")
                else:
                    st.caption("No quantifiable metrics (%, $, multipliers, or counts) detected.")

            with col2:
                st.subheader("Missing Sections")
                if ats["missing_sections"]:
                    for sec in ats["missing_sections"]:
                        st.write(f"- {sec}")
                else:
                    st.caption("All core ATS sections are present.")

                st.subheader("Action Verbs Detected")
                if ats["action_verbs_found"]:
                    _render_skill_pills(ats["action_verbs_found"][:15])
                else:
                    st.caption("No strong ATS action verbs detected.")

            st.subheader("Deterministic Recommendations")
            for suggestion in ats["suggestions"]:
                st.write(f"- {suggestion}")

            st.markdown("---")
            st.subheader("AI Resume Critique & Bullet Rewriter")
            if st.button("Run AI Resume Critique & Bullet Rewriter", type="primary"):
                with st.spinner("Analyzing resume with Gemini 3.8 Flash..."):
                    try:
                        resolve_api_key(_get_active_api_key())
                        st.session_state["ai_resume_analysis"] = analyze_resume_with_ai(
                            resume_text,
                            api_key=_get_active_api_key(),
                        )
                    except Exception as exc:
                        st.error(str(exc))

            ai_result = st.session_state.get("ai_resume_analysis")
            if ai_result:
                s_col, w_col, m_col = st.columns(3)
                with s_col:
                    st.markdown("**Strengths**")
                    for item in ai_result.get("strengths", []):
                        st.write(f"- {item}")
                with w_col:
                    st.markdown("**Weaknesses**")
                    for item in ai_result.get("weaknesses", []):
                        st.write(f"- {item}")
                with m_col:
                    st.markdown("**Missing Impact Metrics**")
                    for item in ai_result.get("missing_impact_metrics", []):
                        st.write(f"- {item}")

                st.markdown("#### ATS-Optimized Bullet Rewrites (Before vs. After)")
                for idx, bullet in enumerate(ai_result.get("rewritten_bullets", []), start=1):
                    before_txt = html.escape(bullet.get("before", ""))
                    after_txt = html.escape(bullet.get("after", ""))
                    rationale_txt = html.escape(bullet.get("rationale", ""))
                    st.markdown(
                        f"""
                        <div class="rewrite-box">
                            <div class="rewrite-label">Rewrite {idx} - Before</div>
                            <p>{before_txt}</p>
                            <div class="rewrite-label">After (ATS-Optimized)</div>
                            <p><strong>{after_txt}</strong></p>
                            <div class="rewrite-label">Rationale</div>
                            <p>{rationale_txt}</p>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

        with tab_orphan:
            st.subheader("Orphan Skill Auditor (Claim vs. Proof Detector)")
            st.caption(
                "Cross-references skills claimed on your resume against your Projects, Experience, Internships, and Research sections."
            )
            audit = audit_orphan_skills(resume_text, extracted_skills)

            o1, o2, o3 = st.columns(3)
            o1.metric("Proof Ratio", f"{audit['proof_ratio']}%")
            o2.metric("Verified Skills", len(audit["verified_skills"]))
            o3.metric("Orphan Skills", len(audit["orphan_skills"]))

            if not audit["evidence_section_found"]:
                st.info(
                    "Explicit Projects/Experience headers were not detected; scanned non-skills resume text as evidence fallback."
                )

            st.markdown("---")
            v_col, u_col = st.columns(2)
            with v_col:
                st.subheader("Verified Skills (Backed by Evidence)")
                if audit["verified_skills"]:
                    _render_skill_pills(
                        [item["skill"] for item in audit["verified_skills"]],
                        variant="matched",
                    )
                    st.markdown("")
                    for item in audit["verified_skills"]:
                        with st.expander(f"Evidence: {item['skill']}"):
                            st.write(item["snippet"])
                else:
                    st.caption("No extracted skills were found in project or experience sections.")

            with u_col:
                st.subheader("Orphan Skills (Zero Evidence Found)")
                if audit["orphan_skills"]:
                    _render_skill_pills(audit["orphan_skills"], variant="missing")
                    st.caption(
                        "These skills are listed on your resume without supporting project or work experience bullets."
                    )
                else:
                    st.success("All detected skills have supporting evidence in your experience or projects.")

            if audit["orphan_skills"]:
                st.markdown("---")
                if st.button("Generate Proof Bullets for Orphan Skills", type="primary"):
                    with st.spinner("Drafting project-backed proof suggestions with Gemini 3.8 Flash..."):
                        try:
                            resolve_api_key(_get_active_api_key())
                            st.session_state["orphan_proof_suggestions"] = generate_orphan_proof_bullets(
                                resume_text=resume_text,
                                orphan_skills=audit["orphan_skills"],
                                api_key=_get_active_api_key(),
                            )
                        except Exception as exc:
                            st.error(str(exc))

                proof_suggestions = st.session_state.get("orphan_proof_suggestions")
                if proof_suggestions:
                    st.markdown("#### Suggested Proof Bullets")
                    for item in proof_suggestions:
                        skill_name = html.escape(item.get("orphan_skill", ""))
                        target_sec = html.escape(item.get("target_project_or_role", ""))
                        bullet_txt = html.escape(item.get("suggested_bullet", ""))
                        tip_txt = html.escape(item.get("verification_tip", ""))
                        st.markdown(
                            f"""
                            <div class="rewrite-box">
                                <div class="rewrite-label">{skill_name} | Target: {target_sec}</div>
                                <p><strong>{bullet_txt}</strong></p>
                                <p>Verification Tip: {tip_txt}</p>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )

        with tab_brutal:
            st.subheader("Brutal Recruiter Red-Flag & Skeptic Audit")
            st.caption(
                "Simulates a strict 6-second hiring manager screen to expose buzzword inflation, vague claims, and interview traps."
            )
            brutal_role = st.text_input(
                "Optional Target Role for Strict Screening",
                placeholder="e.g., Backend Software Engineer, Data Scientist, ML Engineer",
                key="brutal_target_role",
            )

            if st.button("Run Brutal Recruiter Audit", type="primary"):
                with st.spinner("Running skeptical hiring-manager audit with Gemini 3.8 Flash..."):
                    try:
                        resolve_api_key(_get_active_api_key())
                        st.session_state["brutal_audit_result"] = run_brutal_recruiter_audit(
                            resume_text=resume_text,
                            target_role=brutal_role,
                            api_key=_get_active_api_key(),
                        )
                    except Exception as exc:
                        st.error(str(exc))

            brutal_res = st.session_state.get("brutal_audit_result")
            if brutal_res:
                verdict = brutal_res.get("overall_verdict", "Borderline")
                banner_cls = "verdict-borderline"
                if verdict == "Strong Pass":
                    banner_cls = "verdict-pass"
                elif verdict == "High Rejection Risk":
                    banner_cls = "verdict-risk"

                st.markdown(
                    f"""
                    <div class="verdict-banner {banner_cls}">
                        <strong>Overall Verdict: {html.escape(verdict)}</strong>
                        <p style="margin-top: 0.4rem; margin-bottom: 0;">
                            <strong>6-Second Impression:</strong> {html.escape(brutal_res.get("six_second_impression", ""))}
                        </p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                st.markdown("#### Red Flags Identified")
                for idx, flag in enumerate(brutal_res.get("red_flags", []), start=1):
                    title = f"Flag {idx} [{flag.get('category', 'Issue')}]: {flag.get('issue', '')}"
                    with st.expander(title, expanded=True):
                        st.markdown(f"**Trigger / Gap:** {flag.get('exact_quote_or_gap', '')}")
                        st.markdown(f"**How to Fix:** {flag.get('fix', '')}")

                st.markdown("#### Trap Interview Questions")
                for idx, trap in enumerate(brutal_res.get("trap_interview_questions", []), start=1):
                    q_title = f"Trap Question {idx}: {trap.get('question', '')}"
                    with st.expander(q_title, expanded=(idx == 1)):
                        st.markdown(
                            f"**What the Interviewer Is Testing:** {trap.get('what_interviewer_is_testing', '')}"
                        )
                        st.markdown(f"**Defense Strategy:** {trap.get('how_to_defend', '')}")


# PAGE 3: JOB MATCHING
elif page == "Job Matching":
    st.header("Job Matching")

    if _require_resume():
        resume_text = st.session_state["resume_text"]

        job_description = st.text_area(
            "Paste Target Job Description",
            height=220,
            placeholder="Paste the full job description including requirements and responsibilities...",
        )
        include_ai_keywords = st.checkbox(
            "Generate AI-Tailored Resume Keywords (Gemini 3.8 Flash)",
            value=bool(_get_active_api_key()),
        )

        if st.button("Analyze Match", type="primary"):
            if not job_description.strip():
                st.warning("Please paste a job description first.")
            else:
                score, matched_skills, missing_skills = calculate_job_match(
                    resume_text,
                    job_description,
                )
                st.session_state["job_match_result"] = {
                    "score": score,
                    "matched": matched_skills,
                    "missing": missing_skills,
                    "jd": job_description,
                }
                st.session_state["job_keywords_result"] = None

                if include_ai_keywords:
                    with st.spinner("Generating AI-tailored resume keywords..."):
                        try:
                            resolve_api_key(_get_active_api_key())
                            st.session_state["job_keywords_result"] = generate_tailored_keywords(
                                resume_text,
                                job_description,
                                missing_skills=missing_skills,
                                api_key=_get_active_api_key(),
                            )
                        except Exception as exc:
                            st.error(str(exc))

        match_res = st.session_state.get("job_match_result")
        if match_res:
            m1, m2, m3 = st.columns(3)
            m1.metric("Verified Skill Match Score", f"{match_res['score']}%")
            m2.metric("Matched JD Skills", len(match_res["matched"]))
            m3.metric("Missing JD Skills", len(match_res["missing"]))

            st.progress(min(max(match_res["score"], 0), 100) / 100)

            col1, col2 = st.columns(2)
            with col1:
                st.subheader("Matched Skills")
                _render_skill_pills(match_res["matched"], variant="matched")
            with col2:
                st.subheader("Missing Skills")
                _render_skill_pills(match_res["missing"], variant="missing")

            kw_res = st.session_state.get("job_keywords_result")
            if kw_res:
                st.markdown("---")
                st.subheader("AI-Tailored Resume Keywords & Positioning")
                _render_skill_pills(kw_res.get("priority_keywords", []))
                st.markdown("**Tailoring Recommendations**")
                for tip in kw_res.get("tailoring_tips", []):
                    st.write(f"- {tip}")


# PAGE 4: SKILL GAP & 4-WEEK ROADMAP
elif page == "Skill Gap & 4-Week Roadmap":
    st.header("Skill Gap & 4-Week Roadmap")

    if _require_resume():
        resume_text = st.session_state["resume_text"]

        role_mode = st.radio(
            "Target Role Input Mode",
            ["Select Preset Domain", "Enter Custom Target Role"],
            horizontal=True,
        )

        if role_mode == "Select Preset Domain":
            target_role = st.selectbox("Target Domain", list(ROLE_SKILLS.keys()))
            is_preset = True
        else:
            target_role = st.text_input(
                "Custom Target Role",
                placeholder="e.g., Generative AI Engineer, MLOps Engineer, Product Analyst",
            )
            is_preset = False

        existing_skills = []
        missing_skills = []
        if is_preset and target_role:
            existing_skills, missing_skills = analyze_skill_gap(resume_text, target_role)
            total = len(existing_skills) + len(missing_skills)
            readiness = round((len(existing_skills) / total) * 100) if total > 0 else 0

            r1, r2, r3 = st.columns(3)
            r1.metric("Preset Readiness Score", f"{readiness}%")
            r2.metric("Existing Role Skills", len(existing_skills))
            r3.metric("Missing Role Skills", len(missing_skills))

            col1, col2 = st.columns(2)
            with col1:
                st.subheader("Existing Skills")
                _render_skill_pills(existing_skills, variant="matched")
            with col2:
                st.subheader("Missing Skills")
                _render_skill_pills(missing_skills, variant="missing")
            st.markdown("---")
        elif not is_preset:
            st.caption(
                "Custom roles use Gemini semantic evaluation to infer role-specific competencies and build a tailored 4-week roadmap."
            )

        tab_roadmap, tab_roi = st.tabs(
            ["4-Week Roadmap & Capstone", "What-If Skill ROI Simulator"]
        )

        with tab_roadmap:
            if st.button("Generate 4-Week Roadmap & Capstone Plan", type="primary"):
                if not target_role or not target_role.strip():
                    st.warning("Please select or enter a target role.")
                else:
                    with st.spinner("Building prioritized skill gap & 4-week roadmap..."):
                        try:
                            resolve_api_key(_get_active_api_key())
                            st.session_state["roadmap_result"] = semantic_skill_gap_and_roadmap(
                                resume_text=resume_text,
                                target_role=target_role.strip(),
                                missing_skills=missing_skills,
                                api_key=_get_active_api_key(),
                            )
                        except Exception as exc:
                            st.error(str(exc))

            roadmap = st.session_state.get("roadmap_result")
            if roadmap:
                st.subheader("Role Fit Summary")
                st.markdown(
                    f"""
                    <div class="iq-card">
                        <p>{html.escape(roadmap.get("role_fit_summary", ""))}</p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                st.subheader("Prioritized Missing Skills")
                prioritized = roadmap.get("prioritized_missing_skills", [])
                if prioritized:
                    table_rows = [
                        {
                            "Skill": item.get("skill", ""),
                            "Priority": item.get("priority", ""),
                            "Prerequisites": ", ".join(item.get("prerequisites", [])) or "None",
                            "Rationale": item.get("reason", ""),
                        }
                        for item in prioritized
                    ]
                    st.dataframe(pd.DataFrame(table_rows), use_container_width=True, hide_index=True)

                st.subheader("4-Week Learning Roadmap")
                for week_item in roadmap.get("four_week_roadmap", []):
                    week_title = f"{week_item.get('week', 'Week')}: {week_item.get('focus_topic', '')}"
                    with st.expander(week_title, expanded=True):
                        st.markdown(
                            f"**Recommended Resource Type:** {week_item.get('recommended_resource_type', '')}"
                        )
                        st.markdown("**Action Items:**")
                        for action in week_item.get("action_items", []):
                            st.write(f"- {action}")

                capstone = roadmap.get("capstone_mini_project", {})
                if capstone:
                    st.subheader("Capstone Mini-Project")
                    st.markdown(
                        f"""
                        <div class="iq-card">
                            <h4>{html.escape(capstone.get("title", "Portfolio Capstone"))}</h4>
                            <p>{html.escape(capstone.get("description", ""))}</p>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
                    st.markdown("**Target Tech Stack**")
                    _render_skill_pills(capstone.get("tech_stack", []))
                    st.markdown("**Acceptance Criteria**")
                    for idx, criterion in enumerate(capstone.get("acceptance_criteria", []), start=1):
                        st.write(f"{idx}. {criterion}")

        with tab_roi:
            st.subheader("What-If Skill ROI Simulator")
            st.caption(
                "Toggle missing skills to simulate projected readiness gains and total study hours without extra API calls."
            )

            sim_skills = list(missing_skills)
            if not is_preset:
                roadmap_state = st.session_state.get("roadmap_result") or {}
                inferred = [
                    item.get("skill", "")
                    for item in roadmap_state.get("prioritized_missing_skills", [])
                    if item.get("skill")
                ]
                custom_roi_input = st.text_input(
                    "Skills to Simulate (comma-separated)",
                    value=", ".join(inferred),
                    placeholder="e.g., Kubernetes, Terraform, LangChain, PyTorch",
                )
                sim_skills = [s.strip() for s in custom_roi_input.split(",") if s.strip()]

            if not sim_skills:
                if is_preset and existing_skills:
                    st.success("All required skills for this preset domain are already present on your resume.")
                else:
                    st.info("Select a preset domain with missing skills, generate a custom roadmap, or enter skills above to simulate ROI.")
            else:
                roi_cache_key = f"{target_role.strip()}|{','.join(sim_skills)}"
                cached_roi = st.session_state.get("roi_sim_data")
                if not cached_roi or cached_roi.get("cache_key") != roi_cache_key:
                    st.session_state["roi_sim_data"] = {
                        "cache_key": roi_cache_key,
                        "source": "Baseline Offline Estimates (15 hrs, Weight 5)",
                        "items": fallback_skill_roi(sim_skills),
                    }

                btn_col1, btn_col2 = st.columns([2, 3])
                with btn_col1:
                    if st.button("Estimate Skill ROI with Gemini 3.8 Flash", type="primary"):
                        with st.spinner("Estimating learning hours and role impact weights..."):
                            items = estimate_skill_roi(
                                missing_skills=sim_skills,
                                target_role=target_role.strip() or "Target Role",
                                api_key=_get_active_api_key(),
                            )
                            has_key = bool(_get_active_api_key())
                            st.session_state["roi_sim_data"] = {
                                "cache_key": roi_cache_key,
                                "source": (
                                    "Gemini 3.8 Flash Calibrated Estimates"
                                    if has_key
                                    else "Baseline Offline Estimates (No API Key)"
                                ),
                                "items": items,
                            }
                with btn_col2:
                    st.caption(f"Estimate Source: {st.session_state['roi_sim_data']['source']}")

                roi_items = st.session_state["roi_sim_data"]["items"]
                st.markdown("**Select Skills to Acquire:**")
                checked_items = []
                unchecked_items = []
                for idx, item in enumerate(roi_items):
                    skill_name = item["skill"]
                    est_hrs = int(item["estimated_hours"])
                    weight = int(item["impact_weight"])
                    chk_key = f"roi_chk_{abs(hash(roi_cache_key))}_{idx}"
                    label = f"{skill_name} | Est. {est_hrs} hrs | Impact Weight: {weight}/10"
                    is_checked = st.checkbox(label, key=chk_key)
                    if item.get("why_it_matters"):
                        st.caption(item["why_it_matters"])
                    if is_checked:
                        checked_items.append(item)
                    else:
                        unchecked_items.append(item)

                total_pool = max(len(existing_skills) + len(roi_items), 1)
                baseline_score = round((len(existing_skills) / total_pool) * 100)
                projected_score = round(((len(existing_skills) + len(checked_items)) / total_pool) * 100)
                score_delta = projected_score - baseline_score
                total_hours = sum(int(item["estimated_hours"]) for item in checked_items)

                st.markdown("---")
                m1, m2, m3 = st.columns(3)
                m1.metric("Baseline Readiness Score", f"{baseline_score}%")
                m2.metric(
                    "Projected Readiness Score",
                    f"{projected_score}%",
                    delta=f"+{score_delta}%" if score_delta > 0 else "0%",
                )
                m3.metric("Total Learning Hours Selected", f"{total_hours} hrs")

                if unchecked_items:
                    best_skill = max(
                        unchecked_items,
                        key=lambda x: int(x["impact_weight"]) / max(int(x["estimated_hours"]), 1),
                    )
                    best_ratio = round(
                        int(best_skill["impact_weight"]) / max(int(best_skill["estimated_hours"]), 1),
                        2,
                    )
                    st.markdown(
                        f"""
                        <div class="roi-callout">
                            <strong>Highest-ROI Next Skill: {html.escape(best_skill['skill'])}</strong>
                            (Impact-to-Effort Ratio: {best_ratio} | Impact Weight: {best_skill['impact_weight']}/10 | Est. {best_skill['estimated_hours']} hrs)<br/>
                            {html.escape(best_skill.get('why_it_matters', ''))}
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
                else:
                    st.success("All simulated missing skills are selected (100% projected coverage).")


# PAGE 5: CAREER RECOMMENDATIONS
elif page == "Career Recommendations":
    st.header("Career Recommendations")

    tab_career, tab_pivot = st.tabs(
        ["Career Path Recommendations", "Transferable Skill Pivot Bridge"]
    )

    with tab_career:
        st.subheader("Academic Specialization Baseline")
        spec_col1, spec_col2 = st.columns([3, 1])
        with spec_col1:
            specialization = st.text_input(
                "Enter Specialization (e.g., CSE, IT, ECE, MECH, AI/ML)",
                value="CSE",
            )
        with spec_col2:
            st.write("")
            st.write("")
            run_spec = st.button("Map Specialization", use_container_width=True)

        if run_spec or specialization:
            roles, req_skills = recommend_career(specialization)
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**Baseline Roles**")
                for role_name in roles:
                    st.write(f"- {role_name}")
            with col2:
                st.markdown("**Foundational Skills**")
                _render_skill_pills(req_skills)

        st.markdown("---")
        st.subheader("AI-Powered Personalized Career Paths")
        if _require_resume():
            resume_text = st.session_state["resume_text"]
            extracted_skills = st.session_state["extracted_skills"]

            if st.button("Generate Personalized Career Paths", type="primary"):
                with st.spinner("Evaluating resume fit across career trajectories..."):
                    try:
                        resolve_api_key(_get_active_api_key())
                        st.session_state["ai_career_paths"] = recommend_careers_with_ai(
                            resume_text=resume_text,
                            extracted_skills=extracted_skills,
                            api_key=_get_active_api_key(),
                        )
                    except Exception as exc:
                        st.error(str(exc))

            career_paths = st.session_state.get("ai_career_paths")
            if career_paths:
                cols = st.columns(len(career_paths))
                for col, path in zip(cols, career_paths):
                    with col:
                        match_pct = int(path.get("match_percentage", 0))
                        st.markdown(
                            f"""
                            <div class="iq-card">
                                <h4>{html.escape(path.get("role_title", ""))}</h4>
                                <p><strong>Fit Score:</strong> {match_pct}%</p>
                                <p><strong>Why It Fits:</strong> {html.escape(path.get("why_it_fits", ""))}</p>
                                <p><strong>Immediate Next Step:</strong> {html.escape(path.get("next_step", ""))}</p>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )
                        st.progress(min(max(match_pct, 0), 100) / 100)

    with tab_pivot:
        st.subheader("Transferable Skill Pivot Bridge")
        st.caption(
            "Translate your existing skills and projects into interview-ready equivalents for any target pivot role."
        )
        if _require_resume():
            resume_text = st.session_state["resume_text"]
            extracted_skills = st.session_state["extracted_skills"]

            pivot_role = st.text_input(
                "Target Pivot Role",
                placeholder="e.g., Data Analyst, Product Manager, ML Engineer, FinOps Analyst",
                key="pivot_target_role",
            )

            if st.button("Build Pivot Bridge", type="primary"):
                if not pivot_role.strip():
                    st.warning("Please enter a target pivot role.")
                else:
                    with st.spinner("Mapping transferable skills with Gemini 3.8 Flash..."):
                        try:
                            resolve_api_key(_get_active_api_key())
                            st.session_state["pivot_bridge_result"] = generate_pivot_bridge(
                                resume_text=resume_text,
                                extracted_skills=extracted_skills,
                                target_role=pivot_role.strip(),
                                api_key=_get_active_api_key(),
                            )
                        except Exception as exc:
                            st.error(str(exc))

            pivot_res = st.session_state.get("pivot_bridge_result")
            if pivot_res:
                st.markdown("#### Pivot Feasibility Summary")
                st.markdown(
                    f"""
                    <div class="iq-card">
                        <p>{html.escape(pivot_res.get("pivot_feasibility_summary", ""))}</p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                bridges = pivot_res.get("transferable_bridges", [])
                if bridges:
                    st.markdown("#### Transferable Competency Mapping")
                    bridge_rows = [
                        {
                            "Your Current Asset": b.get("existing_project_or_skill", ""),
                            "Target Role Equivalent": b.get("target_role_equivalent", ""),
                            "How to Reframe": b.get("how_to_reframe_in_interview", ""),
                        }
                        for b in bridges
                    ]
                    st.dataframe(pd.DataFrame(bridge_rows), use_container_width=True, hide_index=True)

                st.markdown("#### Tailored 3-Sentence Pivot Pitch")
                st.markdown(
                    f"""
                    <div class="rewrite-box">
                        <div class="rewrite-label">Tell Me About Yourself - Pivot Pitch</div>
                        <p>{html.escape(pivot_res.get("reframed_pitch", ""))}</p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )


# PAGE 6: SKILL VERIFICATION QUIZ
elif page == "Skill Verification Quiz":
    st.header("Skill Verification Quiz")

    if _require_resume():
        extracted_skills = st.session_state["extracted_skills"]

        st.markdown("**Skills Claimed on Resume**")
        if extracted_skills:
            _render_skill_pills(extracted_skills)
        else:
            st.warning(
                "No built-in taxonomy skills were detected automatically. Enter skills manually below to generate a quiz."
            )

        custom_skills_input = st.text_input(
            "Optional: Specify or Override Skills to Test (comma-separated)",
            placeholder="e.g., Python, SQL, Docker, Machine Learning",
        )

        if st.button("Generate 3-Question Verification Quiz", type="primary"):
            if custom_skills_input.strip():
                quiz_skills = [s.strip() for s in custom_skills_input.split(",") if s.strip()]
            else:
                quiz_skills = extracted_skills

            if not quiz_skills:
                st.warning("Provide at least one skill to generate the quiz.")
            else:
                with st.spinner("Generating technical verification questions..."):
                    try:
                        resolve_api_key(_get_active_api_key())
                        st.session_state["quiz_questions"] = generate_skill_verification_quiz(
                            claimed_skills=quiz_skills,
                            api_key=_get_active_api_key(),
                        )
                        st.session_state["quiz_submitted"] = False
                        st.session_state["quiz_user_answers"] = {}
                    except Exception as exc:
                        st.error(str(exc))

        questions = st.session_state.get("quiz_questions")
        if questions:
            st.markdown("---")
            with st.form("skill_verification_form"):
                selected_answers = {}
                for idx, q in enumerate(questions):
                    st.markdown(f"**Question {idx + 1}. {q.get('question', '')}**")
                    selected_answers[idx] = st.radio(
                        f"Select answer for Question {idx + 1}",
                        options=q.get("options", []),
                        index=None,
                        key=f"quiz_q_{idx}",
                        label_visibility="collapsed",
                    )
                    st.markdown("")

                submitted = st.form_submit_button("Submit Quiz Answers", type="primary")
                if submitted:
                    if any(selected_answers[i] is None for i in range(len(questions))):
                        st.warning("Please answer all 3 questions before submitting.")
                    else:
                        st.session_state["quiz_user_answers"] = selected_answers
                        st.session_state["quiz_submitted"] = True

            if st.session_state.get("quiz_submitted"):
                answers = st.session_state["quiz_user_answers"]
                correct_count = sum(
                    1
                    for idx, q in enumerate(questions)
                    if answers.get(idx) == q.get("correct_answer")
                )
                st.metric("Diagnostic Score", f"{correct_count} / {len(questions)}")

                for idx, q in enumerate(questions):
                    user_ans = answers.get(idx)
                    correct_ans = q.get("correct_answer")
                    is_correct = user_ans == correct_ans

                    if is_correct:
                        st.success(f"Question {idx + 1}: Correct ({correct_ans})")
                    else:
                        st.error(
                            f"Question {idx + 1}: Incorrect | Your answer: {user_ans} | Correct answer: {correct_ans}"
                        )
                    st.caption(f"Explanation: {q.get('explanation', '')}")
