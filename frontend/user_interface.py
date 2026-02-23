"""
Streamlit UI for the Career Architect: Cyclic Multi-Agent Career Strategist.
Handles resume upload, job analysis, HITL approval, and cold email display.
"""
import os
from uuid import uuid4
import requests
import streamlit as st
from dotenv import load_dotenv

load_dotenv()
BASE_URL = os.getenv("BASE_URL")
JOB_ANALYSIS_ENDPOINT = os.getenv("JOB_ANALYSIS_ENDPOINT")
RESUME_ENDPOINT = os.getenv("RESUME_ENDPOINT")

st.set_page_config(page_title="Career Architect", page_icon="📧", layout="centered")
st.title("The Career Architect")
st.caption("Research → Analyze → Approve → Cold Email")

# Session state
if "thread_id" not in st.session_state:
    st.session_state.thread_id = None
if "analysis_state" not in st.session_state:
    st.session_state.analysis_state = None
if "awaiting_decision" not in st.session_state:
    st.session_state.awaiting_decision = False


def _resume_with(decision: str):
    tid = st.session_state.thread_id
    if not tid:
        return
    with st.spinner("Generating..." if decision == "yes" else "Skipping..."):
        try:
            r = requests.post(
                f"{BASE_URL}{RESUME_ENDPOINT}",
                json={"thread_id": tid, "decision": decision},
                timeout=60,
            )
            r.raise_for_status()
            data = r.json()
            st.session_state.analysis_state = data
            st.session_state.awaiting_decision = data.get("interrupted", False)
            st.rerun()
        except requests.RequestException as e:
            st.error(f"Resume failed: {e}")


MAJOR_CITIES = [
    "New York, USA",
    "San Francisco, USA",
    "London, UK",
    "Toronto, Canada",
    "Berlin, Germany",
    "Bangalore, India",
    "Mumbai, India",
    "Singapore",
    "Sydney, Australia",
]

# --- Form ---
with st.form("career_form", clear_on_submit=False):
    st.subheader("Upload & Configure")
    uploaded_file = st.file_uploader("Upload your resume (PDF)", type=["pdf"])
    location = st.selectbox("Location", MAJOR_CITIES)
    role = st.text_input("Target role", placeholder="e.g. Backend Engineer, AI Engineer")
    submitted = st.form_submit_button("Analyze Jobs")

if submitted:
    if not uploaded_file or not role or not location:
        st.error("Please upload a resume, select a location, and enter a role.")
    else:
        file_path = f"{uuid4()}.pdf"
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        with st.spinner("Researching jobs and analyzing your profile..."):
            try:
                import base64
                with open(file_path, "rb") as f:
                    b64 = base64.b64encode(f.read()).decode()
                r = requests.post(
                    f"{BASE_URL}{JOB_ANALYSIS_ENDPOINT}",
                    json={
                        "resume_base64": b64,
                        "location": location,
                        "role": role,
                    },
                    timeout=120,
                )
                r.raise_for_status()
                data = r.json()
                st.session_state.thread_id = data["thread_id"]
                st.session_state.analysis_state = data
                st.session_state.awaiting_decision = data.get("interrupted", False)
            except requests.RequestException as e:
                err_detail = ""
                if hasattr(e, "response") and e.response is not None:
                    try:
                        err_detail = e.response.json().get("detail", e.response.text)
                    except Exception:
                        err_detail = e.response.text
                st.error(f"Request failed: {err_detail or str(e)}")
            finally:
                if os.path.exists(file_path):
                    os.remove(file_path)

# --- HITL: Show analysis and Yes/No ---
if st.session_state.awaiting_decision and st.session_state.analysis_state:
    st.divider()
    st.subheader("Review & Decide")
    data = st.session_state.analysis_state
    interrupt_val = data.get("interrupt_value") or []
    state = data.get("state") or {}
    analysis = ""
    for iv in interrupt_val:
        v = iv.get("value", iv)
        if isinstance(v, dict):
            analysis = v.get("message", v.get("analysis", str(v)))
        else:
            analysis = str(v)
    if not analysis and state.get("message"):
        msgs = state["message"]
        if msgs:
            analysis = msgs[-1].get("content", str(msgs[-1])) if isinstance(msgs[-1], dict) else str(msgs[-1])
    st.markdown(analysis)
    col1, col2, _ = st.columns([1, 1, 3])
    with col1:
        if st.button("Yes, generate cold email"):
            _resume_with("yes")
    with col2:
        if st.button("No, skip"):
            _resume_with("no")

# --- Display final email (after resume with yes) ---
if st.session_state.thread_id and st.session_state.analysis_state and not st.session_state.awaiting_decision:
    state = st.session_state.analysis_state.get("state") or {}
    email = state.get("generated_email")
    if email:
        st.divider()
        st.subheader("Your Cold Email")
        st.text_area(
            "Select and copy (Ctrl+C / Cmd+C)",
            value=email,
            height=200,
            disabled=False,
            key="cold_email_display",
        )
        st.download_button(
            label="Download as .txt",
            data=email,
            file_name="cold_email.txt",
            mime="text/plain",
            key="cold_email_download",
        )
        st.success("Ready to copy and send.")
    else:
        st.info("You chose not to generate a cold email.")
