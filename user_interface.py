import streamlit as st
import app_controller 
from uuid import uuid4
import os

st.header("The Career Architect")

st.text("Upload your resume")


uploaded_file = st.file_uploader("Upload resume", type=["pdf"])
file_path = ""

if uploaded_file is not None:
    
    os.makedirs("data/user", exist_ok=True)

    unique_name = f"{uuid4()}.pdf"
    file_path = os.path.join("data/user", unique_name)

    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    st.success(f"File saved to {file_path}")

MAJOR_CITIES = [
    "New York, USA",
    "London, UK",
    "Toronto, Canada",
    "Berlin, Germany",
    "Bangalore, India",
    "Mumbai, India",
    "Singapore",
    "Sydney, Australia",
]

location = st.selectbox("Select City", MAJOR_CITIES)

role = st.text_input("Enter job role")


if st.button("Analyze My Resume"):
    with st.spinner("Analysing..."):
        if not location or not role:
            st.error("Please upload a resume and enter a job description.")
        else:
            request_body = app_controller.job_analysis_request(
                resume_file=file_path,
                role=role,
                location=location
            )
            
            response = app_controller.job_analysis(request=request_body)
            st.write(response["result"])
            st.write(" Do you to generate cold-email?")
            res = st.radio("Do you to generate cold-email?",
                           ["Yes","No"])
            if res=="Yes":
                req = app_controller.email_generation_request(
                    job_analysis= response["result"],
                    thread_id= response["thread_id"]
                )
            
                result = app_controller.generate_cold_email(req)
                st.write(result)