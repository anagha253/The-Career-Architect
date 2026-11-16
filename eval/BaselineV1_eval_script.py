from backend.utility import ingest_pdfs_to_chroma, get_vector_store
from backend.agents import model
from langchain_core.messages import HumanMessage

# -------------------------------
# Evaluation configuration
# -------------------------------

TEST_CASE1 = {
    "id": "D01",
    "profile_path": "eval/D01.pdf",
    "thread_id": "eval_D01_run1",
    "job_title": "Python Backend Developer",
    "requirement": "Hands-on Python API development",
    "ground_truth": "demonstrated",
}

TEST_CASE2 = {
    "id": "U02",
    "profile_path": "eval/U02.pdf",
    "thread_id": "eval_U02_run1",
    "job_title": "Distributed Systems Architect",
    "requirement": "Distributed systems architecture design",
    "ground_truth": "uncertain",
}

TEST_CASE3 = {
    "id": "M03",
    "profile_path": "eval/M03.pdf",
    "thread_id": "eval_M03_run1",
    "job_title": "Machine Learning Engineer",
    "requirement": "PyTorch development experience",
    "ground_truth": "missing",
}


TEST_CASE4 = {
    "id": "D03",
    "profile_path": "eval/D03.pdf",
    "thread_id": "eval_D03_run1",
    "job_title": "Applied ML Engineer",
    "requirement": "Hands-on classical ML model training",
    "ground_truth": "demonstrated",
}

TEST_CASE5 = {
    "id": "U05",
    "profile_path": "eval/U05.pdf",
    "thread_id": "eval_U05_run1",
    "job_title": "Machine Learning (ML) Engineer",
    "requirement": "Production ML model deployment",
    "ground_truth": "uncertain",
}

TEST_CASE6 = {
    "id": "M01",
    "profile_path": "eval/M01.pdf",
    "thread_id": "eval_M01_run1",
    "job_title": "Python developer",
    "requirement": "Python programming experience",
    "ground_truth": "missing",
}


D02 = {
    "id": "D02",
    "profile_path": "eval/D02.pdf",
    "thread_id": "eval_D02_run1",
    "job_title": "RAG Engineer",
    "requirement": "Experience developing RAG systems",
    "ground_truth": "demonstrated",
}

D04 = {
    "id": "D04",
    "profile_path": "eval/D04.pdf",
    "thread_id": "eval_D04_run1",
    "job_title": "Agentic AI Engineer",
    "requirement": "Experience with LangGraph agent orchestration",
    "ground_truth": "demonstrated",
}

D05 = {
    "id": "D05",
    "profile_path": "eval/D05.pdf",
    "thread_id": "eval_D05_run1",
    "job_title": "Deep Learning Engineer",
    "requirement": "Hands-on PyTorch model development",
    "ground_truth": "demonstrated",
}

U01 = {
    "id": "U01",
    "profile_path": "eval/U01.pdf",
    "thread_id": "eval_U01_run1",
    "job_title": "Python Backend Engineer",
    "requirement": "Production Python backend development",
    "ground_truth": "uncertain",
}

U03 = {
    "id": "U03",
    "profile_path": "eval/U03.pdf",
    "thread_id": "eval_U03_run1",
    "job_title": "Deep Learning Engineer",
    "requirement": "Experience training and optimizing deep learning models",
    "ground_truth": "uncertain",
}

U04 = {
    "id": "U04",
    "profile_path": "eval/U04.pdf",
    "thread_id": "eval_U04_run1",
    "job_title": "LangGraph Engineer",
    "requirement": "Experience developing stateful LangGraph workflows",
    "ground_truth": "uncertain",
}

M02 = {
    "id": "M02",
    "profile_path": "eval/M02.pdf",
    "thread_id": "eval_M02_run1",
    "job_title": "RAG Engineer",
    "requirement": "Hands-on RAG implementation",
    "ground_truth": "missing",
}

M04 = {
    "id": "M04",
    "profile_path": "eval/M04.pdf",
    "thread_id": "eval_M04_run1",
    "job_title": "Agentic AI Engineer",
    "requirement": "LangGraph development experience",
    "ground_truth": "missing",
}

M05 = {
    "id": "M05",
    "profile_path": "eval/M05.pdf",
    "thread_id": "eval_M05_run1",
    "job_title": "ML Engineer",
    "requirement": "Production ML deployment experience",
    "ground_truth": "missing",
}


# -------------------------------
# Step 1: Ingest candidate profile
# -------------------------------


def ingest_profile(case):
    ingest_pdfs_to_chroma(case["profile_path"], thread_id=case["thread_id"])


# -------------------------------
# Step 2: Retrieve candidate evidence
# -------------------------------


def retrieve_profile(case):
    vector_store = get_vector_store(case["thread_id"])
    retriever = vector_store.as_retriever(search_kwargs={"k": 5})
    print(retriever)

    query = (
        "Summarize: projects with one-line descriptions, "
        "academic background, work experience in 5 bullet points. "
        "Be concise."
    )

    documents = retriever.invoke(query)

    resume_text = "\n".join(document.page_content for document in documents)
    print(
        f"Retrieved {len(documents)} documents from ChromaDB for thread_id: {case['thread_id']}"
    )

    return resume_text


# -------------------------------
# Step 3: Run baseline analyser
# -------------------------------


def run_analyser(case, resume_text):

    job_context = (
        f"Job Title: {case['job_title']}\n" f"Required skill: {case['requirement']}"
    )

    # Original analyser prompt — unchanged
    analysis_prompt = (
        f"Resume/projects summary:\n{resume_text}\n\n"
        f"Found jobs:\n{job_context}\n\n"
        "Compare the candidate profile to these jobs. "
        """Demonstrated: Explicit evidence satisfies the requirement. 
        Uncertain: Evidence is incomplete or only indirectly relevant.
        Missing: The candidate explicitly states they lack the required experience.
        provide Supporting evidence for your decision, 
        and \provide 1-2 line brief explanation.

        Finally provide  a label fr the analysis: 'demonstrated', 'uncertain', or 'missing' based on the evidence and explanation.
        Final output format:
        - Requirement: <requirement>
        - Label: demonstrated, uncertain, or missing
        - Supporting evidence: <evidence>
        - Brief explanation: <1-2 line explanation>

        """
    )

    response = model.invoke([HumanMessage(content=analysis_prompt)])

    return response.content


# -------------------------------
# Step 4: Execute evaluation
# -------------------------------


def evaluate(case):

    print(f"\nEvaluating: {case['id']}")
    ingest_profile(case)

    resume_text = retrieve_profile(case)

    analysis = run_analyser(case, resume_text)

    print("\nRetrieved Evidence:")
    print(resume_text)

    print("\nAnalyser Response:")
    print(analysis)

    print("\nGround Truth:")
    print(case["ground_truth"])


def delete_vector_store(list_of_thread_ids):
    for thread_id in list_of_thread_ids:
        vector_store = get_vector_store(thread_id)
        vector_store.delete_collection()


if __name__ == "__main__":
    delete_vector_store(
        [
            TEST_CASE1["thread_id"],
            TEST_CASE2["thread_id"],
            TEST_CASE3["thread_id"],
            TEST_CASE4["thread_id"],
            TEST_CASE5["thread_id"],
            TEST_CASE6["thread_id"],
            D02["thread_id"],
            D04["thread_id"],
            D05["thread_id"],
            U01["thread_id"],
            U03["thread_id"],
            U04["thread_id"],
            M02["thread_id"],
            M04["thread_id"],
            M05["thread_id"],
        ]
    )
    # Run only once for a fresh collection.
    # ingest_profile(TEST_CASE)

    evaluate(TEST_CASE1)
    evaluate(TEST_CASE2)
    evaluate(TEST_CASE3)
    evaluate(TEST_CASE4)
    evaluate(TEST_CASE5)
    evaluate(TEST_CASE6)
    evaluate(D02)
    evaluate(D04)
    evaluate(D05)
    evaluate(U01)
    evaluate(U03)
    evaluate(U04)
    evaluate(M02)
    evaluate(M04)
    evaluate(M05)
