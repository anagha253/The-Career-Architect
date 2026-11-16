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
        "Recommend whether to draft a cold email. "
        "Output exactly: 'Yes' or 'No' with a 1–2 sentence reason."
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
        [TEST_CASE1["thread_id"], TEST_CASE2["thread_id"], TEST_CASE3["thread_id"]]
    )
    # Run only once for a fresh collection.
    # ingest_profile(TEST_CASE)

    evaluate(TEST_CASE1)
    evaluate(TEST_CASE2)
    evaluate(TEST_CASE3)
