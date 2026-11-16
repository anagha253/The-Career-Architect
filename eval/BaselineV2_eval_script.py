from backend.utility import ingest_pdfs_to_chroma, get_vector_store, invoke_with_retry
from backend.agents import model
from langchain_core.messages import HumanMessage

# -------------------------------
# Evaluation configuration
# -------------------------------

E01 = {
    "id": "E01",
    "profile_path": "eval/E01.pdf",
    "thread_id": "eval_E01_run1",
    "job_title": "Applied AI Engineer",
    "requirements": [
        "Python API development",
        "RAG implementation",
        "Production PyTorch model deployment",
        "Kubernetes experience",
        "Distributed systems architecture design",
    ],
    "ground_truth": [
        "demonstrated",
        "demonstrated",
        "uncertain",
        "missing",
        "uncertain",
    ],
}

E02 = {
    "id": "E02",
    "profile_path": "eval/E02.pdf",
    "thread_id": "eval_E02_run1",
    "job_title": "ML Platform Engineer",
    "requirements": [
        "Python ML data pipeline development",
        "Hands-on XGBoost model training",
        "Production ML model deployment",
        "Production AWS infrastructure experience",
        "Apache Spark experience",
    ],
    "ground_truth": [
        "demonstrated",
        "demonstrated",
        "uncertain",
        "uncertain",
        "missing",
    ],
}

E03 = {
    "id": "E03",
    "profile_path": "eval/E03.pdf",
    "thread_id": "eval_E03_run1",
    "job_title": "Agentic AI Engineer",
    "requirements": [
        "LangGraph agent orchestration",
        "RAG implementation",
        "Human-in-the-loop workflow implementation",
        "Hands-on LLM benchmarking",
        "LLM fine-tuning experience",
    ],
    "ground_truth": [
        "demonstrated",
        "demonstrated",
        "demonstrated",
        "uncertain",
        "missing",
    ],
}

E04 = {
    "id": "E04",
    "profile_path": "eval/E04.pdf",
    "thread_id": "eval_E04_run1",
    "job_title": "AI Research Engineer",
    "requirements": [
        "PyTorch model implementation",
        "Model evaluation",
        "Reproducing published ML baselines",
        "Novel deep learning architecture research",
        "3D point-cloud processing",
    ],
    "ground_truth": [
        "demonstrated",
        "demonstrated",
        "demonstrated",
        "uncertain",
        "missing",
    ],
}

E05 = {
    "id": "E05",
    "profile_path": "eval/E05.pdf",
    "thread_id": "eval_E05_run1",
    "job_title": "Production AI Engineer",
    "requirements": [
        "Production Python backend development",
        "Redis caching implementation",
        "Kafka-based event processing",
        "Production LLM observability",
        "Azure cloud experience",
    ],
    "ground_truth": [
        "demonstrated",
        "demonstrated",
        "demonstrated",
        "uncertain",
        "missing",
    ],
}


# -------------------------------
# Step 1: Ingest candidate profile
# -------------------------------


def ingest_profile(case):
    vector_store = get_vector_store(case["thread_id"])

    if vector_store._collection.count() > 0:
        print(f"{case['id']} already ingested")
        return

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
    requirements_text = "\n".join(
        f"- {requirement}" for requirement in case["requirements"]
    )

    job_context = (
        f"Job Title: {case['job_title']}\n" f"Required skills:\n{requirements_text}"
    )

    # Original analyser prompt — unchanged
    analysis_prompt = (
        f"Resume/projects summary:\n{resume_text}\n\n"
        f"Found jobs:\n{job_context}\n\n"
        "Compare the candidate profile to these jobs. "
        """
            Evaluate EACH job requirement independently.

            For every requirement:
            1. Identify whether the candidate demonstrates the required skill.
            2. Classify it as demonstrated, uncertain, or missing using the Classification policy 
            Classification policy:

            DEMONSTRATED:
            The candidate profile explicitly provides evidence satisfying the
            complete job requirement, including the required proficiency level.

            MISSING:
            The candidate explicitly states that they do not have the required
            skill or experience.

            UNCERTAIN:
            The profile contains insufficient, indirect, partial, or ambiguous
            evidence to establish the requirement.

            Critical rule:
            Absence of evidence is NOT evidence of absence.
            Do not classify a requirement as MISSING merely because the
            candidate has not demonstrated it.
            When there is no explicit denial and insufficient positive evidence,
            classify it as UNCERTAIN.

            Evaluate each requirement independently.


            3. Provide supporting evidence from the candidate profile.
            4. Explain the classification briefly.



            Return exactly one result for every job requirement.
            Do not combine requirements into a single classification.
            Do not invent skills or experience.


        """
    )

    response = invoke_with_retry(model, [HumanMessage(content=analysis_prompt)])

    return response.content


# -------------------------------
# Step 4: Execute evaluation
# -------------------------------


import time


def evaluate(case):
    print(f"\nEvaluating: {case['id']}", flush=True)

    start = time.perf_counter()
    ingest_profile(case)
    print(f"1. Ingestion completed: {time.perf_counter()-start:.2f}s", flush=True)

    start = time.perf_counter()
    resume_text = retrieve_profile(case)
    print(f"2. Retrieval completed: {time.perf_counter()-start:.2f}s", flush=True)

    if not resume_text.strip():
        print("ERROR: Empty retrieval. Skipping inference.")
        return

    start = time.perf_counter()
    print("3. Calling Gemini...", flush=True)
    analysis = run_analyser(case, resume_text)
    print(f"4. Gemini completed: {time.perf_counter()-start:.2f}s", flush=True)

    print(analysis)


def delete_vector_store(list_of_thread_ids):
    for thread_id in list_of_thread_ids:
        vector_store = get_vector_store(thread_id)
        vector_store.delete_collection()


if __name__ == "__main__":
    # delete_vector_store(
    #     [
    #         E02["thread_id"],
    #         # E04["thread_id"],
    #         # E05["thread_id"],
    #         # E01["thread_id"],
    #         # E03["thread_id"],
    #     ]
    # )
    # Run only once for a fresh collection.
    # ingest_profile(TEST_CASE)
    evaluate(E01)
    evaluate(E02)
    evaluate(E03)
    evaluate(E04)
    evaluate(E05)
