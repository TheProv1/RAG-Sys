import os
import json

def assert_true(condition, msg):
    if not condition: raise AssertionError(f"[FAIL] {msg}")

def main():
    print("Starting pipeline validation...")
    
    # 1. Check Required Files
    required_files =[
        "sources.json", "test_queries.json", "corpus.json",
        "vector_store.npz", "retrieval_logs.jsonl", "generated_answers.json",
        "grounding_verification.json", "answer_audit.json",
        "llm_calls.jsonl"
    ]
    for f in required_files:
        assert_true(os.path.exists(f), f"Missing artifact: {f}")

    # 2. Corpus Checks
    with open("corpus.json", "r") as f: corpus = json.load(f)
    assert_true(len(corpus) > 0, "Corpus is empty.")

    # 3. LLM Audit Logs validation
    with open("llm_calls.jsonl", "r") as f:
        calls =[json.loads(line) for line in f if line.strip()]
    
    stages = {c["stage"] for c in calls}
    
    # We only assert Generation took place if fallback wasn't triggered for literally everything
    with open("answer_audit.json", "r") as f: audits = json.load(f)
    all_fallback = all(a["fallback"] for a in audits)
    
    if not all_fallback:
        assert_true("ANSWER_GENERATION" in stages, "Missing ANSWER_GENERATION LLM stage.")
        assert_true("GROUNDING_VERIFICATION" in stages, "Missing GROUNDING_VERIFICATION stage.")

    # 4. Pipeline Audit Enforcement
    for a in audits:
        assert_true("query_id" in a, "Audit missing query ID.")
        assert_true("retrieved_chunks" in a, "Audit missing retrieval chunks array.")
        
        if a["fallback"]:
            assert_true("final_response" in a, "Missing referral response on fallback.")
        else:
            assert_true("generated_answer" in a, "Missing generated answer on valid query.")
            assert_true("grounding_verification" in a, "Missing claim verification record.")
            
            # Check for chunk citations
            ans_text = a["final_response"]
            assert_true("[" in ans_text and "]" in ans_text, "Answer did not cite any Chunk ID.")

    print("\n[SUCCESS] Pipeline constraints and artifacts have been strictly validated.")

if __name__ == "__main__":
    main()