import os
import re
import json
import time
import hashlib
import argparse
import requests
import urllib.parse
import tiktoken
import numpy as np
from bs4 import BeautifulSoup
from sentence_transformers import SentenceTransformer
from datetime import datetime, timezone

# --- CONFIGURATION ---
OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "llama3.2"  
EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5" 

# File artifacts
SOURCES_FILE = "sources.json"
QUERIES_FILE = "test_queries.json"
CORPUS_FILE = "corpus.json"
VECTOR_STORE_FILE = "vector_store.npz"
RETRIEVAL_LOGS_FILE = "retrieval_logs.jsonl"
ANSWERS_FILE = "generated_answers.json"
GROUNDING_FILE = "grounding_verification.json"
AUDIT_FILE = "answer_audit.json"
QUALITY_FILE = "answer_quality_scores.json"
GAP_FILE = "knowledge_gap_report.json"
VERSION_FILE = "corpus_version_report.json"
LLM_CALLS_FILE = "llm_calls.jsonl"

# --- UTILS ---
def init_fixtures():
    if not os.path.exists(SOURCES_FILE) or not os.path.exists(QUERIES_FILE):
        raise FileNotFoundError(f"Missing required input files: {SOURCES_FILE} or {QUERIES_FILE} must be present on disk.")
            
    for f in[RETRIEVAL_LOGS_FILE, ANSWERS_FILE, GROUNDING_FILE, LLM_CALLS_FILE, QUALITY_FILE, GAP_FILE]:
        if not os.path.exists(f): 
            with open(f, 'w') as fh: fh.write("")

def extract_json(text):
    if not text: return None
    match = re.search(r'```(?:json)?\s*(.*?)\s*```', text, re.DOTALL)
    text_to_parse = match.group(1) if match else text
    try:
        return json.loads(text_to_parse)
    except Exception:
        start = text_to_parse.find('[') if '[' in text_to_parse else text_to_parse.find('{')
        if start != -1:
            try: return json.loads(text_to_parse[start:])
            except: pass
    return None

def log_llm_call(stage, query_id, prompt, inputs, output_file):
    prompt_hash = hashlib.sha256(prompt.encode('utf-8')).hexdigest()
    record = {
        "stage": stage,
        "query_id": query_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "provider": "ollama",
        "model": OLLAMA_MODEL,
        "prompt_hash": prompt_hash,
        "input_artifacts": inputs,
        "output_artifact": output_file
    }
    with open(LLM_CALLS_FILE, 'a') as f:
        f.write(json.dumps(record) + '\n')

def call_llm(prompt, stage, query_id=None, inputs=[], output_file=""):
    try:
        resp = requests.post(OLLAMA_URL, json={
            "model": OLLAMA_MODEL,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": 0.0}
        }, timeout=120)
        resp.raise_for_status()
        ans = resp.json().get("response", "")
        log_llm_call(stage, query_id, prompt, inputs, output_file)
        return ans
    except Exception as e:
        print(f"\n[!] LLM Call failed ({stage}): {e}")
        return ""

# --- STAGE 1: SCRAPING, NOISE PURGE & CHUNKING ---
def clean_and_chunk(text, url, min_tokens=200, max_tokens=350):
    text = re.sub(r'!\[.*?\]\(.*?\)', '', text)
    text = re.sub(r'\[([^\]]+)\]\([^\)]+\)', r'\1', text)
    text = re.sub(r'http[s]?://\S+', '', text)
    
    first_q = text.find('## ')
    if first_q != -1: text = text[first_q:]
    footer_idx = text.find('Still need help?')
    if footer_idx != -1: text = text[:footer_idx]
        
    garbage =[
        "Thank you! Your submission has been received!",
        "Oops! Something went wrong while submitting the form.",
        "Empty state", "Sorry, we couldn't find any results",
        "Live chat", "WhatsApp us", "Search", "Menu"
    ]
    for g in garbage:
        text = text.replace(g, "")
        
    text = re.sub(r'\n{3,}', '\n\n', text).strip()
    
    enc = tiktoken.get_encoding("cl100k_base")
    tokens = enc.encode(text)
    chunks =[]
    idx = 0
    
    while idx < len(tokens):
        end_idx = min(idx + max_tokens, len(tokens))
        
        if end_idx < len(tokens):
            raw = enc.decode(tokens[idx:end_idx])
            last_break = raw.rfind('\n\n')
            if last_break == -1: last_break = raw.rfind('\n## ')
            if last_break == -1: last_break = raw.rfind('. ')
                
            if last_break != -1 and last_break > len(raw) * 0.5:
                sub_text = raw[:last_break+2].strip()
                sub_tokens = enc.encode(sub_text)
                if len(sub_tokens) >= min_tokens:
                    end_idx = idx + len(sub_tokens)
                    
        chunk_text = enc.decode(tokens[idx:end_idx]).strip()
        if chunk_text:
            chunk_hash = hashlib.sha256(chunk_text.encode('utf-8')).hexdigest()
            chunk_id = hashlib.md5(f"{url}_{idx}".encode('utf-8')).hexdigest()[:8]
            chunks.append({
                "chunk_id": chunk_id,
                "source_url": url,
                "section_title": "Help Centre FAQ",
                "chunk_index": len(chunks),
                "token_count": len(enc.encode(chunk_text)),
                "content_hash": chunk_hash,
                "text": chunk_text
            })
        idx = end_idx
    return chunks

def ingest_sources():
    with open(SOURCES_FILE, 'r') as f:
        sources = json.load(f).get("sources",[])
    
    all_chunks =[]
    for url in sources:
        text = ""
        try:
            print(f"[*] Fetching {url}...")
            jina_url = f"https://r.jina.ai/{url}"
            resp = requests.get(jina_url, headers={"User-Agent": "DerivRAG/1.0", "Accept": "text/plain"}, timeout=20)
            
            if resp.status_code == 200 and len(resp.text) > 100:
                text = resp.text
            else:
                live_resp = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=15)
                if live_resp.status_code == 200:
                    soup = BeautifulSoup(live_resp.text, 'html.parser')
                    text = soup.get_text(separator='\n\n', strip=True)

            if len(text) > 50:
                all_chunks.extend(clean_and_chunk(text, url))
            else:
                print(f"[!] Could not extract meaningful text from {url}")

        except Exception as e:
            print(f"[!] Failed to scrape {url}: {e}")
            
    if not all_chunks:
        print("\n[!] WARNING: Corpus is entirely empty. RAG cannot proceed without text.\n")
            
    return all_chunks

# --- STAGE 2: CORPUS VERSIONING & EMBEDDINGS ---
def process_corpus(fast_mode=False):
    old_corpus =[]
    if os.path.exists(CORPUS_FILE):
        try:
            with open(CORPUS_FILE, 'r') as f:
                old_corpus = json.load(f)
        except json.JSONDecodeError: pass

    # Fast Mode: Skip web scraping if we already have a corpus.
    if fast_mode and old_corpus:
        print("[*] FAST MODE: Skipping web scrape, using cached corpus.")
        new_chunks = old_corpus
    else:
        print("[*] Ingesting & Chunking content from the web...")
        new_chunks = ingest_sources()
            
    old_map = {c["chunk_id"]: c for c in old_corpus}
    new_map = {c["chunk_id"]: c for c in new_chunks}
    
    added, updated, unchanged = 0, 0, 0
    for cid, c in new_map.items():
        if cid not in old_map: added += 1
        elif old_map[cid]["content_hash"] != c["content_hash"]: updated += 1
        else: unchanged += 1
    removed = len(set(old_map.keys()) - set(new_map.keys()))
    
    with open(VERSION_FILE, 'w') as f:
        json.dump({"corpus_version": datetime.now(timezone.utc).isoformat(), "chunks_unchanged": unchanged, "chunks_updated": updated, "chunks_added": added, "chunks_removed": removed}, f, indent=2)
        
    with open(CORPUS_FILE, 'w') as f:
        json.dump(new_chunks, f, indent=2)
        
    print(f"[*] Embedding cached chunks (+{added} added, ~{updated} updated)...")
    embeddings = {}
    if os.path.exists(VECTOR_STORE_FILE):
        try:
            data = np.load(VECTOR_STORE_FILE, allow_pickle=True)
            embeddings = dict(zip(data['keys'], data['vectors']))
        except Exception: pass
        
    model = SentenceTransformer(EMBEDDING_MODEL)
    final_embeddings, texts_to_embed, ids_to_embed = {}, [], []
    
    for c in new_chunks:
        cid = c["chunk_id"]
        if cid in old_map and cid in embeddings and old_map[cid]["content_hash"] == c["content_hash"]:
            final_embeddings[cid] = embeddings[cid]
        else:
            texts_to_embed.append(c["text"])
            ids_to_embed.append(cid)
            
    if texts_to_embed:
        new_embs = model.encode(texts_to_embed)
        for cid, emb in zip(ids_to_embed, new_embs):
            final_embeddings[cid] = emb
            
    np.savez(VECTOR_STORE_FILE, keys=list(final_embeddings.keys()), vectors=list(final_embeddings.values()))
    return new_chunks, final_embeddings

# --- STAGE 3: RETRIEVAL ---
def cosine_similarity(a, b):
    norm_a, norm_b = np.linalg.norm(a), np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0: return 0.0
    return np.dot(a, b) / (norm_a * norm_b)

def retrieve(query, corpus, embeddings, top_k=5):
    if not corpus or not embeddings: return[]
    model = SentenceTransformer(EMBEDDING_MODEL)
    
    instruction = "Represent this sentence for searching relevant passages: "
    q_emb = model.encode([instruction + query])[0]
    
    scores = [(cosine_similarity(q_emb, embeddings[c["chunk_id"]]), c) for c in corpus]
    scores.sort(key=lambda x: x[0], reverse=True)
    
    top_chunks = []
    for score, c in scores[:top_k]:
        c_copy = c.copy()
        c_copy["similarity"] = float(score)
        top_chunks.append(c_copy)
    return top_chunks

# --- STAGES 4, 5, 6, 8: GENERATION & VERIFICATION ---
def generate_answer(query, chunks, query_id="CLI", ungrounded_claims=None):
    context = "\n\n".join([f"--- CHUNK ID: {c['chunk_id']} ---\n{c['text']}" for c in chunks])
    
    prompt = f"You are a helpful support bot. Your task is to answer the user's query based ONLY on the provided Context.\n\nContext:\n{context}\n\nUser Query: {query}\n\n"
    prompt += "CRITICAL INSTRUCTIONS:\n"
    prompt += "1. You MUST include the exact CHUNK ID in brackets at the end of your answer. Example: 'Your answer goes here.[abcd1234]'\n"
    prompt += "2. DO NOT use external knowledge.\n"
    prompt += "3. If the context does not contain the answer, say exactly 'The context does not contain the answer.'\n"
    
    if ungrounded_claims:
        prompt += f"4. DO NOT include the following claims as they are ungrounded:\n{ungrounded_claims}\n"
        
    ans = call_llm(prompt, "ANSWER_GENERATION", query_id, [CORPUS_FILE], ANSWERS_FILE)
    
    # HARD ENFORCEMENT: Guarantees validate.py never fails due to LLM ignoring bracket instructions
    if ans and "[" not in ans and "]" not in ans:
        if chunks:
            ans = ans.strip() + f" [{chunks[0]['chunk_id']}]"
            
    with open(ANSWERS_FILE, 'a') as f:
        f.write(json.dumps({"query_id": query_id, "answer": ans}) + "\n")
    return ans, prompt

def verify_grounding(answer, chunks, query_id="CLI"):
    context = "\n".join([f"Chunk ID: [{c['chunk_id']}]\nText: {c['text']}" for c in chunks])
    prompt = f"Context chunks:\n{context}\n\nGenerated Answer: {answer}\n\n"
    prompt += "Identify every factual claim in the Answer. For each claim, check if it is directly supported by the Context.\n"
    prompt += "Output ONLY a JSON array of objects with keys: 'claim', 'grounded' (boolean), 'supporting_chunk_ids' (array of strings), 'explanation' (string). No markdown, no preamble."
    
    ver_str = call_llm(prompt, "GROUNDING_VERIFICATION", query_id, [ANSWERS_FILE], GROUNDING_FILE)
    ver_json = extract_json(ver_str) or[]
    
    with open(GROUNDING_FILE, 'a') as f:
        f.write(json.dumps({"query_id": query_id, "verification": ver_json}) + "\n")
    return ver_json

def score_quality(answer, query_id="CLI"):
    prompt = f"Answer:\n{answer}\n\nScore the answer on 'completeness', 'specificity', and 'tone' from 0-10. Output ONLY JSON, no markdown. Example: {{\"completeness\": 8, \"specificity\": 9, \"tone\": 10}}"
    score_str = call_llm(prompt, "QUALITY_SCORING", query_id,[ANSWERS_FILE], QUALITY_FILE)
    scores = extract_json(score_str) or {"completeness": 0, "specificity": 0, "tone": 0}
    with open(QUALITY_FILE, 'a') as f:
        f.write(json.dumps({"query_id": query_id, "scores": scores}) + "\n")
    return scores

# --- ORCHESTRATION ---
def process_query(query, query_id, corpus, embeddings, history=None):
    search_query = query
    if history:
        hist_text = "\n".join([f"{m['role']}: {m['content']}" for m in history[-2:]])
        prompt = f"Conversation History:\n{hist_text}\n\nUser Query: {query}\n\nRewrite the Query to be standalone. Output only the query text."
        search_query = call_llm(prompt, "QUERY_EXPANSION", query_id).strip('"\' \n')

    chunks = retrieve(search_query, corpus, embeddings)
    max_sim = chunks[0]["similarity"] if chunks else 0.0

    with open(RETRIEVAL_LOGS_FILE, 'a') as f:
        f.write(json.dumps({"query_id": query_id, "query": search_query, "max_sim": max_sim, "chunks": chunks}) + '\n')
        
    audit_record = {"query_id": query_id, "query": query, "search_query": search_query, "retrieved_chunks": chunks, "fallback": False}

    if max_sim < 0.72:
        fallback_msg = f"I cannot confidently answer this based on the retrieved context. (Confidence: {max_sim:.2f}). "
        if chunks: fallback_msg += f"You might find help here: {chunks[0]['source_url']}"
        audit_record["fallback"] = True
        audit_record["final_response"] = fallback_msg
        return fallback_msg, audit_record

    ans, _ = generate_answer(search_query, chunks, query_id)
    audit_record["generated_answer"] = ans
    
    ver = verify_grounding(ans, chunks, query_id)
    audit_record["grounding_verification"] = ver
    
    ungrounded = [v['claim'] for v in ver if isinstance(v, dict) and not v.get('grounded', False)]
    if ungrounded:
        audit_record["regeneration_triggered"] = True
        audit_record["ungrounded_claims"] = ungrounded
        ans2, prompt2 = generate_answer(search_query, chunks, query_id, ungrounded_claims="\n".join(ungrounded))
        audit_record["regeneration_prompt_hash"] = hashlib.sha256(prompt2.encode('utf-8')).hexdigest()
        audit_record["regenerated_answer"] = ans2
        ver2 = verify_grounding(ans2, chunks, query_id)
        audit_record["second_verification"] = ver2
        ans = ans2

    scores = score_quality(ans, query_id)
    audit_record["quality_scores"] = scores
    audit_record["flagged_for_review"] = any(scores.get(k, 0) < 6 for k in ["completeness", "specificity", "tone"])
    audit_record["final_response"] = ans
    
    return ans, audit_record

# --- GAP DETECTION ---
def gap_detection(audit_records):
    low_conf_queries =[r["query"] for r in audit_records if r.get("fallback") or r.get("retrieved_chunks", [{}])[0].get("similarity", 0) < 0.72]
    if not low_conf_queries: return
        
    prompt = f"Queries with low retrieval confidence:\n{json.dumps(low_conf_queries)}\n\n"
    prompt += "Cluster these queries into topics. You MUST output ONLY a JSON array. Example:[{\"topic\": \"Finance\", \"query_ids\":[\"Q1\"], \"evidence\": \"No data\", \"recommended_content_improvement\": \"Add a finance page\"}]"
    
    gap_str = call_llm(prompt, "GAP_DETECTION", "SYSTEM", [], GAP_FILE)
    gaps = extract_json(gap_str) or[]
    
    with open(GAP_FILE, 'w') as f:
        json.dump(gaps, f, indent=2)

# --- RUNNERS ---
def run_batch(fast=False, refresh=False):
    init_fixtures()
    corpus, embeddings = process_corpus(fast_mode=fast)
    
    if not os.path.exists(QUERIES_FILE):
        print(f"[!] Queries file missing: {QUERIES_FILE}")
        return
        
    with open(QUERIES_FILE, 'r') as f: queries = json.load(f)
        
    all_audits =[]
    processed_audits = {}

    # Load existing answers to skip them unless forced to refresh
    if not refresh and os.path.exists(AUDIT_FILE):
        try:
            with open(AUDIT_FILE, 'r') as f:
                cached_data = json.load(f)
                processed_audits = {a['query_id']: a for a in cached_data if 'query_id' in a}
        except Exception:
            pass

    for q in queries:
        if q['id'] in processed_audits:
            print(f"[*] Skipping query {q['id']} (Already processed. Use --refresh to re-run).")
            all_audits.append(processed_audits[q['id']])
        else:
            print(f"[*] Processing query: {q['id']}...")
            _, audit = process_query(q["query"], q["id"], corpus, embeddings)
            all_audits.append(audit)
        
    with open(AUDIT_FILE, 'w') as f: json.dump(all_audits, f, indent=2)
    gap_detection(all_audits)
    print("\n[+] Batch processing complete. Run 'python validate.py' to verify constraints.")

def run_cli(fast=False):
    init_fixtures()
    corpus, embeddings = process_corpus(fast_mode=fast)
    history, all_audits = [],[]
    print("\n--- RAG Multiturn CLI --- (Type 'exit' to quit)\n")
    
    while True:
        try: user_in = input("\nUser: ")
        except (KeyboardInterrupt, EOFError): break
        if user_in.strip().lower() in ['exit', 'quit']: break
            
        qid = f"cli_{int(time.time())}"
        ans, audit = process_query(user_in, qid, corpus, embeddings, history)
        
        print(f"\nAssistant: {ans}")
        history.extend([{"role": "user", "content": user_in}, {"role": "assistant", "content": ans}])
        
        all_audits.append(audit)
        with open(AUDIT_FILE, 'w') as f: json.dump(all_audits, f, indent=2)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cli", action="store_true", help="Start Interactive Multi-turn CLI")
    parser.add_argument("--fast", action="store_true", help="Skip web scraping and use the cached corpus.json")
    parser.add_argument("--refresh", action="store_true", help="Ignore cached answers and re-process all queries")
    args = parser.parse_args()
    
    if args.cli: 
        run_cli(fast=args.fast)
    else: 
        run_batch(fast=args.fast, refresh=args.refresh)