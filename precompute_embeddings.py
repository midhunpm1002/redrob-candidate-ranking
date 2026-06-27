#!/usr/bin/env python3
"""
Phase 1: Local Embedding Generation.
Generates sentence-transformers embeddings for all candidates and the fixed JD.
Saves candidate embeddings as numpy array in artifacts/embeddings.npy,
and saves the list of candidate IDs (in index order) in artifacts/embeddings_index.pkl.
"""

import os
import json
import gzip
import pickle
import argparse
import numpy as np
from datetime import datetime
from tqdm import tqdm
from sentence_transformers import SentenceTransformer

# Fixed Job Description text
JOB_DESCRIPTION = """
Senior AI Engineer — Founding Team
Company: Redrob AI (Series A AI-native talent intelligence platform)
Location: Pune/Noida, India (Hybrid — flexible cadence) | Open to relocation candidates from Tier-1 Indian cities
Employment Type: Full-time
Experience Required: 5–9 years

We're building a new AI Engineering org from scratch. We need someone who is simultaneously comfortable with deep technical depth in modern ML systems (embeddings, retrieval, ranking, LLMs, fine-tuning) and a scrappy product-engineering attitude (willing to ship a working ranker in a week).
The high-level mandate: own the intelligence layer of Redrob's product (ranking, retrieval, matching systems).
Things you absolutely need:
- Production experience with embeddings-based retrieval systems (sentence-transformers, OpenAI embeddings, BGE, E5, or similar) deployed to real users.
- Production experience with vector databases or hybrid search infrastructure (Pinecone, Weaviate, Qdrant, Milvus, OpenSearch, Elasticsearch, FAISS).
- Strong Python.
- Hands-on experience designing evaluation frameworks for ranking systems (NDCG, MRR, MAP, A/B test interpretation).
"""

def clean_and_truncate_text(text, max_words=300):
    if not text:
        return ""
    words = text.split()
    if len(words) > max_words:
        words = words[:max_words]
    return " ".join(words)

def build_candidate_text(cand):
    """
    Constructs a text blob for the candidate by combining headline, summary, and career history.
    """
    profile = cand.get("profile", {})
    headline = profile.get("headline", "")
    summary = profile.get("summary", "")
    
    career = cand.get("career_history", [])
    career_texts = []
    for job in career:
        title = job.get("title", "")
        desc = job.get("description", "")
        company = job.get("company", "")
        career_texts.append(f"Worked as {title} at {company}. {desc}")
        
    full_text = f"Headline: {headline}. Summary: {summary}. Experience: " + " ".join(career_texts)
    return clean_and_truncate_text(full_text)

def main():
    parser = argparse.ArgumentParser(description="Precompute candidate embeddings.")
    parser.add_argument("--candidates", type=str, default="candidates.jsonl", help="Path to candidates jsonl file")
    parser.add_argument("--model", type=str, default="all-MiniLM-L6-v2", help="SentenceTransformer model name")
    parser.add_argument("--batch-size", type=int, default=128, help="Batch size for encoding")
    parser.add_argument("--out-dir", type=str, default="artifacts", help="Directory to save precomputed embeddings")
    args = parser.parse_args()

    print(f"Loading local SentenceTransformer model: {args.model}...")
    start_time = datetime.now()
    
    # Load model (forces download to local cache if not present, then runs completely offline)
    model = SentenceTransformer(args.model)
    
    # Disable network for validation that model runs locally hereafter
    os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
    
    print(f"Reading candidates from {args.candidates}...")
    candidate_ids = []
    texts_to_embed = []
    
    open_func = gzip.open if args.candidates.endswith(".gz") else open
    mode = "rt" if args.candidates.endswith(".gz") else "r"
    
    with open_func(args.candidates, mode, encoding="utf-8") as f:
        # Get total line count for progress bar if not gzipped
        total_lines = None
        if not args.candidates.endswith(".gz"):
            try:
                with open(args.candidates, "r", encoding="utf-8") as temp_f:
                    total_lines = sum(1 for _ in temp_f)
            except Exception:
                pass

        for line in tqdm(f, total=total_lines, desc="Building candidate texts"):
            if not line.strip():
                continue
            cand = json.loads(line)
            candidate_ids.append(cand.get("candidate_id"))
            texts_to_embed.append(build_candidate_text(cand))
            
    print(f"Encoding {len(texts_to_embed)} candidate profiles on CPU (batch size = {args.batch_size})...")
    # Generate embeddings
    embeddings = model.encode(
        texts_to_embed, 
        batch_size=args.batch_size, 
        show_progress_bar=True,
        convert_to_numpy=True
    )
    
    print("Encoding fixed job description...")
    jd_embedding = model.encode(JOB_DESCRIPTION, convert_to_numpy=True)
    
    # Save files
    os.makedirs(args.out_dir, exist_ok=True)
    
    npy_path = os.path.join(args.out_dir, "embeddings.npy")
    idx_path = os.path.join(args.out_dir, "embeddings_index.pkl")
    jd_path = os.path.join(args.out_dir, "jd_embedding.npy")
    
    print(f"Saving candidate embeddings array to {npy_path}...")
    np.save(npy_path, embeddings)
    
    print(f"Saving JD embedding array to {jd_path}...")
    np.save(jd_path, jd_embedding)
    
    print(f"Saving candidate IDs index to {idx_path}...")
    with open(idx_path, "wb") as f:
        pickle.dump(candidate_ids, f, protocol=pickle.HIGHEST_PROTOCOL)
        
    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()
    print(f"Done precomputing embeddings in {duration:.2f} seconds.")

if __name__ == "__main__":
    main()
