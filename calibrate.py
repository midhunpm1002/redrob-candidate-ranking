#!/usr/bin/env python3
"""
Calibration Workflow.
Loads sample_candidates.json, scores them, and displays details side-by-side.
Allows the user to enter 1-5 ratings to calculate agreement and tune weights.
"""

import json
import sys
import numpy as np
from score import score_candidate
from precompute_features import extract_features_from_candidate

def print_candidate_summary(idx, feat, score, is_hp, hp_reason):
    print(f"\n[{idx+1}/50] Candidate ID: {feat['candidate_id']}")
    print(f"Name: {feat['current_title']} | Years of Exp: {feat['years_of_experience']} | Co: {feat['current_company']}")
    print(f"Headline: {feat['headline']}")
    print(f"Location: {feat['location']} | Notice Period: {feat['notice_period_days']} days")
    print(f"Skills: {', '.join([s.get('name') for s in feat['skills'][:5]])}")
    if is_hp:
        print(f"--> gated as HONEYPOT: {hp_reason}")
    else:
        print(f"--> computed score: {score:.4f}")

def main():
    print("--- CALIBRATION WORKFLOW ---")
    print("Loading sample_candidates.json...")
    try:
        with open("sample_candidates.json", "r", encoding="utf-8") as f:
            candidates = json.load(f)
    except FileNotFoundError:
        print("Error: sample_candidates.json not found in workspace.")
        sys.exit(1)

    print(f"Loaded {len(candidates)} sample candidates.")
    
    # Process features locally for the sample candidates
    features_list = []
    for cand in candidates:
        feat = extract_features_from_candidate(cand)
        # For sample calibration, we don't have precomputed sentence embeddings,
        # so we default similarity to a baseline (e.g. 0.5) or calculate it using a fast term match.
        # Let's use a simple keyword overlap ratio as a mock semantic similarity for calibration, or just 0.5.
        jd_keywords = ["ai", "ml", "machine learning", "search", "ranking", "retrieval", "embeddings", "pinecone", "weaviate"]
        desc_text = " ".join(feat["career_descriptions"] + [feat["summary"], feat["headline"]]).lower()
        overlap = sum(1 for kw in jd_keywords if kw in desc_text)
        sim = min(overlap / len(jd_keywords), 1.0)
        
        score, base_fit, mult, penalty, is_hp, hp_reason = score_candidate(feat, semantic_similarity=sim)
        features_list.append({
            "feat": feat,
            "score": score,
            "is_hp": is_hp,
            "hp_reason": hp_reason,
            "sim": sim
        })

    # Sort candidates by score descending (honeypots at bottom)
    features_list.sort(key=lambda x: (-x["score"], x["feat"]["candidate_id"]))

    print("\nHow would you like to run calibration?")
    print("1. View current rankings and feature breakdown for all 50 candidates.")
    print("2. Run interactive 1-5 rating workflow (rate a few candidates, then check correlation).")
    
    choice = input("Enter 1 or 2 (default 1): ").strip()
    if choice != "2":
        # Choice 1: Print ranked list
        print("\n=== CURRENT MODEL RANKINGS FOR SAMPLE CANDIDATES ===")
        print(f"{'Rank':<5} | {'Candidate ID':<13} | {'Score':<8} | {'Exp':<5} | {'Notice':<6} | {'Title':<25}")
        print("-" * 75)
        for i, item in enumerate(features_list):
            feat = item["feat"]
            score = item["score"]
            is_hp = item["is_hp"]
            status_str = "HONEYPOT" if is_hp else f"{score:.4f}"
            print(f"{i+1:<5} | {feat['candidate_id']:<13} | {status_str:<8} | {feat['years_of_experience']:<5} | {feat['notice_period_days']:<6} | {feat['current_title'][:25]:<25}")
        print("\nUse these rankings to tune weights in score.py.")
        sys.exit(0)

    # Choice 2: Interactive rating
    ratings = {}
    print("\n--- Interactive Rating Mode ---")
    print("We will show you a few candidates. Enter your judgment score (1 = poor fit, 5 = excellent fit).")
    print("Type 'q' to stop rating and view results.")

    # Show a mix of high, mid, and low model ranked candidates
    indices_to_rate = list(range(0, 10)) + list(range(20, 25)) + list(range(45, 50))
    # Remove duplicates and clamp
    indices_to_rate = sorted(list(set([i for i in indices_to_rate if i < len(features_list)])))

    for idx in indices_to_rate:
        item = features_list[idx]
        print_candidate_summary(idx, item["feat"], item["score"], item["is_hp"], item["hp_reason"])
        user_input = input("Enter your rating (1-5) or 'q': ").strip()
        if user_input.lower() == 'q':
            break
        try:
            rating = float(user_input)
            if 1 <= rating <= 5:
                ratings[item["feat"]["candidate_id"]] = rating
            else:
                print("Invalid range. Skipping candidate.")
        except ValueError:
            print("Invalid input. Skipping candidate.")

    if not ratings:
        print("No ratings provided. Exiting.")
        sys.exit(0)

    print("\n=== CALIBRATION RESULTS ===")
    print(f"{'Candidate ID':<13} | {'User Rating (1-5)':<18} | {'Model Rank':<10} | {'Model Score':<11} | {'Title':<20}")
    print("-" * 80)
    
    user_scores = []
    model_scores = []
    
    for i, item in enumerate(features_list):
        cid = item["feat"]["candidate_id"]
        if cid in ratings:
            user_rating = ratings[cid]
            model_rank = i + 1
            model_score = item["score"]
            user_scores.append(user_rating)
            model_scores.append(model_score)
            print(f"{cid:<13} | {user_rating:<18} | {model_rank:<10} | {model_score:<11.4f} | {item['feat']['current_title'][:20]:<20}")

    if len(user_scores) > 1:
        # Compute correlation
        corr = np.corrcoef(user_scores, model_scores)[0, 1]
        print("-" * 80)
        print(f"Pearson Correlation between User Ratings and Model Scores: {corr:.2f}")
        print("A correlation near 1.0 indicates high agreement. If negative or low, adjust weights in score.py.")
    else:
        print("Not enough rated candidates to compute correlation.")

if __name__ == "__main__":
    main()
