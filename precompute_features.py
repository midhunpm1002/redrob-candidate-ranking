#!/usr/bin/env python3
"""
Phase 1: Feature and Honeypot Extraction.
Parses candidates.jsonl, extracts rule-based features, and checks consistency.
Saves features to artifacts/features.pkl.gz.
"""

import os
import json
import gzip
import pickle
import argparse
from datetime import datetime
from tqdm import tqdm

# Constants
CURRENT_DATE = datetime(2026, 6, 27)  # Current date per metadata

def parse_date(date_str):
    if not date_str:
        return None
    try:
        return datetime.strptime(date_str, "%Y-%m-%d")
    except ValueError:
        return None

def check_honeypot(cand):
    """
    Evaluates consistency constraints to flag honeypots.
    Returns (is_honeypot, reason).
    """
    profile = cand.get("profile", {})
    career = cand.get("career_history", [])
    skills = cand.get("skills", [])
    signals = cand.get("redrob_signals", {})

    # 1. Experience consistency
    years_exp = profile.get("years_of_experience", 0)
    if years_exp < 0:
        return True, "Negative years of experience"
    
    total_career_months = sum(item.get("duration_months", 0) for item in career)
    stated_months = years_exp * 12
    # Wild mismatches: allow reasonable slack for gaps or overlaps, but flag extreme discrepancies
    # For example, if career history months are less than 20% or more than 200% of stated experience, and the difference is large.
    # Let's check: if stated experience is > 1 year (12 months), and total career history is less than 35% of stated or more than 250%.
    if stated_months > 12:
        if total_career_months < stated_months * 0.35:
            return True, f"Career history duration ({total_career_months}m) is far less than stated experience ({years_exp}y / {stated_months}m)"
        if total_career_months > stated_months * 2.5:
            return True, f"Career history duration ({total_career_months}m) is far greater than stated experience ({years_exp}y / {stated_months}m)"
    elif years_exp == 0 and total_career_months > 12:
        return True, f"0 years of experience stated but career history duration is {total_career_months}m"

    # 2. Expert skill with 0 duration
    for s in skills:
        if s.get("proficiency") == "expert" and s.get("duration_months", 0) <= 0:
            return True, f"Expert proficiency claimed for skill '{s.get('name')}' with 0 duration"

    # 3. Expected salary min > max
    salary_range = signals.get("expected_salary_range_inr_lpa", {})
    sal_min = salary_range.get("min", 0)
    sal_max = salary_range.get("max", 0)
    if sal_min > sal_max:
        return True, f"Expected salary min ({sal_min}) is greater than max ({sal_max})"

    # 4. Overlapping date ranges
    # Build list of intervals (start, end)
    intervals = []
    current_jobs = 0
    for idx, job in enumerate(career):
        start = parse_date(job.get("start_date"))
        end = parse_date(job.get("end_date"))
        is_cur = job.get("is_current", False)
        
        if is_cur:
            current_jobs += 1
            if not end:
                end = CURRENT_DATE
        elif not end:
            end = CURRENT_DATE
            
        if start and end:
            if start > end:
                return True, f"Job {idx} has start date after end date"
            intervals.append((start, end, job.get("company", "")))

    if current_jobs > 1:
        return True, f"Multiple concurrent 'current' jobs ({current_jobs})"

    # Check for overlaps between unrelated companies
    intervals.sort(key=lambda x: x[0])
    for i in range(len(intervals)):
        for j in range(i + 1, len(intervals)):
            s1, e1, c1 = intervals[i]
            s2, e2, c2 = intervals[j]
            # If they overlap significantly
            if s2 < e1:
                # Calculate overlap in days
                overlap_days = (min(e1, e2) - s2).days
                # If overlap is more than 90 days (3 months) at different companies
                if overlap_days > 90 and c1 != c2:
                    return True, f"Overlapping jobs at '{c1}' and '{c2}' for {overlap_days} days"

    # 5. Timeline inversion
    signup = parse_date(signals.get("signup_date"))
    last_act = parse_date(signals.get("last_active_date"))
    if signup and last_act and signup > last_act:
        return True, f"Signup date ({signals.get('signup_date')}) is after last active date ({signals.get('last_active_date')})"
    if signup and signup > CURRENT_DATE:
        return True, f"Signup date ({signals.get('signup_date')}) is in the future"

    return False, ""

def extract_features_from_candidate(cand):
    """
    Extracts tabular features for rule-based scoring.
    """
    profile = cand.get("profile", {})
    career = cand.get("career_history", [])
    skills = cand.get("skills", [])
    signals = cand.get("redrob_signals", {})
    education = cand.get("education", [])

    cid = cand.get("candidate_id")
    is_hp, hp_reason = check_honeypot(cand)

    # Basic fields
    years_exp = profile.get("years_of_experience", 0)
    current_title = profile.get("current_title", "")
    current_company = profile.get("current_company", "")
    current_company_size = profile.get("current_company_size", "")
    current_industry = profile.get("current_industry", "")
    location = profile.get("location", "")
    country = profile.get("country", "")

    # career history details
    career_titles = [job.get("title", "") for job in career]
    career_companies = [job.get("company", "") for job in career]
    career_industries = [job.get("industry", "") for job in career]
    career_durations = [job.get("duration_months", 0) for job in career]
    career_descriptions = [job.get("description", "") for job in career]

    # Education tiers
    edu_tiers = [edu.get("tier", "unknown") for edu in education]
    
    # Skills list
    skills_list = []
    for s in skills:
        skills_list.append({
            "name": s.get("name", ""),
            "proficiency": s.get("proficiency", "beginner"),
            "duration_months": s.get("duration_months", 0),
            "endorsements": s.get("endorsements", 0)
        })

    # Redrob signals
    notice_period = signals.get("notice_period_days", 90)
    willing_relocate = signals.get("willing_to_relocate", False)
    preferred_work_mode = signals.get("preferred_work_mode", "flexible")
    expected_salary = signals.get("expected_salary_range_inr_lpa", {})
    expected_salary_min = expected_salary.get("min", 0.0)
    expected_salary_max = expected_salary.get("max", 0.0)
    
    # Behavioral
    recruiter_resp_rate = signals.get("recruiter_response_rate", 0.0)
    last_act_str = signals.get("last_active_date", "")
    open_to_work = signals.get("open_to_work_flag", False)
    interview_comp_rate = signals.get("interview_completion_rate", 0.0)
    github_score = signals.get("github_activity_score", -1.0)
    offer_acc_rate = signals.get("offer_acceptance_rate", -1.0)
    saved_recruiters_30d = signals.get("saved_by_recruiters_30d", 0)
    search_appearance_30d = signals.get("search_appearance_30d", 0)

    # Compile the feature record
    feature_record = {
        "candidate_id": cid,
        "is_honeypot": is_hp,
        "honeypot_reason": hp_reason,
        "years_of_experience": years_exp,
        "current_title": current_title,
        "current_company": current_company,
        "current_company_size": current_company_size,
        "current_industry": current_industry,
        "location": location,
        "country": country,
        "career_titles": career_titles,
        "career_companies": career_companies,
        "career_industries": career_industries,
        "career_durations": career_durations,
        "career_descriptions": career_descriptions,
        "edu_tiers": edu_tiers,
        "skills": skills_list,
        "notice_period_days": notice_period,
        "willing_to_relocate": willing_relocate,
        "preferred_work_mode": preferred_work_mode,
        "expected_salary_min": expected_salary_min,
        "expected_salary_max": expected_salary_max,
        "recruiter_response_rate": recruiter_resp_rate,
        "last_active_date_str": last_act_str,
        "open_to_work_flag": open_to_work,
        "interview_completion_rate": interview_comp_rate,
        "github_activity_score": github_score,
        "offer_acceptance_rate": offer_acc_rate,
        "saved_by_recruiters_30d": saved_recruiters_30d,
        "search_appearance_30d": search_appearance_30d,
        "headline": profile.get("headline", ""),
        "summary": profile.get("summary", "")
    }
    return feature_record

def main():
    parser = argparse.ArgumentParser(description="Precompute candidate features and check honeypots.")
    parser.add_argument("--candidates", type=str, default="candidates.jsonl", help="Path to candidates jsonl file")
    parser.add_argument("--out", type=str, default="artifacts/features.pkl.gz", help="Path to save output features pickle")
    args = parser.parse_args()

    print(f"Reading candidates from {args.candidates}...")
    start_time = datetime.now()

    # Create artifacts directory if not exists
    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    features = {}
    honeypot_count = 0
    total_count = 0

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

        for line in tqdm(f, total=total_lines, desc="Extracting features"):
            if not line.strip():
                continue
            cand = json.loads(line)
            feat = extract_features_from_candidate(cand)
            features[feat["candidate_id"]] = feat
            total_count += 1
            if feat["is_honeypot"]:
                honeypot_count += 1

    print(f"Writing features to {args.out}...")
    with gzip.open(args.out, "wb") as f:
        pickle.dump(features, f, protocol=pickle.HIGHEST_PROTOCOL)

    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()
    print(f"Precomputed features for {total_count} candidates.")
    print(f"Flagged {honeypot_count} honeypots ({honeypot_count/total_count:.2%} of total).")
    print(f"Done in {duration:.2f} seconds.")

if __name__ == "__main__":
    main()
