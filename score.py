#!/usr/bin/env python3
"""
Scoring and Gating Logic for Redrob Challenge.
Contains constants, honeypot consistency gate, rule-based features,
behavioral multiplier, semantic similarity integration, and the final scoring function.
"""

import numpy as np
from datetime import datetime

# ==========================================
# Named Weights & Constants
# ==========================================
# Rule-based weights (sum = 0.85)
WEIGHT_TITLE = 0.25
WEIGHT_PRODUCTION = 0.20
WEIGHT_INFRA = 0.15
WEIGHT_EVAL = 0.10
WEIGHT_EXPERIENCE = 0.05
WEIGHT_LOCATION = 0.05
WEIGHT_NOTICE = 0.05

# Semantic weight (0.15)
WEIGHT_SEMANTIC = 0.15

# Penalties (subtracted directly from base fit score)
PENALTY_TITLE_CHASER = 0.15
PENALTY_CONSULTING_ONLY = 0.25
PENALTY_CV_SPEECH_ROBOTICS_ONLY = 0.20

# Behavioral Multiplier Bounds
BEHAVIORAL_MIN = 0.70
BEHAVIORAL_MAX = 1.15

# Evaluation Date Reference
CURRENT_DATE = datetime(2026, 6, 27)

# ==========================================
# Gating Logic (Honeypot Detector)
# ==========================================
def parse_date(date_str):
    if not date_str:
        return None
    try:
        return datetime.strptime(date_str, "%Y-%m-%d")
    except ValueError:
        return None

def check_honeypot(cand_feat):
    """
    Checks candidate features for contradictory or physically impossible values.
    Returns (is_honeypot, reason).
    """
    years_exp = cand_feat.get("years_of_experience", 0)
    if years_exp < 0:
        return True, "Negative years of experience"

    # 1. Experience consistency
    durations = cand_feat.get("career_durations", [])
    total_career_months = sum(durations)
    stated_months = years_exp * 12

    if stated_months > 12:
        if total_career_months < stated_months * 0.35:
            return True, f"Career history duration ({total_career_months}m) is far less than stated experience ({years_exp}y / {stated_months}m)"
        if total_career_months > stated_months * 2.5:
            return True, f"Career history duration ({total_career_months}m) is far greater than stated experience ({years_exp}y / {stated_months}m)"
    elif years_exp == 0 and total_career_months > 12:
        return True, f"0 years of experience stated but career history duration is {total_career_months}m"

    # 2. Expert skill with 0 duration
    skills = cand_feat.get("skills", [])
    for s in skills:
        if s.get("proficiency") == "expert" and s.get("duration_months", 0) <= 0:
            return True, f"Expert proficiency claimed for skill '{s.get('name')}' with 0 duration"

    # 3. Overlapping date ranges
    # Build list of intervals (start, end)
    intervals = []
    current_jobs = 0
    
    # We reconstruct intervals from career history if needed, but in precompute we do it.
    # To make score.py independent, let's copy the overlap date check logic here.
    # Note: we check this during feature extraction in precompute_features.py,
    # but we will also store the precomputed result.
    is_hp_precomputed = cand_feat.get("is_honeypot", False)
    hp_reason_precomputed = cand_feat.get("honeypot_reason", "")
    if is_hp_precomputed:
        return True, hp_reason_precomputed

    return False, ""


# ==========================================
# Scoring Component Functions
# ==========================================
def score_single_title(title):
    if not title:
        return 0.0
    title_lower = title.lower()
    
    # Check unrelated first
    unrelated_keywords = [
        "marketing", "hr", "recruiter", "accountant", "mechanical", "civil",
        "sales", "support", "operations", "graphic", "designer", "brand",
        "business analyst", "finance", "legal", "product manager", "project manager",
        "civil engineer", "mechanical engineer", "sales executive", "customer support"
    ]
    if any(k in title_lower for k in unrelated_keywords):
        return 0.0
        
    strong_keywords = [
        "ai engineer", "ml engineer", "machine learning engineer", 
        "search engineer", "retrieval engineer", "ranking engineer", 
        "recommendation engineer", "nlp engineer", "llm engineer",
        "ai/ml engineer", "founding ai engineer", "staff ml engineer"
    ]
    if any(k in title_lower for k in strong_keywords):
        return 1.0
        
    moderate_keywords = [
        "applied scientist", "data scientist", "research scientist", 
        "ml researcher", "nlp researcher", "deep learning", "ai scientist"
    ]
    if any(k in title_lower for k in moderate_keywords):
        return 0.8
        
    baseline_keywords = [
        "backend", "software engineer", "data engineer", 
        "technical lead", "tech lead", "systems engineer", "principal engineer"
    ]
    if any(k in title_lower for k in baseline_keywords):
        return 0.6
        
    weak_keywords = [
        "full stack", "full-stack", "frontend", "front-end", "developer"
    ]
    if any(k in title_lower for k in weak_keywords):
        return 0.4
        
    return 0.2

def compute_title_relevance(cand_feat):
    """
    Computes title relevance score: max(current_title_score, 0.7 * max(career_titles_score)).
    """
    current_title = cand_feat.get("current_title", "")
    current_score = score_single_title(current_title)
    
    career_titles = cand_feat.get("career_titles", [])
    if career_titles:
        career_score = max(score_single_title(t) for t in career_titles)
        return max(current_score, 0.7 * career_score)
    return current_score

def compute_production_ml_evidence(cand_feat):
    """
    Computes score for production ML/search/ranking deployment evidence.
    Looks at career descriptions and company profiles.
    """
    descriptions = cand_feat.get("career_descriptions", [])
    summary = cand_feat.get("summary", "")
    headline = cand_feat.get("headline", "")
    
    all_text = " ".join(descriptions + [summary, headline]).lower()
    
    # Positive production/scale keywords
    prod_keywords = [
        "deployed", "production", "shipped", "real users", "scale", "in production",
        "recommendation system", "ranking system", "search engine", "retrieval engine",
        "serving", "optimized", "optimised", "latency", "throughput", "ab test", "a/b test",
        "ndcg", "mrr", "map", "vector database", "millions of users", "hybrid search"
    ]
    
    matches = sum(1 for kw in prod_keywords if kw in all_text)
    # Simple log scale to reward matching multiple terms, capped at 1.0
    text_score = min(matches / 4.0, 1.0)
    
    # Company multiplier: product experience is valued higher than pure consulting
    # Check current industry and company size
    industry = cand_feat.get("current_industry", "").lower()
    comp_size = cand_feat.get("current_company_size", "")
    
    is_consulting = any(c in industry for c in ["consulting", "it services", "staffing"])
    
    # Recognize product company indicators (size, software/internet industry)
    is_product_indicator = any(p in industry for p in ["software", "internet", "e-commerce", "technology", "financial services", "computer software"])
    
    multiplier = 1.0
    if is_consulting:
        multiplier = 0.7
    elif is_product_indicator:
        multiplier = 1.2
        
    return min(text_score * multiplier, 1.0)

def compute_infra_score(cand_feat):
    """
    Detects mentions of embeddings/retrieval infrastructure.
    Weights each by duration, proficiency, and endorsements.
    """
    infra_skills = [
        "sentence-transformers", "openai embeddings", "bge", "e5", "pinecone",
        "weaviate", "qdrant", "milvus", "opensearch", "elasticsearch", "faiss",
        "vector database", "hybrid search", "bm25"
    ]
    
    skills = cand_feat.get("skills", [])
    total_val = 0.0
    
    for s in skills:
        name = s.get("name", "").lower().replace(" ", "").replace("-", "")
        # Match names
        matched = False
        for infra in infra_skills:
            clean_infra = infra.replace(" ", "").replace("-", "")
            if clean_infra in name or name in clean_infra:
                matched = True
                break
                
        if matched:
            dur = s.get("duration_months", 0)
            prof = s.get("proficiency", "beginner")
            ends = s.get("endorsements", 0)
            
            prof_mult = {"beginner": 0.25, "intermediate": 0.5, "advanced": 0.8, "expert": 1.0}.get(prof, 0.25)
            dur_score = min(dur / 24.0, 1.5)
            ends_score = min(ends / 10.0, 1.0)
            
            total_val += (dur_score + ends_score) * prof_mult
            
    # Normalize: if they have at least two well-developed infra skills, score is 1.0
    return min(total_val / 3.0, 1.0)

def compute_eval_score(cand_feat):
    """
    Detects evaluation framework mentions (NDCG, MRR, MAP, A/B test, offline/online eval).
    Weights similarly by duration/endorsements in skills, plus text presence.
    """
    eval_skills = ["ndcg", "mrr", "map", "a/b test", "evaluation", "ab test"]
    skills = cand_feat.get("skills", [])
    total_val = 0.0
    
    for s in skills:
        name = s.get("name", "").lower()
        matched = any(ev in name for ev in eval_skills)
        if matched:
            dur = s.get("duration_months", 0)
            prof = s.get("proficiency", "beginner")
            ends = s.get("endorsements", 0)
            
            prof_mult = {"beginner": 0.25, "intermediate": 0.5, "advanced": 0.8, "expert": 1.0}.get(prof, 0.25)
            dur_score = min(dur / 24.0, 1.5)
            ends_score = min(ends / 10.0, 1.0)
            
            total_val += (dur_score + ends_score) * prof_mult
            
    # Also scan text descriptions
    descriptions = cand_feat.get("career_descriptions", [])
    all_text = " ".join(descriptions).lower()
    text_matches = sum(1 for ev in eval_skills if ev in all_text)
    text_score = min(text_matches * 0.25, 0.5)
    
    return min((total_val / 2.0) + text_score, 1.0)

def compute_experience_fit(cand_feat):
    """
    Smooth Gaussian-like experience years fit score.
    Peaks at 5-9 years, decays gradually outside.
    """
    years = cand_feat.get("years_of_experience", 0)
    if 5.0 <= years <= 9.0:
        return 1.0
    elif years < 5.0:
        return float(np.exp(-0.5 * ((years - 5.0) / 1.5) ** 2))
    else:
        return float(np.exp(-0.5 * ((years - 9.0) / 3.0) ** 2))

def compute_location_fit(cand_feat):
    """
    Noida/Pune preferred, other Tier-1 Indian cities welcome with relocation.
    """
    loc = cand_feat.get("location", "").lower()
    reloc = cand_feat.get("willing_to_relocate", False)
    country = cand_feat.get("country", "").lower()
    
    # Check country
    is_india = "india" in country or "india" in loc or loc in ["pune", "noida", "bangalore", "bengaluru", "delhi", "mumbai", "hyderabad", "chennai", "gurgaon"]
    
    if "pune" in loc or "noida" in loc:
        return 1.0
        
    tier1_cities = ["hyderabad", "mumbai", "delhi", "bangalore", "bengaluru", "chennai", "gurgaon", "ncr", "kolkata"]
    is_tier1 = any(c in loc for c in tier1_cities)
    
    if is_tier1:
        if reloc or "india" in country:
            return 0.8
        return 0.4
        
    if is_india:
        if reloc:
            return 0.6
        return 0.3
        
    return 0.1  # Outside India

def compute_notice_fit(cand_feat):
    """
    Smooth notice period score. Prefers <= 30 days.
    """
    notice = cand_feat.get("notice_period_days", 30)
    if notice <= 30:
        return 1.0
    return float(np.exp(-0.5 * ((notice - 30.0) / 45.0) ** 2))


# ==========================================
# Negative Signal Penalties
# ==========================================
def compute_penalties(cand_feat):
    """
    Calculates total penalties to subtract.
    """
    penalty = 0.0
    
    # 1. Title Chaser Penalty
    durations = cand_feat.get("career_durations", [])
    if len(durations) >= 3:
        avg_tenure = sum(durations) / len(durations)
        if avg_tenure < 18.0:
            # Check title seniority inflation (e.g. going from junior/se to lead/staff rapidly)
            titles = [t.lower() for t in cand_feat.get("career_titles", [])]
            # Simple check: does the list contain junior/intern at start and senior/staff later?
            # Since index 0 is usually most recent in career_history:
            has_junior = any("junior" in t or "intern" in t or "associate" in t for t in titles[1:])
            has_senior = any("senior" in t or "lead" in t or "staff" in t or "principal" in t for t in titles[:2])
            if has_junior and has_senior:
                penalty += PENALTY_TITLE_CHASER
                
    # 2. Consulting Only Penalty
    companies = [c.lower() for c in cand_feat.get("career_companies", [])]
    current_company = cand_feat.get("current_company", "").lower()
    all_companies = companies + [current_company]
    
    services_firms = ["tcs", "infosys", "wipro", "accenture", "cognizant", "capgemini", "tech mahindra", "hcl", "lti", "mindtree"]
    
    # Check if currently at a services firm AND has only services/consulting industry in career history
    industries = [ind.lower() for ind in cand_feat.get("career_industries", [])]
    current_industry = cand_feat.get("current_industry", "").lower()
    all_industries = industries + [current_industry]
    
    is_consulting_industry = all(any(x in ind for x in ["services", "consulting", "staffing"]) for ind in all_industries if ind)
    is_services_companies = all(any(firm in comp for firm in services_firms) for comp in all_companies if comp)
    
    if is_consulting_industry and is_services_companies and len(all_companies) > 0:
        penalty += PENALTY_CONSULTING_ONLY

    # 3. Computer Vision/Speech/Robotics Only Penalty
    skills = [s.get("name", "").lower() for s in cand_feat.get("skills", [])]
    descriptions = cand_feat.get("career_descriptions", [])
    summary = cand_feat.get("summary", "").lower()
    all_text = " ".join(descriptions + [summary]).lower()
    
    cv_keywords = [
        "computer vision", "image classification", "object detection", "speech recognition",
        "text-to-speech", "tts", "stt", "robotics", "cv", "speech", "vision", "opencv",
        "yolo", "ros", "pointcloud", "lidar", "segmentation", "pytorch3d"
    ]
    nlp_search_keywords = [
        "nlp", "natural language", "retrieval", "search", "ranking", "recommendation",
        "rag", "embedding", "vector search", "bm25", "milvus", "pinecone", "weaviate",
        "qdrant", "elasticsearch", "solr", "lucene", "information retrieval", "ir"
    ]
    
    has_cv = any(kw in all_text for kw in cv_keywords) or any(any(kw in s for kw in cv_keywords) for s in skills)
    has_nlp_search = any(kw in all_text for kw in nlp_search_keywords) or any(any(kw in s for kw in nlp_search_keywords) for s in skills)
    
    if has_cv and not has_nlp_search:
        penalty += PENALTY_CV_SPEECH_ROBOTICS_ONLY
        
    return penalty


# ==========================================
# Behavioral Availability Multiplier
# ==========================================
def compute_behavioral_multiplier(cand_feat):
    """
    Computes behavioral multiplier based on engagement metrics.
    Bounds the result to [0.70, 1.15].
    Sentinels (-1) are mapped to neutral.
    """
    # 1. last_active_date recency
    last_act_str = cand_feat.get("last_active_date_str", "")
    last_act = parse_date(last_act_str)
    
    recency_factor = 0.9  # Default neutral
    if last_act:
        days_inactive = (CURRENT_DATE - last_act).days
        if days_inactive <= 30:
            recency_factor = 1.1
        elif days_inactive <= 90:
            recency_factor = 1.0
        elif days_inactive <= 180:
            recency_factor = 0.8
        else:
            recency_factor = 0.6
            
    # 2. open_to_work_flag
    open_to_work = cand_feat.get("open_to_work_flag", False)
    otw_factor = 1.05 if open_to_work else 0.95
    
    # 3. recruiter_response_rate
    resp_rate = cand_feat.get("recruiter_response_rate", 0.5)
    # Neutral is around 0.5, maps to 1.0
    resp_factor = 0.85 + 0.3 * resp_rate  # Range: [0.85, 1.15]
    
    # 4. interview_completion_rate
    interview_rate = cand_feat.get("interview_completion_rate", 0.8)
    int_factor = 0.9 + 0.2 * interview_rate  # Range: [0.9, 1.1]
    
    # 5. github_activity_score
    github = cand_feat.get("github_activity_score", -1.0)
    git_factor = 1.0
    if github >= 0:
        git_factor = 0.95 + 0.15 * (github / 100.0)  # Range: [0.95, 1.10]
        
    # 6. offer_acceptance_rate
    offer_acc = cand_feat.get("offer_acceptance_rate", -1.0)
    offer_factor = 1.0
    if offer_acc >= 0:
        offer_factor = 0.9 + 0.2 * offer_acc  # Range: [0.9, 1.1]
        
    # 7. saved_by_recruiters and search appearances
    saved = cand_feat.get("saved_by_recruiters_30d", 0)
    saved_factor = 1.0 + 0.02 * min(saved, 5)
    
    # Combine multiplicatively
    mult = recency_factor * otw_factor * resp_factor * int_factor * git_factor * offer_factor * saved_factor
    
    # Bounded between 0.70 and 1.15
    return float(np.clip(mult, BEHAVIORAL_MIN, BEHAVIORAL_MAX))


# ==========================================
# Complete Hybrid Scorer
# ==========================================
def score_candidate(cand_feat, semantic_similarity=0.5):
    """
    Computes the final composite score for a candidate.
    Returns (score, base_fit_score, multiplier, penalty, is_honeypot, honeypot_reason)
    """
    # 1. Honeypot check
    is_hp, hp_reason = check_honeypot(cand_feat)
    if is_hp:
        return 0.0, 0.0, 0.0, 0.0, True, hp_reason
        
    # 2. Rule sub-scores
    title = compute_title_relevance(cand_feat)
    prod = compute_production_ml_evidence(cand_feat)
    infra = compute_infra_score(cand_feat)
    eval_score = compute_eval_score(cand_feat)
    exp = compute_experience_fit(cand_feat)
    loc = compute_location_fit(cand_feat)
    notice = compute_notice_fit(cand_feat)
    
    # 3. Combine base fit rules
    base_rules_score = (
        WEIGHT_TITLE * title +
        WEIGHT_PRODUCTION * prod +
        WEIGHT_INFRA * infra +
        WEIGHT_EVAL * eval_score +
        WEIGHT_EXPERIENCE * exp +
        WEIGHT_LOCATION * loc +
        WEIGHT_NOTICE * notice
    )
    
    # 4. Integrate semantic similarity
    # Ensure similarity is bounded and scaled
    sem_score = float(np.clip(semantic_similarity, 0.0, 1.0))
    
    # Base fit score before penalties
    base_fit = base_rules_score + WEIGHT_SEMANTIC * sem_score
    
    # 5. Apply penalties
    penalties = compute_penalties(cand_feat)
    final_base_fit = max(0.0, base_fit - penalties)
    
    # 6. Apply behavioral availability multiplier
    multiplier = compute_behavioral_multiplier(cand_feat)
    final_score = final_base_fit * multiplier
    
    return final_score, final_base_fit, multiplier, penalties, False, ""
