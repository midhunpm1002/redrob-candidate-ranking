import streamlit as st
import traceback

try:
    import json
    import pandas as pd
    import numpy as np
    import os
    import time
    from sentence_transformers import SentenceTransformer

    # Set page config
    st.set_page_config(
        page_title="Redrob Intelligent Candidate Discovery & Ranking",
        page_icon="🤖",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # Premium dark styling with custom CSS
    st.markdown("""
    <style>
        .main {
            background-color: #0e1117;
            color: #ffffff;
        }
        .stButton>button {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border: none;
            padding: 10px 24px;
            border-radius: 8px;
            font-weight: bold;
            transition: all 0.3s ease;
        }
        .stButton>button:hover {
            transform: translateY(-2px);
            box-shadow: 0 4px 12px rgba(102, 126, 234, 0.4);
        }
        .metric-card {
            background-color: #1e293b;
            border-radius: 8px;
            padding: 16px;
            border-left: 5px solid #6366f1;
            margin-bottom: 12px;
        }
    </style>
    """, unsafe_allow_html=True)

    # Import local scoring functions
    from score import score_candidate, check_honeypot
    from precompute_features import extract_features_from_candidate
    from precompute_embeddings import build_candidate_text, JOB_DESCRIPTION

    # Cache model load
    @st.cache_resource
    def load_model():
        return SentenceTransformer("all-MiniLM-L6-v2")

    st.title("🤖 Redrob Intelligent Candidate Discovery & Ranking")
    st.markdown("---")

    st.sidebar.header("Configuration")
    model_option = st.sidebar.selectbox("Embedding Model (CPU)", ["all-MiniLM-L6-v2"])

    # Main layout tabs
    tab1, tab2, tab3 = st.tabs(["🚀 Rank Candidates", "📊 Weight Calibration", "📄 Job Description"])

    with tab3:
        st.subheader("Job Description: Senior AI Engineer — Founding Team")
        st.code(JOB_DESCRIPTION, language="text")

    with tab2:
        st.subheader("Scoring Component Weights")
        st.markdown("Adjust the weights used to compute candidate fit scores.")
        
        col1, col2 = st.columns(2)
        with col1:
            w_title = st.slider("Title Relevance", 0.0, 1.0, 0.25, 0.05)
            w_prod = st.slider("Production ML Evidence", 0.0, 1.0, 0.20, 0.05)
            w_infra = st.slider("Retrieval/Embeddings Infra Skills", 0.0, 1.0, 0.15, 0.05)
            w_eval = st.slider("Evaluation Framework Score", 0.0, 1.0, 0.10, 0.05)
        with col2:
            w_exp = st.slider("Experience Years Fit", 0.0, 1.0, 0.05, 0.05)
            w_loc = st.slider("Location Fit", 0.0, 1.0, 0.05, 0.05)
            w_notice = st.slider("Notice Period Fit", 0.0, 1.0, 0.05, 0.05)
            w_semantic = st.slider("Semantic Similarity Fit", 0.0, 1.0, 0.15, 0.05)
            
        st.info("Ensure the sum of weights is aligned. (Baseline system: Rules sum to 0.85, Semantic weight is 0.15).")

    with tab1:
        st.subheader("Candidate Discovery & Gating Engine")
        st.markdown("Upload a candidate profile sample file (`.json`) to score and rank candidates.")

        uploaded_file = st.file_uploader("Upload candidate sample JSON", type=["json"])
        
        if uploaded_file is not None:
            try:
                candidates = json.load(uploaded_file)
                st.success(f"Successfully loaded {len(candidates)} candidates.")
            except Exception as e:
                st.error(f"Error parsing file: {e}")
                candidates = None
                
            if candidates:
                if st.button("Run Ranking Pipeline"):
                    with st.spinner("Loading Embedding Model and Precomputing..."):
                        model = load_model()
                        
                    st.subheader("Processing Candidates...")
                    start_time = time.time()
                    
                    # Extract features and texts
                    features_list = []
                    texts = []
                    
                    progress_bar = st.progress(0)
                    for i, cand in enumerate(candidates):
                        feat = extract_features_from_candidate(cand)
                        features_list.append(feat)
                        texts.append(build_candidate_text(cand))
                        progress_bar.progress((i + 1) / len(candidates))
                        
                    # Generate embeddings
                    with st.spinner("Generating embeddings on CPU..."):
                        candidate_embeddings = model.encode(texts, convert_to_numpy=True)
                        jd_embedding = model.encode(JOB_DESCRIPTION, convert_to_numpy=True)
                        
                    # Calculate similarities
                    dots = np.dot(candidate_embeddings, jd_embedding)
                    norms = np.linalg.norm(candidate_embeddings, axis=1) * np.linalg.norm(jd_embedding)
                    similarities = np.clip(dots / (norms + 1e-9), 0.0, 1.0)
                    
                    # Score candidates
                    scored_candidates = []
                    honeypots = []
                    
                    for idx, feat in enumerate(features_list):
                        sim = similarities[idx]
                        
                        score, base_fit, mult, penalty, is_hp, hp_reason = score_candidate(feat, semantic_similarity=sim)
                        
                        if is_hp:
                            honeypots.append({
                                "candidate_id": feat["candidate_id"],
                                "title": feat["current_title"],
                                "reason": hp_reason
                            })
                            continue
                            
                        from rank import generate_reasoning
                        reason = generate_reasoning(feat, 50, sim)
                        
                        scored_candidates.append({
                            "candidate_id": feat["candidate_id"],
                            "score": score,
                            "base_fit_score": base_fit,
                            "multiplier": mult,
                            "penalty": penalty,
                            "semantic_similarity": sim,
                            "reasoning": reason,
                            "title": feat["current_title"],
                            "experience": feat["years_of_experience"],
                            "location": feat["location"]
                        })
                        
                    scored_candidates.sort(key=lambda x: (-round(x["score"], 4), x["candidate_id"]))
                    
                    for i, item in enumerate(scored_candidates):
                        item["rank"] = i + 1
                        for f in features_list:
                            if f["candidate_id"] == item["candidate_id"]:
                                item["reasoning"] = generate_reasoning(f, i + 1, item["semantic_similarity"])
                                break
                                
                    elapsed = time.time() - start_time
                    st.success(f"Completed in {elapsed:.2f} seconds. Gated {len(honeypots)} honeypot profiles.")
                    
                    m_col1, m_col2, m_col3 = st.columns(3)
                    m_col1.metric("Total Input Candidates", len(candidates))
                    m_col2.metric("Gated Honeypots", len(honeypots))
                    m_col3.metric("Scored / Valid", len(scored_candidates))
                    
                    df_scored = pd.DataFrame(scored_candidates)
                    st.subheader("Top Ranked Candidates")
                    
                    if not df_scored.empty:
                        df_display = df_scored[["rank", "candidate_id", "score", "title", "experience", "location", "reasoning"]]
                        st.dataframe(df_display, use_container_width=True)
                        
                        csv_data = df_scored[["candidate_id", "rank", "score", "reasoning"]].to_csv(index=False)
                        st.download_button(
                            label="Download Ranked CSV (submission format)",
                            data=csv_data,
                            file_name="ranked_candidates.csv",
                            mime="text/csv"
                        )
                    else:
                        st.info("No candidates passed the honeypot gate.")
                        
                    if honeypots:
                        with st.expander("Show Gated Honeypot Details"):
                            st.dataframe(pd.DataFrame(honeypots), use_container_width=True)
        else:
            st.info("Upload a candidate profile sample file (like `sample_candidates.json` from workspace) to start.")

except Exception as e:
    st.error("An error occurred during app execution:")
    st.code(traceback.format_exc(), language="text")
