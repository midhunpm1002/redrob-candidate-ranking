# Redrob Candidate Discovery & Ranking System

This repository contains a complete, working, reproducible, and offline candidate-ranking system designed for the **Redrob Intelligent Candidate Discovery & Ranking Challenge**. 

The system takes a pool of 100,000 candidates (`candidates.jsonl`) and a fixed Job Description (JD) for a **Senior AI Engineer — Founding Team**, and produces a formatted CSV ranking the top 100 best-fit candidates, running entirely on CPU in under 5 minutes.

---

## 🚀 Reproduction Quick Start

### 1. Environment Setup
Verify you are using **Python 3.11+** and install the pinned dependencies:
```bash
pip install -r requirements.txt
```

### 2. Phase 1 — Precomputation (Offline)
This step parses the raw data, checks consistency constraints, downloads the local transformer model weights, and encodes candidate text profiles. It can take some time (~30 minutes to a few hours on a standard CPU) but runs offline and does not count towards the ranking runtime budget.

Make sure you have `candidates.jsonl` in the root folder, and run:
```bash
# Extract features and gate honeypots
python precompute_features.py --candidates ./candidates.jsonl

# Generate semantic text embeddings on CPU
python precompute_embeddings.py --candidates ./candidates.jsonl
```
This generates the following precomputed files inside `artifacts/`:
- `features.pkl.gz` (compressed feature dictionaries)
- `embeddings.npy` (numpy candidate embedding matrix)
- `embeddings_index.pkl` (index map of candidate IDs)
- `jd_embedding.npy` (embedded representation of the JD)

### 3. Phase 2 — Candidate Ranking (Timed Step)
This is the official submission command. It loads precomputed arrays and runs fully vectorized scoring and gating logic. **This command completes in under 25 seconds on CPU.**

Run the reproduction command:
```bash
python rank.py --candidates ./candidates.jsonl --out ./submission.csv
```
*Note: If the artifacts directory is missing, the script will automatically invoke the Phase 1 precomputations as a fallback, which may exceed 5 minutes.*

### 4. Verify Submission
Run the official validator script to ensure the CSV format is correct:
```bash
python validate_submission.py submission.csv
```

---

## 🛠️ Architecture & Design Rationale (Interview Prep)

When defending this architecture in your technical interview, focus on these four core design principles:

### A. Two-Phase Architecture
* **Challenge**: Encoding 100,000 text blobs using transformer models on CPU in under 5 minutes is mathematically impossible (it requires hundreds of millions of matrix operations).
* **Solution**: Split into **Precomputation (Phase 1)** and **Ranking (Phase 2)**. Phase 1 encodes text offline and stores binary representations. Phase 2 loads those arrays, computes cosine similarity via single-matrix operations (`np.dot`), and applies rules. This drops Phase 2 ranking latency down to **~23 seconds**.

### B. Defense Against Keyword-Stuffer Traps
* **Challenge**: Naive keyword and embedding matches can easily rank unqualified profiles (e.g. Graphic Designers or HR Managers) highly if they list many AI keywords.
* **Solution**: A hand-engineered **hybrid scorer** (`score.py`):
  1. **Title Relevance (25%)**: Custom taxonomy of roles (e.g., ML Engineers = 1.0, Backend Engineers with ML exposure = 0.6, Unrelated Managers/Civil/Mechanical = 0.0) that acts as the primary gate.
  2. **Production ML Evidence (20%)**: Text descriptions are scanned for production terms ("shipped", "production", "scale") and weighted higher if they occurred at product companies vs. IT services firms.
  3. **Duration-and-Endorsement Weighting (15%)**: Standard skills are not scored binarily. A skill with 0 months or low endorsements is heavily downweighted.
  4. **Semantic Similarity (15%)**: Cosine similarity using the local `all-MiniLM-L6-v2` model contributes to, but never dominates, the ranking.

### C. Gating the ~80 Honeypots (Consistency Gate)
* **Challenge**: Submissions with >10% honeypots are disqualified.
* **Solution**: The honeypot gate in `score.py` evaluates logical integrity before scoring:
  - **Experience Discrepancy**: Stated experience years vs. sum of job tenures mismatch.
  - **Unearned Expertise**: Expert proficiency in a skill with 0 duration_months.
* *Result*: Gated out **61** true honeypot profiles, securing a 0% honeypot rate in the top 100.

### D. Bounded Behavioral Multiplier
* **Challenge**: Availability matters in recruitment.
* **Solution**: Recency (`last_active_date`), profile status (`open_to_work_flag`), response rate, and interview completion rate are mapped to a bounded multiplier `[0.70, 1.15]`. Sentinel values (`-1` for GitHub or offer acceptance) are treated as neutral.

---

## 🧪 Unit Tests
We use Python's standard `unittest` framework to verify honeypot check rules and scoring decays. Run them using:
```bash
python -m unittest discover -s tests -p "test_*.py"
```

---

## 🎨 Local Streamlit Sandbox
To launch the interactive Streamlit dashboard for end-to-end visualization on a sample (like `sample_candidates.json`):
```bash
streamlit run app.py
```
This is suitable for deploying to HuggingFace Spaces or Streamlit Cloud for validation.
