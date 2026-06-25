"""
AI Candidate Ranking System - India Runs / Redrob Challenge
=============================================================
Hybrid scoring:
  1. TF-IDF Cosine Similarity  (semantic layer via sklearn)
  2. Keyword skill matching     (explicit skill taxonomy)
  3. Career history quality     (product company, ML experience)
  4. Experience & seniority     (YoE, title seniority)
  5. Redrob behavioral signals  (availability, responsiveness, activity)
  6. Disqualifier penalties     (bad signals from JD)

Output: team_antigravity.csv  (top 100 candidates)
"""

import os
import json
import csv
import time
import math
import re
from datetime import datetime
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# ─────────────────────────────────────────────────────────────
# PATHS
# ─────────────────────────────────────────────────────────────
BASE_DIR = r"C:\Users\gopar\OneDrive\Desktop\India Runs\extracted_data\[PUB] India_runs_data_and_ai_challenge\India_runs_data_and_ai_challenge"
CANDIDATES_JSONL = os.path.join(BASE_DIR, "candidates.jsonl")
OUTPUT_CSV = r"C:\Users\gopar\OneDrive\Desktop\India Runs\team_antigravity.csv"

# ─────────────────────────────────────────────────────────────
# JOB DESCRIPTION  (full extracted content)
# ─────────────────────────────────────────────────────────────
JOB_DESCRIPTION = """
Senior Machine Learning Engineer AI Platform Redrob
Own the intelligence layer of Redrob product ranking retrieval matching systems
5 to 9 years experience applied ML AI product companies
Production experience embeddings based retrieval systems sentence transformers BGE E5 OpenAI embeddings
Production experience vector databases hybrid search Pinecone Weaviate Qdrant Milvus FAISS Elasticsearch OpenSearch
Strong Python production code quality
Hands on experience designing evaluation frameworks ranking NDCG MRR MAP A B testing offline evaluation
End to end ranking search recommendation system shipped real users scale
LLM fine tuning LoRA QLoRA PEFT learning to rank XGBoost neural LTR
HR tech recruiting tech marketplace product experience
Distributed systems large scale inference optimization
Open source AI ML contributions
Shipped end to end ranking search recommendation system production real users meaningful scale
Strong opinions retrieval hybrid dense evaluation offline online LLM integration fine tune prompt
NLP information retrieval natural language processing
Transformer BERT encoder decoder attention mechanism
Information retrieval semantic search vector embeddings cosine similarity
Recommendation system collaborative filtering content based filtering
A B testing online evaluation experiment design statistical significance
MLOps model deployment serving inference optimization
Python PyTorch TensorFlow scikit learn
Machine learning deep learning neural network
Data pipeline feature engineering model training
Production deployment scaling monitoring
Product company engineering culture shipping fast iteration
"""

# ─────────────────────────────────────────────────────────────
# SKILL TAXONOMIES
# ─────────────────────────────────────────────────────────────
CORE_ML_SKILLS = {
    # Embeddings & Retrieval
    "sentence-transformers", "sentence transformers", "embeddings", "embedding",
    "vector search", "dense retrieval", "bi-encoder", "cross-encoder", "bge", "e5",
    "openai embeddings", "semantic search", "semantic similarity",
    # Vector DBs
    "pinecone", "weaviate", "qdrant", "milvus", "faiss", "elasticsearch", "opensearch",
    "vector database", "vector db", "ann", "approximate nearest neighbor",
    # Search & Ranking
    "ranking", "information retrieval", "search", "recommendation",
    "bm25", "hybrid search", "reranking", "re-ranking",
    "ndcg", "mrr", "map", "learning to rank", "ltr", "xgboost", "lightgbm",
    # NLP/ML Core
    "nlp", "natural language processing", "transformers", "bert", "pytorch",
    "tensorflow", "python", "machine learning", "deep learning", "scikit-learn",
    "neural network", "fine-tuning", "transfer learning",
    # LLM & Generative
    "llm", "large language model", "rag", "retrieval augmented",
    "gpt", "fine-tuning", "lora", "qlora", "peft", "langchain", "huggingface",
    # MLOps & Infra
    "mlops", "kubernetes", "docker", "airflow", "spark", "kafka",
    "aws", "gcp", "azure", "model serving", "inference optimization",
    # Evaluation
    "a/b testing", "a/b test", "experiment", "evaluation framework",
}

CONSULTING_COMPANIES = {
    "tcs", "tata consultancy", "infosys", "wipro", "accenture", "cognizant",
    "capgemini", "tech mahindra", "hexaware", "mphasis", "ltimindtree",
    "mindtree", "birlasoft", "niit tech", "mphasis",
}

NON_ML_TITLES = {
    "marketing manager", "hr manager", "graphic designer", "accountant",
    "civil engineer", "mechanical engineer", "content writer", "customer support",
    "sales executive", "operations manager", "finance manager", "supply chain",
    "project manager", "business analyst", "legal", "teacher", "nurse",
}

ML_POSITIVE_TITLES = {
    "machine learning", "ml engineer", "ai engineer", "data scientist",
    "nlp engineer", "search engineer", "ranking engineer", "research engineer",
    "applied scientist", "recommendation", "retrieval", "platform engineer",
    "backend engineer", "software engineer", "senior engineer", "principal engineer",
    "staff engineer", "data engineer", "mlops", "deep learning", "scientist",
}

ML_CAREER_KEYWORDS = {
    "machine learning", "ml", " ai ", "nlp", "data scientist", "research",
    "ranking", "search", "recommendation", "applied", "embedding", "retrieval",
    "neural", "model", "inference", "training",
}

PRODUCTION_KEYWORDS = {
    "deployed", "production", "shipped", "scale", "billion", "million users",
    "million requests", "ranking", "retrieval", "recommendation", "embedding",
    "vector", "a/b test", "experiment", "real users", "launched",
}

# ─────────────────────────────────────────────────────────────
# REFERENCE DATE
# ─────────────────────────────────────────────────────────────
TODAY = datetime(2025, 6, 25)


# ─────────────────────────────────────────────────────────────
# SCORING FUNCTIONS
# ─────────────────────────────────────────────────────────────

def score_experience(cand: dict) -> float:
    """Score YoE with sweet spot at 5-9 years per JD."""
    yoe = cand["profile"].get("years_of_experience", 0)
    if yoe < 3:
        return 15.0
    elif yoe < 4:
        return 35.0
    elif yoe < 5:
        return 60.0
    elif yoe <= 9:
        return 80.0 + (yoe - 5) * 4.0   # 80 → 96
    elif yoe <= 12:
        return 96.0 - (yoe - 9) * 3.0   # 96 → 87
    else:
        return max(72.0, 87.0 - (yoe - 12) * 2.0)


def score_seniority(cand: dict) -> float:
    """Score title seniority and ML/AI relevance."""
    title = (cand["profile"].get("current_title", "") + " " +
             cand["profile"].get("headline", "")).lower()
    score = 25.0
    if any(t in title for t in ML_POSITIVE_TITLES):
        score += 45.0
    if any(x in title for x in ["senior", "sr.", "lead", "principal", "staff", "architect"]):
        score += 20.0
    if any(t in title for t in NON_ML_TITLES):
        score -= 45.0
    if any(x in title for x in ["junior", "jr.", "intern", "trainee", "fresher"]):
        score -= 25.0
    return max(0.0, min(100.0, score))


def score_career(cand: dict) -> float:
    """Score career history for ML/AI product company experience."""
    career = cand.get("career_history", [])
    if not career:
        return 20.0
    score = 45.0
    consulting_count = 0
    ml_months = 0
    for job in career:
        comp_lower = job.get("company", "").lower()
        title_lower = job.get("title", "").lower()
        desc_lower = job.get("description", "").lower()
        dur = job.get("duration_months", 0)

        if any(dc in comp_lower for dc in CONSULTING_COMPANIES):
            consulting_count += 1

        if any(t in title_lower for t in ML_CAREER_KEYWORDS):
            ml_months += dur

        # Reward production ML descriptions
        prod_hits = sum(1 for kw in PRODUCTION_KEYWORDS if kw in desc_lower)
        if prod_hits >= 3:
            score += min(12.0, dur * 0.4)
        elif prod_hits >= 1:
            score += min(6.0, dur * 0.2)

    total = len(career)
    if total > 0:
        if consulting_count == total:
            score -= 30.0
        elif consulting_count / total > 0.6:
            score -= 15.0

    if ml_months >= 36:
        score += 22.0
    elif ml_months >= 18:
        score += 12.0
    elif ml_months >= 6:
        score += 4.0

    return max(0.0, min(100.0, score))


def score_skills(cand: dict) -> float:
    """Score skill match, proficiency, endorsements, and platform assessments."""
    skills = cand.get("skills", [])
    if not skills:
        return 8.0

    prof_map = {"beginner": 1, "intermediate": 2, "advanced": 3, "expert": 4}
    matched = 0
    prof_sum = 0
    end_sum = 0

    for s in skills:
        name = s.get("name", "").lower()
        prof = s.get("proficiency", "beginner").lower()
        ends = s.get("endorsements", 0)
        if any(cs in name for cs in CORE_ML_SKILLS):
            matched += 1
            prof_sum += prof_map.get(prof, 1)
            end_sum += min(ends, 30)

    assessments = cand.get("redrob_signals", {}).get("skill_assessment_scores", {})
    ml_scores = [
        v for k, v in assessments.items()
        if any(cs in k.lower() for cs in CORE_ML_SKILLS)
    ]
    avg_assess = (sum(ml_scores) / len(ml_scores)) if ml_scores else 62.0

    if matched == 0:
        return 8.0

    breadth  = min(matched / 7.0, 1.0) * 40.0   # 40 pts max
    depth    = (prof_sum / matched / 4.0) * 35.0  # 35 pts max
    social   = min(end_sum / 80.0, 1.0) * 10.0   # 10 pts max
    platform = avg_assess * 0.15                   # 15 pts max

    return min(100.0, breadth + depth + social + platform)


def score_signals(cand: dict) -> float:
    """Score Redrob platform signals for availability & engagement."""
    sig = cand.get("redrob_signals", {})
    total = 0.0

    # Open to work (20 pts)
    total += 20.0 if sig.get("open_to_work_flag", False) else 0.0

    # Notice period (20 pts)
    n = sig.get("notice_period_days", 90)
    if n <= 15:   total += 20.0
    elif n <= 30: total += 18.0
    elif n <= 60: total += 10.0
    elif n <= 90: total += 4.0

    # Recruiter response rate (15 pts)
    total += sig.get("recruiter_response_rate", 0.0) * 15.0

    # Interview completion rate (10 pts)
    total += sig.get("interview_completion_rate", 0.0) * 10.0

    # Response time (8 pts)
    rt = sig.get("avg_response_time_hours", 48)
    if rt <= 4:   total += 8.0
    elif rt <= 12: total += 6.0
    elif rt <= 24: total += 4.0
    else:          total += 1.0

    # GitHub activity (7 pts)
    gh = sig.get("github_activity_score", -1)
    total += (gh * 0.07) if gh >= 0 else 3.0

    # Last active recency (8 pts)
    try:
        dt = datetime.strptime(sig.get("last_active_date", "2020-01-01"), "%Y-%m-%d")
        days = (TODAY - dt).days
        if days <= 7:    total += 8.0
        elif days <= 30: total += 6.0
        elif days <= 90: total += 3.0
        elif days <= 180: total += 1.0
        # else 0 – clearly inactive
    except Exception:
        total += 3.0

    # Verification (7 pts)
    if sig.get("verified_email", False):   total += 2.5
    if sig.get("verified_phone", False):   total += 2.5
    if sig.get("linkedin_connected", False): total += 2.0

    # Profile completeness (5 pts)
    completeness = sig.get("profile_completeness_score", 70)
    total += completeness * 0.05

    return min(100.0, total)


def disqualifier_penalty(cand: dict) -> float:
    """Return 0-45 penalty for major JD red flags."""
    penalty = 0.0
    profile = cand["profile"]
    title = (profile.get("current_title", "") + " " +
             profile.get("headline", "")).lower()
    summary = profile.get("summary", "").lower()
    career  = cand.get("career_history", [])
    sig     = cand.get("redrob_signals", {})

    # Non-ML current title
    if any(t in title for t in NON_ML_TITLES):
        penalty += 22.0

    # Very low recruiter response (not reachable)
    rr = sig.get("recruiter_response_rate", 1.0)
    if rr < 0.10:
        penalty += 18.0
    elif rr < 0.20:
        penalty += 9.0

    # Very inactive on platform
    try:
        dt = datetime.strptime(sig.get("last_active_date", "2020-01-01"), "%Y-%m-%d")
        days = (TODAY - dt).days
        if days > 180:
            penalty += 12.0
        elif days > 90:
            penalty += 5.0
    except Exception:
        pass

    # Pure consulting background
    if career:
        consulting_cnt = sum(
            1 for j in career
            if any(dc in j.get("company", "").lower() for dc in CONSULTING_COMPANIES)
        )
        if consulting_cnt == len(career):
            penalty += 15.0

    # No production ML evidence
    has_prod_ml = any(kw in summary for kw in PRODUCTION_KEYWORDS)
    if not has_prod_ml:
        penalty += 5.0

    return min(penalty, 45.0)


# ─────────────────────────────────────────────────────────────
# TEXT BUILDER FOR TF-IDF
# ─────────────────────────────────────────────────────────────

def build_text(cand: dict) -> str:
    p = cand["profile"]
    parts = [
        p.get("headline", ""),
        p.get("summary", ""),
        p.get("current_title", ""),
        p.get("current_industry", ""),
    ]
    # Skills (repeat expert/advanced skills 2x for TF-IDF boost)
    for s in cand.get("skills", []):
        name = s.get("name", "")
        prof = s.get("proficiency", "")
        parts.append(name)
        if prof in ("expert", "advanced"):
            parts.append(name)
    # Career titles & descriptions
    for job in cand.get("career_history", []):
        parts.append(job.get("title", ""))
        parts.append(job.get("description", ""))
    # Certifications
    for cert in cand.get("certifications", []):
        parts.append(cert.get("name", ""))
    return " ".join(filter(None, parts))


# ─────────────────────────────────────────────────────────────
# REASONING GENERATOR
# ─────────────────────────────────────────────────────────────

def build_reasoning(cand: dict, tfidf_score: float) -> str:
    p = cand["profile"]
    sig = cand.get("redrob_signals", {})
    title = p.get("current_title", "Engineer")
    yoe   = p.get("years_of_experience", 0)
    co    = p.get("current_company", "N/A")
    open_w = sig.get("open_to_work_flag", False)
    notice = sig.get("notice_period_days", 90)
    gh     = sig.get("github_activity_score", -1)
    rr     = sig.get("recruiter_response_rate", 0.0)

    matched = [s.get("name","") for s in cand.get("skills",[])
               if any(cs in s.get("name","").lower() for cs in CORE_ML_SKILLS)][:4]
    skills_str = ", ".join(matched) if matched else "applied ML/AI"

    parts = [f"{title} with {yoe:.1f} yrs exp at {co}."]
    if matched:
        parts.append(f"Core AI/ML skills: {skills_str}.")
    if tfidf_score > 0.20:
        parts.append("Strong semantic alignment with JD (ranking, retrieval, embeddings).")
    if open_w and notice <= 30:
        parts.append(f"Immediately available (Open to Work, {notice}d notice).")
    elif notice <= 30:
        parts.append(f"Notice period {notice}d - ready to start soon.")
    if gh >= 60:
        parts.append(f"Active GitHub contributor (score {gh:.0f}/100).")
    if rr >= 0.70:
        parts.append(f"Excellent recruiter responsiveness ({rr:.0%}).")
    return " ".join(parts)


# ─────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────

def main():
    t0 = time.time()

    # ── Load candidates ───────────────────────────────────────
    print("[1/5] Loading candidates from JSONL...", flush=True)
    candidates = []
    with open(CANDIDATES_JSONL, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                candidates.append(json.loads(line))
    print(f"      Loaded {len(candidates):,} candidates in {time.time()-t0:.1f}s", flush=True)

    # ── Build text representations ───────────────────────────
    print("[2/5] Building text representations...", flush=True)
    cand_texts = [build_text(c) for c in candidates]

    # ── TF-IDF Vectorization ─────────────────────────────────
    print("[3/5] Computing TF-IDF cosine similarity...", flush=True)
    all_texts = [JOB_DESCRIPTION] + cand_texts
    vectorizer = TfidfVectorizer(
        max_features=20000,
        ngram_range=(1, 2),
        sublinear_tf=True,
        min_df=2,
    )
    tfidf_matrix = vectorizer.fit_transform(all_texts)
    jd_vec  = tfidf_matrix[0]
    cand_vecs = tfidf_matrix[1:]
    tfidf_scores = cosine_similarity(jd_vec, cand_vecs).flatten()   # shape (N,)
    print(f"      TF-IDF done. Max sim={tfidf_scores.max():.4f}  Min sim={tfidf_scores.min():.4f}", flush=True)

    # ── Compute all scores ────────────────────────────────────
    print("[4/5] Computing hybrid scores...", flush=True)
    results = []
    for i, cand in enumerate(candidates):
        sem  = float(tfidf_scores[i]) * 100.0   # 0-100

        exp_s  = score_experience(cand)
        sen_s  = score_seniority(cand)
        car_s  = score_career(cand)
        skl_s  = score_skills(cand)
        sig_s  = score_signals(cand)
        pen    = disqualifier_penalty(cand)

        # Weights tuned to JD emphasis
        raw = (
            sem    * 0.35 +   # TF-IDF semantic relevance
            skl_s  * 0.25 +   # Technical skill match
            car_s  * 0.15 +   # Career trajectory & product experience
            ((exp_s + sen_s) / 2.0) * 0.15 +  # Experience level
            sig_s  * 0.10     # Platform availability signals
        )
        final = max(0.0, raw - pen)

        results.append({
            "candidate_id": cand["candidate_id"],
            "raw_score": final,
            "sem_score": float(tfidf_scores[i]),
            "idx": i,
        })

    # ── Sort by score, then candidate_id (tie-break) ──────────
    results.sort(key=lambda x: (-x["raw_score"], x["candidate_id"]))

    # ── Assign ranks ──────────────────────────────────────────
    for rank, r in enumerate(results, 1):
        r["rank"] = rank

    top100 = results[:100]

    # ── Normalize scores to [0.2000, 1.0000] ──────────────────
    hi = top100[0]["raw_score"]
    lo = top100[-1]["raw_score"]
    for r in top100:
        if hi > lo:
            r["norm_score"] = round(0.2 + (r["raw_score"] - lo) / (hi - lo) * 0.8, 4)
        else:
            r["norm_score"] = round(1.0 - (r["rank"] - 1) * (0.8 / 99), 4)

    # Guarantee non-increasing (challenge rule)
    for i in range(1, len(top100)):
        if top100[i]["norm_score"] > top100[i-1]["norm_score"]:
            top100[i]["norm_score"] = top100[i-1]["norm_score"]

    # ── Generate reasoning ────────────────────────────────────
    for r in top100:
        cand = candidates[r["idx"]]
        r["reasoning"] = build_reasoning(cand, r["sem_score"])

    # ── Write CSV ─────────────────────────────────────────────
    print(f"[5/5] Writing {OUTPUT_CSV}...", flush=True)
    with open(OUTPUT_CSV, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["candidate_id", "rank", "score", "reasoning"])
        for r in top100:
            writer.writerow([r["candidate_id"], r["rank"], r["norm_score"], r["reasoning"]])

    elapsed = time.time() - t0
    print(f"\nDONE! Processed {len(candidates):,} candidates in {elapsed:.1f}s", flush=True)
    print(f"Output written: {OUTPUT_CSV}", flush=True)
    print("\nTop 15 Candidates:", flush=True)
    for r in top100[:15]:
        cand = candidates[r["idx"]]
        title = cand["profile"].get("current_title", "N/A")
        yoe   = cand["profile"].get("years_of_experience", 0)
        print(f"  #{r['rank']:3d}  {r['candidate_id']}  score={r['norm_score']:.4f}  {title}  {yoe:.1f}yrs", flush=True)


if __name__ == "__main__":
    main()
