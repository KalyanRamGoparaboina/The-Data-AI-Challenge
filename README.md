# The Data & AI Challenge — Intelligent Candidate Ranking System

> **India Runs x Redrob | Intelligent Candidate Discovery & Ranking Challenge**

A production-grade AI system that ranks 100,000 candidates for a **Senior ML Engineer – AI Platform** role, the way a great recruiter would — not by matching keywords, but by genuinely understanding who fits.

---

## Problem Statement

Recruiters go through hundreds of profiles and still miss the right person — not because talent isn't there, but because keyword filters can't see what actually matters. This system:

- **Reads the job description and understands what the role actually needs**
- **Looks at the full picture** — career history, skills, behavioral signals, platform activity
- **Delivers a shortlist a recruiter can trust**

---

## Solution Architecture

### Hybrid 5-Layer Scoring Model

| Layer | Weight | What It Measures |
|---|---|---|
| **Semantic Similarity** (TF-IDF bigrams, sublinear TF) | 35% | Contextual alignment of candidate's career narrative with the JD |
| **Skill Match** (taxonomy + proficiency + endorsements + assessments) | 25% | Hands-on competence in embeddings, vector DBs, ranking, NLP |
| **Career Trajectory** (product co. ML experience + production keywords) | 15% | Real production ML work at product companies (not pure consulting) |
| **Experience & Seniority** (YoE sweet spot 5–9yrs + title seniority) | 15% | Correct seniority level per JD requirements |
| **Behavioral Signals** (Redrob platform data) | 10% | Genuine availability — open to work, response rate, recency, GitHub |

### Disqualifier Penalties (0–45 pts deducted)

The system explicitly implements the JD's anti-patterns:

| Red Flag | Penalty |
|---|---|
| Non-ML current title (Marketing Manager, Accountant, etc.) | −22 pts |
| Recruiter response rate < 10% (not reachable) | −18 pts |
| Platform inactive > 180 days | −12 pts |
| Entire career at consulting firms (TCS/Infosys/Wipro/Accenture/etc.) | −15 pts |
| No evidence of production ML deployment | −5 pts |

---

## Files

| File | Description |
|---|---|
| `rank_candidates.py` | Main pipeline — reads 100K candidates, scores, ranks, outputs CSV |
| `team_antigravity.csv` | **Final submission** — top 100 ranked candidates |
| `submission_metadata.yaml` | Full approach documentation and metadata |

---

## Results

- **100,000 candidates** processed in **35.1 seconds**
- **Validation: PASSED** (official `validate_submission.py` — all challenge rules met)
- Score range: **1.0000** (rank 1) → **0.2000** (rank 100)

### Top 10 Shortlisted Candidates

| Rank | Candidate ID | Title | YoE | Company | Score |
|---|---|---|---|---|---|
| 1 | CAND_0039754 | Senior Applied Scientist | 16.2 | Meta | 1.0000 |
| 2 | CAND_0018499 | Senior ML Engineer | 7.2 | Zomato | 0.9736 |
| 3 | CAND_0055905 | Senior ML Engineer | 8.1 | Flipkart | 0.9151 |
| 4 | CAND_0046064 | Senior NLP Engineer | 8.9 | Salesforce | 0.9099 |
| 5 | CAND_0081846 | Lead AI Engineer | 6.7 | Razorpay | 0.9020 |
| 6 | CAND_0011687 | Senior NLP Engineer | 7.8 | Niramai | 0.8826 |
| 7 | CAND_0071974 | Senior AI Engineer | 7.8 | Netflix | 0.8332 |
| 8 | CAND_0002025 | Senior AI Engineer | 5.9 | Apple | 0.8169 |
| 9 | CAND_0088025 | Staff ML Engineer | 8.6 | Yellow.ai | 0.8015 |
| 10 | CAND_0008425 | Senior NLP Engineer | 7.8 | Ola | 0.7553 |

---

## Key Design Decisions

### Avoiding the "Keyword Trap"
The JD explicitly warns: *"The right answer is not 'find candidates whose skills section contains the most AI keywords.'"*

This system avoids that trap:
- A **Marketing Manager** with 8 AI skills listed → scores LOW (non-ML title penalty)
- An **ML Engineer** with 3 AI skills but production retrieval system experience → scores HIGH

### Behavioral Signal Integration
Per the JD: *"A perfect-on-paper candidate who hasn't logged in for 6 months with a 5% recruiter response rate is not actually available."*

The signals layer penalizes inactive/unresponsive candidates and boosts:
- `open_to_work_flag = True`
- `notice_period_days ≤ 30`
- `recruiter_response_rate ≥ 0.70`
- `last_active_date` within 30 days
- High `github_activity_score`

---

## How to Run

```bash
# Install dependencies
pip install scikit-learn numpy

# Run the ranking pipeline
python rank_candidates.py

# Validate output
python validate_submission.py team_antigravity.csv
```

**Requirements:** Python 3.9+, scikit-learn, numpy

The script expects `candidates.jsonl` in:
```
extracted_data/[PUB] India_runs_data_and_ai_challenge/India_runs_data_and_ai_challenge/candidates.jsonl
```

---

## Submission Format

```csv
candidate_id,rank,score,reasoning
CAND_0039754,1,1.0,"Senior Applied Scientist with 16.2 yrs exp at Meta..."
...
CAND_0052335,100,0.2,"Applied ML Engineer with 6.4 yrs exp at Aganitha..."
```

---

## Team
**Antigravity** | India Runs x Redrob AI Challenge 2025
