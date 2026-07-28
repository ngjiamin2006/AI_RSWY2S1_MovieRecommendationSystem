# Spec Compliance Audit

Running record comparing the current code against the assignment requirements
(`ASSIGNMENT.md`, Topic 3 — Recommender System). Like `AI_DISCLOSURE_LOG.md`,
this is a **versioned log** — append a new dated entry each time the code
changes meaningfully, do not overwrite past entries. This lets you track
whether refinements actually closed the gaps identified earlier.

---

## Version 1 — 2026-07-21 — Initial scaffold

**Code state this entry describes:** the initial project scaffold — `data_loader.py`,
`algorithms/content_based.py`, `algorithms/collaborative_filtering.py`,
`algorithms/hybrid.py`, `evaluation.py`, `app.py`. No changes made since
creation; this is the very first working version. See `AI_DISCLOSURE_LOG.md`
Entry 1 for provenance of this same code.

### 1. Topic 3 (Recommender System) requirements vs. code

| Requirement (from spec) | Status | Notes |
|---|---|---|
| a. Real-life scenario (streaming/movies) | Met | Movie recommendation, Netflix-style onboarding |
| b. Background study (scenario, type, functionality/benefits) | Not code | Documentation-only task, not yet started |
| c. Each member implements a *different* method (CF / content-based / hybrid) | Met | All three exist, run correctly, produce distinct outputs |
| d. Evaluate via precision/recall/F1 **or** MSE/RMSE **or** satisfaction survey | Partial | precision/recall/F1@10 done for all 3; RMSE done for CF only (correct, since CB/hybrid don't predict ratings); satisfaction survey not implemented |

### 2. Prototype rubric vs. code

| Criteria (weight) | Current state | Gap |
|---|---|---|
| UI/Output (10%) | Functional but plain — text list, no posters/images, no search | To reach "Good" tier: visual polish (poster thumbnails, better layout, genre filter chips) |
| Programming (20%) | Weakest area right now | No exception handling anywhere (no try/except on file loads, no guard if a CSV is malformed/missing). Validation is implicit (e.g. button disabled if no genre picked) rather than explicit with error messages. Rubric explicitly grades "thorough validations, business rules" |
| Degree of completion (10%) | Core MVP complete | Missing: satisfaction survey mechanism, session persistence (likes reset on refresh) |
| System implementation (10%) | Matches what was proposed | Onboarding → algorithm tabs → like-button loop → evaluation tab, as discussed |
| Presentation (10%) | Not code | Depends on each member understanding their own file |

### 3. What's already good (working as intended, don't "fix")

- **Cold-start handling** — content-based and hybrid gracefully use onboarding genres when there are no likes yet; CF correctly has no cold-start ability and falls back to a popularity list. This is a real, citable finding for the report, not a bug.
- **Evaluation pipeline** — real train/test split, real numbers: CF precision@10 ≈ 0.108, hybrid ≈ 0.094, content-based ≈ 0.004, RMSE ≈ 0.89 (collaborative filtering). Legitimate results for Results & Discussion.
- **Algorithm separation** — `recommend(...)` signatures are consistent across all three files, so each member can independently improve their own algorithm without breaking the others.

### 4. Gaps needing more research/refinement (priority order)

1. **Exception handling & validation** (Programming, 20% — highest-weighted gap). No `try/except` anywhere; no handling for missing files, empty datasets, or bad input beyond UI-level disabling.
2. **Content-based precision is very weak (0.004@10).** Genre-only overlap is a known-weak signal. Fix: bring in `tags.csv` (downloaded, currently unused) via TF-IDF for richer features.
3. **Hybrid `alpha` is a manually-set slider value, not tuned.** Should grid-search alpha against precision@k using `evaluation.py` and report the best value.
4. **No satisfaction-survey evaluation path**, though the spec allows it as a third acceptable metric. Not required (precision/recall/F1 + RMSE already satisfy the requirement), but a 👍/👎 in-app feedback widget would strengthen the live demo.
5. **Single random train/test split (seed=42).** Statistically thin; repeating across multiple seeds and reporting mean±std would strengthen the methodology section.
6. **`tags.csv` and `links.csv` downloaded but unused.** Tags feed into gap #2; `links.csv` (IMDb/TMDb IDs) could support poster images for the UI gap.

### 5. Explicitly not code, still outstanding

Background study, related work/literature comparison, methodology write-up, and the satisfaction survey (if used) — none of this lives in the codebase; still needed for Documentation (40%).

---

## Version 2 — (add next entry here after the next round of refinement)
