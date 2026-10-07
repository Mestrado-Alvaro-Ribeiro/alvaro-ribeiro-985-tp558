# HPO Search Comparison Notebook Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a runnable Seminar 3 notebook that compares grid search, random search, and TPE on one deterministic simulated objective.

**Architecture:** A single notebook defines a synthetic validation-loss function, executes each search strategy, normalizes their trial histories into pandas data frames, and displays a results table and convergence figure. Grid search is implemented with `itertools.product`; Optuna supplies the random and TPE samplers.

**Tech Stack:** Python, NumPy, pandas, Matplotlib, Optuna, Jupyter Notebook.

**Spec:** `docs/superpowers/specs/2026-09-23-hpo-search-comparison-design.md`

## Global Constraints

- Use only `numpy`, `pandas`, `matplotlib`, and `optuna` beyond the Python standard library.
- Use a fixed seed and no external datasets, GPU, or model training.
- Clearly state that this is a didactic illustration, not a reproduction of BOHB or Auto-PyTorch.

## Review Focus

- Missing Optuna should produce a direct installation instruction instead of an opaque import failure.
- The grid must have a comparable, documented trial budget rather than silently using a much larger search.
- Repeated notebook execution must recreate all state and yield the same summary.
- All methods must optimize the identical objective and parameter ranges.
- The convergence plot must use the running minimum, since lower validation loss is better.

---

### Task 1: Build the executable comparison notebook

**Files:**
- Create: `seminar3/notebooks/hpo_search_comparison.ipynb`

**Interfaces:**
- Produces: `objective(learning_rate, dropout, width) -> float` and a results data frame with `method`, `trial`, `loss`, and hyperparameter columns.

- [ ] **Step 1: Create a notebook validation check**

Run:

```bash
jupyter nbconvert --to notebook --execute seminar3/notebooks/hpo_search_comparison.ipynb --output /tmp/hpo_search_comparison.executed.ipynb
```

Expected: it initially fails because the notebook does not exist.

- [ ] **Step 2: Create the minimal notebook**

Include cells that:

```python
SEED = 42
TRIALS = 60

def objective(learning_rate, dropout, width):
    log_lr = np.log10(learning_rate)
    return (log_lr + 3) ** 2 + (dropout - 0.2) ** 2 + ((width - 128) / 256) ** 2
```

Run Grid Search using a 5 × 3 × 4 grid and Optuna studies using
`RandomSampler(seed=SEED)` and `TPESampler(seed=SEED)`. Consolidate trial
records, display each method's best loss/configuration, and plot grouped
running-minimum loss curves. Add Portuguese markdown cells with the scope and
BOHB/Auto-PyTorch distinction.

- [ ] **Step 3: Run the notebook end-to-end**

Run:

```bash
jupyter nbconvert --to notebook --execute seminar3/notebooks/hpo_search_comparison.ipynb --output /tmp/hpo_search_comparison.executed.ipynb
```

Expected: success; the executed copy contains the summary table and convergence plot.

- [ ] **Step 4: Verify the tracked notebook is structurally valid**

Run:

```bash
python -m json.tool seminar3/notebooks/hpo_search_comparison.ipynb > /dev/null
```

Expected: success.

- [ ] **Step 5: Commit**

```bash
git add seminar3/notebooks/hpo_search_comparison.ipynb
git commit -m "feat(seminar3): add HPO comparison notebook"
```
