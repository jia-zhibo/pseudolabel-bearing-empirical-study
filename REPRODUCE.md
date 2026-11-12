# Reproduction protocol

## 0. What can be reproduced, and what cannot

| Item | Status |
|---|---|
| Tables 1-4 (all means, SDs, deltas, p-values, 95% CI) | **Fully reproducible** from the shipped JSON with `python compute_statistics.py`. No scipy/numpy needed. |
| FastCNN Tables 2-4 and Figures 3-5, per-epoch curves | **Bit-exact** under the pinned protocol. |
| BiLSTM Table 1 (n=20) | Reproducible under the pinned protocol; this is the only execution used in the paper. |
| BiLSTM n=5 "first execution" (+4.01 pp, p=0.0352) | **Not reproducible by design.** Its baseline branch was unseeded, so it inherited the process RNG state. Retained only as the documented negative control in Section 7. |

## 1. Environment

```bash
python -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Authoritative runs used Python 3.10, `torch 2.14.1+cpu`, CPU only, **10 threads**.
The thread pin must happen **before** numpy/torch are imported:

```bash
export OMP_NUM_THREADS=10 MKL_NUM_THREADS=10 NUMEXPR_NUM_THREADS=10 \
       OPENBLAS_NUM_THREADS=10 VECLIB_MAXIMUM_THREADS=10
```

On Windows (cmd):

```bat
set OMP_NUM_THREADS=10
set MKL_NUM_THREADS=10
set NUMEXPR_NUM_THREADS=10
set OPENBLAS_NUM_THREADS=10
set VECLIB_MAXIMUM_THREADS=10
```

Every script also calls `torch.set_num_threads(10)` explicitly. Changing the thread
count changes the numerical trajectory on CPU, so keep it fixed.

## 2. Datasets

Not redistributed. Download from the official sites:

```bash
python data/download_cwru_real.py      # -> data/cwru_real/  (40 .mat files, ~134 MB)
python download_pu_data.py             # -> data/pu_data/    (~5.7 GB)
python data/verify_cwru_real.py        # sanity check
```

Exact settings: CWRU 10 classes, 12 kHz, 0 hp -> 3 hp, window 512, stride 128,
50 labelled source samples per class. PU 3 classes, 64 kHz, 1500 rpm -> 900 rpm,
window 512, stride 1024. BiLSTM 80 epochs; FastCNN 40 epochs.

## 3. Fast check (seconds, no data needed)

```bash
python compute_statistics.py
python _final_audit_v7.py        # optional; needs python-docx
```

`compute_statistics.py` recomputes Tables 1-4 and the Section 7 negative control
directly from `results/*.json`. Expected Table 1 baseline / CORAL / MMD / pseudo:
79.91 / 83.25 / 80.55 / 80.80 %, with pseudo delta +0.89 pp, p = 0.3044,
95% CI -2.69 to +4.48 pp.

## 4. Full re-runs (hours of CPU time)

Run from the repository root. Each script writes its output into `results/`.

```bash
# Table 1 - BiLSTM, n=20, ~9 h CPU on the hardware used in the study
python _auth_bilstm_n20_v2.py       # -> results/authoritative_bilstm_n50_n20.json

# Table 2 - FastCNN, CWRU + PU, n=10
python _auth_fastcnn_main.py        # -> results/authoritative_fastcnn_main.json

# Tables 3 & 4 - ablation, n=3, two loss configurations
python _auth_fastcnn_ablation.py    # -> results/authoritative_fastcnn_ablation.json

# Section 6.1 loss-matched comparison
python _run_matched_pseudo.py       # -> results/revision_pseudo_matched_loss.json

# Section 6.4 training dynamics
python _run_training_curves.py      # -> results/revision_training_curves.json
```

Supporting modules used by the scripts above (do not run directly):
`_run_revision_experiments.py` (split / FastCNN / DANN / evaluation),
`_run_curves_full.py` (identical per-epoch instrumentation for every branch),
`_run_full_instrumented.py` (instrumented full re-run), `data/`, `models/`, `utils/`.

## 5. Statistical convention

* One-sided paired t-test, alternative fixed **a priori** as `H1: mu(method) > mu(baseline)`.
* Significance marker: `* <=> (p < 0.05 AND Cohen's d_z > 0)`.
* SD with `ddof=1`; 95% CI from the t distribution; TOST equivalence margin +/-5 pp;
  power from the noncentral t distribution.
* No comparison was tested in more than one direction; no post-hoc direction changes.

## 6. Known caveats

* `_auth_stats.py` in `legacy/` is the **n=5-era** script and prints `TABLE 1 ... 5 repeats`.
  It does **not** match the final manuscript. Use `compute_statistics.py`.
* `revision_full_instrumented.json` and `revision_ablation_instrumented.json` are
  bit-identical to the two `authoritative_fastcnn_*.json` files; they are not shipped.
* The manuscript states "all 3545 per-epoch curve points". The shipped authoritative
  JSONs contain 4286 points per metric (FastCNN main), 3217 (ablation) and 597
  (training curves); the exact 3545 figure appears to come from an earlier revision of
  the figure set. Verify this count against the final figure scripts before submission.
