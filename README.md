# When Does Pseudo-Label Self-Training Help Few-Shot Cross-Domain Bearing Fault Diagnosis?

**Public repository:** https://github.com/jia-zhibo/pseudolabel-bearing-empirical-study

Reproduction code and raw results for a **controlled, reproducible empirical study** on
few-shot cross-domain bearing fault diagnosis. The study compares an unadapted baseline,
CORAL, MMD, DANN and confidence-guided pseudo-label self-training (plus a loss-matched
pseudo-label variant) on CWRU (cross-load, 0 hp -> 3 hp) and Paderborn / PU (cross-speed,
1500 rpm -> 900 rpm) under an explicit, thread-pinned and fully seeded protocol.

> This repository accompanies two manuscript versions of the **same** study:
> * *When Does Pseudo-Label Self-Training Help Few-Shot Cross-Domain Bearing Fault
>   Diagnosis? A Controlled and Reproducible Empirical Study* (empirical-study positioning,
>   v6.6 revised) -- the version whose code/data paths are documented here.
> * *Confidence-Guided Pseudo-Label Self-Training for Few-Shot Cross-Domain Bearing Fault
>   Diagnosis* (method positioning, v7).
> Both versions consume the **same** result files listed below.

## Key finding

Under the pinned protocol the pseudo-label branch does **not** yield a statistically
significant gain over the unadapted baseline (BiLSTM n=20: +0.89 pp, one-sided p = 0.3044,
Cohen's d = 0.116; 95% CI -2.69 to +4.48 pp; TOST +/-5 pp equivalence p = 0.0135).
An earlier n=5 execution in which the baseline branch was left unseeded had reported a
spurious +4.01 pp gain (p = 0.0352); that execution is retained in `results/` solely as
the documented negative control (Section 7 of the manuscript) and **no value from it is
used in any table**.

## Repository layout

```
.
├── _auth_bilstm_n20_v2.py        # Table 1: BiLSTM, n=20, CWRU 0->3 hp  (AUTHORITATIVE)
├── _auth_bilstm.py               # n=5 seeded control run
├── _auth_fastcnn_main.py         # Table 2: FastCNN, n=10, CWRU + PU
├── _auth_fastcnn_ablation.py     # Tables 3 & 4: ablation, n=3, two loss configs
├── _run_revision_experiments.py  # shared split / FastCNN / DANN / evaluation
├── _run_curves_full.py           # shared per-epoch instrumentation
├── _run_matched_pseudo.py        # loss-matched pseudo-label comparison
├── _run_full_instrumented.py     # instrumented full re-run
├── _run_training_curves.py       # Section 6.4 training dynamics
├── compute_statistics.py         # recompute every table from the raw JSON (see below)
├── _final_audit_v7.py            # 304-point consistency audit
├── download_pu_data.py           # PU download helper
├── legacy/_auth_stats.py         # DEPRECATED: n=5-era statistics, kept for provenance
├── data/                         # dataset loaders + CWRU download helper (NO raw data)
├── models/                       # BiLSTM-Attention, FastCNN, domain-adaptation losses
├── utils/                        # seeding / training helpers, quality gate
├── results/                      # raw per-repetition results (see results/README.md)
├── figures/                      # the 9 manuscript figures + 2 roadmap figures
├── REPRODUCE.md                  # step-by-step reproduction protocol
├── DATA_AVAILABILITY.md          # replacement text for the manuscript statement
├── requirements.txt
├── .gitignore
└── LICENSE (MIT)
```

## Numerical environment (material for reproducibility)

All authoritative runs were executed with:

* CPU threads pinned to **10**: `OMP_NUM_THREADS`, `MKL_NUM_THREADS`,
  `NUMEXPR_NUM_THREADS`, `OPENBLAS_NUM_THREADS`, `VECLIB_MAXIMUM_THREADS` set **before**
  importing numpy/torch, plus `torch.set_num_threads(10)`.
* Explicit seeds: BiLSTM `n=20` -> `[0, 1000, ..., 19000]`; FastCNN -> `[42, 1042, ..., 9042]`;
  ablation -> `[42, 1042, 2042]`.
* CWRU: 10 classes, 12 kHz, window 512, stride 128, 50 labelled source samples per class,
  0 hp -> 3 hp. PU: 3 classes, 64 kHz, window 512, stride 1024, 1500 rpm -> 900 rpm.
* BiLSTM 80 epochs; FastCNN 40 epochs; PyTorch `2.14.1+cpu` (recorded in the JSON meta).

Under this protocol the FastCNN results of Tables 2-4 and Figures 3-5 reproduce
bit-exactly. The BiLSTM experiment does **not** (see the finding above).

## Datasets (NOT included in this repository)

The raw datasets are **not** redistributed here; obtain them from the official sources and
place them under `data/`:

| Dataset | Official source | Local directory after download |
|---|---|---|
| CWRU | https://engineering.case.edu/bearingdatacenter | `data/cwru_real/` (40 `.mat` files, ~134 MB) |
| PU / Paderborn | https://groups.uni-paderborn.de/kat/BearingDataCenter/ | `data/pu_data/` (~5.7 GB) |

Helpers: `python data/download_cwru_real.py` and `python download_pu_data.py`.
Both datasets are third-party; please cite the original providers and respect their terms.

## Quick start

```bash
pip install -r requirements.txt
python compute_statistics.py        # recompute Tables 1-4 from the shipped JSON
python _final_audit_v7.py           # optional: full consistency audit
```

Full re-runs (hours of CPU time) are documented in `REPRODUCE.md`.

## Result files -> manuscript tables

| File | Used in |
|---|---|
| `results/authoritative_bilstm_n50_n20.json` | Table 1 (FINAL, n=20) |
| `results/cwru_n50_da_comparison.json` | Section 7, first (unseeded) execution, negative control |
| `results/authoritative_bilstm_n50.json` | n=5 seeded intermediate; audit lineage |
| `results/authoritative_fastcnn_main.json` | Table 2; Figures 3-5 per-epoch curves |
| `results/authoritative_fastcnn_ablation.json` | Tables 3 & 4; ablation curves |
| `results/revision_pseudo_matched_loss.json` | Section 6.1 loss-matched comparison |
| `results/revision_training_curves.json` | Section 6.4 training dynamics |
| `results/revision_{cwru,pu}_10repeats.json`, `revision_ablation_3repeats.json` | cross-checks / Appendix |

## Statistics convention

One-sided paired t-test with the pre-specified alternative `H1: mu(method) > mu(baseline)`;
`* <=> (p < 0.05 AND Cohen's d > 0)`; all standard deviations use `ddof=1`;
Holm-Bonferroni, 95% CI, TOST equivalence and noncentral-t power as reported in the paper.

## License & funding

MIT (see `LICENSE`). This work was supported by the Shandong Provincial Key Research and
Development Program (Science and Technology-based SME Innovation Capacity Improvement
Project), Grant No. 2025TSGCCZZB0555.
