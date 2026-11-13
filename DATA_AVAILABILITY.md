# Data availability statement - replacement text

## Why this file exists

The current manuscript (Section "Data availability", paragraph [163] of the v6.6
revised version) ends with:

> *The experiment code and the raw result files are available from the author on reasonable request.*

That sentence is incompatible with publishing the code on GitHub. The public repository
has already been created; the replacement text below is ready to paste as-is.

## Replacement text (paste into the manuscript, English)

```
The CWRU and Paderborn (PU) bearing datasets analysed in this study are publicly
available from the Case Western Reserve University Bearing Data Center
(https://engineering.case.edu/bearingdatacenter) and the Paderborn University
Bearing Data Center (https://groups.uni-paderborn.de/kat/BearingDataCenter/),
respectively. All per-repetition accuracies, the per-epoch training-dynamics records
and the statistical test outputs generated in this study are reported in full in the
Supplementary Material associated with this article. The complete experiment code,
the raw per-repetition result files, the figure-generation scripts and a step-by-step
reproduction protocol are publicly available at
https://github.com/jia-zhibo/pseudolabel-bearing-empirical-study under the MIT licence.
```

**Repository (already created, public)**

`https://github.com/jia-zhibo/pseudolabel-bearing-empirical-study`

The replacement text above is ready to paste as-is. Once the Zenodo GitHub integration
is enabled and a release has been archived, append: *"A permanent versioned archive is
available at `<VERSION_DOI>` (concept DOI: `<CONCEPT_DOI>`)."*

## Zenodo integration

Zenodo reads `.zenodo.json` from the repository root when archiving a GitHub release.
Both `.zenodo.json` and `CITATION.cff` are present: Zenodo uses **only** `.zenodo.json`,
while GitHub uses `CITATION.cff` for its citation panel. Zenodo archives releases created
**after** the repository is enabled; earlier releases are not ingested retroactively, so
a fresh release must be published once the integration is active.

## Two more places to update

1. **Appendix A** ("Supplementary data ... can be found in the online version"): add
   the same repository URL so readers can find the machine-readable records, e.g.
   *"The machine-readable per-repetition records and the statistical test outputs are
   additionally deposited at https://github.com/jia-zhibo/pseudolabel-bearing-empirical-study
   (see results/README.md)."*
2. **README / repository description**: state the exact manuscript title and version
   (v6.6 revised, empirical-study positioning) so the repository is citable and not
   confused with the v7 method-positioning manuscript of the same study.

## What is deliberately NOT in the repository

| Excluded | Reason |
|---|---|
| `data/cwru_real/` (134 MB), `data/pu_data/` (5.7 GB) | Third-party datasets, too large for GitHub (100 MB/file limit); download via the official sites and the shipped helpers. |
| `checkpoints/` (88 MB) | Trained weights are not needed to reproduce any table; regeneration is cheap. |
| `revision_cwru_visualization.json` (7.5 MB), `revision_pu_visualization.json` (44 MB) | Intermediate t-SNE / feature dumps; not cited in the paper. |
| Manuscript generators `_gen_*.py`, cover-letter generator, Word patches | Contain personal contact details (e-mail / phone) and are not part of the experimental method. |
| `topjournal_*.json`, `cwru_real_qgcda_*.json`, `models/diffusion/*` | Belong to the **other** manuscript (tower-clock / conditional-diffusion RESS paper). Mixing them in would create a data-provenance error. The conditional-diffusion implementation is therefore deliberately not shipped; only the detector / domain-adaptation modules used by this empirical study are included. |
| `legacy/_auth_stats.py` | Deprecated n=5 statistics; kept only under `legacy/` with an explicit deprecation note. |
