# 3D Modeling of the Lamina Cribrosa in OCT Data

Deep-learning pipeline that segments lamina cribrosa (LC) pores in 2D optical
coherence tomography (OCT) slices, reconstructs them into a 3D volume, and
computes quantitative morphometric biomarkers — with an interactive dashboard
and an automated, radiology-style PDF/PNG report.

> **Research output only — not a diagnostic device.** All measures are
> uncalibrated against normative clinical data and depend on the chosen
> segmentation threshold. See [Limitations](#limitations).

---

## What it does

The lamina cribrosa is a porous connective-tissue structure at the optic nerve
head; its pore morphology is of interest in glaucoma research. This project
automates what is otherwise a manual, slow analysis:

1. **2D segmentation** — an Attention U-Net (PyTorch) segments pores in each OCT
   slice (256×256), trained with a combined Dice + weighted-BCE loss to handle
   the heavy class imbalance (pores are sparse).
2. **3D reconstruction** — per-slice masks are stacked into a `(Z, H, W)` volume.
3. **Biomarkers** — within an LC region of interest, the pipeline computes:
   - **CTVF** (connective tissue volume fraction) and **porosity** (dimensionless)
   - **3D pore count** (26-connectivity)
   - **pore-size distribution** (mean / median / max, in voxels)
4. **Interactive dashboard** — pick patient/eye/orientation, sweep the
   segmentation threshold and minimum pore size live, recolour the 3D view by
   depth / pore size / pore ID, and scroll the 2D slice viewer
   (input / ground-truth / prediction).
5. **Report** — a one-click A4 report (PNG) summarising the case, with the 3D
   views, the 2D triptych, the biomarkers, and cohort-relative ranking.

---

## Methodology highlights

- **Patient-based train/val split.** Whole patients are assigned to train or
  validation (not individual images), preventing leakage of adjacent slices
  from the same eye into both sets.
- **Attention U-Net** over a plain U-Net: attention gates help the decoder focus
  on the small, sparse pore structures.
- **ROI-based CTVF.** CTVF is computed inside a per-slice LC region of interest
  (derived by morphological closing of the pore mask), not over the whole image
  bounding box — the bounding-box version is dominated by empty background and
  is not meaningful.
- **Cohort-relative reporting.** Because the data are image-only (no physical
  voxel spacing), absolute voxel volumes are not clinically comparable. The
  report therefore ranks each eye *within the analysed dataset* rather than
  against external norms.

---

## Results (reference run)

On the reference dataset (~2,300 image–mask pairs, 10 patients, 8/2 split):

| Metric            | Value     |
|-------------------|-----------|
| Best validation IoU  | ~0.52  |
| Validation Dice      | ~0.67  |

Qualitative prediction-vs-ground-truth panels show close agreement on pore
location and density. (See the report figures.)

---

## Repository layout

```
.
├── README.md
├── LICENSE
├── .gitignore
├── requirements.txt
├── notebooks/
│   └── lamina_cribrosa_OCT.ipynb     # full end-to-end pipeline (Colab-ready)
├── docs/
│   ├── ABOUT.md                      # extended project description
│   └── USAGE.md                      # step-by-step run guide + cell order
└── src/                              # optional standalone module versions
    ├── dataset.py
    ├── model.py
    └── losses.py
```

> The dataset and trained model weights are **not** included — see
> [Data & weights](#data--weights).

---

## Quickstart (Google Colab)

1. Open `notebooks/lamina_cribrosa_OCT.ipynb` in
   [Google Colab](https://colab.research.google.com/) (*File → Open notebook →
   GitHub* tab, or upload it).
2. **Runtime → Change runtime type → GPU** (a T4 is sufficient).
3. **Run the kaleido install cell first**, then *Runtime → Restart session*
   (this is required for the report's 3D static export; see
   [docs/USAGE.md](docs/USAGE.md)).
4. Mount Drive, edit the path constants to point at your data, and run the cells
   top to bottom.
5. Use the dashboard to explore a patient, then build the report.

Full run instructions and the exact cell order are in
[docs/USAGE.md](docs/USAGE.md).

---

## Data & weights

This repository contains **code only**. The OCT image dataset and the trained
model checkpoint are intentionally excluded because:

- medical imaging data should not be redistributed without appropriate consent
  and data-use agreements;
- GitHub is not suited to large binary files (100 MB/file limit).

To run the pipeline you will need your own OCT slice dataset organised as
`patient / eye / orientation / *.png` with matching label masks, and you will
train your own weights (or place an existing checkpoint at the path configured
in the notebook).

---

## Intended use

This is a **research and educational tool** for exploring lamina cribrosa pore
morphology from OCT images. It is appropriate for:

- methodological experimentation with segmentation + 3D morphometry pipelines;
- *relative* comparison of pore structure across eyes within a single, uniformly
  processed dataset;
- generating hypotheses and visualisations for further, properly validated study.

It is **not** intended for, and must not be used for, clinical diagnosis,
treatment decisions, screening, or any patient-facing purpose. It is not a
medical device and has not been validated, calibrated, or regulatory-cleared.

## Limitations

Understanding these is essential to interpreting any output correctly.

1. **Image-only data → no physical units.**
   Voxel spacing (µm per pixel laterally, µm between slices) is a property of the
   scanner acquisition and is **not recoverable from exported image files**.
   Consequently, absolute pore *volumes* are reported in **voxels**, which are
   comparable **only within a single dataset processed identically** — never
   against published values, other scanners, or clinical references. Porosity and
   CTVF (dimensionless ratios) and pore *count* are more robust because they do
   not depend on voxel size.

2. **No clinical validation.**
   The pipeline has not been validated against clinical diagnoses, outcomes, or
   expert manual segmentation at scale. There are no normative ranges built in,
   so the report describes *what was measured*, not whether it is normal or
   abnormal. All "findings" are descriptive; all comparisons are relative.

3. **Small cohort.**
   The reference split uses roughly 10 patients. Any patterns are exploratory and
   not statistically powered. Cohort-relative rankings in the report describe
   position *within this specific dataset only*.

4. **Threshold sensitivity.**
   Biomarkers (especially CTVF/porosity and pore count) shift with the
   segmentation probability threshold. The threshold used is recorded in every
   report; results at different thresholds are not directly comparable.

5. **ROI is pore-derived, not anatomically defined.**
   CTVF is computed inside an LC region of interest approximated by morphological
   closing of the predicted pores — not from an independently annotated anatomical
   boundary. The absolute CTVF value therefore depends on the ROI parameters.

6. **Segmentation is imperfect.**
   At the reference run, validation IoU is ~0.52 / Dice ~0.67 — good for a sparse
   small-object task, but predictions contain errors. Downstream 3D biomarkers
   inherit any segmentation bias (e.g. systematic under- or over-segmentation).

## Responsible-use note on data

This repository contains **code only**; no patient imaging is included. If you
apply this to real OCT scans, ensure you have the appropriate consent and
data-use rights for that dataset. Coded identifiers (e.g. patient folder names)
are still potentially sensitive — handle accordingly and do not publish patient
data without authorisation.

---

## License

Released under the MIT License (code). See [LICENSE](LICENSE). The license covers
the code only and makes no claim over any dataset or clinical use.

## Acknowledgements

Architecture based on the Attention U-Net (Oktay et al., 2018). Built and
iterated in Google Colab.
