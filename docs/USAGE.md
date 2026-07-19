# Usage Guide

Step-by-step instructions for running the pipeline in Google Colab.

## 1. Environment

- Open `notebooks/lamina_cribrosa_OCT.ipynb` in Google Colab.
- **Runtime → Change runtime type → GPU** (T4 is sufficient; ~30–45 min to train
  25 epochs on ~1,850 images).

## 2. Data layout

Point the notebook's path constants at a dataset organised like this:

```
OCT_Project/
├── dataset/ROI/enh/            # input OCT images (.png)
│   └── <PATIENT>/<EYE>/<ORIENT>/<name>.png
└── pred/                       # label masks (.jpg), same relative paths
    └── <PATIENT>/<EYE>/<ORIENT>/<name>.jpg
```

- `PATIENT` e.g. `ALLV`, `BRUN`; `EYE` e.g. `OD`/`OG`/`OS`; `ORIENT` e.g. `H`/`V`.
- Pairing is by matching relative path with the label extension swapped to `.jpg`.

Edit in the setup cell:
```python
PARENT_FOLDER     = "/content/drive/MyDrive/OCT_Project"   # <-- your path
IMAGE_ROOT_FOLDER = ".../dataset/ROI/enh"
LABEL_ROOT_FOLDER = ".../pred"
```
The setup cell prints `exists: True/False` for each path — fix any `False`
before continuing.

## 3. The kaleido caveat (important)

The report's 3D panels are exported as static images via **kaleido**, which must
be installed **before** the kernel needs it and **does not survive a runtime
restart**. To avoid the install-restart-rerun loop:

1. Run the kaleido install cell (`!pip install -q "kaleido==0.2.1"`) **first**.
2. **Runtime → Restart session.**
3. Then run everything top to bottom.

If you ever see `ValueError: ... requires the kaleido package`, the kernel was
restarted after the install — repeat the three steps above.

## 4. Cell order

Run top to bottom:

1. **Mount Drive**
2. **Setup** — imports, paths, hyperparameters, device
3. **collect_pairs** — patient-based 80/20 split
4. **OCTDataset** + DataLoaders
5. **Attention U-Net** (`AttentionBlock`, `AttentionUNet`)
6. **Loss + metrics** (`DiceBCELoss`, `dice_score`, `iou_score`)
7. **Training** — saves best checkpoint (by val IoU) to Drive
8. **`!pip install plotly -q`**
9. **`output.enable_custom_widget_manager()`** — enables interactive widgets
10. **Dashboard** — patient selector + threshold/pore sliders + 3D + 2D viewer
11. **Cohort batch** — run **once** to build the dataset-wide reference table
12. **Report** — one-click A4 report with cohort-relative ranking

> After a restart you can skip training (step 7) if the checkpoint already
> exists on Drive — the dashboard and report load weights from there.

## 5. Using the dashboard

- Pick **Patient / Eye / Orient**, click **Load patient** (runs inference once,
  caches raw probabilities).
- **Threshold** and **Min pore** sliders update the 3D view, biomarkers, and the
  2D prediction live (they re-threshold the cached probabilities — no model
  re-run).
- **Colour by** switches the 3D colouring (depth / pore size / pore ID).
- **2D slice** scrolls input / ground-truth / prediction.

## 6. Building the report

1. Run the **cohort batch** cell once (rebuild it if you change the threshold).
2. In the report cell, click **Build report** — it uses the currently loaded
   patient and the current slider settings, and offers a PNG download.

## 7. Tuning the report layout

All layout knobs are constants at the top of the report cell or inside
`build_report`:

- `LH`, `GAP_HD`, `GAP_PAR`, `GAP_SEC` — vertical rhythm (line height + gaps).
- `BODY_WRAP`, `BULLET_WRAP` — characters per line (keeps text within margins).
- `heading(...)` `fontsize` — all section headings; `para(...)` `size` — body text.
- `PANEL_W`, `PANEL_H_3D`, `PANEL_H_2D`, `ROW_GAP`, `col_step` — image sizes/positions.
- Images are pinned to the bottom (`FOOTER_Y`) and grow upward; if you add text,
  watch the gap above the image band so they don't overlap.
