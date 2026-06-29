# Malaria Cell Classification

A two-phase deep-learning pipeline that diagnoses malaria from single red-blood-cell (RBC)
microscopy images. **Phase 1** screens a cell as *Healthy* vs *Infected*; **Phase 2** takes an
infected cell and classifies the *Plasmodium* developmental stage as *Ring*, *Trophozoite*,
*Schizont*, or *Gametocyte*. Both phases use transfer learning.

```
RBC image
   │
   ▼
[Phase 1] Binary classifier (ResNet-50)
   ├── Healthy  → stop
   └── Infected → [Phase 2] Stage classifier (EfficientNet-B0)
                     → Ring | Trophozoite | Schizont | Gametocyte
```

The project is **two phases only**: an infected cell goes straight into the stage classifier, with no
separate parasite-detection or cropping step at inference.

## Why two phases

Microscopy of Giemsa-stained blood smears is the diagnostic gold standard for malaria but is slow and
depends on scarce expert microscopists. Splitting the task lets each model use the data best suited to
it: screening has a large, balanced dataset, while staging has only small, heavily imbalanced ones and
leans on transfer learning to compensate.

## Repository structure

```
.
├── phase1_binary_healthy_infected.ipynb   # Phase 1 — screening (ResNet-50)
├── phase4_stage_classifier__3_.ipynb      # Phase 2 — staging (EfficientNet-B0)
├── artifacts/                             # saved checkpoints (created at runtime)
│   ├── binary_resnet50.pt
│   └── stage_efficientnet_b0.pt
├── assets/                                # figures / diagrams
└── README.md
```

> Note: the staging notebook is filenamed `phase4_…` for historical reasons (an earlier design had
> intermediate detect/crop phases that were dropped). Conceptually it is **Phase 2**.

## Datasets

| Phase | Dataset | Content | Source |
|---|---|---|---|
| 1 | NIH / Kaggle "Cell Images for Detecting Malaria" | 27,558 single cells, balanced Parasitized/Uninfected | pulled via `kagglehub` |
| 2 | MP-IDB | full-slide PNG, 4 species, mask + stage annotations | `git clone` |
| 2 | IML-Malaria | 345 thin-smear images, bounding-box annotations | `git clone` |
| 2 | MD-2019 (optional) | pre-cropped, stage manifest (XLSX) | manual download |

For Phase 2, parasite images are built once from each dataset's mask / bounding-box annotations and
merged into one folder per stage (`stage_merged/<stage>/`). The merged set is heavily imbalanced
(≈ 21.6×, rings dominant), which is countered with ring undersampling, minority oversampling to 1,000
each, a weighted sampler, and a class-weighted loss.

## How to run

Both notebooks are written for **Google Colab with a GPU runtime**.

**Phase 1 — Screening**
1. Open `phase1_binary_healthy_infected.ipynb` and select a GPU runtime.
2. Run all cells. The NIH dataset is downloaded automatically via `kagglehub`.
3. Ensure the data path resolves to the folder directly containing `Parasitized/` and `Uninfected/`
   (the loader should report two classes).
4. Output: `artifacts/binary_resnet50.pt`.

**Phase 2 — Staging**
1. Open `phase4_stage_classifier__3_.ipynb`. Run the install cell, then **Runtime → Restart**, then
   continue from the next cell.
2. (Optional) Mount Google Drive and set the MD-2019 paths to include that dataset; otherwise it is
   skipped automatically.
3. Run all cells. MP-IDB and IML-Malaria are cloned from GitHub; crops are built into `stage_merged/`.
4. Output: `artifacts/stage_efficientnet_b0.pt`.

### Requirements

```
torch torchvision scikit-learn scipy opencv-python pillow matplotlib tqdm openpyxl kagglehub
```

## Method (both phases)

ImageNet-pretrained backbone, trained in two steps: freeze the backbone and train a new head, then
fine-tune the whole network at a lower learning rate. Adam (`weight_decay=1e-4`), learning rate
reduced on validation-loss plateau, images at 224×224, stratified 70/15/15 split (seed 42). Phase 2
adds stronger augmentation and the class-balancing strategy above.

## Results

**Phase 1 — Screening.** A ResNet-50 separates Healthy from Infected cells on a 4,134-cell held-out
test set. *Screening metrics will be reported once the evaluation run completes.*

**Phase 2 — Staging** (held-out test set, n = 294):

| Metric | Value |
|---|---|
| Accuracy | 74.2% |
| Macro-F1 | 0.67 |
| Macro-recall | 0.81 |

| Stage | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| gametocyte | 0.92 | 0.77 | 0.84 | 44 |
| ring | 0.97 | 0.72 | 0.83 | 217 |
| schizont | 0.48 | 1.00 | 0.65 | 10 |
| trophozoite | 0.23 | 0.74 | 0.35 | 23 |

Ring and gametocyte classify reliably; trophozoite is the hardest class (rare, and morphologically
between ring and schizont). The balancing strategy favours recall over precision, so rare stages are
rarely missed but sometimes over-predicted. Schizont and trophozoite have few test cells (10 and 23),
so their per-class scores carry wide error bars.

## Limitations & future work

- Small, skewed stage data; tiny test support for the rarest classes.
- MD-2019 not yet included; adding it should most help the rare stages.
- Cross-dataset domain shift (species, staining, resolution differ).
- The two phases are evaluated separately; end-to-end accuracy is not yet measured.
- Next: complete Phase 1 evaluation, add MD-2019, run a held-out cross-dataset test, add Grad-CAM, and
  measure the full Phase 1 → Phase 2 pipeline.

## References

- NIH/Kaggle — Cell Images for Detecting Malaria
- MP-IDB — Loddo et al., 2019
- IML-Malaria — Arshad et al., 2021 (arXiv:2102.08708)
- MD-2019 — Mendeley `5bf2kmwvfn`
- CDC DPDx — Malaria reference library
- ResNet-50 (He et al., 2015); EfficientNet-B0 (Tan & Le, 2019), ImageNet-pretrained via torchvision

## License

Add a license of your choice (e.g. MIT). Dataset usage is subject to each dataset's own license.
