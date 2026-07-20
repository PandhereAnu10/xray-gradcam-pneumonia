# Does an AI Model Actually See Pneumonia or Just the X-Ray Machine?

Fine-tuning a real ResNet18 on the **NIH Chest X-ray14** dataset (the
benchmark dataset behind Stanford's CheXNet research) to classify
Pneumonia vs. No Pneumonia - then using **Grad-CAM**, implemented from
scratch, to visually check *where* the model is actually looking when it
makes a prediction. Correct or wrong.

High accuracy on medical images doesn't guarantee a model learned the
right thing. It could be keying off scan positioning, equipment
artifacts, or dataset quirks instead of the actual disease. This project
is a real, honest test of that — including a real failure case, shown
below, not just the wins.

---

## Results

### Confusion Matrix

![Confusion Matrix](https://raw.githubusercontent.com/PandhereAnu10/xray-gradcam-pneumonia/main/outcome/confusion_matrix.png)

| | Precision | Recall |
|---|---|---|
| No Pneumonia | 80% | 87% |
| Pneumonia | 46% | 33% |

In plain terms: the model correctly identifies healthy X-rays most of
the time, but **misses roughly 2 out of every 3 real pneumonia cases** -
191 pneumonia cases were called "healthy" outright. That's the honest
result of a quick 3-epoch fine-tune on a rare, hard-to-spot condition
using labels auto-extracted from radiology reports (not hand-verified by
radiologists) - and it's a well-documented pattern in real medical AI
research, not a broken experiment.

### Grad-CAM: Correctly Classified X-Rays

![Grad-CAM Correct](https://raw.githubusercontent.com/PandhereAnu10/xray-gradcam-pneumonia/main/outcome/gradcam_correct.png)

When the model gets it right, the heatmap concentrates over the central
chest/lung area - a reassuring sign it's looking in the right
anatomical place.

### Grad-CAM: Misclassified X-Rays

![Grad-CAM Misclassified](https://raw.githubusercontent.com/PandhereAnu10/xray-gradcam-pneumonia/main/outcome/gradcam_misclassified.png)

This is the interesting one. Comparing where the model looks on its
mistakes against where it looks when it's right is the actual
diagnostic value of Grad-CAM, a heatmap that's still centered on the
lungs suggests a genuinely hard/ambiguous case, while one that's drifted
toward the edges, shoulders, or scan artifacts suggests the model is
picking up on the wrong signal entirely.

---

## Why this dataset

**NIH Chest X-ray14**: 100,000+ chest X-rays, 14 possible findings,
released publicly by the NIH - the real benchmark dataset used in
published medical AI research (it's what Stanford's CheXNet was built
on), not a toy beginner dataset. It's structured as flat image folders
(`images_001/images/*.png` ... `images_012/images/*.png`) plus a
`Data_Entry_2017.csv` with a multi-label `Finding Labels` column (e.g.
`"Infiltration|Pneumonia"`). This project collapses that down to a
clean binary Pneumonia-vs-not task and balances the classes for a fast,
meaningful demo.

## How it works

1. **Index the data**: walk all 12 image folders, build a filename -
   filepath lookup, join it against `Data_Entry_2017.csv` labels.
2. **Balance the classes**: pneumonia is only ~1-2% of the full
   dataset, so training on it directly would just teach the model to
   always guess "no pneumonia." Every available positive case is kept,
   plus a random matched sample of negatives.
3. **Transfer learning**: start from a ResNet18 pretrained on
   ImageNet, replace the final layer for this 2-class task, fine-tune
   for 3 epochs.
4. **Evaluate honestly**: test only on held-out images the model never
   saw during training; report precision/recall per class, not just
   overall accuracy (accuracy alone would hide the pneumonia recall
   problem entirely).
5. **Grad-CAM, from scratch**: forward/backward hooks on the last
   convolutional layer (`model.layer4[-1]`), no external explainability
   library, so every step (which layer, which gradient, how the weights
   are computed) is inspectable rather than trusted as another black box
   explaining the first one.

## Run it yourself

1. Create a Kaggle notebook, turn on GPU (T4) and Internet in Settings.
2. Add the **NIH Chest X-rays** dataset via "Add Input" (organization:
   `nih-chest-xrays`).
3. Paste `gradcam.py`'s contents into the first cell.
4. Paste `kaggle_xray_gradcam.py`'s cells in order (split at the
   `# --- CELL n ---` markers).
5. Run all. Three images land in `/kaggle/working/outputs/`.

## Files

- [`gradcam.py`](./gradcam.py) — Grad-CAM implemented from scratch with
  forward/backward hooks, verified against a real ResNet18 architecture
- [`kaggle_xray_gradcam.py`](./kaggle_xray_gradcam.py) - the full
  pipeline: data indexing, class balancing, transfer learning,
  evaluation, confusion matrix, and Grad-CAM diagnostics

## Honest limitations

- This is a learning/demo project, not a validated clinical tool - do
  not read diagnostic reliability into these results.
- The binary "Pneumonia vs. not" framing collapses the dataset's
  original multi-label structure; a real clinical model would need to
  handle co-occurring findings.
- Training on a balanced subsample rather than the full imbalanced
  dataset is a deliberate simplification for a fast, clean demo.
- Grad-CAM shows correlation with the prediction, not a causal or
  medically verified explanation - a heatmap over the right anatomical
  region doesn't guarantee clinically sound reasoning, only that the
  model's evidence was concentrated there.
- NIH Chest X-ray14's labels were extracted from radiology reports with
  NLP, not manually verified by radiologists for every image - a
  documented limitation of this dataset.

## License

MIT
