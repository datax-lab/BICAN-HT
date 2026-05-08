# Step 1: Senescence Patch Extraction

The code extracts overlapping nucleus and non-nucleus patches from cellular senescence images.

The extracted patches are later used for:

- Contrastive learning with encoder training
- Ranking patches
- Learning nucleus-cytoplasm relationships

## Cell Lines

- HFF
- HMC3

## Senescence Conditions

- H2O2
- Dox
- Pal

Users can change the folder names and paths according to their own dataset structure.

---

## Input

The patch extraction code reads:

- Control images
- Senescence images

and generates:

- Overlapping nucleus patches
- Overlapping non-nucleus patches

---

## Input Paths

Update the folder paths before running the code:

```python
control_path = "/path/to/control_images"
senescence_path = "/path/to/senescence_images"
output_path = "/path/to/output_folder"
```

Example:

```python
control_path = "/data/HFF/control"
senescence_path = "/data/HFF/h2o2"
output_path = "/data/HFF/output"
```

---

## Run

Open and run:

```text
Patch_extractor.ipynb
```
# Step 2: Contrastive Learning with E-SupConLoss

After patch extraction, the extracted nucleus and non-nucleus patches are used for contrastive learning.

This step trains an encoder with an MLP projection head using E-SupConLoss. The goal is to learn meaningful patch-level embeddings for control and senescence samples.

The training uses:
- Control nucleus patches
- Control non-nucleus patches
- Senescence nucleus patches
- Senescence non-nucleus patches

The trained encoder is later used for patch ranking and interaction learning.

## Input

This step uses the extracted patches from Step 1.

---

## Run

```bash
python training_Esupconloss.py
```

---

## Output

```text
esupconloss_encoder.pth
```
# Step 3: Ranking Patches

After contrastive learning, the saved encoder from Step 2 is used to rank nucleus and non-nucleus patches.

This step trains a patch-level classification model using the frozen encoder learned from E-SupConLoss training.

The model learns to classify whether a patch belongs to:

- Control
- Senescence

The trained model is then used to rank the most important nucleus and non-nucleus patches.

---

## Input

This step uses:

- Extracted patches from Step 1
- Saved encoder from Step 2

---

## Run

```bash
python Ranking_patches.py
```

---

## Output

```text
encoder_mlp_patch_classifier.pth
```
# Step 4: Senescence Scoring with Cross-Attention

After ranking the patches, the top-k nucleus and non-nucleus patches are selected for senescence prediction.

This step uses a cross-attention network to learn relationships between nucleus and non-nucleus patches and generates a senescence probability score for each image.

The workflow includes:

- Patch extraction
- Patch ranking
- Top-k patch selection
- Cross-attention interaction learning
- Senescence probability prediction

---

## Input

This step uses:

- Control and senescence images

---

## Run

```bash
python cross_attention_senescence_scoring.py
```

---

## Output

The model generates a probability score indicating whether the image belongs to:

- Control
- Senescence
