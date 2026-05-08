# Step 1: Senescence Patch Extraction

The code extracts overlapping nucleus and non-nucleus patches from cellular senescence images.

The extracted patches are later used for:

- Contrastive learning
- Encoder training
- Senescence classification

## Supported Cell Lines

- HFF
- HMC3

## Supported Senescence Conditions

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

The learned embeddings bring related nucleus and non-nucleus representations closer within the same condition, while separating control and senescence representations.

## Input

This file uses the extracted patches from Step 1.

Expected patch folders:

```text
output_folder/
├── control/
│   ├── overlapping_nucleus/
│   └── overlapping_non_nucleus/
│
└── senescence/
    ├── overlapping_nucleus/
    └── overlapping_non_nucleus/
```

Users should update the patch folder paths according to the respective dataset.

---

## Run

Run the training file using:

```bash
python training_Esupconloss.py
```

---

## Output


```text
esupconloss_encoder.pth
```
