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
