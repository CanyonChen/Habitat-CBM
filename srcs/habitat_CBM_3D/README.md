# Habitat-CBM 3D UCSF-PDGM Pipeline

This folder contains the 3D companion pipeline for `repo/models/habitat_CBM_3D.py`.

## Data Contract

- Dataset root: `/root/autodl-tmp/habitat_CBM/PKG _UCSF_PDGM_Version_5`.
- Pipeline input is a manifest CSV, not materialized `train/val/test` image folders.
- Follow-up IDs containing `_FU` are excluded by default. Use `--include-followup true` to include them.
- Model input modalities: `T1/T1c/T2/FLAIR`, shaped `[4,D,H,W]`.
- Habitat/concept modalities: `ADC + tumor_segmentation` only.
- ADC-only habitat mapping: K-means in tumor VOI, with cluster centers ordered low-to-high as `H1/H2/H3`.
- Default downstream habitat mask: `h23` (`H2+H3`).
- Concept assets may contain `C1..C8`; training defaults to `C1/C2/C3/C4/C6`.
- NIfTI-reading steps require `nibabel`; install the main environment with `pip install -r repo/requirements.txt` or the radiomics subset with `pip install -r repo/requirements_pyradiomics.txt`.

## End-To-End Commands

```bash
# 1) Build manifest split
python repo/srcs/habitat_CBM_3D/data_split_ucsf_pdgm_3D.py \
  --ucsf-root "/root/autodl-tmp/habitat_CBM/PKG _UCSF_PDGM_Version_5" \
  --output-root /root/autodl-tmp/habitat_CBM/dataset/ucsf_pdgm_3d \
  --overwrite
```

```bash
# 2) Build ADC-only habitat masks
python repo/srcs/habitat_CBM_3D/build_habitat_ucsf_3D.py \
  --manifest-csv /root/autodl-tmp/habitat_CBM/dataset/ucsf_pdgm_3d/manifest_ucsf_pdgm_3d.csv \
  --output-root /root/autodl-tmp/habitat_CBM/dataset/ucsf_pdgm_3d/habitat_masks \
  --selected-habitat h23 \
  --overwrite true
```

```bash
# 3) Extract radiomics proxies and concept proxy features
python repo/srcs/habitat_CBM_3D/get_habitat_radiomics_ucsf_3D.py \
  --manifest-csv /root/autodl-tmp/habitat_CBM/dataset/ucsf_pdgm_3d/manifest_ucsf_pdgm_3d.csv \
  --habitat-root /root/autodl-tmp/habitat_CBM/dataset/ucsf_pdgm_3d/habitat_masks \
  --output-root /root/autodl-tmp/habitat_CBM/results/ucsf_pdgm_3d_radiomics \
  --selected-habitat h23 \
  --run-id ucsf_h23
```

```bash
# 4) Build CBM concept labels and scaler
python repo/srcs/habitat_CBM_3D/build_habitat_cbm_labels_ucsf_3D.py \
  --concept-proxy-csv /root/autodl-tmp/habitat_CBM/results/ucsf_pdgm_3d_radiomics/ucsf_h23/concept_proxy_features_ucsf_h23.csv \
  --source-mapping-json /root/autodl-tmp/habitat_CBM/results/ucsf_pdgm_3d_radiomics/ucsf_h23/concept_source_mapping_ucsf_h23.json \
  --output-label-csv /root/autodl-tmp/habitat_CBM/dataset/ucsf_pdgm_3d/concept_label/concept_labels.csv \
  --output-stats-csv /root/autodl-tmp/habitat_CBM/dataset/ucsf_pdgm_3d/concept_label/concept_statistics.csv \
  --output-scaler-json /root/autodl-tmp/habitat_CBM/dataset/ucsf_pdgm_3d/concept_label/concept_scaler_stats.json
```

```bash
# 5) Loader smoke test
python repo/srcs/habitat_CBM_3D/data_loader_habitat_CBM_3D.py \
  --manifest-csv /root/autodl-tmp/habitat_CBM/dataset/ucsf_pdgm_3d/manifest_ucsf_pdgm_3d.csv \
  --split train \
  --concept-label-csv /root/autodl-tmp/habitat_CBM/dataset/ucsf_pdgm_3d/concept_label/concept_labels.csv \
  --concept-scaler-json /root/autodl-tmp/habitat_CBM/dataset/ucsf_pdgm_3d/concept_label/concept_scaler_stats.json \
  --max-samples 2
```

```bash
# 6) Train 3D Habitat-CBM
python repo/srcs/habitat_CBM_3D/train_habitat_CBM_3D.py \
  --config repo/srcs/habitat_CBM_3D/args_train_habitat_CBM_3D.json \
  --run-id ucsf_h23_run01 \
  --device cuda:0
```

## Evaluation And Intervention

```bash
python repo/srcs/habitat_CBM_3D/eval_habitat_CBM_3D.py \
  --config repo/srcs/habitat_CBM_3D/args_train_habitat_CBM_3D.json \
  --checkpoint /path/to/stage3_best.pt \
  --run-id ucsf_h23_eval \
  --device cuda:0 \
  --splits val,test
```

```bash
python repo/srcs/habitat_CBM_3D/eval_habitat_cbm_3D_concepts.py \
  --patient-concepts-csv /path/to/patient_concepts_habitat_cbm_3d_ucsf_h23_run01.csv \
  --output-dir /path/to/concept_eval \
  --split test \
  --scale raw
```

```bash
python repo/srcs/habitat_CBM_3D/intervene_habitat_cbm_3D.py \
  --checkpoint /path/to/stage3_best.pt \
  --patient-predictions-csv /path/to/patient_predictions_habitat_cbm_3d_ucsf_h23_run01.csv \
  --patient-concepts-csv /path/to/patient_concepts_habitat_cbm_3d_ucsf_h23_run01.csv \
  --concept-scaler-json /root/autodl-tmp/habitat_CBM/dataset/ucsf_pdgm_3d/concept_label/concept_scaler_stats.json \
  --output-dir /path/to/intervention \
  --split test
```

## Smoke-Test Tips

- Use `--max-patients 2` on habitat/radiomics scripts for quick checks.
- Use `data.target_shape=[8,32,32]` and one epoch per stage for CPU/CUDA smoke tests.
- Set `train.use_monai_augmentation=false` if MONAI is not installed.
