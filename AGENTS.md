# AGENTS.md

This file provides guidance to agents when working with code in this repository.

## Critical Path Issues
- Hardcoded Windows paths in [`breast-cancer-cnn.ipynb`](breast-cancer-cnn.ipynb:40) MUST be updated before running: `r'c:\Users\Admin\Downloads\csv\...'`
- Image paths use forward slashes in replacements but backslashes in hardcoded paths - causes inconsistency
- Model files saved to project root with specific names: `breast_cancer_cnn_model.h5`, `model_gwo_selected_feature.h5`

## Custom GWO Implementation  
- Grey Wolf Optimization algorithm in [`gwo_feature_selection_cnn()`](breast-cancer-cnn.ipynb:527) is project-specific, not standard library
- Function expects `cnn_model_builder` callback that returns compiled model
- Feature selection runs for limited epochs (3) for fitness evaluation - NOT full training

## Image Processing Pipeline
- Target size hardcoded as `(224, 224, 3)` throughout - changing requires multiple updates
- [`image_processor()`](breast-cancer-cnn.ipynb:428) expects absolute paths via `os.path.abspath()`
- Binary classification uses specific mapper: `{'MALIGNANT': 1, 'BENIGN': 0, 'BENIGN_WITHOUT_CALLBACK': 0}`

## Model Architecture Constraints
- Transfer learning model (`vgg_model`) referenced but not defined in visible code
- Feature extraction uses `flatten_2` layer name specifically - layer must exist
- [`cnn_model_builder()`](breast-cancer-cnn.ipynb:781) for GWO expects 1D input, not image input

## Data Dependencies  
- Requires CSV files: `meta.csv`, `dicom_info.csv`, `mass_case_description_train_set.csv`, `mass_case_description_test_set.csv`
- Image directory structure: `CBIS-DDSM/jpeg/` gets replaced with local paths
- [`fix_image_path()`](breast-cancer-cnn.ipynb:135) modifies dataframe in-place using hardcoded column indices (11, 12, 13)

## Model Loading Pattern
- Prediction workflow requires loading feature extractor AND trained GWO model separately
- New images must go through: preprocess → feature extraction → GWO feature selection → prediction