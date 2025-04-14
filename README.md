# MultimodalMed

## Overview
MultimodalMed is a research project focused on exploring the effectiveness of combining text and vision modalities for classification tasks. The project compares the performance of multimodal approaches against using only text or vision modalities, using specific datasets for medical data analysis.

## Features
- **Multimodal Analysis**: Investigates the integration of vision and text data for improved classification.
- **Comparative Evaluation**: Benchmarks multimodal models against single-modality models.
- **Hyperparameter Optimization**: Utilizes Optuna for efficient tuning of model parameters.
- **Evaluation Tools**: Provides metrics, confusion matrices, and ROC curves for detailed analysis.
- **Logging**: Supports TensorBoard and Weights & Biases (W&B) for experiment tracking.

## Installation
1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd multimodalmed
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Training
To train a model, run:
```bash
python train.py
```

### Hyperparameter Optimization
To perform hyperparameter optimization for all modes (vision, text, multimodal), run:
```bash
python run_all_modes.py
```

### Evaluation
To evaluate a trained model, use:
```bash
python evaluate.py --checkpoint <path-to-checkpoint>
```

## Configuration
The `total_config.py` file contains the default configuration for the project. You can modify it to customize the dataset, model, and training parameters.

## File Structure
- `train.py`: Script for training models.
- `run_all_modes.py`: Script for hyperparameter optimization.
- `evaluate.py`: Script for evaluating trained models.
- `models.py`: Model definitions and utilities.
- `data.py`: Data loading and preprocessing.
- `total_config.py`: Default configuration settings.

## License
This project is licensed under the MIT License.