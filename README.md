# MLP-for-CR

[![License: CC BY 4.0](https://img.shields.io/badge/License-CC_BY_4.0-lightgrey.svg)](https://creativecommons.org/licenses/by/4.0/)
[![Python 3.9](https://img.shields.io/badge/Python-3.9-blue.svg)](https://www.python.org/)

Official implementation of the MLP framework for detecting careless responses in online questionnaires, as presented in:
**"A Multilayer Perceptron Model to Detect Careless Responses in Online Likert Questionnaires"**  
*Yangruiyu Zeng, Hui Yuan, Ligang Wang*, et al. (2025)

## Features
- **Noise Injection System**: Simulates 4 careless response patterns
- **Adaptive MLP Architecture**: 4 hidden layers with ReLU activation
- **Benchmarking Suite**: Compares with 5 traditional psychometric indicators
- **High Accuracy**: 94.49% classification performance
- **Cross-scale Design**: Raw score processing

## Installation
```bash
git clone https://github.com/zengyangruiyu0113/careless-response-detection.git
cd careless-response-detection
pip install -r requirements.txt

## Data Preparation
| Item1 | Item2 | ... | Item20 |
|-------|-------|-----|--------|
|   3   |   1   | ... |   2    |


## Training Pipeline
# Initialize pipeline
project = DataDenoising()
project.read_excel(path="Depression.xlsx")

# Generate synthetic noise patterns
project.make_noise_data(data_size=(10000,20), data_scale=[0,1,2,3])

# Build and train model
project.train_test_data(itemsnum=20)
project.build_model(
    middle_struction=(30,30,30,30),
    activation="relu",
    epochs=30
)

# Save trained model
project.model.save("careless_detection_model.h5")

## Detection Workflow

# Load new responses
new_data = pd.read_excel("new_surveys.xlsx").values

# Predict response quality
predictions = project.model_predict(new_data)

# Output: ["Careful", "Random", "Straightlining", "Pattern1", "Pattern2"]
print(predictions)
