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

