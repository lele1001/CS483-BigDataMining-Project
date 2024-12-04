# CS483 - Big Data Mining Project: Diabetes Prediction and Analysis
This repository contains the implementation of a predictive modeling project for diabetes classification and risk analysis. Using machine learning techniques and advanced interpretability methods, the project aims to enhance healthcare decision-making by analyzing the CDC Diabetes Health Indicators dataset.

## Table of Contents

- [Overview](#overview)  
- [Features](#features)  
- [Technologies](#technologies)  
- [Setup](#setup)  
- [Usage](#usage)  
- [Results](#results)  
- [Acknowledgments](#acknowledgments)  
- [License](#license)  

---

## Overview

Diabetes is a growing public health challenge, making early identification of at-risk individuals crucial for effective intervention. This project develops machine learning models to classify individuals as healthy, pre-diabetic, or diabetic based on health, lifestyle, and demographic data. 

Key objectives:
- Develop interpretable models for diabetes prediction.  
- Identify significant risk factors using SHAP analysis.  
- Generate personalized health reports with Large Language Models (LLMs).  

The dataset includes 21 features, such as BMI, physical activity, cholesterol levels, and demographic factors, providing comprehensive insight into diabetes risks.

---

## Features

- **Dimensionality Reduction**: PCA, t-SNE, and autoencoders reduce dataset complexity while preserving critical insights.  
- **Clustering**: K-means and stratified K-means clustering uncover distinct health profiles.  
- **Classification**: Decision trees, Random Forests, AdaBoost, and Neural Networks for diabetes classification.  
- **Interpretability**: SHAP analysis provides feature importance and transparency for predictive models.  
- **Generative Reports**: LLM integration produces patient-specific, understandable medical reports.  

---

## Technologies

- **Programming Language**: Python  
- **Libraries**: Scikit-learn, TensorFlow, SHAP, Matplotlib  
- **Data Source**: [CDC Diabetes Health Indicators Dataset](https://archive.ics.uci.edu/dataset/891/cdc+diabetes+health+indicators)  
- **Generative AI**: Google Generative AI API  

---

## Setup

1. Clone the repository:
   ```
   git clone <repository-url>
   cd <repository-directory>
   ```
2. Install the dependencies: 
    ```
    pip install -r requirements.txt
    ```
3. Download and place the datase in the ```data/``` directory (if not already present)

---

## Usage 

1. **Preprocess the Dataset:** clean and prepare the data for modeling
    ```
    python preprocess.py
    ```
2. Run Models: execute the scripts for different analyses:
    - Dimensionality Reduction: ```python dimensionality_reduction.py```
    - Clustering: ```python clustering.py```
    - Classification: ```python classification.py```
3. Generate Reports: create personalized reports with the LLM tool
    ```
    python generate_reports.py
    ```

---

## Results

- **Model Performance:** Achieved ~85% accuracy for diabetes classification using neural networks.
- **Key Findings:** Physical health, BMI, and age were the most influential features for diabetes prediction.
**Reports:** Patient-specific recommendations were successfully generated using LLMs.

---

## Acknowledgments
This project was completed as part of CS483 coursework under the guidance of Prof. Lu Cheng. Special thanks to team Spaghetti Coders:

- Eleonora Cabai
- Filippo Corna
- Davide Ettori
- Patrik Poggi
