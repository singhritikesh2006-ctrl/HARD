# HARD - Heart Attack Risk Detection

![Languages](https://img.shields.io/badge/Language-Python-blue)
![Notebook](https://img.shields.io/badge/Jupyter-Notebook-orange)
![ML](https://img.shields.io/badge/ML-Scikit--Learn-green)

A machine learning-based project that predicts heart attack risk using clinical and lifestyle features. This project was developed as part of coursework at **KIIT University**.

## Project Overview

Cardiovascular diseases remain one of the leading causes of mortality globally. **HARD** leverages supervised learning techniques to analyze patient data and classify individuals based on their risk of heart attack, enabling early detection and timely medical intervention.

## Datasets Used

This project uses two publicly available heart disease datasets:

| Dataset | Source | Records | Features | Target |
|---------|--------|---------|----------|--------|
| **cardio_train_separated.csv** | Kaggle (Cardiovascular Disease) | 70,000 | 13 (age, gender, height, weight, BP, cholesterol, etc.) | `cardio` (binary) |
| **HeartDiseaseTrain-Test.csv** | UCI Heart Disease Dataset | 1,025 | 14 (age, sex, chest pain, BP, cholesterol, max heart rate, etc.) | `target` (binary) |

Both datasets are included in the `Datasets/` folder.

## Models Implemented

Three machine learning algorithms were trained and evaluated on both datasets:

1. **Logistic Regression** - Baseline linear model
2. **Support Vector Machine (LinearSVC)** - Margin-based classifier
3. **Random Forest** - Ensemble tree-based model

## Results Summary

Performance was evaluated using **Accuracy**, **Precision**, and **Cross-Validation (5-fold)** scores.

### Dataset 1 (cardio_train_separated.csv)

| Model | Accuracy | Precision | CV Score | Overall Score |
|-------|----------|-----------|----------|---------------|
| Logistic Regression | 72.36% | 74.58% | 71.86% | 72.94% |
| SVM (LinearSVC) | 65.62% | 66.82% | 65.00% | 65.82% |
| Random Forest | 71.41% | 71.91% | 71.61% | 71.64% |

### Dataset 2 (HeartDiseaseTrain-Test.csv)

| Model | Accuracy | Precision | CV Score | Overall Score |
|-------|----------|-----------|----------|---------------|
| Logistic Regression | 79.51% | 76.99% | 84.64% | 80.38% |
| SVM (LinearSVC) | 79.51% | 76.99% | 84.88% | 80.46% |
| **Random Forest** | **98.54%** | **100.00%** | **97.93%** | **98.82%** |

### Best Model

**Random Forest on Dataset 2** achieved the best overall performance with **98.82%** overall score, making it the most reliable predictor for heart attack risk in this project.

## Key Features (by Importance)

### Dataset 1:
- Age (30.9%)
- Systolic Blood Pressure (17.6%)
- Weight (17.2%)
- Height (15.7%)

### Dataset 2:
- Chest Pain Type (13.0%)
- Vessels Colored by Fluoroscopy (12.7%)
- Oldpeak (ST Depression) (12.3%)
- Thalassemia (12.1%)

## Repository Structure

```
HARD/
├── Codes/
│   └── HARD.ipynb           # Main Jupyter notebook with all code
├── Datasets/
│   ├── cardio_train_separated.csv
│   └── HeartDiseaseTrain-Test.csv
├── Research Papers/
│   ├── PAPER1.pdf
│   ├── PAPER2CARDIO.pdf
│   ├── PAPER3HEARTATTACKPREDICTION.pdf
│   ├── paper4heartdisease.pdf
│   ├── paper5.pdf
│   ├── paper6.pdf
│   ├── paper7.pdf
│   ├── paper8heartattackprediction.pdf
│   ├── paper9Prediction of Heart Disease UCI....pdf
│   └── paper10Effective_Heart_Disease_Predi....pdf
├── PAPER WORK9&10.d...      # Coursework drafts
├── PROJECT TITLE - Hear...  # Project title page
├── Project_Report-forma...  # Report template
├── Research Paper Revie...  # Research paper reviews
└── README.md                # This file
```

## How to Run

### Prerequisites

```bash
pip install numpy pandas scikit-learn matplotlib joblib jupyter
```

### Steps

1. **Clone the repository:**
   ```bash
   git clone https://github.com/singhritikesh2006-ctrl/HARD.git
   cd HARD/Codes
   ```

2. **Launch Jupyter Notebook:**
   ```bash
   jupyter notebook HARD.ipynb
   ```

3. **Run all cells** in the notebook sequentially.

4. **Pre-trained models** (`rf_model_dataset1.pkl`, `rf_model_dataset2.pkl`, `scaler.pkl`) are saved automatically after running the notebook.

## Pipeline Overview

```
Data Loading -> Preprocessing -> Feature Scaling -> Train/Test Split ->
Model Training -> Evaluation -> Feature Importance -> Model Saving ->
Risk Prediction Function
```

The notebook also includes a `predict_risk()` function that takes user input and returns:
- **Risk percentage** (0-100%)
- **Risk category**: Low Risk (<25%), Medium Risk (25-50%), High Risk (50-75%), or Extremely Dangerous (>=75%)

## Limitations

- Dataset 1 contains outliers in blood pressure values (e.g., negative and extremely high readings) that may affect model performance.
- Class imbalance exists in Dataset 1 (~50-50 split).
- Models were evaluated on the same datasets they were trained on; external validation is recommended.
- The prediction function assumes input features match the dataset schema.

## Future Improvements

- [ ] Hyperparameter tuning using GridSearchCV
- [ ] Handle class imbalance with SMOTE or undersampling
- [ ] Add more evaluation metrics (Recall, F1-Score, ROC-AUC)
- [ ] Deploy the model as a web application using Flask/Streamlit
- [ ] Integrate SHAP/LIME for model explainability
- [ ] Add data visualization and EDA notebooks

## Contributors

- **Bhanu Kumar Dev** - [@bhanukumardev](https://github.com/bhanukumardev)
- Project team from **KIIT University**

## License

This project is created for academic purposes. Feel free to use and modify the code.

---

**Note**: This project is for educational and research purposes only. It should not be used as a substitute for professional medical advice, diagnosis, or treatment.
