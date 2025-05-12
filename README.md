Parkinson's Disease Classification
Overview
This project implements a machine learning pipeline to classify Parkinson's Disease using vocal and speech features from the parkinson_disease.csv dataset. The pipeline includes data preprocessing, feature selection, handling class imbalance, model training, evaluation, and hyperparameter tuning. Three models—Logistic Regression, XGBoost, and Support Vector Classifier (SVC)—are trained and evaluated, with a focus on achieving high AUC scores for binary classification.
The dataset contains features derived from voice recordings, such as pitch, jitter, shimmer, and Mel-frequency cepstral coefficients (MFCCs), with a target column (class) indicating the presence (1) or absence (0) of Parkinson's Disease.
Features
Data Preprocessing: Removes duplicates, drops highly correlated features, and selects features using Chi-squared tests.

Class Imbalance Handling: Uses RandomOverSampler to balance the minority class.

Model Training: Trains Logistic Regression, XGBoost, and SVC models.

Evaluation Metrics: Reports AUC scores, confusion matrices, classification reports, and ROC curves.

Hyperparameter Tuning: Optimizes Logistic Regression parameters using GridSearchCV.

Visualization: Displays class distribution (pie chart), confusion matrices, and ROC curves.

Results
Logistic Regression: Training AUC: 0.50, Validation AUC: 0.50

XGBoost: Training AUC: 1.00, Validation AUC: 0.73

SVC: Training AUC: 0.72, Validation AUC: 0.57

Tuned Logistic Regression: Best Training AUC: 0.75, Validation AUC: 0.50

The XGBoost model outperforms others on the validation set, though Logistic Regression shows poor performance, possibly due to class imbalance or feature selection issues.
Dataset
The dataset (parkinson_disease.csv) includes:
Features: 753 columns, including gender, vocal features (e.g., PPE, DFA, RPDE), MFCCs, and wavelet-based features.

Target: Binary class column (0 = Healthy, 1 = Parkinson's).

Preprocessing Steps:
Averages rows with the same id to remove duplicates.

Removes features with correlation > 0.7.

Selects top 30 features using Chi-squared tests after MinMax scaling.

Installation
Prerequisites
Python 3.7+

Libraries: numpy, pandas, matplotlib, seaborn, imblearn, scikit-learn, xgboost

Setup
Clone the repository:
bash

git clone https://github.com/Vijaybattula26/parkinsons-disease-classification.git
cd parkinsons-disease-classification

Install dependencies:
bash

pip install -r requirements.txt

Place the parkinson_disease.csv dataset in the project directory or update the file path in the script.

Requirements File
Create a requirements.txt with the following content:

numpy
pandas
matplotlib
seaborn
imbalanced-learn
scikit-learn
xgboost

Usage
Ensure the parkinson_disease.csv dataset is in the project directory or update the path in the script.

Run the script:
bash

python parkinsons_classification.py

The script will:
Load and preprocess the data.

Train and evaluate Logistic Regression, XGBoost, and SVC models.

Display AUC scores, confusion matrices, classification reports, and ROC curves.

Perform hyperparameter tuning for Logistic Regression and show results.

Project Structure

parkinsons-disease-classification/
│
├── parkinson_disease.csv        # Dataset (not included in repo)
├── parkinsons_classification.py # Main script
├── requirements.txt             # Dependencies
├── README.md                    # Project documentation
└── outputs/                     # Directory for saving plots (optional)
    ├── class_distribution.png
    ├── confusion_matrix_lr.png
    ├── roc_curves.png
    ├── roc_tuned_lr.png

Code Explanation
The script (parkinsons_classification.py) performs the following steps:
Data Loading: Loads the dataset and identifies the target column (class).

Preprocessing:
Removes duplicates by averaging rows with the same id.

Drops highly correlated features (correlation > 0.7).

Selects top 30 features using Chi-squared tests after MinMax scaling.

Class Distribution: Visualizes class distribution with a pie chart.

Train/Test Split: Splits data into 80% training and 20% validation sets.

Oversampling: Balances the training set using RandomOverSampler.

Model Training: Trains Logistic Regression, XGBoost, and SVC models.

Evaluation:
Computes training and validation AUC scores.

Generates confusion matrices and classification reports for Logistic Regression.

Plots ROC curves for all models.

Hyperparameter Tuning: Uses GridSearchCV to tune Logistic Regression parameters (C, penalty, solver, class_weight).

Tuned Model Evaluation: Evaluates the tuned Logistic Regression model and plots its ROC curve.

Visualizations
Class Distribution: Pie chart showing the proportion of healthy vs. Parkinson's cases.

Confusion Matrix: For Logistic Regression, displaying true positives, false positives, etc.

ROC Curves: Comparing AUC scores across all models and the tuned Logistic Regression model.

Limitations
Class Imbalance: Logistic Regression performs poorly despite oversampling, suggesting issues with feature selection or model suitability.

Overfitting: XGBoost's perfect training AUC (1.0) indicates potential overfitting.

Feature Selection: Limiting to 30 features may discard valuable information.

Dataset Size: A small dataset may limit model generalization.

Future Improvements
Experiment with alternative feature selection methods (e.g., Recursive Feature Elimination).

Explore additional models (e.g., Random Forest, Neural Networks).

Mitigate XGBoost overfitting using regularization or cross-validation.

Test SMOTE or other oversampling techniques.

Augment the dataset or acquire more data for better generalization.

Portfolio Links
Explore more of my work:
GitHub: github.com/Vijaybattula26

LinkedIn: linkedin.com/in/vijay-battula-29a131336

HackerRank: hackerrank.com/profile/Vijaybattula1426

LeetCode: leetcode.com/u/vijaybattula26

Contact
For questions or collaboration, reach out via LinkedIn or email at vijaybattula@example.com (mailto:vijaybattula1426@example.com).

