# Loan Prediction using Data Mining and Ensemble Learning
<img width="400" alt="image" src="https://github.com/user-attachments/assets/ccac858d-46ea-46ee-90b5-85470b4cc9a1" />      <img width="420" alt="image" src="https://github.com/user-attachments/assets/b2431c26-e273-4c96-b789-eab36dfcd14b" />

## 📌 Project Description
Comprehensive loan prediction system using data mining techniques including EDA, anomaly detection, PCA, and association rule mining. Ensemble learning pipeline integrates models such as Decision Trees, SVM, XGBoost, and Keras-based deep networks for high-accuracy predictions. Emphasis on handling imbalanced data, detecting outliers, and reducing dimensionality for robust modeling. Includes extensive model evaluation and Streamlit framework.

## 🧾 Dataset
- Source: `loan-train.csv` and `loan-test.csv`
- Contains loan application details like income, education, property area, and loan status
<img width="500" alt="image" src="https://github.com/user-attachments/assets/b2431c26-e273-4c96-b789-eab36dfcd14b" />

## 🔍 Features
- **Exploratory Data Analysis (EDA)**: Visualized approval trends, income distribution, and categorical breakdowns
- **Data Preprocessing**: Null handling, encoding, scaling with MinMaxScaler
- **Anomaly Detection**: Used Isolation Forest and Local Outlier Factor to filter out noisy samples
- **Dimensionality Reduction**: Applied PCA for feature compression and insight visualization
- **Association Rule Mining**: Captured hidden patterns in categorical features
- **Model Training**:
  - Classical ML: Decision Tree, Random Forest, Logistic Regression, SVM, KNN, AdaBoost, GradientBoosting, XGBoost
  - Neural Networks: Conv1D-based Keras model with regularization and callbacks
- **Ensemble Learning**: Final model built using stacking (meta-learner over base classifiers)
- **Model Persistence**: Saved trained models using `pickle`, `joblib`, and `keras.models`

## 🧰 Tools & Technologies
- Python (pandas, numpy, matplotlib, seaborn, scikit-learn)
- XGBoost, Keras, TensorFlow
- PCA, IsolationForest, LocalOutlierFactor
- Streamlit (for UI integration if applicable)


    ```

## 📈 Output
- Visual analytics and statistics
- Evaluation metrics (accuracy, confusion matrix, ROC curves)
- Saved models

## ⚠️ Disclaimer
This model is built on historical loan data and may not generalize to all real-world scenarios. Always validate using domain-specific checks.

---

