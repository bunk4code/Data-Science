# 🧑‍💻 **Data Science - Machine Learning Algorithms Practice**  
### Repository of Machine Learning Projects and Practice Files  
---
![Machine Learning](https://img.shields.io/badge/Machine%20Learning-Practice%20Files-orange) ![Python](https://img.shields.io/badge/Python-Code-blue) ![Algorithms](https://img.shields.io/badge/Algorithms-Learner-yellow)

Welcome to my **Data Science** repository! Here, I upload practice files where I explore and implement various **Machine Learning** algorithms. Each file contains step-by-step code, explanations, and examples to understand the algorithm at a deeper level.

---

## 🛠️ **Methodology of Learning**  
My approach to learning **Machine Learning** is based on:

1. **Concept Understanding**: Exploring the theory and mathematical formulation of algorithms.
2. **Preprocessing**: Handling missing data, scaling, encoding, and feature engineering.
3. **Implementation**: Coding the algorithm in Python using `scikit-learn` and tuning its hyperparameters.
4. **Evaluation**: Analyzing the performance with various metrics and visualizations.

---

## 📂 **Algorithms in this Repository**

### 1. 🔍 **Anomaly Detection**  
- **Files:**  
  - `AnomlyDetection.ipynb`  
  - `DBScan_AnomlyDetection.ipynb`  
  - `RandomForest_AnomalyDetection.ipynb`
  
- **Overview**: Anomaly detection algorithms help identify unusual data points, which may indicate fraud, network intrusion, or other rare events.
  
- **Techniques Used**:  
  - **DBScan**: A density-based clustering algorithm to detect anomalies. It identifies clusters based on the density of points and marks outliers as anomalies.
  - **Random Forest**: An ensemble method where decision trees are trained and combined to classify whether a point is an anomaly.

- **Mathematics Behind**:  
  **DBScan Algorithm**:  
  - A point is a core point if at least `minPts` points are within a distance `ε` (epsilon).
  - Any point within the neighborhood of a core point is part of a cluster, while isolated points are labeled as anomalies.



---![network-anomalies](https://github.com/user-attachments/assets/a2aa825c-1f7e-4954-a7f2-7065bf0f87ce)


### 2. 🌳 **Decision Tree**  
- **Files:**  
  - `DecisionTree_Classification.ipynb`  
  - `DecisionTree_Regression.ipynb`

- **Overview**: A **Decision Tree** splits the dataset into branches at each node based on feature values, making predictions at the leaf nodes.

- **Techniques Used**:  
  - **Classification Trees**: Predict a categorical outcome.
  - **Regression Trees**: Predict continuous values.

- **Mathematics Behind**:  
  - Decision Trees minimize the **Gini impurity** or **Information Gain** at each split.
  - For **Regression Trees**, the splitting criterion is based on reducing the **mean squared error (MSE)**.

  **Formulas**:  
  - **Gini Impurity**:  
    \[
    Gini = 1 - \sum_{i=1}^{n} p_i^2
    \]
  - **MSE (for Regression Trees)**:  
    \[
    MSE = \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y})^2
    \]
![0_ZhAmjNNuUUDU3kc0](https://github.com/user-attachments/assets/95d64432-0e25-44ca-a259-24bec2b82a20)


---

### 3. 🤖 **Support Vector Machines (SVM)**  
- **Files:**  
  - `SupportVectorMachineClassification.ipynb`  
  - `SuportVectorMachine_Regression.ipynb`

- **Overview**: SVM finds a **hyperplane** that maximizes the margin between data points of different classes or predicts continuous outcomes using a linear or non-linear approach.

- **Mathematics Behind**:  
  - For **classification**, SVM solves the optimization problem:
    \[
    \text{Maximize} \ \frac{2}{||w||} \ \text{subject to} \ y_i(w^T x_i + b) \geq 1
    \]
  - For **regression**, SVM minimizes the error while fitting a function that has deviations within a certain margin \( \epsilon \).
  - ![1718266257027]
  (https://github.com/user-attachments/assets/cb0fe2a4-a7e3-40a1-8b82-dc03ba76145e)


---

### 4. 📊 **Linear & Logistic Regression**  
- **Files:**  
  - `LinearRegressionPractice.ipynb`  
  - `LogisticRegression.ipynb`

- **Overview**:  
  - **Linear Regression** models the relationship between a dependent variable and one or more independent variables.
  - **Logistic Regression** is used for binary classification problems where the output is a probability.

- **Mathematics Behind**:  
  - **Linear Regression**:  
    \[
    y = \beta_0 + \beta_1 x_1 + \beta_2 x_2 + \dots + \beta_n x_n + \epsilon
    \]
  - **Logistic Regression**:  
    \[
    P(y=1|x) = \frac{1}{1 + e^{-(\beta_0 + \beta_1 x_1 + \dots + \beta_n x_n)}}
    \]
![images](https://github.com/user-attachments/assets/f8a1181f-d6a8-4430-a386-13f5017c6643)



---

### 5. 🧮 **Ridge & Lasso Regression**  
- **Files:**  
  - `Ridge&LassoRegression.ipynb`

- **Overview**: Ridge and Lasso are regularization techniques used to prevent overfitting by adding a penalty to the size of the coefficients.

- **Mathematics Behind**:  
  - **Ridge Regression**:  
    \[
    \min ||y - X\beta||^2_2 + \lambda ||\beta||^2_2
    \]
  - **Lasso Regression**:  
    \[
    \min ||y - X\beta||^2_2 + \lambda ||\beta||_1
    \]
  These equations add regularization to the loss function to reduce overfitting.

![1603631920705](https://github.com/user-attachments/assets/19ef5d86-f532-4987-b2c3-43a4856493ae)


---

### 6. 🌲 **Random Forest**  
- **Files:**  
  - `RandomForest.ipynb`

- **Overview**: Random Forest is an ensemble learning method that builds multiple decision trees and aggregates their predictions to improve accuracy and reduce overfitting.

- **Mathematics Behind**:  
  - The output of a Random Forest is the **majority vote** for classification or the **average** of predictions for regression from multiple trees.
![VPLDRjim48NtEiN0NNILmXewTG84HcB7MXLTf5vcGqkK9PWc3Of6a1GzJU_JCvNKZpNwYRmGu3UFJpEVxD5ZORcnbvCCYcEkpjpm4yXS2Vj-5g2DpJGf92Bb5saZhUol_4D0k215fHDHY3FWd8Y6gwCN5rGiCfxejrmT8EMo4DIkBohPQAE4WV2M5](https://github.com/user-attachments/assets/ad5ddb0c-2199-4872-85d4-b50f0eedb84d)


---

### 7. 📘 **Naive Bayes**  
- **Files:**  
  - `NaiveBayes.ipynb`

- **Overview**: Naive Bayes is a probabilistic algorithm based on **Bayes' Theorem**, assuming independence between features. It’s particularly useful for **text classification**.

- **Mathematics Behind**:  
  - **Bayes' Theorem**:  
    \[
    P(y|x) = \frac{P(x|y) P(y)}{P(x)}
    \]

 ![1685792014092](https://github.com/user-attachments/assets/6576f5b8-fc2a-41fb-bde4-0c14f1c3f2a0)


---

## 🎯 **Next Steps**  
I am constantly updating this repository as I learn new techniques and algorithms. Check back for more updates! Feel free to collaborate and share your ideas.

---

## 📈 **Connect with Me!**  
[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-blue)](https://www.linkedin.com/in/your-profile)  
[![GitHub](https://img.shields.io/badge/GitHub-Follow-lightgrey)](https://github.com/your-github-username)

---
