# Titanic Survival Prediction from Scratch

An end-to-end Machine Learning pipeline developed to analyze the historical Titanic passenger manifest and predict passenger survival. The core of this project is a **custom Decision Tree Classifier built completely from scratch** using Python and NumPy, emphasizing foundational algorithmic design, data tidying, and hyperparameter tuning.

## 📌 Project Overview
This project avoids high-level abstractions like Scikit-Learn's `tree` module to demonstrate the underlying mechanics of supervised learning algorithms. By evaluating data distributions and managing chaotic classes, the custom model maps out paths of binary questions using mathematical metrics to determine historical survival outcomes.

### Core Objectives:
* **Exploratory Data Analysis (EDA):** Identify distinct survival gaps across categorical features.
* **Data Tidying & Transforming:** Address missing structural indicators and execute One-Hot Encoding for numeric translation.
* **Algorithmic Construction:** Implement Gini Impurity, Information Gain, and Recursive Tree Generation routines using raw NumPy array manipulation.
* **Performance Optimization:** Evaluate boundaries across various hyperparameters to combat the Bias-Variance Tradeoff (Overfitting vs. Underfitting).

---

## 📊 Pipeline Architecture

The machine learning pipeline progresses through the following stages:

1.  **Data Intake:** Loads the 891-row passenger dataset containing 15 initial feature metrics.
2.  **Feature Cleansing:** Drops target leakage rows (`alive`) and columns missing critical volume (`deck`). Fills missing metrics with stable estimators (`median`, `mode`).
3.  **One-Hot Encoding:** Translates qualitative columns (`sex`, `embarked`) into binary indicator flags.
4.  **Stratification:** Shuffles and reserves a controlled 20% validation split (`X_test`, `y_test`) to guarantee strict evaluation integrity.
5.  **Model Induction:** Executes recursive binary splits maximizing Information Gain.
6.  **Hyperparameter Tuning:** Sweeps depth parameters to isolate optimal test accuracy.

---

## 🛠️ Algorithmic Mechanics (Under the Hood)

### 1. Gini Impurity ($\text{Gini}$)
Measures statistical "chaos" or misclassification probability within any given group node. A perfectly homogenous node equals `0`, while a perfectly split 50/50 distribution peaks at `0.5`.
$$\text{Gini} = 1 - \sum_{i=1}^{C} (p_i)^2$$

### 2. Information Gain ($\text{IG}$)
Calculates the numerical reduction in chaos achieved by breaking a dataset down into two sub-branches via a specific condition mask. The algorithm iterates through every threshold boundary to isolate the highest available value.

$$\text{IG} = \text{Gini}_{\text{parent}} - \left( \frac{N_{\text{left}}}{N_{\text{total}}} \cdot \text{Gini}_{\text{left}} + \frac{N_{\text{right}}}{N_{\text{total}}} \cdot \text{Gini}_{\text{right}} \right)$$

### 3. Structural Constraints (Stopping Rules)
To ensure the tree maintains generalization capability on unseen records, recursion halts immediately when any of the following parameters are satisfied:
* `max_depth`: Maximum depth levels allowed from the root node.
* `min_samples_split`: The minimal passenger volume required to warrant a distinct sub-split.
* `Purity`: The target node contains solely survivors (`1`) or victims (`0`).

---

## 📈 Performance & Tuning

### Baseline Comparison
* **Majority Class Baseline:** ~62% Accuracy (Blindly guessing "Died" for all test passengers based on majority class distributions).
* **Custom Model (Depth = 3):** **79.89%** Accuracy | **0.7391** F1-Score.

### The Bias-Variance Tradeoff (Depth Optimization)
Testing the tree across depths ranging from 1 to 10 reveals the classic machine learning optimization curve:

* **Depths 1–2 (Underfitting):** High bias. The tree is structurally restricted from mapping multi-layered feature dependencies (e.g., assessing Passenger Class and Age simultaneously).
* **Depth 7 (The Sweet Spot):** Achieves peak model validation performance of **81.01% Testing Accuracy**.
* **Depths 8–10 (Overfitting):** High variance. The tree gains additional parameters and begins memorizing unique, noisy traits isolated to individuals in the training set, causing testing performance on hidden samples to degrade sharply.

---

## 🚀 Repository Structure

```text
├── .gitignore
├── README.md                 <- Project documentation and analysis
└── <roll_number>_S<sec>_A3.ipynb   <- Complete Jupyter Notebook code & visualizations
