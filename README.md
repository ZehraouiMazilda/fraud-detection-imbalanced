# Check Fraud Detection - Imbalanced Learning Project

This repository contains the source code and analysis for the **Big Data Mining** project (Master 2 SISE - University of Lyon 2), supervised by Guillaume Metzler.

## 📋 Project Overview
[cite_start]The goal of this project is to develop a machine learning pipeline to detect fraudulent check transactions (unpaid or stolen checks) within a massive, real-world retail dataset[cite: 1, 6].

[cite_start]The primary challenge lies in the **imbalanced nature of the data** (frauds are rare compared to normal transactions)[cite: 1], requiring specific strategies such as sampling techniques and cost-sensitive learning.

## 🎯 Objectives
The project is divided into two main performance goals:
1.  [cite_start]**Technical Performance:** Maximize the **F-Measure** to balance Precision and Recall[cite: 24].
2.  **Business Performance:** Maximize the **Profit Margin** by implementing a custom cost matrix. [cite_start]We aim to optimize the threshold to balance the gain of accepted valid transactions against the financial loss of missed frauds[cite: 35, 37].

## 🗂️ Data & Experimental Protocol
[cite_start]The dataset covers 10 months of transactions (Feb 2017 – Nov 2017)[cite: 18].
*Note: Raw data is not included in this repository due to size and confidentiality.*

### Strict Temporal Split
[cite_start]To simulate a real-world production environment and avoid temporal data leakage, we strictly adhere to the following split[cite: 20, 21, 22]:
* **Train Set:** Feb 1, 2017 – Aug 31, 2017
* **Test Set:** Sep 1, 2017 – Nov 30, 2017

[cite_start]⚠️ **Critical Constraint:** The variable `CodeDecision` is removed from the predictive modeling phase as it represents post-transaction information[cite: 16].

## 🛠️ Methodology
[cite_start]We implement a robust workflow including at least 5 distinct procedures[cite: 28]:
* **Preprocessing:** Feature selection and exclusion of data leakage variables.
* [cite_start]**Sampling Strategies:** Testing Undersampling (Random, Tomek Links) and Oversampling (SMOTE, ADASYN) to handle class imbalance[cite: 30].
* [cite_start]**Modeling:** Comparing various algorithms (e.g., Random Forest, Gradient Boosting/XGBoost, SVM, Deep Learning)[cite: 32].
* [cite_start]**Cost-Sensitive Learning:** Adjusting class weights and decision thresholds to maximize business gain[cite: 30].

## 📂 Project Structure
* `/data`: (Excluded via .gitignore) Raw and processed data files.
* `/notebooks`: Jupyter notebooks for EDA (Exploratory Data Analysis) and experiments.
* `/src`: Python scripts for modular code (preprocessing, training, evaluation).
* `/reports`: Generated figures and final analysis reports.

## 👥 Authors
* **Mazilda**
* *[Name of your teammate]*

---
*Project carried out as part of the Master 2 SISE curriculum (2025-2026).*
