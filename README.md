# **Loan Default Prediction Using Deep Learning (Keras & TensorFlow)**
*A full end-to-end machine learning project exploring credit risk modelling.*

---

## **Project Overview**

This project builds a neural network using **Keras (TensorFlow)** to predict whether a loan issued by **LendingClub** will be **fully repaid** or **charged off**.  
The project includes complete data cleaning, feature engineering, exploratory analysis, and model evaluation.

It follows a **real-world data science workflow** and is one of my most comprehensive projects.

---

## **Key Objectives**

- Clean and preprocess real financial data  
- Handle missing values, categorical features, outliers, and skewed variables  
- Perform feature engineering to create the target label  
- Build and train a neural network classifier  
- Compare training/validation performance  
- Evaluate the final model using accuracy, confusion matrix, and classification report  

---

## **Dataset**

This project uses two LendingClub files:

- **`lending_club_info.csv`** — metadata describing each feature  
- **`lending_club_loan_two.csv`** — loan records used for training the model  

These include variables such as:

- loan amount  
- interest rate  
- credit grade  
- FICO scores  
- employment length  
- home ownership  
- verification status  
- debt-to-income ratios  
- payment history  

---

## **Data Preprocessing Steps**

✔ Removed irrelevant or data-leakage columns  
✔ Created binary target variable:

```python
df['loan_repaid'] = df['loan_status'].map({'Fully Paid': 1, 'Charged Off': 0})
```

✔ Dropped the original `loan_status` column afterwards  
✔ Explored distributions with visualisations  
✔ Performed one-hot encoding for categorical variables  
✔ Normalised numerical features  
✔ Split data into training and test sets  

---

## **Model Architecture (Keras Sequential Model)**

The neural network uses:

- Multiple Dense layers with ReLU activation  
- Dropout layers to reduce overfitting  
- A final sigmoid layer for binary classification  

Simplified architecture:

```
Input → Dense(128) → Dense(64) → Dense(32) → Dense(1, activation='sigmoid')
```

Loss: **binary crossentropy**  
Optimizer: **Adam**  
Evaluation: accuracy, confusion matrix, classification report  

---

## **Model Performance**

Typical performance seen in training:

- **~82–86% validation accuracy**  
- Balanced training and validation curves  
- Strong ability to distinguish between fully repaid loans and charged-off loans  

Final evaluation metrics (accuracy + confusion matrix) are included in the notebook.

---

##**What This Project Demonstrates**

- Real-world handling of large, messy financial datasets  
- Proper preprocessing pipeline  
- Neural network design, training, tuning, and evaluation  
- Understanding of credit-risk modelling  
- Full machine-learning workflow from raw data → clean model  

---

## **Repository Structure**

```
Loan-Default-Prediction-Neural-Network/
│
├── README.md
├── loan_default_prediction.ipynb
├── data/
│     ├── lending_club_info.csv
│     └── lending_club_loan_two.csv
└── models/
      └── final_model.h5   (optional)
```

If data files are too large to upload, include a note on where to download them.

---

## **How to Run**

1. Install dependencies:
   ```bash
   pip install pandas numpy matplotlib seaborn scikit-learn tensorflow
   ```

2. Open the notebook:
   ```bash
   jupyter notebook loan_default_prediction.ipynb
   ```

3. Run all cells in order.

---

## **Contact**

**Eduart Cakaj**  
📧 cakajeduart5@gmail.com  
🔗 GitHub: https://github.com/cakajeduart5-bot  

---
