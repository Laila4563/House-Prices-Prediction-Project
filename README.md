# House Price Prediction Pipeline

## 📌 Project Overview
This project implements a **machine learning pipeline** to predict house prices using various regression models. It follows a structured approach with **data preprocessing, feature engineering, model training, and inference** components. The pipeline ensures modularity, reusability, and scalability.

---

## 📂 Project Structure

```
Pipeline Dev/
│-- data/                     # Stores raw and processed datasets (not included in repo)
│-- models/                   # Stores trained machine learning models
│   │-- linear_regression_model.pkl
│   │-- ridge_regression_model.pkl
│   │-- elastic_net_regression_model.pkl
│   │-- ...
│-- pipeline/                 # Contains scripts for model training and testing
│   │-- training_pipeline.ipynb    # Jupyter Notebook for training models
│   │-- test_pipeline.py          # Script to test model predictions
│-- results/                   # Stores evaluation metrics, plots, and reports
│-- utils/                     # Contains utility scripts for data processing
│   │-- encoding.py            # Handles categorical encoding
│   │-- feat_eng.py            # Implements feature engineering techniques
│   │-- helpers.py             # Contains helper functions
│   │-- model_training.py      # Handles model training and evaluation
│   │-- preprocessing.py       # Performs data cleaning and transformations
│   │-- scaling.py             # Normalization and standardization techniques
│   │-- skewness.py            # Fixes skewed data distributions
│-- .gitignore                 # Specifies files and folders to be ignored in version control
```

---

## 🚀 How the Pipeline Works

### 1️⃣ **Data Preprocessing**
- Missing values are handled.
- Skewness in numerical features is corrected.
- Categorical variables are encoded using `encoding.py`.
- Features are scaled using `scaling.py`.

### 2️⃣ **Feature Engineering**
- Feature transformations and new feature creation are done using `feat_eng.py`.
- Feature selection techniques are applied to retain important variables.

### 3️⃣ **Model Training**
- The training pipeline (`training_pipeline.ipynb`) trains multiple regression models:
  - **Linear Regression**
  - **Ridge Regression** (including tuned version)
  - **Lasso Regression**
  - **Elastic Net Regression** (including tuned version)
  - **Polynomial Regression** (degree 2)
- The models are trained using the functions defined in `model_training.py`.
- The trained models are stored as `.pkl` files in the `models/` directory.

### 4️⃣ **Inference & Testing**
- The `test_pipeline.py` script loads the trained models and runs predictions on test data.
- Performance metrics such as MAE and R² are evaluated and saved in the `results/` directory.

---

## 🛠️ How to Run the Project

### 🔹 **Step 1: Install Dependencies**
Ensure you have Python installed, then install the required libraries

### 🔹 **Step 2: Train the Model**
Run the Jupyter Notebook to train the models:
```bash
jupyter notebook pipeline/training_pipeline.ipynb
```

### 🔹 **Step 3: Test the Model**
Run the test script to check predictions:
```bash
python pipeline/test_pipeline.py
```

---

## 📊 Results & Performance
- The trained models are evaluated using standard regression metrics.
- Model performance reports and graphs are saved in `results/`.
- The best-performing model is selected based on validation metrics.


---

### 🔥 Authors
* **[Fady Nabil](https://github.com/FadyNF)**

* **[Omneya Osama](https://github.com/omneyaosama1)**

* **[Laila Amgad](https://github.com/Laila4563)**

* **[Angie ElSegeiny](https://github.com/elsegeinyangie)**
