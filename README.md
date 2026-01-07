# Customer Churn Prediction Project

## 📌 Project Overview

This project focuses on predicting **customer churn** using machine learning techniques. Customer churn refers to the likelihood of customers discontinuing a service. The objective of this project is to build an end-to-end ML solution that can help businesses identify high-risk customers and take proactive retention measures.

The project is designed and implemented with **industry best practices**, including clean project structure, version control hygiene, and deployment readiness using **Streamlit**.

---

## 🚀 Key Features

* End-to-end **Customer Churn Prediction** system
* Multiple ML models trained during experimentation
* Best-performing model selected for production
* Interactive **Streamlit web application**
* Deployment-ready structure (GitHub + Streamlit Cloud)
* Clean repository using a professional `.gitignore`

---

## 🧠 Machine Learning Workflow

1. Data loading and preprocessing
2. Feature engineering and selection
3. Model training and evaluation
4. Model comparison
5. Selection of the best-performing model
6. Model serialization using `joblib`
7. Web app deployment using Streamlit

---

## 📂 Project Structure

```
customer_churn/
│
├── models/
│   ├── best_model.joblib          # Final selected ML model
│   └── feature_columns.json       # Feature metadata used during training
│
├── src/                            # Source code (preprocessing, training, utils)
├── data/                           # Dataset (excluded from GitHub if large/sensitive)
├── app.py                          # Streamlit application entry point
├── requirements.txt                # Project dependencies
├── .gitignore                      # Git ignore rules
└── README.md                       # Project documentation
```

> ⚠️ Note: Other experimental models are stored **outside the repository** to keep the project lightweight and deployment-friendly.

---

## 🛠️ Tech Stack

* **Programming Language:** Python
* **Libraries:**

  * pandas
  * numpy
  * scikit-learn
  * joblib
  * streamlit
* **Version Control:** Git & GitHub
* **Deployment:** Streamlit Cloud

---

## ▶️ How to Run the Project Locally

### 1️⃣ Clone the Repository

```bash
git clone <repository-url>
cd customer_churn
```

### 2️⃣ Create a Virtual Environment

```bash
python -m venv venv
source venv/bin/activate   # For Windows: venv\\Scripts\\activate
```

### 3️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

### 4️⃣ Run the Streamlit App

```bash
streamlit run app.py
```

---

## 🌐 Deployment

The application is deployed using **Streamlit Cloud**. Only the final production-ready model (`best_model.joblib`) is included in the repository to ensure smooth and fast deployment.

---

## 📈 Use Case

This project can be used by:

* Telecom companies
* Subscription-based businesses
* SaaS platforms
* Banking & financial services

To identify customers who are likely to churn and take preventive actions.

---

## 🧪 Model Management Strategy

* Multiple models were trained and evaluated during experimentation
* Only the **best-performing model** is version-controlled
* Experimental models are excluded to reduce repository size and avoid deployment issues

This approach follows **real-world ML engineering practices**.


---

⭐ If you find this project useful, consider giving it a star!
