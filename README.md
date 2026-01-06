# churn-customer-streamlit
Customer churn analysis and prediction project using Python, XGBoost, Streamlit, and Power BI.

# Customer Churn Analysis & Prediction
##  Project Overview
This project focuses on analyzing and predicting customer churn in the telecommunications industry.
Beyond model building, the project emphasizes **business insights, decision-making, and real-world deployment**.

An interactive Streamlit application and Power BI dashboards are used to present insights and support churn prevention strategies.

---

## Dataset
- **Source:** IBM Telco Customer Churn (Kaggle)
- **Size:** 7,043 customers
- **Features:** Demographics, services, contract details, payment methods, and charges
- **Target variable:** Churn (Yes / No)

---

## Key Business Insights
- Customers on **month-to-month contracts** have the highest churn risk
- **Fiber optic internet users** churn more frequently than DSL users
- Customers without **Online Security or Tech Support** are more likely to churn
- Long-tenured customers with **low monthly charges** also show churn tendencies

---

## Modeling Approach
- Compared multiple models: Logistic Regression, Decision Tree, Random Forest, and XGBoost
- Feature selection and optimization were applied
- **XGBoost** achieved the best performance after hyperparameter tuning

**Final Model Performance:**
- Accuracy: ~77%
- Balanced precision and recall

---

## Application & Visualization
- **Streamlit App:** Interactive churn analysis and prediction
- **Power BI Dashboard:** Business-focused insights and churn drivers

---

##  Tech Stack
- Python, Pandas, NumPy
- Scikit-learn, XGBoost
- Streamlit
- Power BI
- Plotly
