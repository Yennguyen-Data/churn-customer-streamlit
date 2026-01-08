
import streamlit as st
import pandas as pd
import joblib
import plotly.express as px
from ortools.linear_solver import pywraplp


# load data
df = pd.read_csv('Telco_Customer_Churn.csv')
df_model_comparison = pd.read_csv('model_compare_full.csv')
df_model_comparison_10f = pd.read_csv('model_compare_10f.csv')

report_xgb = pd.read_csv('report_model_xgb.csv')
y_train = pd.read_csv("y_train.csv").iloc[:,0]
y_test = pd.read_csv("y_test.csv").iloc[:,0]
model = joblib.load("churn_xgb_model.pkl")

features_importance = joblib.load("base_features.pkl")


predict_pipeline = joblib.load("churn_xgb_model.pkl")
#Building function

# GUI

st.title('Churn Prediction Project')
st.write('## Telco Data Analysis')

menu = ['Project Overview', 'Data Insights', 'Build Model', 'New Customer Prediction']
#Sidebar Menu
choice = st.sidebar.selectbox('Menu', menu)
#Sidebar Content
st.sidebar.markdown('''
### ***About the Project***
- Project title: Customer Churn Prediction
- Conducted Date: 01/02/2026 
- Implemented By: Hai Yen Nguyen
                              ''')

if choice == 'Project Overview':
    tab1, tab2, tab3 = st.tabs(['Project Introduction', 'Dataset', 'Algorithms'])
    with tab1:
        st.image('tab1_1.png')
        st.write("""
                    ### Project Background:
                    In the telecommunications (Telco) industry, customer loyalty plays a 
                 critical role in determining a company’s long-term sustainability. 
                 This project was developed with the objective of building an early warning system to 
                 identify customers who are at high risk of leaving the telecom service. Such a system 
                 enables relevant departments to intervene in a timely manner through targeted promotional offers or service upgrades.\n
                    By utilizing customer information and telecom usage behavior data, this project analyzes overall usage patterns 
                 and trends, helping businesses gain a more comprehensive understanding of their customer landscape. Based on these insights,
                  a predictive model is constructed to identify customers with a high likelihood of churn.
                 Finally, the next stage is to recommend optimal retention strategies. These strategies aim to reduce the likelihood of churn while maximizing the company’s net revenue.
                    #### General Information:
                    - Dataset: Telco Customer Churn dataset
                    - Programming Language: Python
                    - Model Algorithm: XGBoost""")
    
    with tab2:
        st.write("""
The dataset used in this project is provided by IBM Business Analytics and is publicly available on the Kaggle platform. Although it is a sample dataset, it accurately reflects many real-world challenges currently faced by telecommunications service providers.

- Scale: The dataset contains information on 7,043 customers located in California during the third quarter.
- Features: The dataset consists of 21 variables, covering the following main categories:
   - Demographics: Gender, senior citizen status, dependents, and related attributes.
   - Services: Phone service, multiple lines, Internet service (DSL/Fiber optic), online security, online backup, technical support, and streaming TV/movies.
   - Account Information: Customer tenure, contract type, payment method, monthly charges, and total charges.
- Target Variable: Churn (Yes/No), indicating whether a customer left the service in the previous month.
""")

    with tab3:
        st.markdown("""
The entire data analysis and model development process was conducted using Python and popular libraries.

- **Pandas** and **NumPy** were used for data manipulation, storage, and display.
- **Matplotlib** was used for data visualization.
- **Scikit-learn (sklearn)** was used for building pipelines, data preprocessing, handling class imbalance (via **imblearn**), and model evaluation.
- The main algorithm for prediction is **XGBoost**, while other algorithms such as **Logistic Regression, Decision Tree,** and **Random Forest** were also implemented for performance comparison.
- **OR-Tools** was used for optimization of retention strategy allocation to maximize expected net revenue under budget constraints.
                    
Besides, for supporting visualization, **Power BI** was also used.
""")


if choice == 'Data Insights':
    tab4, tab5, tab6, tab7 = st.tabs(
        ['Demographic', 'Services', 'Account Information', 'Target Variable']
    )

    # ==============================
    # TAB 4: DEMOGRAPHIC ANALYSIS
    # ==============================
    with tab4:
        st.header("Demographic Analysis")
        st.caption("Understanding how customer demographics relate to churn behavior")

        # --------------------------------
        # Overview Metrics
        # --------------------------------
        total_customers = df.shape[0]
        churn_rate = (df['Churn'] == 'Yes').mean() * 100

        col_m1, col_m2 = st.columns(2)
        col_m1.metric("Total Customers", f"{total_customers:,}")
        col_m2.metric("Overall Churn Rate", f"{churn_rate:.1f}%")

        st.divider()

        # =====================================================
        # 1. Gender & Churn
        # =====================================================
        st.subheader("1. Gender vs Churn")

        gender_churn = (
            df.groupby('gender')['Churn']
            .value_counts(normalize=True)
            .unstack() * 100
        )

        col1, col2 = st.columns([2, 3])

        with col1:
            st.write("**Key Findings**")
            st.write(
                f"""
                - Gender distribution is nearly **balanced**.
                - Churn rate difference between males and females is **minimal**.
                - Gender is **not a strong churn driver** in this dataset.
                """
            )

        with col2:
            fig_gender = px.bar(
                gender_churn,
                barmode='group',
                labels={'value': 'Percentage (%)', 'gender': 'Gender'},
                title="Churn Rate by Gender"
            )
            st.plotly_chart(fig_gender, use_container_width=True)

        st.info(
            """
            Gender-based targeting is unlikely to significantly reduce churn.
            Retention strategies should prioritize behavioral and contractual factors instead.
            """
        )

        st.divider()

        # =====================================================
        # 2. Senior Citizens & Churn
        # =====================================================
        st.subheader("2. Senior Citizens vs Churn")

        senior_churn = (
            df.groupby('SeniorCitizen')['Churn']
            .value_counts(normalize=True)
            .unstack() * 100
        )

        senior_rate = senior_churn.loc[1, 'Yes']
        non_senior_rate = senior_churn.loc[0, 'Yes']
        churn_ratio = senior_rate / non_senior_rate

        col3, col4 = st.columns([2, 3])

        with col3:
            st.write("**Key Findings**")
            st.write(
                f"""
                - **Senior customers** show a churn rate of **{senior_rate:.1f}%**.
                - **Non-senior customers** churn at **{non_senior_rate:.1f}%**.
                - Seniors are **{churn_ratio:.1f}x more likely** to churn.
                """
            )

        with col4:
            fig_senior = px.bar(
                senior_churn,
                barmode='group',
                labels={
                    'value': 'Percentage (%)',
                    'SeniorCitizen': 'Is Senior Citizen?'
                },
                title="Churn Rate: Senior vs Non-Senior Customers"
            )
            st.plotly_chart(fig_senior, use_container_width=True)

        st.info(
            """
            Senior citizens represent a **high-risk churn segment**.
            
            **Recommended Actions**
            - Introduce simplified plans with transparent pricing
            - Provide senior-specific loyalty programs
            - Improve customer support accessibility for older users
            """
        )

        st.divider()

   #-------------------------------------------------------------------
        # 3. Family Status & Churn
        
        st.subheader("3. Family Status vs Churn")

        col5, col6 = st.columns(2)

        with col5:
            partner_churn = (
                df.groupby('Partner')['Churn']
                .value_counts(normalize=True)
                .unstack() * 100
            )

            fig_partner = px.bar(
                partner_churn,
                barmode='group',
                labels={'value': 'Percentage (%)', 'Partner': 'Has Partner?'},
                title="Churn Rate by Partner Status"
            )
            st.plotly_chart(fig_partner, use_container_width=True)

        with col6:
            dependents_churn = (
                df.groupby('Dependents')['Churn']
                .value_counts(normalize=True)
                .unstack() * 100
            )

            fig_dependents = px.bar(
                dependents_churn,
                barmode='group',
                labels={'value': 'Percentage (%)', 'Dependents': 'Has Dependents?'},
                title="Churn Rate by Dependents Status"
            )
            st.plotly_chart(fig_dependents, use_container_width=True)

        st.info(
            """
            Customers **without partners or dependents** are more likely to churn.
            
            **Interpretation**
            - Lower switching costs
            - Less emotional or contractual attachment
            
            **Recommendation**
            - Target single customers with bundled offers or long-term incentives
            """
        )
    #-------------------------------------------------------------------

    # TAB 5: SERVICES ANALYSIS

    with tab5:
        st.header("Service Usage & Churn Analysis")
        st.caption("Identifying service-related churn drivers in the telecom business")

        #-------------------------------------------------------------------
        # 1. Internet Service vs Churn
        
        st.subheader("1. Internet Service Type")

        internet_churn = (
            df.groupby('InternetService')['Churn']
            .value_counts(normalize=True)
            .unstack() * 100
        )

        col1, col2 = st.columns([2, 3])

        with col1:
            st.write("**Key Findings**")
            st.write(
                """
                - Fiber Optic customers exhibit the **highest churn rate**.
                - DSL users show a significantly **lower churn tendency**.
                - Customers without internet service have the **lowest churn risk**.
                """
            )

        with col2:
            fig_internet = px.bar(
                internet_churn,
                barmode='group',
                labels={'value': 'Percentage (%)', 'InternetService': 'Internet Service Type'},
                title="Churn Rate by Internet Service"
            )
            st.plotly_chart(fig_internet, use_container_width=True)

        st.info(
            """
            Fiber Optic is a premium service, yet it shows the highest churn rate.
            This suggests potential issues related to pricing, service stability, or unmet customer expectations.
            
            **Recommendation**
            - Review Fiber pricing strategy
            - Improve service reliability and customer support for Fiber users
            """
        )

        st.divider()

        #-------------------------------------------------------------------
        # 2. Support & Protection Services
        
        st.subheader("2. Support & Protection Services")

        support_services = [
            'OnlineSecurity',
            'OnlineBackup',
            'DeviceProtection',
            'TechSupport'
        ]

        col3, col4 = st.columns(2)

        with col3:
            selected_service = st.selectbox(
                "Select a service to analyze:",
                support_services
            )

        service_churn = (
            df.groupby(selected_service)['Churn']
            .value_counts(normalize=True)
            .unstack() * 100
        )

        with col4:
            fig_support = px.bar(
                service_churn,
                barmode='group',
                labels={'value': 'Percentage (%)', selected_service: 'Service Status'},
                title=f"Churn Rate by {selected_service}"
            )
            st.plotly_chart(fig_support, use_container_width=True)

        st.info(
            """
            Customers who do **NOT** subscribe to support or protection services
            consistently show **significantly higher churn rates**.
            
            **Recommendation**
            - Bundle security and tech support with internet plans
            - Promote add-on services during onboarding to reduce early churn
            """
        )

        st.divider()

        #-------------------------------------------------------------------
        # 3. Entertainment Services (Streaming)
        
        st.subheader("3. Entertainment Services")

        col5, col6 = st.columns(2)

        streaming_tv_churn = (
            df.groupby('StreamingTV')['Churn']
            .value_counts(normalize=True)
            .unstack() * 100
        )

        streaming_movie_churn = (
            df.groupby('StreamingMovies')['Churn']
            .value_counts(normalize=True)
            .unstack() * 100
        )

        with col5:
            fig_tv = px.bar(
                streaming_tv_churn,
                barmode='group',
                labels={'value': 'Percentage (%)', 'StreamingTV': 'Streaming TV'},
                title="Churn Rate by Streaming TV"
            )
            st.plotly_chart(fig_tv, use_container_width=True)

        with col6:
            fig_movies = px.bar(
                streaming_movie_churn,
                barmode='group',
                labels={'value': 'Percentage (%)', 'StreamingMovies': 'Streaming Movies'},
                title="Churn Rate by Streaming Movies"
            )
            st.plotly_chart(fig_movies, use_container_width=True)

        st.info(
            """
            Entertainment services (Streaming TV & Movies) show **limited impact** on churn.
            While popular, they do **not strongly improve customer retention**.
            
            **Recommendation**
            Streaming services should be treated as **value-added features**, not primary retention levers.
            """
        )

        st.divider()

        #-------------------------------------------------------------------
        # 4. Service Bundling Effect
        
        st.subheader("4. Service Bundling Effect")

        service_columns = [
            'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
            'TechSupport', 'StreamingTV', 'StreamingMovies'
        ]

        df['TotalServices'] = df[service_columns].apply(
            lambda x: (x == 'Yes').sum(), axis=1
        )

        bundle_churn = (
            df.groupby('TotalServices')['Churn']
            .value_counts(normalize=True)
            .unstack() * 100
        )

        fig_bundle = px.line(
            bundle_churn,
            markers=True,
            labels={'value': 'Percentage (%)', 'TotalServices': 'Number of Services'},
            title="Churn Rate by Number of Subscribed Services"
        )

        st.plotly_chart(fig_bundle, use_container_width=True)

        st.info(
            """
            Customers with **more subscribed services churn significantly less**.
            
            **Business Recommendation**
            - Encourage multi-service subscriptions
            - Design bundled offers to increase customer stickiness
            """
        )
        #-------------------------------------------------------------------
    # TAB 6: ACCOUNT INFORMATION
    
    with tab6:
        st.header("Account Information & Churn Drivers")
        st.caption("Analyzing contractual, lifecycle, pricing, and payment-related churn factors")

        #-------------------------------------------------------------------
        # 1. Contract Type (Strongest Churn Driver)
        
        st.subheader("1. Contract Type vs Churn")

        contract_churn = (
            df.groupby('Contract')['Churn']
            .value_counts(normalize=True)
            .unstack() * 100
        )

        mtm_rate = contract_churn.loc['Month-to-month', 'Yes']
        one_year_rate = contract_churn.loc['One year', 'Yes']
        two_year_rate = contract_churn.loc['Two year', 'Yes']

        col1, col2 = st.columns([2, 3])

        with col1:
            st.write("**Key Findings**")
            st.write(
                f"""
                - Month-to-month customers churn at **{mtm_rate:.1f}%**
                - One-year contracts churn at **{one_year_rate:.1f}%**
                - Two-year contracts churn at **{two_year_rate:.1f}%**
                """
            )

        with col2:
            fig_contract = px.bar(
                contract_churn,
                barmode='group',
                labels={'value': 'Percentage (%)', 'Contract': 'Contract Type'},
                title="Churn Rate by Contract Type"
            )
            st.plotly_chart(fig_contract, use_container_width=True)

        st.info(
            """
            Contract type is the **single strongest churn driver**.
            Customers with low commitment (month-to-month) are significantly more likely to churn.
            
            **Recommendation**
            - Incentivize long-term contracts
            - Offer loyalty discounts for contract upgrades
            """
        )

        st.divider()

        #-------------------------------------------------------------------
        # 2. Customer Tenure (Lifecycle Churn)
        
        st.subheader("2. Customer Tenure & Lifecycle Risk")

        tenure_bins = [0, 6, 12, 24, 60]
        tenure_labels = ['0–6 months', '6–12 months', '12–24 months', '24+ months']
        df['TenureGroup'] = pd.cut(df['tenure'], bins=tenure_bins, labels=tenure_labels)

        tenure_churn = (
            df.groupby('TenureGroup')['Churn']
            .value_counts(normalize=True)
            .unstack() * 100
        )

        fig_tenure = px.line(
            tenure_churn,
            markers=True,
            labels={'value': 'Percentage (%)', 'TenureGroup': 'Tenure Group'},
            title="Churn Rate Across Customer Lifecycle"
        )

        st.plotly_chart(fig_tenure, use_container_width=True)

        st.info(
            """
            Churn risk is **highest during the early customer lifecycle** and declines sharply over time.
            
            **Recommendation**
            - Focus retention efforts within the first 6 months
            - Improve onboarding and early customer experience
            """
        )

        st.divider()

        #-------------------------------------------------------------------
        # 3. Pricing Sensitivity
        
        st.subheader("3. Pricing & Charges")

        col3, col4 = st.columns(2)

        with col3:
            fig_monthly = px.box(
                df,
                x='Churn',
                y='MonthlyCharges',
                labels={'MonthlyCharges': 'Monthly Charges ($)'},
                title="Monthly Charges vs Churn"
            )
            st.plotly_chart(fig_monthly, use_container_width=True)

        with col4:
            fig_total = px.box(
                df,
                x='Churn',
                y='TotalCharges',
                labels={'TotalCharges': 'Total Charges ($)'},
                title="Total Charges vs Churn"
            )
            st.plotly_chart(fig_total, use_container_width=True)

        st.info(
            """
            Customers who churn tend to have **higher monthly charges** but **lower total charges**,
            indicating early churn driven by **price sensitivity**.
            
            **Recommendation**
            - Review pricing for high-cost plans
            - Offer early-stage discounts for high-risk customers
            """
        )

        st.divider()

        #-------------------------------------------------------------------
        # 4. Payment Method
        
        st.subheader("4. Payment Method vs Churn")

        payment_churn = (
            df.groupby('PaymentMethod')['Churn']
            .value_counts(normalize=True)
            .unstack() * 100
        )

        fig_payment = px.bar(
            payment_churn,
            barmode='group',
            labels={'value': 'Percentage (%)', 'PaymentMethod': 'Payment Method'},
            title="Churn Rate by Payment Method"
        )

        st.plotly_chart(fig_payment, use_container_width=True)

        st.info(
            """
            Customers using **Electronic Check** have the highest churn rate.
            
            **Interpretation**
            - Higher payment friction
            - Lower commitment compared to automatic payment methods
            
            **Recommendation**
            - Encourage auto-pay adoption
            - Provide incentives for switching payment methods
            """
        )
    #-------------------------------------------------------------------
    # TAB 7: EXECUTIVE SUMMARY
    
    with tab7:
        st.header("Executive Summary & Key Churn Drivers")
        st.caption("High-level insights and actionable recommendations derived from the analysis")

        #-------------------------------------------------------------------
        # 1. Churn Snapshot
        
        st.subheader("Churn Snapshot")
        st.image('general_dashboard.png')
        st.divider()

        st.markdown("""
## Based on the dashboard and insights, key information is summarized as follows:

### 1. Key Performance Indicators (KPIs)
- **Churn scale:** 1,869 customers have left, representing **27% of the total 7,043 customers**.  
- **Financial impact:** These churns resulted in a **monthly revenue loss of $139.13K**.  
- **Critical tenure:** The average customer tenure is **32.37 months**, but the scatter plot indicates that the **highest risk is concentrated among customers with very low tenure (<6 months)**.

### 2. Detailed Table Insights
The table highlights the **highest-risk customer segments**:

- **Most at-risk configuration:** Customers on **Month-to-month contracts**, using **Fiber optic Internet**, and **without Online Security or Tech Support**.  
- **Leading churn segment:** Within this group, customers using **Electronic Check payments** (with no security or support) contribute **over 550 churned customers** out of the total 1,869.

### 3. Cost vs. Loyalty Correlation
- **Low Monthly Charges:** Even long-tenured customers tend to churn if they lack additional support services.  
- **High Total Charges:** Represented by large light-blue bubbles at the high end of the tenure axis, these **high-value customers** are critical to protect, as they contribute significantly to total revenue but show early signs of churn risk.
""")

        st.divider()

        #-------------------------------------------------------------------
        # 2. Top Churn Drivers (EDA-Based Ranking)
        
        st.subheader("Top Churn Drivers")

        churn_drivers = {
            "Contract Type (Month-to-Month)": "★★★★★",
            "Short Tenure (< 6 months)": "★★★★★",
            "High Monthly Charges": "★★★★☆",
            "Fiber Optic Internet Service": "★★★★☆",
            "Lack of Support Services": "★★★★☆",
            "Payment via Electronic Check": "★★★☆☆",
            "Customer Demographics": "★☆☆☆☆"
        }

        for driver, importance in churn_drivers.items():
            st.write(f"- **{driver}**: {importance}")

        st.info(
            """
            Churn is driven primarily by **account-related and service-related factors**,
            not by customer demographics.
            """
        )

        st.divider()

        #-------------------------------------------------------------------
        # 3. High-Risk Customer Profile
        
        st.subheader("High-Risk Customer Profile")

        st.markdown(
            """
            **A customer with a high likelihood of churn typically has the following characteristics:**
            
            - On a **month-to-month contract**
            - Tenure of **less than 6 months**
            - Subscribed to **Fiber Optic internet**
            - Pays **higher-than-average monthly charges**
            - Does **not** use support or protection services
            - Uses **Electronic Check** as the payment method
            """
        )

        st.success(
            """
            This profile can be directly used to:
            - Design targeted retention campaigns
            - Define business rules for churn prevention
            - Guide feature selection for predictive modeling
            """
        )

        st.divider()

        #-------------------------------------------------------------------
        # 4. Business Recommendations
        
        st.subheader("Actionable Business Recommendations")

        st.markdown(
            """
            **Retention & Lifecycle Management**
            - Focus retention efforts within the **first 6 months**
            - Improve onboarding and early customer experience
            
            **Contract & Pricing Strategy**
            - Incentivize upgrades from month-to-month to long-term contracts
            - Review pricing for premium Fiber Optic plans
            
            **Product & Service Bundling**
            - Bundle Online Security and Tech Support with internet services
            - Promote multi-service subscriptions to increase customer stickiness
            
            **Payment Optimization**
            - Encourage automatic payment methods
            - Reduce friction associated with electronic check payments
            """
        )

        st.divider()

        #-------------------------------------------------------------------
        # 5. Bridge to Predictive Modeling
        
        st.subheader("From Insights to Predictive Modeling")

        st.markdown(
            """
            The identified churn drivers were later used as **key input features**
            in a machine learning model to predict customer churn risk.
            
            This ensures that the model is:
            - **Business-aligned**
            - **Interpretable**
            - **Actionable in real-world deployment**
            """
        )


if choice == "Build Model":
    
    
    tab_choose_model, tab_Build_Model = st.tabs(['Choose Model', 'Build Model'])
    
    
    with tab_choose_model:
        st.subheader("Model Performance Comparison")
        st.write("""
        Before selecting the algorithm for the predictive model, I tested Logistic Regression, Decision Tree, Random Forest, and XGBoost. \n
        Using all encoded features""")
        
        # Displaying the table
        st.dataframe(df_model_comparison)
        st.image('tab1_2.png')

        st.write('''
                 - Random Forest demonstrated the most stable performance, achieving the highest Accuracy, Precision, and F1-score. Its main drawback is the relatively long training time.
                - Logistic Regression had the highest Recall (0.933), meaning it captures most churn cases, but its Precision was low (0.53), leading to many false positives.
                 Practically, for every 10 customers predicted as churn by Logistic Regression, only about 5 are actual churners. Meanwhile, Random Forest, with a Precision of 0.83, correctly identifies approximately 8 out of 10 predicted churners.
                 Therefore, despite its longer training time, Random Forest is selected for its superior predictive reliability.
                 
                 The next step is to analyze the top 10 most important features identified by Random Forest and re-evaluate the model using only these features.

                 Reducing features lowers Random Forest’s performance, as it relies on the diversity of all variables for its ensemble.
                    Surprisingly, XGBoost improves significantly when limited to 20 features, with Recall rising from 0.46 to 0.67 and F1-score from 0.58 to 0.62, indicating that the original dataset contained noisy features.
                Logistic Regression still achieves the highest Recall (0.93) but has low Precision (0.44).
                Next, I will evaluate XGBoost using the top 10 important features.  
                ''')
        st.image('tab_3_feature_important.png')
        st.dataframe(df_model_comparison_10f)

        st.write('''
                 After training the model using the top 10 important features identified by XGBoost, I can simplify the model while still achieving stable and reliable results.
                 I plan to further optimize this model by improving Precision and F1-score using Grid Search for hyperparameter tuning and adjusting the classification threshold to select the optimal cutoff for real-world application.''')

        st.success("**Conclusion:** XGBoost offers the highest 'Intelligence Ceiling.' I chose it because its default state is stable, its speed allows for deep optimization, and its architecture is specifically designed to handle complex, imbalanced business problems like Customer Churn.")

    
    with tab_Build_Model:
        st.title("Build Model")
        
        st.subheader("1. Modeling Overview")
        st.write("""
        After testing multiple algorithms, the final decision was to build the predictive
                  model using XGBoost with the top 10 important features, aiming to simplify the model and enhance its practical applicability.    """)
        
        st.subheader("2. Data Preprocessing")
        st.write("""
        - **Numeric features:** PowerTransformer (Yeo-Johnson) applied to normalize skewed distributions.
        - **Categorical features:** One-Hot Encoding (first level dropped) applied to all categorical columns.
        - **Target variable:** Label-encoded 'Churn' (0 = No, 1 = Yes).
        - **Feature matrix:** Combined numeric and categorical features for modeling.
        """)
        
        st.subheader("3. Handling Class Imbalance")
        st.write("Churn is a minority class (~26% of customers). SMOTE was applied to the training set to balance the classes.")
        
        st.info("After SMOTE, the classes are balanced, which helps the model better detect churners.")
        st.subheader("4. Top 10 Feature Importance")
        st.write(features_importance)
        st.image('tab_3_feature_important.png')

       

# build model
        st.subheader("5. Modeling")
        st.write('''I built a pipeline where the data passes through preprocessing, including feature engineering and encoding, and SMOTE to handle class imbalance.
                The model uses XGBoost with tuned hyperparameters, optimized via a 5-fold Grid Search, resulting in significant performance improvement.''')
        st.code("""
pipe_xgb = Pipeline([
    ('preprocess', preprocessor),
    ('smote', SMOTE(random_state=42)),
    ('model', XGBClassifier(
        n_estimators=300,
        max_depth=5,
        learning_rate=0.01,
        subsample=0.7,
        colsample_bytree=0.8,
        scale_pos_weight=1,
        random_state=42,
        use_label_encoder=False,
        eval_metric='logloss'
    ))
])
    
""", language='python') 
        
        st.dataframe(report_xgb)
        st.write('''After performing Grid Search, the XGBoost model shows significant improvement.
        The Recall increased from 0.67 to 0.74, meaning the model can correctly identify 74% of customers who are likely to churn.
        The Precision of 0.52 indicates the model is better at reducing false positives, while still capturing most churn cases.
        The ROC AUC of 0.829 demonstrates that the model can effectively distinguish between customers who will churn and those who will not.
        I will use this optimized XGBoost model for the prediction phase.''')
        st.image('confusion_matrix.png')
        st.image('roc_auc.png')

    
        
        st.subheader("6. Threshold Tuning (Precision / Recall / F1)")
        st.image('threshold.png')
        st.write("""
        After adjusting the classification threshold, we observed that the F1-score reached its highest value of 0.62.
        The model achieved its best balance with a Precision of 0.60 and a Recall of 0.64.
        This means that out of every 10 customers predicted as churn, 6 are correctly identified, and the model captures approximately 64% of all actual churners.
               """)
        
        
        st.success("""
        I chose to use a classification threshold of 0.60 to achieve the best balance between Precision and Recall.
 """)

   
if choice == 'New Customer Prediction':
    import streamlit as st
    import pandas as pd
    import matplotlib.pyplot as plt
    from ortools.linear_solver import pywraplp

    st.title("Customer Churn Prediction & Strategy Optimization")

    # Initialize ----------------------------
    threshold = 0.6
    # Define strategies
    strategies = {
    "A": {"Cost": 50,"Additional_Discount": 0.2, "Churn_Reduction": 0.4, "Desc":"Contract upgrade for Month to Month customer + small Discount for others"},
    "B": {"Cost": 30,"Additional_Discount": 0.3, "Churn_Reduction": 0.3, "Desc":"Service improvement + small Discount for others"},
    "C": {"Cost": 0, "Additional_Discount": 0.5,"Churn_Reduction": 0.35, "Desc":"Discount for all customers"},
    "D": {"Cost": 0,"Additional_Discount": 0, "Churn_Reduction": 0.0, "Desc":"No action"}
                }


    base_features = [
        'Contract', 'InternetService', 'OnlineSecurity', 'StreamingMovies', 
        'TechSupport', 'PaymentMethod', 'PaperlessBilling', 'MultipleLines', 
        'gender', 'StreamingTV'
    ]
    numeric_cols = ['tenure','MonthlyCharges','TotalCharges']
    selected_cols = base_features+ numeric_cols

    # Tabs ----------------------------
    tab1, tab2 = st.tabs(["Churn Prediction", "Strategy Recommendation"])

    # TAB 1 ----------------------------
    with tab1:
        st.subheader("Customer Churn Prediction")
        input_method = st.radio("Input method:", ["Manual Input", "Upload file"], horizontal=True)

        # Manual Input -----------
        if input_method == "Manual Input":  
            with st.form("manual_input_form"):
                tenure = st.number_input("Tenure (months)", 0, 100, 12, 1)
                MonthlyCharges = st.number_input("Monthly Charges ($)", 0.0, 1000.0, 50.0, 0.5)
                TotalCharges = st.number_input("Total Charges ($)", 0.0, 10000.0, 600.0, 1.0)
                contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
                internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
                online_security = st.selectbox("Online Security", ["No", "Yes"])
                streaming_movies = st.selectbox("Streaming Movies", ["No", "Yes"])
                tech_support = st.selectbox("Tech Support", ["No", "Yes"])
                payment_method = st.selectbox("Payment Method", ["Electronic check", "Mailed check", 
                                                                 "Bank transfer (automatic)", "Credit card (automatic)"])
                paperless_billing = st.selectbox("Paperless Billing", ["No", "Yes"])
                multiple_lines = st.selectbox("Multiple Lines", ["No", "Yes"])
                gender = st.selectbox("Gender", ["Female", "Male"])
                streaming_tv = st.selectbox("Streaming TV", ["No", "Yes"])
                submitted = st.form_submit_button("Predict Churn")

                if submitted:
                    input_df = pd.DataFrame({
                        'tenure':[tenure],
                        'MonthlyCharges':[MonthlyCharges],
                        'TotalCharges':[TotalCharges],
                        'Contract':[contract],
                        'InternetService':[internet_service],
                        'OnlineSecurity':[online_security],
                        'StreamingMovies':[streaming_movies],
                        'TechSupport':[tech_support],
                        'PaymentMethod':[payment_method],
                        'PaperlessBilling':[paperless_billing],
                        'MultipleLines':[multiple_lines],
                        'gender':[gender],
                        'StreamingTV':[streaming_tv],
                    })
                    pred_prob = predict_pipeline.predict_proba(input_df)[:,1][0]
                    st.session_state['manual_data'] = {
                                                        "df": input_df,
                                                        "prob": pred_prob,
                                                        "monthly": MonthlyCharges,
                                                        "submitted": True
                                                    }
                    st.metric("**Churn Probability:**", f"{pred_prob:.2%}")
                    if pred_prob >= threshold:
                        st.error(f" High Risk! Potential Revenue Loss: ${MonthlyCharges:.2f}/month")
                    else:
                        st.success(" Customer is likely to stay.")
                    

        # CSV Upload -----------
        else:
            uploaded_file = st.file_uploader("Upload file with customer data", type=["csv","xlsx"])
            st.write('''File must contain columns: 
        'Contract', 'InternetService', 'OnlineSecurity', 'StreamingMovies', 
        'TechSupport', 'PaymentMethod', 'PaperlessBilling', 'MultipleLines', 
        'gender', 'StreamingTV','tenure','MonthlyCharges', 'TotalCharges'. ''')
            if uploaded_file is not None:
                if uploaded_file.name.endswith(".csv"):
                        df_load = pd.read_csv(uploaded_file)
                elif uploaded_file.name.endswith(".xlsx"):
                        df_load = pd.read_excel(uploaded_file)
                else:
                    st.error("File type not supported")
                    df_load = None
                if df_load is not None:
                    st.dataframe(df_load.head())
                    df = df_load[selected_cols]
                    for i in numeric_cols:
                        df[i] =  pd.to_numeric(df[i], errors='coerce').fillna(0)
            
                df["Churn_Prob"] = predict_pipeline.predict_proba(df)[:,1]
                df["Churn_Pred"] = (df["Churn_Prob"] >= threshold).astype(int)
                df["Churn_Pred_Label"] = df["Churn_Pred"].map({1:"Churn",0:"No Churn"})
                st.session_state['df_churn'] = df
                # --- REVENUE AT RISK ANALYSIS ---
                churners = df[df["Churn_Pred"] == 1]
                total_loss = churners["MonthlyCharges"].sum()
                total_current_rev = df["MonthlyCharges"].sum()

                st.divider()
                col_m1, col_m2, col_m3 = st.columns(3)
                col_m1.metric("Predicted Churners", len(churners))
                col_m2.metric("Monthly Revenue at Risk", f"${total_loss:,.2f}")
                col_m3.metric("Risk % of Total", f"{(total_loss/total_current_rev):.1%}")
                
                st.warning(f" If no action is taken, you stand to lose {total_loss:,.2f} out of {total_current_rev:,.2f} monthly revenue.")
                st.dataframe(df.head())



    # TAB 2: RECOMMENDATION ----------------------------

    with tab2:

        st.subheader("Growth Analysis & Personalized Optimization (Churners Only)")
        st.info(
            "This module allocates the optimal retention strategy **only to predicted churners** "
            "under a limited budget to maximize expected net revenue."
        )
        #Strategy Overview ----------
        strategy_table = (
            pd.DataFrame(strategies)
            .T
            .reset_index()
            .rename(columns={"index": "Strategy"})
        )
        st.dataframe(strategy_table, use_container_width=True)

        if input_method == "Manual Input":
            if 'manual_data' in st.session_state and st.session_state['manual_data'].get('submitted', False):
                input_df = st.session_state['manual_data']['df']
                pred_prob = st.session_state['manual_data']['prob']
                MonthlyCharges = st.session_state['manual_data']['monthly']
                df_manual = input_df.copy()
                df_manual["Churn_Prob"] = pred_prob
                df_manual["Churn_Pred"] = int(pred_prob >= threshold)
                if df_manual["Churn_Pred"].iloc[0] == 1:
                # Only optimize if predicted churn
                    df_manual = df_manual.reset_index(drop=True)
                    num_customers = 1
                    s_names = list(strategies.keys())
                    solver = pywraplp.Solver.CreateSolver("SCIP")
                    x = {(0,j): solver.BoolVar(f"x_0_{j}") for j in range(len(s_names))}

                    # Cost matrix
                    cost_matrix = {}
                    monthly = MonthlyCharges
                    for j, s in enumerate(s_names):
                        cost_main, cost_add = 0, 0
                        if s=="A":
                            cost_add = strategies[s]["Additional_Discount"]*monthly
                            if contract=="Month-to-month":
                                cost_main = strategies[s]["Cost"]
                        elif s=="B":
                            cost_add = strategies[s]["Additional_Discount"]*monthly
                            if internet_service=="Fiber optic":
                                cost_main = strategies[s]["Cost"]
                        elif s=="C":
                            cost_add = strategies[s]["Additional_Discount"]*monthly
                        # D = 0
                        total_cost = cost_main + cost_add
                        cost_matrix[(0,j)] = total_cost

                    # Constraint: one strategy
                    solver.Add(sum(x[0,j] for j in range(len(s_names)))==1)

                    # Objective
                    objective = solver.Objective()
                    for j,s in enumerate(s_names):
                        churn_flag = 1 if pred_prob*(1-strategies[s]["Churn_Reduction"])>threshold else 0
                        expected_revenue = 0 if churn_flag==1 else monthly
                        net_revenue = expected_revenue - cost_matrix[(0,j)]
                        if net_revenue<0: net_revenue=-1000
                        objective.SetCoefficient(x[0,j], net_revenue)
                    objective.SetMaximization()

                    # Solve
                    status = solver.Solve()
                    if status==pywraplp.Solver.OPTIMAL:
                        for j,s in enumerate(s_names):
                            if x[0,j].solution_value()>0.5:
                                st.success(f"Recommended Strategy: {s} → {strategies[s]['Desc']}")
                                expected_rev = 0 if churn_flag==1 else monthly
                                cost = cost_matrix[(0,j)]
                                net = expected_rev - cost
                                st.write(f"Expected Revenue: ${expected_rev:.2f}, Cost: ${cost:.2f}, Net: ${net:.2f}")
            
            else:
                st.info("Please submit the manual input form first to run optimization.")
                
            
        # input = file
            
        else:
    # Budget ----------
            budget = st.number_input(
                "Set total budget ($)",
                min_value=0.0,
                max_value=50000.0,
                value=5000.0,
                step=100.0
            )

            # Start Optimization ----------
            if st.button("Optimize Strategy for Churners"):

                # ---------- Filter churners only ----------
                df_all = st.session_state["df_churn"].copy().reset_index(drop=True)
                df = df_all[df_all["Churn_Pred"] == 1].reset_index(drop=True)
                num_customers = len(df)
                s_names = list(strategies.keys())

                if num_customers == 0:
                    st.warning("No predicted churners in this dataset. Nothing to optimize.")
                else:

                    solver = pywraplp.Solver.CreateSolver("SCIP")

                    # Decision Variables ----------
                    x = {
                        (i, j): solver.BoolVar(f"x_{i}_{j}")
                        for i in range(num_customers)
                        for j in range(len(s_names))
                    }

                    # Constraint 1: One strategy per customer ----------
                    for i in range(num_customers):
                        solver.Add(sum(x[i, j] for j in range(len(s_names))) == 1)

                    # COST MATRIX ----------
                    cost_matrix = {}
                    for i in range(num_customers):
                        for j, s in enumerate(s_names):
                            monthly = df.loc[i, "MonthlyCharges"]
                            cost_main = 0
                            cost_add = 0

                            if s == "D":
                                pass
                            elif s == "A":
                                cost_add = strategies[s]["Additional_Discount"] * monthly
                                if df.loc[i, "Contract"] == "Month-to-month":
                                    cost_main = strategies[s]["Cost"]
                            elif s == "B":
                                cost_add = strategies[s]["Additional_Discount"] * monthly
                                if df.loc[i, "InternetService"] == "Fiber optic":
                                    cost_main = strategies[s]["Cost"]
                            elif s == "C":
                                cost_add = strategies[s]["Additional_Discount"] * monthly

                            total_cost = cost_main + cost_add
                            cost_matrix[(i, j)] = total_cost

                    #  Constraint 2: Budget ----------
                    solver.Add(
                        sum(
                            x[i, j] * cost_matrix[(i, j)]
                            for i in range(num_customers)
                            for j in range(len(s_names))
                        ) <= budget
                    )

                    # ---------- OBJECTIVE: MAXIMIZE NET REVENUE ----------
                    def churn_decision(churn_prob, churn_reduction):
                        churn_after = churn_prob * (1 - churn_reduction)
                        return 1 if churn_after > threshold else 0

                    objective = solver.Objective()

                    for i in range(num_customers):
                        for j, s in enumerate(s_names):
                            churn_flag = churn_decision(df.loc[i, "Churn_Prob"], strategies[s]["Churn_Reduction"])
                            expected_revenue = 0 if churn_flag == 1 else df.loc[i, "MonthlyCharges"]
                            net_revenue = expected_revenue - cost_matrix[(i, j)]
                            if net_revenue < 0:
                                net_revenue = -1000
                            objective.SetCoefficient(x[i, j], net_revenue)

                    objective.SetMaximization()

                    # SOLVE ----------
                    status = solver.Solve()

                    if status == pywraplp.Solver.OPTIMAL:

                        recommended = []
                        expected_revenue_list = []
                        cost_list = []
                        net_list = []

                        for i in range(num_customers):
                            for j in range(len(s_names)):
                                if x[i, j].solution_value() > 0.5:
                                    s = s_names[j]
                                    recommended.append(s)
                                    churn_flag = churn_decision(df.loc[i, "Churn_Prob"], strategies[s]["Churn_Reduction"])
                                    expected_revenue = 0 if churn_flag == 1 else df.loc[i, "MonthlyCharges"]
                                    cost = cost_matrix[(i, j)]
                                    net = expected_revenue - cost
                                    expected_revenue_list.append(expected_revenue)
                                    cost_list.append(cost)
                                    net_list.append(net)

                        # Output ----------
                        df["Recommended_Strategy"] = recommended
                        df["Expected_Revenue"] = expected_revenue_list
                        df["Cost"] = cost_list
                        df["Net_Revenue"] = net_list

                        st.success("✅ Optimization completed for churners.")

                        st.dataframe(
                            df[
                                ["Churn_Prob", "MonthlyCharges", "Recommended_Strategy", "Expected_Revenue", "Cost", "Net_Revenue"]
                            ],
                            use_container_width=True
                        )

                        # Summary ----------
                        st.markdown("### Summary for Churners")
                        st.metric("Total Expected Revenue", f"${sum(expected_revenue_list):,.2f}")
                        st.metric("Total Cost", f"${sum(cost_list):,.2f}")
                        st.metric("Total Net Revenue", f"${sum(net_list):,.2f}")

                        # Visuals ----------
                        st.markdown("### Strategy Distribution for Churners")
                        st.bar_chart(df["Recommended_Strategy"].value_counts())

                    else:
                        st.error(" No optimal solution found under the given budget.")

                    