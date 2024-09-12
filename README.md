## **Overview**
This project predicts student exam scores based on a variety of factors such as hours studied, attendance, parental involvement, and extracurricular activities. The goal is to understand which factors contribute most to student performance and create a predictive model using Random Forest and XGBoost regression models.

## **Table of Contents**
1. [Project Motivation](#motivation)
2. [Data source](#source)
3. [Data Description](#data)
4. [Technology used](#technology)
5. [Project Structure](#structure)
6. [Modeling](#modeling)
   - Random Forest Regression
   - XGBoost Regression
7. [Evaluation Metrics](#evaluation)
8. [Feature Importance](#feature-importance)
9. [Streamlit App](#streamlit-app)
10. [Contributing](#contributers)

## **Project Motivation** <a name="motivation"></a>
The purpose of this project is to explore which factors have the highest impact on student exam performance and to develop models that can accurately predict exam scores based on these factors. Understanding these relationships could help educators and policymakers make informed decisions to improve student outcomes.

## **Data Source** <a name="source"></a>
The dataset was taken from **Kaggle**
- link: https://www.kaggle.com/datasets/lainguyn123/student-performance-factors/data

## **Data Description** <a name="data"></a>
The dataset contains the following columns:
- **Numerical Features**: `Hours_Studied`, `Attendance`, `Sleep_Hours`, `Previous_Scores`, `Tutoring_Sessions`, `Physical_Activity`, `Exam_Score`
- **Categorical Features**: `Parental_Involvement`, `Access_to_Resources`, `Extracurricular_Activities`, `Motivation_Level`, `Internet_Access`, `Family_Income`, `Teacher_Quality`, `School_Type`, `Peer_Influence`, `Learning_Disabilities`, `Parental_Education_Level`, `Distance_from_Home`, `Gender`

The target variable for this project is `Exam_Score`.

## **Technology used** <a name="technology"></a>
This project was developed using the following technologies and libraries:

- **Python**: The programming language used for the project.
- **Pandas**: For data manipulation and analysis.
- **NumPy**: For numerical computations.
- **Matplotlib & Seaborn**: For data visualization and plotting.
- **Scikit-learn**: For machine learning models, including Random Forest Regression and cross-validation.
- **XGBoost**: For implementing XGBoost Regression models.
- **Jupyter Notebook**: For interactive development and documentation.
- **GridSearchCV**: For hyperparameter tuning of machine learning models.

## **Project Structure** <a name="structure"></a>
- **Data Cleaning and Preprocessing**: Missing values are handled, and categorical features are encoded.
- **Feature Engineering**: Transformation of features to improve model performance.
- **Modeling**: Two regression models are used for prediction.
    - **Random Forest Regression**
    - **XGBoost Regression**
- **Evaluation**: The models are evaluated using R-squared (R²), Mean Absolute Error (MAE), and Mean Squared Error (MSE).

## **Modeling** <a name="modeling"></a>

### **Random Forest Regression**
The Random Forest model was tuned using grid search with cross-validation. Important hyperparameters like `n_estimators`, `max_depth`, `min_samples_split`, and `min_samples_leaf` were optimized.
- **Best R² Score**: *0.649896*

### **XGBoost Regression**
Similarly, XGBoost was used for regression with hyperparameter tuning to optimize performance. Key parameters such as `learning_rate`, `n_estimators`, and `max_depth` were adjusted for the best results.
- **Best R² Score**: *0.702188*

## **Evaluation Metrics** <a name="evaluation"></a>
The performance of both models was evaluated using three key metrics:
- **Mean Absolute Error (MAE)**: Measures the average magnitude of the errors in predictions.
- **Mean Squared Error (MSE)**: Measures the average squared difference between actual and predicted values.
- **R-squared (R²)**: Explains the proportion of variance in the target variable explained by the features.

## **Feature Importance** <a name="feature-importance"></a>
For the Random Forest and XGBoost models, feature importance was analyzed to understand which factors have the most impact on predicting exam scores. This information can help in focusing on the most critical features in the future.
- Random forest feature importance chart:

![0250df74-1882-44ab-8f99-02196ed93b42](https://github.com/user-attachments/assets/4bf2d8d3-37da-4fa2-8abb-a86ac7d6e4c9)

- XGBoost feature importance chart:

![94a2a04b-d71a-4cc0-b503-cdb69f2647db](https://github.com/user-attachments/assets/226b5b97-4e0d-4cff-b39e-364ce58627fc)

## **Streamlit App** <a name="streamlit-app"></a>
The predictive models, including Random Forest and XGBoost, have been deployed as an interactive app using **Streamlit**. This app allows users to input various features such as hours studied, attendance, and extracurricular activities, and provides real-time predictions of student exam scores based on the trained models.  
- You can access the app [here]([your-streamlit-app-link](https://omar-omar-om-student-perfo-streamlit-student-performance-jlgdra.streamlit.app)).

## **Contributers** <a name="contributers"></a>
Author: Omar Ibrahim
