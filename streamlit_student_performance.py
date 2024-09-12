import io
import streamlit as st
import pandas as pd
import numpy as np
import statsmodels.api as sm

# Train-test split and performance metrics
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Random Forest Regression model
from sklearn.ensemble import RandomForestRegressor

# Data preprocessing
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Visualization
import seaborn as sns
import matplotlib.pyplot as plt

# XgBoost
from xgboost import XGBRegressor  # type: ignore
import sklearn

st.write(f"Scikit-learn version: {sklearn.__version__}")

# Streamlit heading
st.title("Student Performance Regression Project")

# Write a brief description
st.write("""
### Overview
In this project, we explore and analyze student performance data using machine learning models, including Random Forest and XGBoost Regression. The main objective is to predict exam scores based on various features such as hours studied, attendance, parental involvement, etc.
""")

# Dataset Description
st.write("""
### Dataset Description
The dataset used in this project contains various factors that may influence student performance. These factors include:
- **Hours_Studied**: The number of hours a student spends studying.
- **Attendance**: The percentage of classes attended by the student.
- **Parental_Involvement**: The level of involvement from parents in the student's education.
- **Access_to_Resources**: Whether the student has access to educational resources such as textbooks and the internet.
- **Extracurricular_Activities**: Whether the student participates in extracurricular activities.
- **Sleep_Hours**: The number of hours of sleep the student gets on a regular basis.
- **Previous_Scores**: The student's previous exam scores.
- **Motivation_Level**: The student's level of motivation towards their studies.
- **Internet_Access**: Whether the student has access to the internet at home.
- **Tutoring_Sessions**: The number of tutoring sessions attended by the student.
- **Family_Income**: The income level of the student's family.
- **Teacher_Quality**: The perceived quality of the student's teachers.
- **School_Type**: Whether the student attends a public or private school.
- **Peer_Influence**: The level of influence peers have on the student's academic performance.
- **Physical_Activity**: Whether the student engages in regular physical activities.
- **Learning_Disabilities**: Whether the student has any diagnosed learning disabilities.
- **Parental_Education_Level**: The highest level of education achieved by the student's parents.
- **Distance_from_Home**: The distance the student has to travel to school.
- **Gender**: The gender of the student (this feature has been removed from the model to avoid bias).
""")


# Load and display the dataset for exploration
df = pd.read_csv('StudentPerformanceFactors.csv')

# Display the first few rows in the app
st.write("First few rows of the dataset:")
st.write(df.head())

# Display the shape of the dataset
st.write("Shape of the dataset:", df.shape)

# Display statistical summary of the dataset
st.write("Statistical Summary:")
st.write(df.describe())

# Display statistical summary of categorical columns
st.write("Statistical Summary of Categorical Columns:")
st.write(df.describe(include='O'))

# Display dataframe information (similar to df.info())
buffer = io.StringIO()
df.info(buf=buffer)
s = buffer.getvalue()

st.text("DataFrame Info:")
st.text(s)

# Display missing values count per column
st.subheader("Missing Values Count")
missing_values = df.isna().sum()
st.write(missing_values)

# Drop rows with missing values
df_cleaned = df.dropna(axis=0)

# Show the count of missing values in the cleaned DataFrame
st.subheader("Missing Values After Cleaning")
missing_values_cleaned = df_cleaned.isna().sum()
st.write(missing_values_cleaned)

# Display the shape of the cleaned DataFrame
st.subheader("Shape of Cleaned DataFrame")
st.write(f"Original DataFrame shape: {df.shape}")
st.write(f"Cleaned DataFrame shape: {df_cleaned.shape}")

# Calculate and display the percentage of rows removed
rows_removed = df.shape[0] - df_cleaned.shape[0]
percentage_dropped = (rows_removed / df.shape[0]) * 100

st.subheader("Rows Removed")
st.write(f"Total rows removed: {rows_removed}")
st.write(f"Percentage of rows removed: {percentage_dropped:.2f}%")

# Check for duplicated rows in the cleaned DataFrame
num_duplicated = df_cleaned.duplicated().sum()
st.subheader("Duplicate Rows")
st.write(f"Number of duplicated rows in the cleaned DataFrame: {num_duplicated}")

# Exploratory Data Analysis
st.title("Exploratory Data Analysis")

for col in df_cleaned.select_dtypes(include=['object']).columns:
    # Create a countplot for each categorical column
    plt.figure(figsize=(10, 6))
    sns.countplot(x=col, data=df_cleaned, palette='viridis')
    plt.title(f'{col} countplot')

    # Display the plot in the Streamlit app
    st.pyplot(plt)
    plt.clf()  # Clear the figure after displaying to prevent overlap

# Numerical Data Analysis
st.title("Numerical Data Analysis")

for col in df.select_dtypes(include=['int']).columns:
    # Create a histogram for each numerical column
    plt.figure(figsize=(10, 6))
    sns.histplot(df[col], kde=True)
    plt.title(f'{col} histogram')

    # Display the plot in the Streamlit app
    st.pyplot(plt)
    plt.clf()  # Clear the figure after displaying to prevent overlap

# Normality Check with QQ Plots
st.title("Normality Check with QQ Plots")

for col in df.select_dtypes(include=['int']).columns:
    # Create a QQ plot for each numerical column
    plt.figure(figsize=(10, 6))
    sm.qqplot(df[col], line='s')
    plt.title(f'{col} QQ Plot')

    # Display the plot in the Streamlit app
    st.pyplot(plt)
    plt.clf()  # Clear the figure after displaying to prevent overlap

# Correlation Heatmap
st.title("Correlation Heatmap")

# Select only numerical columns 
numerical_df = df_cleaned.select_dtypes(include=['number'])

# Compute correlation matrix for numerical columns
corr_matrix = numerical_df.corr()

# Set up the matplotlib figure
plt.figure(figsize=(12, 8))

# Create a heatmap with seaborn
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', linewidths=0.5)

plt.title('Correlation Heatmap for Numerical Columns in df_cleaned')

# Display the heatmap in the Streamlit app
st.pyplot(plt)
plt.clf()

# Load the preprocessed and encoded dataset for modeling
df_encoded = pd.read_csv('cleaned_student_performance_data.csv')
# Add description for ordinal columns
st.write("""
### Ordinal Feature Mappings
In the dataset, certain features are encoded with numerical values to represent categories. Here's how the values are mapped:

- **Parental Involvement**:  
  - Low: 0  
  - Medium: 1  
  - High: 2

- **Access to Resources**:  
  - Low: 0  
  - Medium: 1  
  - High: 2

- **Motivation Level**:  
  - Low: 0  
  - Medium: 1  
  - High: 2

- **Family Income**:  
  - Low: 0  
  - Medium: 1  
  - High: 2

- **Teacher Quality**:  
  - Low: 0  
  - Medium: 1  
  - High: 2

- **Peer Influence**:  
  - Negative: 0  
  - Neutral: 1  
  - Positive: 2

- **Distance from Home**:  
  - Far: 2  
  - Moderate: 1  
  - Near: 0
""")


# Define features and target variable
X = df_encoded.drop(columns=['Exam_Score'], axis=1)
y = df_encoded['Exam_Score']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0, test_size=0.25)

# Load the trained models (Random Forest and XGBoost from GridSearchCV or manually trained models)
rf_model = RandomForestRegressor(random_state=0)
xgb_model = XGBRegressor(random_state=0)

# Train the models
rf_model.fit(X_train, y_train)
xgb_model.fit(X_train, y_train)

# Streamlit App for Predictions
st.title('Student Performance Prediction')

# Define the input fields
def get_numeric_input(col_name, min_val, max_val):
    return st.slider(col_name, min_value=min_val, max_value=max_val, value=int((max_val + min_val) / 2))

def get_categorical_input(col_name, options):
    return st.radio(col_name, options)

# Define numeric input fields
user_inputs = {}
for col in X.columns:
    if col in ['Extracurricular_Activities_No', 'Extracurricular_Activities_Yes', 'Internet_Access_No', 'Internet_Access_Yes',
               'School_Type_Private', 'School_Type_Public', 'Learning_Disabilities_No', 'Learning_Disabilities_Yes',
               'Parental_Education_Level_College', 'Parental_Education_Level_High School', 'Parental_Education_Level_Postgraduate']:
        continue  # Skip radio button columns for sliders

    min_val, max_val = df_encoded[col].min(), df_encoded[col].max()
    user_inputs[col] = get_numeric_input(col, min_val, max_val)

# Define categorical input fields (radio buttons for one-hot encoded columns)
radio_buttons = {
    'Extracurricular_Activities': ['No', 'Yes'],
    'Internet_Access': ['No', 'Yes'],
    'School_Type': ['Private', 'Public'],
    'Learning_Disabilities': ['No', 'Yes'],
    'Parental_Education_Level': ['College', 'High School', 'Postgraduate']
}

for base_col, options in radio_buttons.items():
    cols = [col for col in X.columns if col.startswith(base_col)]
    if len(cols) > 1:
        selected_option = get_categorical_input(base_col, options)
        for col in cols:
            if selected_option in col:
                user_inputs[col] = 1
            else:
                user_inputs[col] = 0

# Create a DataFrame from user inputs
user_input_df = pd.DataFrame([user_inputs])

# Add a button to run the model
if st.button('Run Model'):
    # Make predictions with both models
    rf_prediction = rf_model.predict(user_input_df)
    xgb_prediction = xgb_model.predict(user_input_df)

    # Display the predictions
    st.subheader("Predictions")
    st.write(f"Random Forest Prediction: {rf_prediction[0]:.2f}")
    st.write(f"XGBoost Prediction: {xgb_prediction[0]:.2f}")
