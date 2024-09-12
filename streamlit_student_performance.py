import io
import streamlit as st
import pandas as pd
import numpy as np
import statsmodels.api as sm

# Train-test split and performance metrics
from sklearn.model_selection import train_test_split
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
st.write(""" This application allows you to explore the student performance dataset, perform exploratory data analysis (EDA), and make predictions on student performance using machine learning models. You can input different features to see how they affect the predicted exam score. """)

# Dataset Description
st.write(""" The dataset contains various features related to student performance, such as study hours, attendance, parental involvement, and more. It is used to predict exam scores using machine learning models. """)

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
    st.pyplot(plt.gcf())  # Use plt.gcf() to specify the current figure
    plt.clf()  # Clear the figure after displaying to prevent overlap
    plt.close()  # Close the figure to free up memory

# Numerical Data Analysis
st.title("Numerical Data Analysis")

for col in df.select_dtypes(include=['int']).columns:
    # Create a histogram for each numerical column
    plt.figure(figsize=(10, 6))
    sns.histplot(df[col], kde=True)
    plt.title(f'{col} histogram')

    # Display the plot in the Streamlit app
    st.pyplot(plt.gcf())  # Use plt.gcf() to specify the current figure
    plt.clf()  # Clear the figure after displaying to prevent overlap
    plt.close()  # Close the figure to free up memory

# Normality Check with QQ Plots
st.title("Normality Check with QQ Plots")

for col in df.select_dtypes(include=['int']).columns:
    # Create a QQ plot for each numerical column
    plt.figure(figsize=(10, 6))
    sm.qqplot(df[col], line='s')
    plt.title(f'{col} QQ Plot')

    # Display the plot in the Streamlit app
    st.pyplot(plt.gcf())  # Use plt.gcf() to specify the current figure
    plt.clf()  # Clear the figure after displaying to prevent overlap
    plt.close()  # Close the figure to free up memory

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
st.pyplot(plt.gcf())  # Use plt.gcf() to specify the current figure
plt.clf()  # Clear the figure after displaying to prevent overlap
plt.close()  # Close the figure to free up memory

# Load the preprocessed and encoded dataset for modeling
df_encoded = pd.read_csv('cleaned_student_performance_data.csv')

# Add description for ordinal columns
st.write("""Description for the user to understand the data for prediction purposes:
- Hours_Studied: Number of hours spent studying per week.
- Attendance: Percentage of classes attended.
- Extracurricular_Activities: Participation in extracurricular activities (Yes, No).
- Sleep_Hours: Average number of hours of sleep per night.
- Previous_Scores: Scores from previous exams.
- Internet_Access: Availability of internet access (Yes, No).
- Tutoring_Sessions: Number of tutoring sessions attended per month.
- School_Type: Type of school attended (Public, Private).
- Physical_Activity: Average number of hours of physical activity per week.
- Learning_Disabilities: Presence of learning disabilities (Yes, No).
- Parental_Education_Level: Highest education level of parents (High School, College, Postgraduate).
- Parental Involvement: ['Low' (0), 'Medium' (1), 'High' (2)]
- Access to Resources: ['Low' (0), 'Medium' (1), 'High' (2)]
- Motivation Level: ['Low' (0), 'Medium' (1), 'High' (2)]
- Family Income: ['Low' (0), 'Medium' (1), 'High' (2)]
- Teacher Quality: ['Low' (0), 'Medium' (1), 'High' (2)]
- Peer Influence: ['Negative' (0), 'Neutral' (1), 'Positive' (2)]
- Distance from Home: ['Close' (0), 'Medium' (1), 'Far' (2)]""")

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

# Initialize session state for button click
if 'run_model' not in st.session_state:
    st.session_state.run_model = False

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
    st.session_state.run_model = True

if st.session_state.run_model:
    # Ensure model is only run when button is clicked
    st.session_state.run_model = False
    # Predict using Random Forest and XGBoost
    rf_prediction = rf_model.predict(user_input_df)
    xgb_prediction = xgb_model.predict(user_input_df)

    st.subheader("Prediction Results")
    st.write(f"Random Forest Prediction: {rf_prediction[0]:.2f}")
    st.write(f"XGBoost Prediction: {xgb_prediction[0]:.2f}")
