import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

# Function to perform linear regression and return a visualization
def linear_regression(data, x, y):
    lr = LinearRegression()
    lr.fit(data[[x]], data[y])
    plt.scatter(data[x], data[y])
    plt.plot(data[x], lr.predict(data[[x]]), color='red')
    plt.title('Linear Regression')
    plt.xlabel(x)
    plt.ylabel(y)
    st.pyplot()

# Function to perform decision tree regression and return a visualization
def decision_tree(data, x, y):
    dt = DecisionTreeRegressor()
    dt.fit(data[[x]], data[y])
    plt.scatter(data[x], data[y])
    plt.plot(data[x], dt.predict(data[[x]]), color='red')
    plt.title('Decision Tree')
    plt.xlabel(x)
    plt.ylabel(y)
    st.pyplot()

# Function to perform random forest regression and return a visualization
def random_forest(data, x, y):
    rf = RandomForestRegressor()
    rf.fit(data[[x]], data[y])
    plt.scatter(data[x], data[y])
    plt.plot(data[x], rf.predict(data[[x]]), color='red')
    plt.title('Random Forest')
    plt.xlabel(x)
    plt.ylabel(y)
    st.pyplot()

# Streamlit code to load and manipulate CSV data
st.title('Regression Analysis')
file = st.file_uploader('Upload CSV file', type=['csv'])
if file is not None:
    data = pd.read_csv(file)
    st.write(data)

    # Allow user to select x and y variables
    x_variable = st.selectbox('Select X Variable', data.columns)
    y_variable = st.selectbox('Select Y Variable', data.columns)

    # Display linear regression visualization and R-squared score
    if st.button('Linear Regression'):
        linear_regression(data, x_variable, y_variable)

    # Display decision tree regression visualization and R-squared score
    if st.button('Decision Tree Regression'):
        decision_tree(data, x_variable, y_variable)

    # Display random forest regression visualization and R-squared score
    if st.button('Random Forest Regression'):
        random_forest(data, x_variable, y_variable)
