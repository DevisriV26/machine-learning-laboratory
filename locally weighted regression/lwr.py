import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

st.title("Locally Weighted Regression (Non-Parametric)")

# Upload dataset
uploaded_file = st.file_uploader("Upload CSV dataset", type=["csv"])

if uploaded_file is not None:

    data = pd.read_csv(uploaded_file)

    st.write("Dataset Preview")
    st.write(data.head())

    X = data.iloc[:,0].values
    y = data.iloc[:,1].values

    tau = st.slider("Select Bandwidth (tau)", 0.1, 5.0, 0.5)

    # LWR Function
    def locally_weighted_regression(x0, X, y, tau):
        
        weights = np.exp(-(X - x0)**2 / (2 * tau**2))
        
        W = np.diag(weights)
        
        X_b = np.c_[np.ones(len(X)), X]
        
        theta = np.linalg.pinv(X_b.T @ W @ X_b) @ X_b.T @ W @ y
        
        x0_b = np.array([1, x0])
        
        return x0_b @ theta

    # Prediction points
    X_test = np.linspace(min(X), max(X), 100)

    y_pred = []

    for x in X_test:
        y_pred.append(locally_weighted_regression(x, X, y, tau))

    # Plot graph
    fig, ax = plt.subplots()

    ax.scatter(X, y, label="Data Points")
    ax.plot(X_test, y_pred, label="LWR Curve")

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.legend()

    st.pyplot(fig)
