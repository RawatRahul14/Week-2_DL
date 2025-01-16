import streamlit as st
import pandas as pd
import pickle
from sklearn.datasets import load_diabetes

st.title("This app is to predict the glucose level in the blood of a diabetic patient.")

model = pickle.load(open("jhsdvgcvg", "rb"))

diab = load_diabetes()
X = pd.DataFrame(diab.data, columns = diab.feature_names)

user_input = {}

for col in X.columns:
    user_input[col] = st.slider(col, X[col].min(), X[col].max(), X[col].mean())

df = pd.DataFrame(user_input, index = [0])

st.write(df)

models = {"Linear Regression 1": model,
          "Linear Regression 2": model}

selected_model = st.selectbox("Select a model", ("Linear Regression 1", "Linear Regression 2"))

if st.button("Predict"):
    prediction = models[selected_model].predict(df)
    st.write(f"Glucose level: {prediction[0]}")