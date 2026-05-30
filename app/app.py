# ---------------------------
# IMPORTS
# ---------------------------
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

st.title("📈 Gold Price Prediction App")

# ---------------------------
# LOAD DATA
# ---------------------------
df = pd.read_csv("data/gold_prices.csv")

df["Date"] = pd.to_datetime(df["Date"])
df = df.sort_values("Date")

df["Price"] = df["Price"].astype(str).str.replace(",", "").astype(float)

# ---------------------------
# FEATURES
# ---------------------------
df["lag_1"] = df["Price"].shift(1)
df["lag_7"] = df["Price"].shift(7)
df["MA_7"] = df["Price"].rolling(7).mean()
df["MA_30"] = df["Price"].rolling(30).mean()
df["volatility_7"] = df["Price"].rolling(7).std()

df = df.dropna()

X = df[["lag_1", "lag_7", "MA_7", "MA_30", "volatility_7"]]
y = df["Price"]

# ---------------------------
# TRAIN TEST SPLIT (IMPORTANT FIX)
# ---------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, shuffle=False
)

# ---------------------------
# MODEL
# ---------------------------
model = LinearRegression()
model.fit(X_train, y_train)

pred_test = model.predict(X_test)

# ---------------------------
# NEXT DAY PREDICTION
# ---------------------------
latest = X.iloc[-1].values.reshape(1, -1)
prediction = model.predict(latest)[0]

USD_TO_INR = 83
prediction_inr = prediction * USD_TO_INR

# ---------------------------
# UI
# ---------------------------
st.subheader("📊 Prediction")

col1, col2 = st.columns(2)

with col1:
    st.metric("USD", f"${prediction:.2f}")

with col2:
    st.metric("INR", f"₹{prediction_inr:.2f}")

# ---------------------------
# MODEL EVALUATION
# ---------------------------
from sklearn.metrics import mean_absolute_error, mean_squared_error

mae = mean_absolute_error(y_test, pred_test)
rmse = np.sqrt(mean_squared_error(y_test, pred_test))

st.subheader("📉 Model Performance")
st.write("MAE:", round(mae, 2))
st.write("RMSE:", round(rmse, 2))

# ---------------------------
# PLOT
# ---------------------------
st.subheader("📈 Actual vs Predicted")

fig, ax = plt.subplots()
ax.plot(y_test.values, label="Actual")
ax.plot(pred_test, label="Predicted")
ax.legend()

st.pyplot(fig)