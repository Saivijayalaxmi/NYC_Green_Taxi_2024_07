# NYC Green Taxi Trip Analysis Streamlit App
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Page configuration
st.set_page_config(page_title="NYC Green Taxi - June 2024", layout="wide")
st.title("🚖 NYC Green Taxi Analysis - June 2024")

@st.cache_data
def load_data():
    df = pd.read_parquet("green_tripdata_2024-07.parquet")

    # Datetime conversion
    df['lpep_pickup_datetime'] = pd.to_datetime(df['lpep_pickup_datetime'])
    df['lpep_dropoff_datetime'] = pd.to_datetime(df['lpep_dropoff_datetime'])

    # Feature engineering
    df['trip_duration'] = (df['lpep_dropoff_datetime'] - df['lpep_pickup_datetime']).dt.total_seconds() / 60
    df['weekday'] = df['lpep_pickup_datetime'].dt.day_name()
    df['hour'] = df['lpep_pickup_datetime'].dt.hour

    # Clean data
    df = df[(df['trip_duration'] > 0) & (df['trip_duration'] < 120)]
    df = df[(df['trip_distance'] > 0) & (df['trip_distance'] < 50)]
    df = df.dropna(subset=['total_amount'])

    return df

# Load data
df = load_data()

# Sidebar filters
st.sidebar.header("🧰 Filters")
selected_day = st.sidebar.selectbox("Select Weekday", ["All"] + sorted(df["weekday"].unique()))
selected_hour = st.sidebar.selectbox("Select Hour", ["All"] + sorted(df["hour"].unique()))

filtered_df = df.copy()
if selected_day != "All":
    filtered_df = filtered_df[filtered_df["weekday"] == selected_day]
if selected_hour != "All":
    filtered_df = filtered_df[filtered_df["hour"] == int(selected_hour)]

# Key metrics
st.subheader("🔢 Key Metrics")
col1, col2, col3 = st.columns(3)
col1.metric("Total Trips", len(filtered_df))
col2.metric("Avg Duration (min)", round(filtered_df["trip_duration"].mean(), 2))
col3.metric("Avg Fare ($)", round(filtered_df["total_amount"].mean(), 2))

# Visualizations
st.subheader("📊 Visualizations")

# Trips per weekday
fig1, ax1 = plt.subplots()
order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
df["weekday"].value_counts().reindex(order).plot(kind="bar", ax=ax1, color="lightgreen")
ax1.set_title("Trips by Weekday")
st.pyplot(fig1)

# Trip Distance vs Total Amount
fig2, ax2 = plt.subplots()
sample = filtered_df.sample(min(1000, len(filtered_df)))
sns.scatterplot(data=sample, x="trip_distance", y="total_amount", ax=ax2, alpha=0.4)
ax2.set_title("Total Amount vs. Trip Distance")
st.pyplot(fig2)

# Regression Modeling
st.subheader("🤖 Predict Fare using Machine Learning")

if st.button("Run Models"):
    if filtered_df.empty:
        st.warning("No data to train the model after applying filters.")
    else:
        try:
            X = filtered_df[["trip_distance", "trip_duration", "hour"]]
            y = filtered_df["total_amount"]
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            def show_results(model, name):
                try:
                    model.fit(X_train, y_train)
                    preds = model.predict(X_test)
                    r2 = r2_score(y_test, preds)
                    rmse = mean_squared_error(y_test, preds) ** 0.5
                    st.write(f"**{name}**")
                    st.write(f"R² Score: {r2:.3f}, RMSE: {rmse:.2f}")
                except Exception as e:
                    st.error(f"{name} failed: {e}")

            show_results(LinearRegression(), "Linear Regression")
            show_results(DecisionTreeRegressor(max_depth=10), "Decision Tree")
            show_results(RandomForestRegressor(n_estimators=100, max_depth=10), "Random Forest")
            show_results(GradientBoostingRegressor(n_estimators=100, max_depth=3), "Gradient Boosting")

        except Exception as e:
            st.error(f"Unexpected error during model execution: {e}")
