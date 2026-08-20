import streamlit as st
import pandas as pd
import numpy as np
import joblib

# # Load the saved model
# MODEL_PATH = "best_whale_prediction_model.pkl"

# @st.cache_resource
# def load_model():
#     return joblib.load("models/whale_prediction_model.pkl")

# def main():
#     st.set_page_config(page_title="Mobile Game Whale Prediction", page_icon="🐋", layout="wide")
#     st.title("🐋 Mobile Game Whale Prediction")
#     st.markdown("""
#     This app predicts whether a player will convert to a paying customer ('whale') based on early game behavior.
#     The model is trained on a dataset of mobile game players.
#     """)

#     st.sidebar.header("About")
#     st.sidebar.markdown("""
#     - **Task**: Binary Classification
#     - **Target**: `converted_to_payer` (1 = Yes, 0 = No)
#     - **Model**: The best model based on recall (identifying potential payers)
#     """)

#     model = load_model()

#     st.header("Enter Player Features")
#     col1, col2 = st.columns(2)

#     with col1:
#         age = st.number_input("Age", min_value=13, max_value=60, value=25, step=1)
#         gender = st.selectbox("Gender", ["Male", "Female", "Other"])
#         country = st.selectbox("Country", ["Brazil", "Canada", "Germany", "India", "Indonesia", "Japan", "Mexico", "Philippines", "UK", "USA"])
#         acquisition_channel = st.selectbox("Acquisition Channel", ["organic", "paid_social", "paid_search", "influencer", "referral"])
#         device_type = st.selectbox("Device Type", ["Android", "iOS"])

#         days_since_install = st.number_input("Days Since Install", min_value=1, max_value=90, value=10, step=1)
#         sessions_last_7d = st.number_input("Sessions in the Last 7 Days", min_value=0, max_value=26, value=7, step=1)
#         avg_session_length_min = st.number_input("Average Session Length (minutes)", min_value=0.5, max_value=32.5, value=10.5, step=0.5)
#         total_playtime_hours = st.number_input("Total Playtime (hours)", min_value=0.0, max_value=111.5, value=5.0, step=0.5)
#         levels_completed = st.number_input("Levels Completed", min_value=0, max_value=51, value=13, step=1)
#         current_level = st.number_input("Current Level", min_value=1, max_value=51, value=14, step=1)

#     with col2:
#         tutorial_completed = st.checkbox("Tutorial Completed", value=True)
#         num_friends_connected = st.number_input("Number of Friends Connected", min_value=0, max_value=15, value=2, step=1)
#         push_notifications_enabled = st.checkbox("Push Notifications Enabled", value=True)
#         ad_views = st.number_input("Ad Views", min_value=0, max_value=24, value=6, step=1)
#         rewarded_ad_views = st.number_input("Rewarded Ad Views", min_value=0, max_value=14, value=2, step=1)
#         store_visits = st.number_input("Store Visits", min_value=0, max_value=12, value=2, step=1)
#         items_viewed_in_store = st.number_input("Items Viewed in Store", min_value=0, max_value=41, value=5, step=1)
#         wishlist_items = st.number_input("Wishlist Items", min_value=0, max_value=9, value=1, step=1)
#         days_active_last_30 = st.number_input("Days Active in Last 30 Days", min_value=0, max_value=29, value=15, step=1)
#         streak_days = st.number_input("Streak Days", min_value=0, max_value=46, value=3, step=1)
#         rage_quit_events = st.number_input("Rage Quit Events", min_value=0, max_value=14, value=3, step=1)
#         level_fail_rate = st.number_input("Level Fail Rate", min_value=0.011, max_value=0.994, value=0.600, step=0.001, format="%.3f")
#         social_shares = st.number_input("Social Shares", min_value=0, max_value=7, value=0, step=1)

#     if st.button("Predict"):
#         try:
#             # Create a DataFrame with the exact feature names expected by the model
#             input_data = pd.DataFrame({
#                 "age": [age],
#                 "gender": [gender],
#                 "country": [country],
#                 "acquisition_channel": [acquisition_channel],
#                 "device_type": [device_type],
#                 "days_since_install": [days_since_install],
#                 "sessions_last_7d": [sessions_last_7d],
#                 "avg_session_length_min": [avg_session_length_min],
#                 "total_playtime_hours": [total_playtime_hours],
#                 "levels_completed": [levels_completed],
#                 "current_level": [current_level],
#                 "tutorial_completed": [1 if tutorial_completed else 0],
#                 "num_friends_connected": [num_friends_connected],
#                 "push_notifications_enabled": [1 if push_notifications_enabled else 0],
#                 "ad_views": [ad_views],
#                 "rewarded_ad_views": [rewarded_ad_views],
#                 "store_visits": [store_visits],
#                 "items_viewed_in_store": [items_viewed_in_store],
#                 "wishlist_items": [wishlist_items],
#                 "days_active_last_30": [days_active_last_30],
#                 "streak_days": [streak_days],
#                 "rage_quit_events": [rage_quit_events],
#                 "level_fail_rate": [level_fail_rate],
#                 "social_shares": [social_shares]
#             })

#             # The preprocessor is inside the pipeline, so we just pass the raw DataFrame
#             # and the pipeline handles preprocessing automatically.
#             prediction = model.predict(input_data)[0]
#             probability = model.predict_proba(input_data)[0][1]

#             if prediction == 1:
#                 st.success("💰 Prediction: This player is **LIKELY TO CONVERT** to a paying customer!")
#                 st.metric("Probability of Conversion", f"{probability*100:.2f}%")
#             else:
#                 st.warning("ℹ️ Prediction: This player is **UNLIKELY TO CONVERT** to a paying customer.")
#                 st.metric("Probability of Conversion", f"{probability*100:.2f}%")

#         except Exception as e:
#             st.error(f"An error occurred during prediction: {e}")

# if __name__ == "__main__":
#     main()
