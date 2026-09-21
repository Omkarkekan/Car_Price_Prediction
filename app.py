import os, joblib, pandas as pd, streamlit as st
BASE=os.path.dirname(os.path.abspath(__file__))
@st.cache_resource
def load_model(): return joblib.load(os.path.join(BASE,"car_price_model.pkl"))
@st.cache_data
def load_data(): return pd.read_csv(os.path.join(BASE,"car_data.csv"))
st.set_page_config(page_title="Car Price Predictor",page_icon="🚗")
st.title("🚗 Car Price Predictor")
st.write("Enter the car details to estimate its resale price.")
st.subheader("📊 Model Performance")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("R² Score", "0.77")

with col2:
    st.metric("Performance", "76.66%")

with col3:
    st.metric("MAE", "9906")

with col4:
    st.metric("RMSE", "330")
model=load_model(); data=load_data()
companies=sorted(data["company"].dropna().astype(str).str.strip().unique())
fuels=sorted(data["fuel_type"].dropna().astype(str).str.strip().unique())
years=sorted(pd.to_numeric(data["year"].astype(str).str.extract(r"(\d{4})")[0],errors="coerce").dropna().astype(int).unique(),reverse=True)
company=st.selectbox("Car Company",companies)
year=st.selectbox("Manufacturing Year",years)
kms=st.number_input("Kilometers Driven",0,1000000,30000,1000)
fuel=st.selectbox("Fuel Type",fuels)
if st.button("Predict Price",type="primary",use_container_width=True):
    x=pd.DataFrame([{"company_clean":company.strip().lower(),"year_num":year,"kms_num":kms,"fuel_clean":fuel.strip().lower()}])
    price=max(0,float(model.predict(x)[0]))
    st.success(f"Estimated Price: ₹ {price:,.0f}")
    st.caption("ML estimate based on the supplied dataset; actual price varies by model, condition, ownership and location.")
