import streamlit as st
import pickle
import numpy as np
import pandas as pd

st.set_page_config(page_title="House Price Predictor")

with open('models/df.pkl','rb') as file:
    df = pickle.load(file)

with open('models/real_estate_predictor.pkl','rb') as file:
    pipeline = pickle.load(file)

st.title("🏘️ House Price Predictor")
st.markdown("""---""")
st.header('Enter your inputs')

col1, col2 = st.columns(2)

with col1:
    # property_type
    property_type = st.selectbox('Property Type', df['property_type'].unique().tolist())
with col2:
    # sector
    sector = st.selectbox('Sector',sorted(df['sector'].unique().tolist()))
    

col1, col2, col3 = st.columns(3)
with col1:
    bedrooms = float(st.selectbox('Number of Bedroom',sorted(df['bedRoom'].unique().tolist())))
with col2:
    bathroom = float(st.selectbox('Number of Bathrooms',sorted(df['bathroom'].unique().tolist())))
with col3:
    balcony = st.selectbox('Balconies',sorted(df['balcony'].unique().tolist()))

col1, col2 = st.columns(2)

with col1:
    property_age = st.selectbox('Property Age',sorted(df['agePossession'].unique().tolist()))
    servant_room = st.selectbox('Servant Room',["No", "Yes"])
with col2:
    built_up_area = float(st.number_input('Built Up Area', value=1250))
    store_room = st.selectbox('Store Room',["No", "Yes"])

col1, col2, col3 = st.columns(3)
with col1:
    furnishing_type = st.selectbox('Furnishing Type',sorted(df['furnishing_type'].unique().tolist()))
with col2:
    luxury_category = st.selectbox('Luxury Category',sorted(df['luxury_category'].unique().tolist()))
with col3:
    floor_category = st.selectbox('Floor Category',sorted(df['floor_category'].unique().tolist()))

st.markdown("""---""")
if st.button('Predict'):
    st.balloons()
    servant_room = 1.0 if servant_room == "Yes" else 0.0
    store_room = 1.0 if store_room == "Yes" else 0.0

    data = [[property_type, sector, bedrooms, bathroom, balcony, property_age, built_up_area, servant_room, store_room, furnishing_type, luxury_category, floor_category]]
    columns = ['property_type', 'sector', 'bedRoom', 'bathroom', 'balcony',
               'agePossession', 'built_up_area', 'servant room', 'store room',
               'furnishing_type', 'luxury_category', 'floor_category']
    
    # Convert to DataFrame
    one_df = pd.DataFrame(data, columns=columns)
    # predict
    base_price = np.expm1(pipeline.predict(one_df))[0]

    def get_ci(price):
        if price < 0.5:
            return 0.05
        elif price < 1:
            return 0.12
        else:
            return 0.22
    
    ci = get_ci(base_price)
    low = base_price - ci
    high = base_price + ci

    # display
    st.write("### `The price is in between {} Cr and {} Cr`".format(round(low,2),round(high,2)))
