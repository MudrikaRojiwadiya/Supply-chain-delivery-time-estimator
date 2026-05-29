import streamlit as st
import pickle
from geopy.geocoders import Nominatim
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta


pipe = pickle.load(open("final_pipeline_RF.pkl", "rb"))

data = pickle.load(open("data.pkl", "rb"))
data['source_city'] = data['source_name'].str.split('_').str[0]
data['destination_city'] = data['destination_name'].str.split('_').str[0]


st.markdown("""
<h1 style='text-align:center;
color:#2E8B57;'>
📦 Shipment Delivery Time Predictor
</h1>

<p style='text-align:center;
font-size:20px;
color:gray;'>
Predict shipment arrival time using Machine Learning & Route Intelligence
</p>

<hr>
""", unsafe_allow_html=True)

st.sidebar.title("📦 About Project")

st.sidebar.info("""
This ML-based application predicts
shipment delivery arrival using:

✔ Random Forest
✔ OSRM Routing
✔ Streamlit
✔ Geopy
""")

route_type = st.selectbox("Select route_type", data["route_type"].unique())


col1, col2 = st.columns(2)

with col1:
    source_city = st.selectbox(
        "📍 Select Source City",
        data['source_city'].unique()
    )

with col2:
    destination_city = st.selectbox(
        "🎯 Select Destination City",
        data['destination_city'].unique()
    )
geolocator = Nominatim(user_agent="delivery_app")

try:
    source_location = geolocator.geocode(source_city, timeout=10)
    destination_location = geolocator.geocode(destination_city, timeout=10)

    if source_location and destination_location:

        src_lat = source_location.latitude
        src_lon = source_location.longitude

        dest_lat = destination_location.latitude
        dest_lon = destination_location.longitude

    else:
        st.error("Could not fetch location coordinates.")
        st.stop()

except Exception as e:
    st.error(f"Actual error fetching coordinates: {e}")
    st.stop()

def get_osrm_data(src_lat, src_lon, dest_lat, dest_lon):

    try:
        url = f"http://router.project-osrm.org/route/v1/driving/{src_lon},{src_lat};{dest_lon},{dest_lat}?overview=false"

        response = requests.get(url, timeout=10)

        if response.status_code != 200:
            st.error("Unable to fetch route information. Please try again later.")
            return None

        data = response.json()

        if not data.get("routes"):
            st.error("No route information available for the selected locations.")
            return None, None

        distance = data['routes'][0]['distance'] / 1000
        duration = data['routes'][0]['duration'] / 60

        return distance, duration

    except Exception:
        st.error("Route service is currently unavailable. Please try again later.")
        return None

# GET OSRM DISTANCE AND TIME
osrm_distance, osrm_time = get_osrm_data(
    src_lat,
    src_lon,
    dest_lat,
    dest_lon
)
if osrm_distance is None:
    st.stop()

# SHOW OUTPUT
col3, col4 = st.columns(2)

with col3:
    st.metric(
        "🛣 Distance",
        f"{round(osrm_distance,2)} KM"
    )

with col4:
    st.metric(
        "⏱ Travel Time",
        f"{round(osrm_time,2)} Minutes"
    )

# THEN groupby
source_dict_center = data.groupby('source_city')['source_center'].first().to_dict()
destination_dict_center = data.groupby('destination_city')['destination_center'].first().to_dict()


source_center = source_dict_center[source_city]
destination_center = destination_dict_center[destination_city]

col5, col6 = st.columns(2)

with col5:
    st.info(f"""
    📍 Source Center

    {source_center}
    """)

with col6:
    st.info(f"""
    🎯 Destination Center

    {destination_center}
    """)

col7, col8 = st.columns(2)

with col7:
    order_datetime = st.datetime_input(
        "📅 Order Date & Time"
    )

with col8:
    dispatch_datetime = st.datetime_input(
        "🚚 Dispatch Date & Time"
    )
od_hour = dispatch_datetime.hour
trip_hour = order_datetime.hour


source_center = source_dict_center.get(source_city, "Unknown")
destination_center = destination_dict_center.get(destination_city, "Unknown")
route = source_center + "_" + destination_center
route_freq_map = data.groupby('route')['route'].count().to_dict()

route_frequency = route_freq_map.get(route, 0)

def traffic_level(hour):

    # Low Traffic Hours
    if hour in [0,1,2,3,4,7,23]:
        return "Low"

    # High Traffic Hours
    elif hour in [10,11,12,13,14,17]:
        return "High"

    # Medium Traffic Hours
    else:
        return "Medium"


data['traffic_level'] = data['od_hour'].apply(traffic_level)
traffic_map = {
    "Low": 0,
    "Medium": 1,
    "High": 2
}

data['traffic_level'] = data['traffic_level'].map(traffic_map)

if st.button("🚀 Predict Delivery Time", use_container_width=True): 
    input_data = pd.DataFrame({
    'source_center': [source_center],
    'destination_center': [destination_center],
    'route_type': [route_type],
    'osrm_time': [osrm_time],
    'osrm_distance': [osrm_distance],
    'trip_hour': [trip_hour],
    'od_hour': [od_hour],
    'route_frequency': [route_frequency],
    'traffic_level': [traffic_map[traffic_level(od_hour)]]
    })
    
    prediction = pipe.predict(input_data)[0]

    processed_input = pipe.named_steps['preprocessing'].transform(input_data)

    tree_predictions = []

    for tree in pipe.named_steps['model'].estimators_:
        pred = tree.predict(processed_input)[0]
        tree_predictions.append(pred)

    std_dev = np.std(tree_predictions)

    confidence = 100 - (std_dev / prediction) * 100

    confidence = max(60, min(99, confidence))

    confidence = round(confidence, 1)

    estimated_arrival = dispatch_datetime + timedelta(minutes=prediction)

    st.markdown(f"""
    <div style="
    padding:20px;
    border-radius:15px;
    background:#e8f5e9;
    border:1px solid #c8e6c9;
    text-align:center;
    ">

    <h2 style="color:#2e7d32;">
    📦 Estimated Delivery Arrival
    </h2>

    <h1 style="color:#1b5e20;">
    {estimated_arrival.strftime('%d-%m-%Y %I:%M %p')}
    </h1>

    <p style="font-size:18px; color:#424242;">
    There is a <b>{confidence}% probability</b> that your shipment
    will arrive by the estimated time.
    </p>

    </div>
    """, unsafe_allow_html=True)