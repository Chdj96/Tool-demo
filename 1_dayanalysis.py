import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import io
from streamlit_folium import folium_static
import folium
from PIL import Image
import tempfile
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
import time


# Page configuration
st.set_page_config(page_title="Multi-Parameter Analysis Tool", layout="wide")
st.title("🌡️ Multi-Parameter Analysis Tool")
st.write("Upload your data file to analyze various parameters such as humidity, temperature, NOx, VOC, and PM.")

# Sidebar for user inputs
st.sidebar.header("User Inputs")
uploaded_file = st.sidebar.file_uploader("Upload CSV File", type=["csv"])
google_drive_link = st.sidebar.text_input("Enter Google Drive Link")
period = st.sidebar.slider("Select Time Interval (Minutes)", 1, 60, 10)

## New sidebar option to enter Google Maps location
st.sidebar.header("Map Settings")
latitude = st.sidebar.number_input("Latitude", value=52.5200, format="%.6f")
longitude = st.sidebar.number_input("Longitude", value=13.4050, format="%.6f")
zoom_level = st.sidebar.slider("Zoom Level", 1, 18, 12)

# Display interactive map
st.subheader("Interactive Map")
map_object = folium.Map(location=[latitude, longitude], zoom_start=zoom_level)
folium.Marker([latitude, longitude], popup="Selected Location").add_to(map_object)
folium_static(map_object)

# Function to save and download map using Selenium
def save_map_screenshot():
    options = Options()
    options.add_argument("--headless")
    options.add_argument("--disable-gpu")
    options.add_argument("--window-size=800x600")
    driver = webdriver.Chrome(options=options)

    temp_html = tempfile.NamedTemporaryFile(delete=False, suffix=".html")
    map_object.save(temp_html.name)
    driver.get("file://" + temp_html.name)
    time.sleep(2)  # Wait for map to render

    temp_screenshot = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
    driver.save_screenshot(temp_screenshot.name)
    driver.quit()
    return temp_screenshot.name

if st.button("📥 Download Map"):
    map_path = save_map_screenshot()
    with open(map_path, "rb") as file:
        btn = st.download_button(
            label="Download Map as Image",
            data=file,
            file_name="map.png",
            mime="image/png"
        )
# Parameter units and threshold values
parameter_units = {
    "Humidity": "%",
    "Temperature": "°C",
    "NOx": "ppm",
    "VOC": "ppb",
    "PM": "µg/m³",
}

threshold_values_pm10 = {
    "Daily Average (UBA)": 50,
    "Daily Average (WHO Recommendation)": 45,
}
threshold_values_pm25 = {
    "Annual Average (UBA)": 25,
    "Daily Average (WHO Recommendation)": 15,
}


# Data analysis function
def analyze_data(column_data, period):
    length_of_segment = round(period * 60)
    total_samples = len(column_data)
    number_of_points = int(np.floor(total_samples / length_of_segment))

    max_values = np.zeros(number_of_points)
    avg_values = np.zeros(number_of_points)
    min_values = np.zeros(number_of_points)

    for i in range(number_of_points):
        segment_data = column_data[i * length_of_segment: (i + 1) * length_of_segment]
        max_values[i] = np.max(segment_data)
        avg_values[i] = np.mean(segment_data)
        min_values[i] = np.min(segment_data)

    return max_values, avg_values, min_values, number_of_points

# Get unit for a column based on its name
def get_unit_for_column(column_name):
    for param, unit in parameter_units.items():
        if param.lower() in column_name.lower():
            return unit
    return "Value"

# Generate dynamic time labels based on data range
def generate_dynamic_time_labels(start_time, end_time, number_of_points):
    time_index = pd.date_range(
        start=start_time,
        end=end_time,
        periods=number_of_points,
        tz='Europe/Berlin'
    )
    return time_index.strftime('%Y-%m-%d %H:%M %Z')


def create_gradient_plot(data_left, data_right=None, title="", param_left="", param_right=None, left_unit="",
                         right_unit=None, show_thresholds=False, thresholds=None, start_time=None, end_time=None):
    fig, ax = plt.subplots(figsize=(10, 6))

    # Clean legend labels by removing "Left_" and "left_"
    param_left_clean = param_left.replace("Left_", "S1_").replace("left_", "S1_")
    param_right_clean = param_right.replace("right_", "S2_").replace("right_",  "S2_") if param_right else None

    ax.plot(data_left, label=f"{param_left_clean}", linestyle="solid", color="green")
    ax.fill_between(range(len(data_left)), data_left, alpha=0.1, color="green")

    if data_right is not None:
        ax.plot(data_right, label=f"{param_right_clean}", linestyle="solid", color="blue")
        ax.fill_between(range(len(data_right)), data_right, alpha=0.1, color="skyblue")

    if show_thresholds and thresholds:
        for label, value in thresholds.items():
            ax.axhline(y=value, color='red' if "UBA" in label else 'yellow', linestyle='--', linewidth=1.5,
                       label=f"{label}: {value} µg/m³")

    # Generate time range
    time_range = pd.date_range(start=start_time, end=end_time, periods=len(data_left))

    # Calculate regular hour intervals (every 3 hours)
    hour_interval = 2
    num_intervals = int(24 / hour_interval)
    tick_indices = np.linspace(0, len(data_left) - 1, num_intervals).astype(int)

    # Format time labels with date and regular hour intervals
    time_labels = [time_range[i].strftime('%Y-%m-%d\n%H:00') for i in tick_indices]

    # Set the last label to 23:59
    time_labels[-1] = time_range[-1].strftime('%Y-%m-%d\n23:59')

    ax.set_xticks(tick_indices)
    ax.set_xticklabels(time_labels, rotation=45, ha='right')

    # Extend Y-axis above the maximum value
    mean_value = max(np.max(data_left), np.max(data_right) if data_right is not None else 0)
    ax.set_ylim(0, mean_value * 1.2)  # Extend 10% above max

    ax.legend(title="Parameters", loc="best")  # Updated legend with cleaned labels
    ax.set_xlabel("Time")
    ax.set_ylabel(f"Value ({left_unit})" if not right_unit else f"Value ({left_unit}, {right_unit})")

    st.pyplot(fig)

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=300)
    buf.seek(0)
    st.download_button(
        label="📥 Download Plot",
        data=buf,
        file_name=f"{title.replace(' ', '_')}.png",
        mime="image/png"
    )
    plt.close(fig)


# Main program logic
if uploaded_file:
    data = pd.read_csv(uploaded_file)
    st.success("File uploaded successfully!")

    if 'timestamp' in data.columns:
        data['ISO8601'] = pd.to_datetime(data['ISO8601'], errors='coerce')
        if data['ISO8601'].dt.tz is None:
            data['ISO8601'] = data['ISO8601'].dt.tz_localize('UTC').dt.tz_convert('Europe/Berlin')
        start_time_column = data['ISO8601']
    else:
        st.error("The dataset must contain a 'ISO8601' column.")
        st.stop()

    st.sidebar.header("Column Selection")
    all_columns = data.columns.tolist()

    left_param = st.sidebar.selectbox("Select Left Column", all_columns, index=0)
    right_column_optional = st.sidebar.checkbox("Compare with Right Column")

    if right_column_optional:
        right_param = st.sidebar.selectbox("Select Right Column", all_columns, index=1)

    left_unit = get_unit_for_column(left_param)
    right_unit = get_unit_for_column(right_param) if right_column_optional else None

    pm_type = st.sidebar.selectbox("Select PM Type", ["PM10.0", "PM2.5"])
    thresholds = threshold_values_pm10 if pm_type == "PM10.0" else threshold_values_pm25
    show_threshold_lines = st.sidebar.checkbox("Show Threshold Lines for PM")

    column_data_left = pd.to_numeric(data[left_param], errors="coerce")
    maxVal_left, AvgVal_left, minVal_left, number_of_points_left = analyze_data(column_data_left, period)

    if right_column_optional:
        column_data_right = pd.to_numeric(data[right_param], errors="coerce")
        maxVal_right, AvgVal_right, minVal_right, number_of_points_right = analyze_data(column_data_right, period)

    start_time = start_time_column.min()
    end_time = start_time_column.max()

    st.subheader("Average Values")
    create_gradient_plot(
        data_left=AvgVal_left,
        data_right=AvgVal_right if right_column_optional else None,
        title="Average Values",
        param_left=left_param,
        param_right=right_param if right_column_optional else None,
        left_unit=left_unit,
        right_unit=right_unit,
        show_thresholds=show_threshold_lines,
        thresholds=thresholds,
        start_time=start_time,
        end_time=end_time
    )
#Option to calculate PM exceedance
    calculate_exceedance = st.sidebar.checkbox("Calculate PM Exceedance")

    column_data_left = pd.to_numeric(data[left_param], errors="coerce")
    maxVal_left, AvgVal_left, minVal_left, number_of_points_left = analyze_data(column_data_left, period)

    if right_column_optional:
        column_data_right = pd.to_numeric(data[right_param], errors="coerce")
        maxVal_right, AvgVal_right, minVal_right, number_of_points_right = analyze_data(column_data_right, period)

    start_time = start_time_column.min()
    end_time = start_time_column.max()

    st.subheader(f"Statistics for {left_param}")
    st.write(f"Maximum Value: {np.max(maxVal_left):.2f} {left_unit}")
    st.write(f"Minimum Value: {np.min(minVal_left):.2f} {left_unit}")
    st.write(f"Average Value: {np.mean(AvgVal_left):.2f} {left_unit}")

    if right_column_optional:
        st.subheader(f"Statistics for {right_param}")
        st.write(f"Maximum Value: {np.max(maxVal_right):.2f} {right_unit}")
        st.write(f"Minimum Value: {np.min(minVal_right):.2f} {right_unit}")
        st.write(f"Average Value: {np.mean(AvgVal_right):.2f} {right_unit}")

    # Exceedance Calculation
    if calculate_exceedance and show_threshold_lines:
        if thresholds:
            exceedance_results = {}
            for label, value in thresholds.items():
                exceedance_results[label] = sum(AvgVal_left > value) / len(AvgVal_left) * 100

            st.subheader(f"📊 PM Exceedance Calculation for {left_param}")
            for label, percentage in exceedance_results.items():
                st.write(f"❌ **{label}:** Exceeded in **{percentage:.2f}%** of the time.")

            if right_column_optional:
                exceedance_results_right = {}
                for label, value in thresholds.items():
                    exceedance_results_right[label] = sum(AvgVal_right > value) / len(AvgVal_right) * 100

                st.subheader(f"📊 PM Exceedance Calculation for {right_param}")
                for label, percentage in exceedance_results_right.items():
                    st.write(f"❌ **{label}:** Exceeded in **{percentage:.2f}%** of the time.")


else:
    st.warning("Please upload a CSV file to get started.")
