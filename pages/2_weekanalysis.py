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

# Load and display the company logo at the top of the sidebar
logo_path = r"D:\chrf\pythonProject1\tool\Logo.jpg"  # Adjust the path accordingly
try:
    logo = Image.open(logo_path)
    st.sidebar.image(logo, use_container_width=True)
except FileNotFoundError:
    st.sidebar.error("⚠️ Logo not found. Please check the file path.")
except Exception as e:
    st.sidebar.error(f"⚠️ An error occurred: {e}")

# Sidebar for user inputs
st.sidebar.header("User Inputs")
uploaded_file = st.sidebar.file_uploader("Upload CSV File", type=["csv"])
google_drive_link = st.sidebar.text_input("Enter Google Drive Link")
period = st.sidebar.slider("Select Time Interval (Minutes)", 1, 60, 120)

# Map Settings
st.sidebar.header("Map Settings")
latitude = st.sidebar.number_input("Latitude", value=52.5200, format="%.6f")
longitude = st.sidebar.number_input("Longitude", value=13.4050, format="%.6f")
zoom_level = st.sidebar.slider("Zoom Level", 1, 18, 12)

# Display interactive map
st.subheader("Interactive Map")
map_object = folium.Map(location=[latitude, longitude], zoom_start=zoom_level)
folium.Marker([latitude, longitude], popup="Selected Location").add_to(map_object)
folium_static(map_object)

def save_map_screenshot():
    options = Options()
    options.add_argument("--headless")
    options.add_argument("--disable-gpu")
    options.add_argument("--window-size=800x600")
    driver = webdriver.Chrome(options=options)

    temp_html = tempfile.NamedTemporaryFile(delete=False, suffix=".html")
    map_object.save(temp_html.name)
    driver.get("file://" + temp_html.name)
    time.sleep(2)

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

def get_unit_for_column(column_name):
    for param, unit in parameter_units.items():
        if param.lower() in column_name.lower():
            return unit
    return "Value"

def round_time(dt, base=15):
    new_minute = (dt.minute // base) * base
    return dt.replace(minute=new_minute, second=0)

def create_gradient_plot(data_left, data_right=None, title="", param_left="", param_right=None, left_unit="",
                        right_unit=None, show_thresholds=False, thresholds=None, start_time=None, end_time=None,
                        rounding_base=30):
    fig, ax = plt.subplots(figsize=(10, 6))

    # Clean legend labels
    param_left_clean = param_left.replace("Left_", "S1_").replace("left_", "S1_")
    param_right_clean = param_right.replace("right_", "S2_").replace("Right_", "S2_") if param_right else None

    x = np.arange(len(data_left))
    y = np.array(data_left)

    # Plot the main line
    ax.plot(x, y, label=f"{param_left_clean} ({left_unit})", color="green", linewidth=2)

    # Get the WHO threshold if thresholds are provided and show_thresholds is True
    who_threshold = thresholds.get("Daily Average (WHO Recommendation)", None) if show_thresholds and thresholds else None

    # Apply gradient coloring if WHO threshold is available
    if who_threshold is not None:
        prev_above = y[0] > who_threshold
        for i in range(len(x) - 1):
            current_above = y[i + 1] > who_threshold
            if prev_above and current_above:
                color = 'red'
            elif not prev_above and not current_above:
                color = 'green'
            else:
                x_intersect = x[i] + (who_threshold - y[i]) / (y[i + 1] - y[i])
                ax.plot([x[i], x_intersect], [y[i], who_threshold],
                        color='green' if not prev_above else 'red', linewidth=2)
                ax.plot([x_intersect, x[i + 1]], [who_threshold, y[i + 1]],
                        color='red' if not prev_above else 'green', linewidth=2)
                prev_above = current_above
                continue

            ax.plot([x[i], x[i + 1]], [y[i], y[i + 1]], color=color, linewidth=2)
            prev_above = current_above

    # Add right-side data if available
    if data_right is not None:
        ax.plot(data_right, label=f"{param_right_clean} ({right_unit})", linestyle="solid", color="blue")
        ax.fill_between(range(len(data_right)), data_right, alpha=0.1, color="skyblue")

    # Add threshold lines if needed
    if show_thresholds and thresholds:
        for label, value in thresholds.items():
            ax.axhline(y=value, color='yellow' if "UBA" in label else 'red', linestyle='--', linewidth=1.5,
                      label=f"{label}: {value} µg/m³")

    # Time axis formatting
    num_segments = 12
    tick_indices = np.linspace(0, len(data_left) - 1, num_segments, dtype=int)
    time_range = pd.date_range(start=start_time, end=end_time, periods=num_segments)
    time_labels = [round_time(t, base=rounding_base).strftime('%d.%m.%Y %H:%M') for t in time_range]
    time_labels[-1] = time_range[-1].strftime('%Y-%m-%d\n23:59')

    ax.set_xticks(tick_indices)
    ax.set_xticklabels(time_labels, rotation=45, ha='right')

    # Adjust y-axis
    mean_value = max(np.max(data_left), np.max(data_right) if data_right is not None else 0)
    ax.set_ylim(0, mean_value * 1.2)

    ax.set_title(title)
    ax.legend(title="Parameters", loc="best")
    ax.set_xlabel("Time")
    ax.set_ylabel(f"Value ({left_unit})" if not right_unit else f"Value ({left_unit}, {right_unit})")

    st.pyplot(fig)

    # Save and add download button
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

    if 'ISO8601' in data.columns:
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

    right_param = None
    if right_column_optional:
        right_param = st.sidebar.selectbox("Select Right Column", all_columns, index=1)

    left_unit = get_unit_for_column(left_param)
    right_unit = get_unit_for_column(right_param) if right_column_optional else None

    pm_type = st.sidebar.selectbox("Select PM Type", ["PM10.0", "PM2.5"])
    thresholds = threshold_values_pm10 if pm_type == "PM10.0" else threshold_values_pm25

    # Inside The sidebar
    show_thresholds = {}
    apply_thresholds = {}

    with st.sidebar.expander(" PM Threshold Options", expanded=True):
        st.markdown("Choose which PM thresholds to **display** as lines and which to **apply** for coloring the plot.")

        for label, value in thresholds.items():
            col1, col2, col3 = st.columns([2, 1, 1])
            with col1:
                st.markdown(f"**{label}** ({value} µg/m³)")
            with col2:
                show_thresholds[label] = st.checkbox("Show", value=True, key=f"show_{label}")
            with col3:
                apply_thresholds[label] = st.checkbox("Apply", value=("WHO" in label), key=f"apply_{label}")

    column_data_left = pd.to_numeric(data[left_param], errors="coerce")
    maxVal_left, AvgVal_left, minVal_left, number_of_points_left = analyze_data(column_data_left, period)

    column_data_right = None
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
        param_right=right_param,
        left_unit=left_unit,
        right_unit=right_unit,
        thresholds=thresholds,
        show_thresholds=any(show_thresholds.values()),
        start_time=start_time,
        end_time=end_time
    )

    # Statistics display
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
    calculate_exceedance = st.sidebar.checkbox("Calculate PM Exceedance")
    if calculate_exceedance and any(show_thresholds.values()):
        exceedance_results = {}
        for label, value in thresholds.items():
            if show_thresholds.get(label, False):
                exceedance_results[label] = sum(AvgVal_left > value) / len(AvgVal_left) * 100

        st.subheader(f"📊 PM Exceedance Calculation for {left_param}")
        for label, percentage in exceedance_results.items():
            st.write(f"❌ **{label}:** Exceeded in **{percentage:.2f}%** of the time.")

        if right_column_optional:
            exceedance_results_right = {}
            for label, value in thresholds.items():
                if show_thresholds.get(label, False):
                    exceedance_results_right[label] = sum(AvgVal_right > value) / len(AvgVal_right) * 100

            st.subheader(f"📊 PM Exceedance Calculation for {right_param}")
            for label, percentage in exceedance_results_right.items():
                st.write(f"❌ **{label}:** Exceeded in **{percentage:.2f}%** of the time.")
else:
    st.warning("Please upload a CSV file to get started.")
