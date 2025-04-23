import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import io
import tempfile
import folium
from streamlit_folium import folium_static
from PIL import Image
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
import time
import requests
import os
import gdown
from datetime import datetime

# Streamlit setup
st.set_page_config(page_title="Multi-Parameter Analysis Tool", layout="wide")
st.title("🌡️ Multi-Parameter Analysis Tool")
st.write("Upload your data file to analyze various parameters such as humidity, temperature, NOx, VOC, and PM.")

# Debug toggle
DEBUG = st.sidebar.checkbox("Enable Debug Logs", value=False)

def load_image_from_url(url):
    response = requests.get(url, stream=True)
    if response.status_code != 200:
        raise FileNotFoundError(f"Could not load image from {url}")
    image = Image.open(response.raw)
    return image

# Logo URL (raw format)
logo_url = "https://raw.githubusercontent.com/Chdj96/Tool-demo/main/images/Logo.jpg"
try:
    logo = load_image_from_url(logo_url)
    st.sidebar.image(logo)
except Exception as e:
    st.sidebar.warning(f"⚠️ Could not load logo: {str(e)}")

# Sidebar: File uploader and GDrive link
st.sidebar.header("User Inputs")
uploaded_files = st.sidebar.file_uploader("Upload CSV Files (One Month)", type=["csv"], accept_multiple_files=True)
google_drive_link = st.sidebar.text_input("Or enter Google Drive link to a CSV file:")
period = st.sidebar.slider("Select Time Interval (minutes)", 1, 60, 180)

def download_large_csv_from_gdrive(gdrive_url):
    try:
        if DEBUG:
            st.write("🔍 Starting Google Drive download process...")

        if "id=" in gdrive_url:
            file_id = gdrive_url.split("id=")[-1].split("&")[0]
        elif "file/d/" in gdrive_url:
            file_id = gdrive_url.split("/file/d/")[1].split("/")[0]
        else:
            file_id = gdrive_url.split("/")[-1]

        if DEBUG:
            st.write(f"🔍 Extracted File ID: {file_id}")

        download_url = f"https://drive.google.com/uc?id={file_id}"
        output_path = "temp_downloaded_file.csv"

        with st.spinner('Downloading large file... This may take several minutes for files >200MB'):
            gdown.download(download_url, output_path, quiet=False)

            if not os.path.exists(output_path):
                st.error("❌ File download failed - no file was created")
                return None

            file_size = os.path.getsize(output_path) / (1024 * 1024)
            if DEBUG:
                st.write(f"🔍 Downloaded {file_size:.1f} MB")

        chunks = []
        for chunk in pd.read_csv(output_path, chunksize=100000):
            chunks.append(chunk)

        data = pd.concat(chunks)
        os.remove(output_path)

        for col in data.select_dtypes(include=['float64']):
            data[col] = pd.to_numeric(data[col], downcast='float')
        for col in data.select_dtypes(include=['int64']):
            data[col] = pd.to_numeric(data[col], downcast='integer')

        if DEBUG:
            st.write("🔍 Data preview:", data.head())
            st.write(f"🔍 Memory usage: {data.memory_usage(deep=True).sum() / (1024 ** 2):.2f} MB")

        return data

    except Exception as e:
        st.error(f"❌ Download failed with error: {str(e)}")
        return None

# Handle Google Drive link or uploaded files
data_list = []

if google_drive_link:
    gdrive_data = download_large_csv_from_gdrive(google_drive_link)
    if gdrive_data is not None:
        data_list = [gdrive_data]
        st.success("✅ File loaded from Google Drive!")
else:
    if uploaded_files:
        for file in uploaded_files:
            try:
                chunks = []
                for chunk in pd.read_csv(file, chunksize=100000):
                    chunks.append(chunk)
                data_list.append(pd.concat(chunks))
            except Exception as e:
                st.error(f"❌ Error reading {file.name}: {str(e)}")
        if data_list:
            st.success(f"✅ {len(uploaded_files)} files uploaded and combined!")

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
    temp_html.close()
    driver.get("file://" + temp_html.name)
    time.sleep(2)

    temp_screenshot = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
    driver.save_screenshot(temp_screenshot.name)
    driver.quit()
    return temp_screenshot.name

if st.button("📥 Download Map"):
    map_path = save_map_screenshot()
    with open(map_path, "rb") as file:
        st.download_button(
            label="Download Map as Image",
            data=file,
            file_name="map.png",
            mime="image/png"
        )

# Thresholds and units
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
    length = round(period * 60)
    total = len(column_data)
    segments = int(np.floor(total / length))

    max_vals = np.zeros(segments)
    avg_vals = np.zeros(segments)
    min_vals = np.zeros(segments)

    for i in range(segments):
        seg = column_data[i * length: (i + 1) * length]
        max_vals[i] = np.max(seg)
        avg_vals[i] = np.mean(seg)
        min_vals[i] = np.min(seg)

    return max_vals, avg_vals, min_vals, segments

def get_unit_for_column(column_name):
    for param, unit in parameter_units.items():
        if param.lower() in column_name.lower():
            return unit
    return "Value"

def round_time(dt, base=30):
    new_minute = (dt.minute // base) * base
    return dt.replace(minute=new_minute, second=0, microsecond=0)

def create_gradient_plot(data_left, data_right=None, title="", param_left="", param_right=None, left_unit="",
                          right_unit=None, show_thresholds=False, thresholds=None, start_time=None, end_time=None,
                          rounding_base=30):
    fig, ax = plt.subplots(figsize=(10, 6))

    param_left_clean = param_left.replace("Left_", "S1_").replace("left_", "S1_")
    param_right_clean = param_right.replace("right_", "S2_").replace("Right_", "S2_") if param_right else None

    x = np.arange(len(data_left))
    y = np.array(data_left)

    ax.plot(x, y, label=f"{param_left_clean} ({left_unit})", color="green", linewidth=2)

    who_threshold = thresholds.get("Daily Average (WHO Recommendation)", None) if show_thresholds and thresholds else None

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

    if data_right is not None:
        ax.plot(data_right, label=f"{param_right_clean} ({right_unit})", linestyle="solid", color="blue")
        ax.fill_between(range(len(data_right)), data_right, alpha=0.1, color="skyblue")

    if show_thresholds and thresholds:
        for label, value in thresholds.items():
            ax.axhline(y=value, color='yellow' if "UBA" in label else 'red', linestyle='--', linewidth=1.5,
                       label=f"{label}: {value} µg/m³")

    num_segments = 15
    tick_indices = np.linspace(0, len(data_left) - 1, num_segments, dtype=int)
    time_range = pd.date_range(start=start_time, end=end_time, periods=num_segments)
    time_labels = [round_time(t, base=rounding_base).strftime('%d.%m.%Y %H:%M') for t in time_range]
    time_labels[-1] = time_range[-1].strftime('%Y-%m-%d\n23:59')

    ax.set_xticks(tick_indices)
    ax.set_xticklabels(time_labels, rotation=45, ha='right')

    mean_value = max(np.max(data_left), np.max(data_right) if data_right is not None else 0)
    ax.set_ylim(0, mean_value * 1.2)

    ax.set_title(title)
    ax.legend(title="Parameters", loc="best")
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

# Your main app logic continues from here...
