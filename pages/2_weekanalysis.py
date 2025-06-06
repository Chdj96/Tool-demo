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
# Clean missing values in a dataframe
def fill_missing_values_df(df):
    """
    Fill missing values in the DataFrame using forward fill,
    then backward fill as a fallback.
    """
    return df.fillna(method='ffill').fillna(method='bfill')
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
                         right_unit=None, show_thresholds=False, apply_thresholds=None, thresholds=None,
                         start_time=None, end_time=None, rounding_base=30):
    fig, ax = plt.subplots(figsize=(10, 6))

    param_left_clean = param_left.replace("Left_", "S1_").replace("left_", "S1_")
    param_right_clean = param_right.replace("right_", "S2_").replace("Right_", "S2_") if param_right else None

    x = np.arange(len(data_left))
    y = np.array(data_left)

    # Handle extremely large values
    if np.max(y) > 1e10:
        scaling_factor = 1e18
        y = y / scaling_factor
        ax.plot(x, y, label=f"{param_left_clean} ({left_unit}) ×10^-18", color="green", linewidth=2)
    else:
        ax.plot(x, y, label=f"{param_left_clean} ({left_unit})", color="green", linewidth=2)

    # Threshold coloring logic - only if apply_thresholds is True for any threshold
    if apply_thresholds and any(apply_thresholds.values()):
        # Find the minimum threshold that is being applied
        min_threshold = min(value for label, value in thresholds.items()
                            if apply_thresholds.get(label, False))

        prev_above = y[0] > min_threshold
        for i in range(len(x) - 1):
            current_above = y[i + 1] > min_threshold
            if prev_above == current_above:
                color = 'red' if current_above else 'green'
                ax.plot([x[i], x[i + 1]], [y[i], y[i + 1]], color=color, linewidth=2)
            else:
                x_inter = x[i] + (min_threshold - y[i]) / (y[i + 1] - y[i])
                ax.plot([x[i], x_inter], [y[i], min_threshold],
                        color='green' if not prev_above else 'red', linewidth=2)
                ax.plot([x_inter, x[i + 1]], [min_threshold, y[i + 1]],
                        color='red' if not prev_above else 'green', linewidth=2)
            prev_above = current_above

    if data_right is not None:
        right_y = np.array(data_right)
        if np.max(y) > 1e10 and np.max(right_y) < 1e10:
            right_y = right_y / scaling_factor
            ax.plot(x, right_y, label=f"{param_right_clean} ({right_unit}) ×10^-18", linestyle="solid", color="blue")
        else:
            ax.plot(x, right_y, label=f"{param_right_clean} ({right_unit})", linestyle="solid", color="blue")
        ax.fill_between(range(len(right_y)), right_y, alpha=0.1, color="skyblue")

    # Show threshold lines if requested
    if show_thresholds and thresholds:
        for label, value in thresholds.items():
            if show_thresholds.get(label, False):
                display_value = value
                if np.max(y) > 1e10 and value < 1e10:
                    value = value / scaling_factor
                    display_value = f"{display_value} ×10^-18"

                color = 'orange' if "UBA" in label else 'red'
                ax.axhline(y=value, color=color, linestyle='--', linewidth=1.5,
                           label=f"{label}: {display_value} µg/m³")

    # X-axis formatting
    num_segments = min(15, len(data_left))
    tick_indices = np.linspace(0, len(data_left) - 1, num_segments, dtype=int)

    if start_time and end_time:
        time_range = pd.date_range(start=start_time, end=end_time, periods=num_segments)
        time_labels = [round_time(t, base=rounding_base).strftime('%d.%m.%Y %H:%M') for t in time_range]
        time_labels[-1] = time_range[-1].strftime('%Y-%m-%d\n23:59')
    else:
        time_labels = [f"Point {i}" for i in range(num_segments)]

    ax.set_xticks(tick_indices)
    ax.set_xticklabels(time_labels, rotation=45, ha='right')

    # Y-axis limits
    y_max = np.max(y)
    if data_right is not None:
        y_max = max(y_max, np.max(right_y))
    ax.set_ylim(0, y_max * 1.2)

    # Labels and title
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(title="Parameters", loc="best", frameon=True, facecolor='white', edgecolor='gray')
    ax.set_xlabel("Time", fontsize=12)
    ax.set_ylabel(f"Value ({left_unit})", fontsize=12)

    plt.tight_layout()
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
# MAIN LOGIC
if data_list:
    for idx, data in enumerate(data_list):
        data = fill_missing_values_df(data)  # Clean missing values
        st.success(f"✅ File {idx+1} cleaned successfully!")

        # Now continue processing each `data`...
        data['ISO8601'] = pd.to_datetime(data['ISO8601'], errors='coerce')
        data.dropna(subset=['ISO8601'], inplace=True)

        if data['ISO8601'].dt.tz is None:
            data['ISO8601'] = data['ISO8601'].dt.tz_localize('UTC').dt.tz_convert('Europe/Berlin')

        start_time_column = data['ISO8601']

        # sidebar selections
        st.sidebar.header(f"Column Selection for File {idx+1}")
        all_columns = [col for col in data.columns if col != 'ISO8601']

        left_param = st.sidebar.selectbox(f"Select Left Column (File {idx+1})", all_columns, index=0, key=f"left_{idx}")
        right_column_optional = st.sidebar.checkbox(f"Compare with Right Column (File {idx+1})", key=f"right_optional_{idx}")

        right_param = None
        if right_column_optional:
            right_param = st.sidebar.selectbox(f"Select Right Column (File {idx+1})", all_columns, index=1, key=f"right_{idx}")

        left_unit = get_unit_for_column(left_param)
        right_unit = get_unit_for_column(right_param) if right_param else None

        pm_type = st.sidebar.selectbox(f"Select PM Type (File {idx+1})", ["PM10.0", "PM2.5"], key=f"pm_type_{idx}")
        thresholds = threshold_values_pm10 if pm_type == "PM10.0" else threshold_values_pm25

        # Threshold config
        show_thresholds = {}
        apply_thresholds = {}

        with st.sidebar.expander(f"PM Threshold Options (File {idx+1})", expanded=True):
            for label, value in thresholds.items():
                col1, col2, col3 = st.columns([2, 1, 1])
                with col1:
                    st.markdown(f"**{label}** ({value} µg/m³)")
                with col2:
                    show_thresholds[label] = st.checkbox("Show", value=True, key=f"show_{label}_{idx}")
                with col3:
                    apply_thresholds[label] = st.checkbox("Apply", value=("WHO" in label), key=f"apply_{label}_{idx}")

        column_data_left = pd.to_numeric(data[left_param], errors='coerce')

    maxVal_left, AvgVal_left, minVal_left, _ = analyze_data(column_data_left, period)

    column_data_right = None
    if right_param:
        column_data_right = pd.to_numeric(data[right_param], errors="coerce").dropna()
        maxVal_right, AvgVal_right, minVal_right, _ = analyze_data(column_data_right, period)

    start_time = start_time_column.min()
    end_time = start_time_column.max()

    st.subheader("📈 Average Values Plot")
    create_gradient_plot(
        data_left=AvgVal_left,
        data_right=AvgVal_right if right_column_optional else None,
        title="Average Values",
        param_left=f"S1. {left_param}",
        param_right=f"S2. {right_param}" if right_param else None,
        left_unit=left_unit,
        right_unit=right_unit,
        thresholds=thresholds,
        show_thresholds=show_thresholds,
        apply_thresholds=apply_thresholds,  # This was missing
        start_time=start_time,
        end_time=end_time
    )

    # Display stats
    st.subheader(f"📊 Statistics for {left_param}")
    st.write(f"Maximum Value: {np.max(maxVal_left):.2f} {left_unit}")
    st.write(f"Minimum Value: {np.min(minVal_left):.2f} {left_unit}")
    st.write(f"Average Value: {np.mean(AvgVal_left):.2f} {left_unit}")

    if right_param:
        st.subheader(f"📊 Statistics for {right_param}")
        st.write(f"Maximum Value: {np.max(maxVal_right):.2f} {right_unit}")
        st.write(f"Minimum Value: {np.min(minVal_right):.2f} {right_unit}")
        st.write(f"Average Value: {np.mean(AvgVal_right):.2f} {right_unit}")

    # Exceedance Calculation
    if st.sidebar.checkbox("Calculate PM Exceedance") and any(show_thresholds.values()):
        st.subheader(f"📊 PM Exceedance for {left_param}")
        for label, value in thresholds.items():
            if show_thresholds.get(label):
                percent = np.sum(AvgVal_left > value) / len(AvgVal_left) * 100
                st.write(f"❌ **{label}** exceeded in **{percent:.2f}%** of the time.")

        if right_param:
            st.subheader(f"📊 PM Exceedance for {right_param}")
            for label, value in thresholds.items():
                if show_thresholds.get(label):
                    percent = np.sum(AvgVal_right > value) / len(AvgVal_right) * 100
                    st.write(f"❌ **{label}** exceeded in **{percent:.2f}%** of the time.")
else:
    st.warning("⚠️ No data loaded. Please upload a CSV file to begin.")

#ChatBot
@st.cache_resource
def load_knowledge_faq():
     return [
        {
            "question": "How long does a typical air quality measurement take?",
            "answer": "Measurement campaigns usually range from a single day up to many months depending on project scope."
        },
        {
            "question": "Do you provide recommendations after the data analysis?",
            "answer": "Yes! Each client receives a tailored report with actionable measures and improvement strategies."
        },
        {
            "question": "What are your main services?",
            "answer": "We specialize in environmental data analytics, air quality monitoring with sensors, and sustainability workshops for cities and companies."
        },
        {
            "question": "How can I contact 90green?",
            "answer": "📧 Email: info@90green.com\n📞 Phone: +49 176 41 989 200"
        },
        {
            "question": "What are your office hours?",
            "answer": "🕒 Monday to Friday, 9:00 – 17:00 CET"
        }
    ]

def run_faq_assistant():
     st.markdown("## 🤖 90green Assistant")
     st.markdown("Welcome! Select a question below to learn more about our services.")

     faq_list = load_knowledge_faq()
     questions = [faq["question"] for faq in faq_list]

     selected_question = st.selectbox("Choose a question:", questions)

     if st.button("Get Answer"):
        for faq in faq_list:
            if faq["question"] == selected_question:
                st.markdown(f"**🟢 Question:** {faq['question']}")
                st.markdown(f"**💬 Answer:** {faq['answer']}")
                break

run_faq_assistant()
