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

# Streamlit setup
st.set_page_config(page_title="Multi-Parameter Analysis Tool", layout="wide")
st.title("🌡️ Multi-Parameter Analysis Tool")
st.write("Upload your data file to analyze various parameters such as humidity, temperature, NOx, VOC, and PM.")

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

# Sidebar Inputs
st.sidebar.header("User Inputs")
uploaded_file = st.sidebar.file_uploader("Upload CSV File", type=["csv"])
period = st.sidebar.slider("Select Time Interval (Minutes)", 1, 60, 10)

# Map Settings
st.sidebar.header("Map Settings")
latitude = st.sidebar.number_input("Latitude", value=52.5200, format="%.6f")
longitude = st.sidebar.number_input("Longitude", value=13.4050, format="%.6f")
zoom_level = st.sidebar.slider("Zoom Level", 1, 18, 12)

# Display Map
st.subheader("Interactive Map")
map_object = folium.Map(location=[latitude, longitude], zoom_start=zoom_level)
folium.Marker([latitude, longitude], popup="Selected Location").add_to(map_object)
folium_static(map_object)

# Screenshot
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
        st.download_button("Download Map as Image", file, file_name="map.png", mime="image/png")

# Thresholds
threshold_values_pm10 = {
    "Daily Average (UBA)": 50,
    "Daily Average (WHO Recommendation)": 45,
}
threshold_values_pm25 = {
    "Annual Average (UBA)": 25,
    "Daily Average (WHO Recommendation)": 15,
}
parameter_units = {
    "Humidity": "%",
    "Temperature": "°C",
    "NOx": "ppm",
    "VOC": "ppb",
    "PM": "µg/m³",
}

# Helper functions
def analyze_data(column_data, period):
    segment_len = round(period * 60)
    total_samples = len(column_data)
    points = int(np.floor(total_samples / segment_len))
    max_values = np.zeros(points)
    avg_values = np.zeros(points)
    min_values = np.zeros(points)

    for i in range(points):
        segment = column_data[i * segment_len:(i + 1) * segment_len]
        max_values[i] = np.max(segment)
        avg_values[i] = np.mean(segment)
        min_values[i] = np.min(segment)

    return max_values, avg_values, min_values, points

def get_unit(column):
    for param, unit in parameter_units.items():
        if param.lower() in column.lower():
            return unit
    return "Value"
def create_gradient_plot(data_left, data_right=None, title="", param_left="", param_right=None, left_unit="",
                         right_unit=None, show_thresholds=False, apply_thresholds=None, thresholds=None,
                         start_time=None, end_time=None, rounding_base=30):
    # Create figure with dynamic sizing
    fig = plt.figure(figsize=(12, 8), dpi=150)
    ax = fig.add_subplot(111)
    
    # Calculate required space for legend (add more rows if many thresholds)
    legend_rows = 1 + sum(1 for label in thresholds if show_thresholds.get(label, False))
    legend_height = 0.08 * legend_rows  # Dynamic height based on legend items
    
    # Adjust subplot to leave space for legend
    plt.subplots_adjust(bottom=0.3) 

    
   
    
    param_left_clean = param_left.replace("Left_", "S1_").replace("left_", "S1_")
    param_right_clean = param_right.replace("right_", "S2_").replace("Right_", "S2_") if param_right else None

    x = np.arange(len(data_left))
    y = np.array(data_left)

    active_thresholds = {k: v for k, v in thresholds.items() if apply_thresholds and apply_thresholds.get(k)}
    min_threshold = min(active_thresholds.values()) if active_thresholds else None

    prev_above = y[0] > min_threshold if min_threshold is not None else False
    for i in range(len(x) - 1):
        current_above = y[i + 1] > min_threshold if min_threshold is not None else False
        if prev_above == current_above:
            color = 'red' if current_above else 'green'
            ax.plot([x[i], x[i + 1]], [y[i], y[i + 1]], color=color, linewidth=2, label=param_left_clean if i == 0 else "")
        else:
            x_inter = x[i] + (min_threshold - y[i]) / (y[i + 1] - y[i]) if min_threshold is not None else x[i]
            ax.plot([x[i], x_inter], [y[i], min_threshold], color='green' if not prev_above else 'red', linewidth=2, label=param_left_clean if i == 0 else "")
            ax.plot([x_inter, x[i + 1]], [min_threshold, y[i + 1]], color='red' if not prev_above else 'green', linewidth=2)
        prev_above = current_above

    for label, value in thresholds.items():
        if show_thresholds and show_thresholds.get(label):
            color = 'orange' if "UBA" in label else 'red'
            ax.axhline(y=value, color=color, linestyle='--', linewidth=1.5, label=f"{label}: {value} µg/m³")

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

    # Dynamic Y-axis limits with buffer
    y_max = max(np.max(data_left), np.max(data_right) if data_right is not None else 0)
    buffer = max(y_max * 0.25, 5)  # Minimum 5-unit buffer
    ax.set_ylim(0, y_max + buffer)

    ax.set_xlabel("Time")
    ax.set_ylabel(f"Value ({left_unit})" if not right_unit else f"Value ({left_unit}, {right_unit})")
    ax.set_title(title)
    
    legend = ax.legend(
    loc='upper right',  # or try 'upper left'
    frameon=True,
    fancybox=True,
    shadow=True,
    fontsize='small'
)

    
    # Dynamic Y-axis limits with extra buffer
    y_max = max(np.max(data_left), np.max(data_right) if data_right is not None else 0)
    buffer = max(y_max * 0.3, 10)  # 30% buffer or minimum 10 units
    ax.set_ylim(0, y_max + buffer)
    
    # Save with tight layout
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=300, bbox_inches='tight', pad_inches=0.5)
    buf.seek(0)
    
    # Display in Streamlit
    st.pyplot(fig)
    
    # Download button
    st.download_button(
        "📥 Download Plot",
        data=buf,
        file_name=f"{title}.png",
        mime="image/png"
    )
    
    plt.close(fig)



# MAIN LOGIC
if uploaded_file:
    data = pd.read_csv(uploaded_file)
    st.success("File uploaded successfully!")

    if 'ISO8601' in data.columns:
        data['ISO8601'] = pd.to_datetime(data['ISO8601'], errors='coerce')
        if data['ISO8601'].dt.tz is None:
            data['ISO8601'] = data['ISO8601'].dt.tz_localize('UTC').dt.tz_convert('Europe/Berlin')
        time_column = data['ISO8601']
    else:
        st.error("The dataset must contain a 'ISO8601' column.")
        st.stop()

    st.sidebar.header("Column Selection")
    columns = data.columns.tolist()
    left_param = st.sidebar.selectbox("Select Left Column", columns, index=0)
    right_enabled = st.sidebar.checkbox("Compare with Right Column")
    right_param = st.sidebar.selectbox("Select Right Column", columns, index=1) if right_enabled else None

    left_unit = get_unit(left_param)
    right_unit = get_unit(right_param) if right_enabled else None

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

    data_left = pd.to_numeric(data[left_param], errors="coerce")
    maxL, avgL, minL, _ = analyze_data(data_left, period)

    if right_enabled:
        data_right = pd.to_numeric(data[right_param], errors="coerce")
        maxR, avgR, minR, _ = analyze_data(data_right, period)

    start_time = time_column.min()
    end_time = time_column.max()

    st.subheader("Average Values")
    create_gradient_plot(
        data_left=avgL,
        data_right=avgR if right_enabled else None,
        title="Average Values",
        param_left=left_param,
        param_right=right_param if right_enabled else None,
        left_unit=left_unit,
        right_unit=right_unit,
        thresholds=thresholds,
        apply_thresholds=apply_thresholds,
        show_thresholds=show_thresholds,
        start_time=start_time,
        end_time=end_time
    )

    st.subheader(f"Statistics for {left_param}")
    st.write(f"Max: {np.max(maxL):.2f} {left_unit}")
    st.write(f"Min: {np.min(minL):.2f} {left_unit}")
    st.write(f"Average: {np.mean(avgL):.2f} {left_unit}")

    if right_enabled:
        st.subheader(f"Statistics for {right_param}")
        st.write(f"Max: {np.max(maxR):.2f} {right_unit}")
        st.write(f"Min: {np.min(minR):.2f} {right_unit}")
        st.write(f"Average: {np.mean(avgR):.2f} {right_unit}")

    if st.sidebar.checkbox("Calculate PM Exceedance"):
        st.subheader(f"📊 PM Exceedance Calculation for {left_param}")
        for label, value in thresholds.items():
            pct = sum(avgL > value) / len(avgL) * 100
            st.write(f"❌ **{label}:** Exceeded in **{pct:.2f}%** of the time.")

        if right_enabled:
            st.subheader(f"📊 PM Exceedance Calculation for {right_param}")
            for label, value in thresholds.items():
                pct = sum(avgR > value) / len(avgR) * 100
                st.write(f"❌ **{label}:** Exceeded in **{pct:.2f}%** of the time.")

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
