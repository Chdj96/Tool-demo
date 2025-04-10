import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import io
from streamlit_folium import folium_static
import folium
from PIL import Image
import requests
import gdown
import os
import gc
from datetime import datetime, timedelta

# Page config
st.set_page_config(page_title="Multi-Parameter Analysis Tool", layout="wide")
st.title("🌡️ Multi-Parameter Analysis Tool")
st.write("Upload your data files (Month) to analyze parameters like humidity, temperature, NOx, VOC, and PM.")

# Debug settings
DEBUG = True  # Set to False in production

# Sidebar: Logo
logo_path = r"D:\chrf\pythonProject1\tool\Logo.jpg"
try:
    logo = Image.open(logo_path)
    st.sidebar.image(logo, use_column_width=True)
except Exception as e:
    st.sidebar.warning("⚠️ Logo not found or failed to load.")

# Sidebar: File uploader and GDrive link
st.sidebar.header("User Inputs")
uploaded_files = st.sidebar.file_uploader("Upload CSV Files (One Month)", type=["csv"], accept_multiple_files=True)
google_drive_link = st.sidebar.text_input("Or enter Google Drive link to a CSV file:")
period = st.sidebar.slider("Select Time Interval (days)", 1, 6, 3)


# Helper function to round time
def round_time(dt, base=15):
    minutes = (dt.minute // base) * base
    return dt.replace(minute=minutes, second=0, microsecond=0)


# Enhanced Google Drive downloader with debugging
def download_large_csv_from_gdrive(gdrive_url):
    try:
        if DEBUG: st.write("🔍 Starting Google Drive download process...")

        # Extract file ID from different URL formats
        if "id=" in gdrive_url:
            file_id = gdrive_url.split("id=")[-1].split("&")[0]
        elif "file/d/" in gdrive_url:
            file_id = gdrive_url.split("/file/d/")[1].split("/")[0]
        else:
            file_id = gdrive_url.split("/")[-1]

        if DEBUG: st.write(f"🔍 Extracted File ID: {file_id}")

        # Create download URL
        download_url = f"https://drive.google.com/uc?id={file_id}"
        output_path = "temp_downloaded_file.csv"

        # Download with progress indication
        with st.spinner('Downloading large file... This may take several minutes for files >200MB'):
            gdown.download(download_url, output_path, quiet=False)

            # Verify download completed
            if not os.path.exists(output_path):
                st.error("❌ File download failed - no file was created")
                return None

            file_size = os.path.getsize(output_path) / (1024 * 1024)  # Size in MB
            if DEBUG: st.write(f"🔍 Successfully downloaded {file_size:.1f} MB file")

        # Read in chunks to handle large files
        chunks = []
        for chunk in pd.read_csv(output_path, chunksize=100000):  # 100k rows per chunk
            chunks.append(chunk)

        # Combine and clean up
        data = pd.concat(chunks)
        os.remove(output_path)  # Remove temporary file

        # Optimize memory usage
        for col in data.select_dtypes(include=['float64']):
            data[col] = pd.to_numeric(data[col], downcast='float')
        for col in data.select_dtypes(include=['int64']):
            data[col] = pd.to_numeric(data[col], downcast='integer')

        if DEBUG:
            st.write("🔍 Data preview:", data.head())
            st.write(f"🔍 Memory usage after loading: {data.memory_usage(deep=True).sum() / (1024 ** 2):.2f} MB")

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
        st.success("✅ Large file loaded successfully from Google Drive!")
else:
    if uploaded_files:
        for file in uploaded_files:
            try:
                # Read uploaded files in chunks
                chunks = []
                for chunk in pd.read_csv(file, chunksize=100000):  # 100k rows per chunk
                    chunks.append(chunk)
                data_list.append(pd.concat(chunks))
            except Exception as e:
                st.error(f"❌ Error reading {file.name}: {str(e)}")

        if data_list:
            st.success(f"✅ {len(uploaded_files)} files uploaded and combined successfully!")

# Sidebar: Map
st.sidebar.header("Map Settings")
latitude = st.sidebar.number_input("Latitude", value=52.5200, format="%.6f")
longitude = st.sidebar.number_input("Longitude", value=13.4050, format="%.6f")
zoom_level = st.sidebar.slider("Zoom Level", 1, 18, 12)

# Show map
st.subheader("🗺️ Interactive Map")
map_object = folium.Map(location=[latitude, longitude], zoom_start=zoom_level)
folium.Marker([latitude, longitude], popup="Selected Location").add_to(map_object)
folium_static(map_object)

# Units and thresholds
parameter_units = {
    "Humidity": "%",
    "Temperature": "°C",
    "NOx": "ppm",
    "VOC": "ppb",
    "PM": "µg/m³",
}
threshold_values_pm10 = {"Daily Average (UBA)": 50, "Daily Average (WHO Recommendation)": 45}
threshold_values_pm25 = {"Annual Average (UBA)": 25, "Daily Average (WHO Recommendation)": 15}


# Data analysis function
def analyze_data(column_data, period):
    length_of_segment = round(period * 3600 * 24)
    total_samples = len(column_data)
    number_of_points = int(np.floor(total_samples / length_of_segment))

    if number_of_points == 0:
        return np.array([]), np.array([]), np.array([]), 0  # Handle empty data

    max_values = np.zeros(number_of_points)
    avg_values = np.zeros(number_of_points)
    min_values = np.zeros(number_of_points)

    for i in range(number_of_points):
        segment_data = column_data[i * length_of_segment: (i + 1) * length_of_segment]
        max_values[i] = np.max(segment_data)
        avg_values[i] = np.mean(segment_data)
        min_values[i] = np.min(segment_data)

    return max_values, avg_values, min_values, number_of_points


# Function to get unit for a column
def get_unit_for_column(column_name):
    for param, unit in parameter_units.items():
        if param.lower() in column_name.lower():
            return unit
    return "Value"


# Optimized plot function with time axis formatting
def create_gradient_plot(data_left, data_right=None, title="", param_left="", param_right=None, left_unit="",
                         right_unit=None, show_thresholds=False, thresholds=None, start_time=None, end_time=None):
    fig, ax = plt.subplots(figsize=(12, 6))

    # Downsample if too large for performance
    max_points = 5000
    if len(data_left) > max_points:
        step = len(data_left) // max_points
        data_left = data_left[::step]
        if data_right is not None:
            data_right = data_right[::step]

    param_left_clean = param_left.replace("Left_", "S1_").replace("left_", "S1_") if param_left else "S1"
    param_right_clean = param_right.replace("Right_", "S2_").replace("right_", "S2_") if param_right else None

    x = np.arange(len(data_left))
    y = np.array(data_left)

    ax.plot(x, y, label=f"{param_left_clean} ({left_unit})", color="green", linewidth=2)

    who_threshold = thresholds.get("Daily Average (WHO Recommendation)",
                                   None) if show_thresholds and thresholds else None
    prev_above = y[0] > who_threshold if who_threshold is not None else False

    for i in range(len(x) - 1):
        current_above = y[i + 1] > who_threshold if who_threshold is not None else False

        if prev_above == current_above:
            color = 'red' if current_above else 'green'
            ax.plot([x[i], x[i + 1]], [y[i], y[i + 1]], color=color, linewidth=2)
        else:
            x_intersect = x[i] + (who_threshold - y[i]) / (y[i + 1] - y[i]) if who_threshold is not None else x[i]
            ax.plot([x[i], x_intersect], [y[i], who_threshold], color='green' if not prev_above else 'red', linewidth=2)
            ax.plot([x_intersect, x[i + 1]], [who_threshold, y[i + 1]], color='red' if not prev_above else 'green',
                    linewidth=2)

        prev_above = current_above

    if data_right is not None:
        ax.plot(data_right, label=f"{param_right_clean}", linestyle="solid", color="blue")
        ax.fill_between(range(len(data_right)), data_right, alpha=0.1, color="skyblue")

    if show_thresholds and thresholds:
        for label, value in thresholds.items():
            ax.axhline(y=value, color='yellow' if "UBA" in label else 'red', linestyle='--', linewidth=1.5,
                       label=f"{label}: {value} µg/m³")

    # Time axis formatting
    if start_time and end_time:
        num_segments = 12
        tick_indices = np.linspace(0, len(data_left) - 1, num_segments, dtype=int)
        time_range = pd.date_range(start=start_time, end=end_time, periods=num_segments)
        time_labels = [t.strftime('%d.%m.%Y\n%H:%M') for t in time_range]

        ax.set_xticks(tick_indices)
        ax.set_xticklabels(time_labels, rotation=45, ha='right')

    # Adjust y-axis
    y_max = max(data_left.max(), data_right.max() if data_right is not None else 0)
    ax.set_ylim(0, y_max * 1.2)

    ax.set_title(title)
    ax.legend()
    ax.set_xlabel("Time")
    ax.set_ylabel(f"Value ({left_unit})" if not right_unit else f"Value ({left_unit}, {right_unit})")

    st.pyplot(fig)

    # Save and add download button
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=300, bbox_inches='tight')
    buf.seek(0)
    st.download_button(
        label="📥 Download Plot",
        data=buf,
        file_name=f"{title.replace(' ', '_')}.png",
        mime="image/png"
    )
    plt.close(fig)


# Main logic
if data_list:
    try:
        # Process data in chunks if very large
        full_data = pd.concat(data_list, ignore_index=True)

        # Optimize memory usage
        for col in full_data.select_dtypes(include=['float64']).columns:
            full_data[col] = pd.to_numeric(full_data[col], downcast='float')
        for col in full_data.select_dtypes(include=['int64']).columns:
            full_data[col] = pd.to_numeric(full_data[col], downcast='integer')

        if 'ISO8601' not in full_data.columns:
            st.error("❌ Your dataset must contain an 'ISO8601' column.")
            st.stop()

        # Convert and handle timezone
        full_data['ISO8601'] = pd.to_datetime(full_data['ISO8601'], errors='coerce')
        if full_data['ISO8601'].dt.tz is None:
            full_data['ISO8601'] = full_data['ISO8601'].dt.tz_localize('UTC').dt.tz_convert('Europe/Berlin')

        start_time = full_data['ISO8601'].min()
        end_time = full_data['ISO8601'].max()

        st.sidebar.header("Column Selection")
        all_columns = full_data.columns.tolist()
        left_param = st.sidebar.selectbox("Select Left Column", all_columns, index=0)
        right_column_optional = st.sidebar.checkbox("Compare with Right Column")
        right_param = st.sidebar.selectbox("Select Right Column", all_columns,
                                           index=1) if right_column_optional else None

        left_unit = get_unit_for_column(left_param)
        right_unit = get_unit_for_column(right_param) if right_param else None

        pm_type = st.sidebar.selectbox("Select PM Type", ["PM10.0", "PM2.5"])
        thresholds = threshold_values_pm10 if pm_type == "PM10.0" else threshold_values_pm25
        show_threshold_lines = st.sidebar.checkbox("Show Threshold Lines for PM")

        column_data_left = pd.to_numeric(full_data[left_param], errors="coerce").dropna()
        column_data_right = pd.to_numeric(full_data[right_param], errors="coerce").dropna() if right_param else None





        st.subheader("📈 Time Series Visualization")
        create_gradient_plot(
            data_left=column_data_left,
            data_right=column_data_right,
            title=f"{left_param} vs {right_param}" if right_param else left_param,
            param_left=left_param,
            param_right=right_param,
            left_unit=left_unit,
            right_unit=right_unit,
            show_thresholds=show_threshold_lines,
            thresholds=thresholds,
            start_time=start_time,
            end_time=end_time
        )
        st.subheader("📊 Data Analysis")
        # Calculate statistics
        max_vals, avg_vals, min_vals, n_points = analyze_data(column_data_left, period)

        if n_points > 0:
            stats_df = pd.DataFrame({
                'Statistic': ['Max', 'Average', 'Min'],
                'Value': [max_vals.mean(), avg_vals.mean(), min_vals.mean()],
                'Unit': [left_unit, left_unit, left_unit]
            })
            st.dataframe(stats_df.style.format({'Value': '{:.2f}'}))
        # Show memory usage (for debugging)
        if DEBUG:
            st.sidebar.write(f"Data size: {len(full_data):,} rows")
            st.sidebar.write(f"Memory usage: {full_data.memory_usage(deep=True).sum() / (1024 ** 2):.2f} MB")

        # Clean up memory
        gc.collect()

    except MemoryError:
        st.error("⚠️ The dataset is too large for available memory. Try sampling the data or using a smaller file.")
    except Exception as e:
        st.error(f"❌ Error processing data: {str(e)}")
else:
    st.info("👈 Upload a CSV file or paste a Google Drive link to begin.")

# Troubleshooting section
st.markdown("---")
st.markdown("### Troubleshooting Guide")
st.markdown("""
1. **Google Drive Links Not Working?**
   - Ensure the file is shared with "Anyone with the link" permission
   - Try the direct download link format: `https://drive.google.com/uc?id=FILE_ID`

2. **Large Files (>500MB)?**
   - Downloads may take several minutes
   - Consider preprocessing your data to reduce file size

3. **Graphs Not Showing?**
   - Check your data contains numeric columns
   - Verify the 'ISO8601' column exists for timestamps
""")
