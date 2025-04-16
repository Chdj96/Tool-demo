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


# FIXED: Data analysis function using proper time-based periods
def analyze_data_by_period(df, column_name, time_column, period_days):
    """Calculate statistics by selected time period"""
    # Ensure datetime type
    df = df.copy()  # Create a copy to avoid modifying the original dataframe
    df[time_column] = pd.to_datetime(df[time_column], errors='coerce')

    # Drop rows with invalid timestamps
    df = df.dropna(subset=[time_column])

    # Sort by time
    df = df.sort_values(by=time_column)

    # Calculate period duration
    start_time = df[time_column].min()
    end_time = df[time_column].max()
    total_days = (end_time - start_time).total_seconds() / (12 * 3600)

    # Calculate number of periods
    num_periods = max(1, int(np.ceil(total_days / period_days)))

    # Initialize result arrays
    period_starts = []
    max_values = []
    avg_values = []
    min_values = []

    # Process each period
    for i in range(num_periods):
        period_start = start_time + timedelta(days=i * period_days)
        period_end = start_time + timedelta(days=(i + 1) * period_days)

        # Filter data for this period
        period_data = df[(df[time_column] >= period_start) & (df[time_column] < period_end)]

        # Skip if no data in this period
        if len(period_data) == 0:
            continue

        # Calculate statistics
        period_starts.append(period_start)
        max_values.append(period_data[column_name].max())
        avg_values.append(period_data[column_name].mean())
        min_values.append(period_data[column_name].min())

    return period_starts, max_values, avg_values, min_values


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

    # Plot the main line
    ax.plot(x, y, label=f"{param_left_clean} ({left_unit})", color="green", linewidth=2)

    # Get the WHO threshold if thresholds are provided and show_thresholds is True
    who_threshold = thresholds.get("Daily Average (WHO Recommendation)",
                                   None) if show_thresholds and thresholds else None

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
            x_right = np.linspace(0, len(data_left) - 1, len(data_right))
            ax.plot(x_right, data_right, label=f"{param_right_clean} ({right_unit})", linestyle="solid", color="blue")
            ax.fill_between(range(len(data_right)), data_right, alpha=0.1, color="skyblue")

        # Add threshold lines if needed
    if show_thresholds and thresholds:
        for label, value in thresholds.items():
            ax.axhline(y=value, color='yellow' if "UBA" in label else 'red', linestyle='--', linewidth=1.5,
                       label=f"{label}: {value} µg/m³")

    # Time axis formatting
    if start_time and end_time:
        num_segments = 15
        tick_indices = np.linspace(0, len(data_left) - 1, num_segments, dtype=int)
        time_range = pd.date_range(start=start_time, end=end_time, periods=num_segments)
        time_labels = [t.strftime('%d.%m.%Y') for t in time_range]

        ax.set_xticks(tick_indices)
        ax.set_xticklabels(time_labels, rotation=45, ha='right')

    # Adjust y-axis
    y_max = max(np.max(data_left), np.max(data_right) if data_right is not None else 0)
    ax.set_ylim(0, y_max * 1.2)

    ax.set_title(title)
    ax.legend(title="Parameters", loc="best")
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


# ADDED: Function to plot time-based periods
def plot_period_stats(period_starts, max_vals, avg_vals, min_vals, title, param_name, unit, show_thresholds=False,
                      thresholds=None):
    fig, ax = plt.subplots(figsize=(12, 6))

    # Check if data exists
    if len(period_starts) == 0:
        st.warning("⚠️ No valid data for the selected period")
        return

    # Format x-axis labels
    time_labels = [t.strftime('%d.%m.%Y') for t in period_starts]

    # Plot lines
    ax.plot(time_labels, avg_vals, label=f"Average {param_name}", color="green", linewidth=1.5)

    # Add threshold lines if applicable
    if show_thresholds and thresholds:
        for label, value in thresholds.items():
            ax.axhline(y=value, color='yellow' if "UBA" in label else 'red', linestyle='--', linewidth=1.5,
                       label=f"{label}: {value} µg/m³")

    # Format the plot
    ax.set_title(title)
    ax.set_xlabel("Period Start Date")
    ax.set_ylabel(f"Value ({unit})")
    ax.tick_params(axis='x', rotation=45)
    ax.legend()

    st.pyplot(fig)

    # Save and add download button
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=300, bbox_inches='tight')
    buf.seek(0)
    st.download_button(
        label="📥 Download Period Plot",
        data=buf,
        file_name=f"{title.replace(' ', '_')}.png",
        mime="image/png"
    )
    plt.close(fig)


# Main logic
# ✅ Corrected main block ending for your Streamlit app

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
            try:
                full_data['ISO8601'] = full_data['ISO8601'].dt.tz_localize('UTC').dt.tz_convert('Europe/Berlin')
            except:
                st.warning("⚠️ Timezone conversion failed, using naive timestamps")

        start_time = full_data['ISO8601'].min()
        end_time = full_data['ISO8601'].max()

        st.sidebar.header("Column Selection")
        all_columns = [col for col in full_data.columns if col != 'ISO8601']
        left_param = st.sidebar.selectbox("Select Left Column", all_columns, index=0)
        right_column_optional = st.sidebar.checkbox("Compare with Right Column")
        right_param = st.sidebar.selectbox("Select Right Column", all_columns,
                                           index=min(1, len(all_columns) - 1)) if right_column_optional else None

        left_unit = get_unit_for_column(left_param)
        right_unit = get_unit_for_column(right_param) if right_param else None

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

        # Analyze period values
        period_starts, maxVal_left, AvgVal_left, minVal_left = analyze_data_by_period(
            full_data, left_param, 'ISO8601', period
        )

        if right_column_optional:
            _, maxVal_right, AvgVal_right, minVal_right = analyze_data_by_period(
                full_data, right_param, 'ISO8601', period
            )

        # Plotting average values (no dots, only smoothed lines)
        st.subheader("📈 Average Values Plot")
        create_gradient_plot(
            data_left=AvgVal_left,
            data_right=AvgVal_right if right_column_optional else None,
            title="Average Values",
            param_left=f"S1. {left_param}",
            param_right=f"S2. {right_param}" if right_column_optional else None,
            left_unit=left_unit,
            right_unit=right_unit,
            thresholds={k: v for k, v in thresholds.items() if show_thresholds.get(k)},
            show_thresholds=any(show_thresholds.values()),
            start_time=period_starts[0],
            end_time=period_starts[-1]
        )

        # Statistics display
        st.subheader(f"📊 Statistics for {left_param}")
        st.write(f"Maximum Value: {np.max(maxVal_left):.2f} {left_unit}")
        st.write(f"Minimum Value: {np.min(minVal_left):.2f} {left_unit}")
        st.write(f"Average Value: {np.mean(AvgVal_left):.2f} {left_unit}")

        if right_column_optional:
            st.subheader(f"📊 Statistics for {right_param}")
            st.write(f"Maximum Value: {np.max(maxVal_right):.2f} {right_unit}")
            st.write(f"Minimum Value: {np.min(minVal_right):.2f} {right_unit}")
            st.write(f"Average Value: {np.mean(AvgVal_right):.2f} {right_unit}")

        # Exceedance Calculation
        calculate_exceedance = st.sidebar.checkbox("Calculate PM Exceedance")
        if calculate_exceedance and any(show_thresholds.values()):
            exceedance_results = {
                label: sum(np.array(AvgVal_left) > value) / len(AvgVal_left) * 100
                for label, value in thresholds.items() if show_thresholds.get(label, False)
            }

            st.subheader(f"📊 PM Exceedance for {left_param}")
            for label, percentage in exceedance_results.items():
                st.write(f"❌ **{label}** exceeded in **{percentage:.2f}%** of the time.")

            if right_column_optional:
                exceedance_results_right = {
                    label: sum(np.array(AvgVal_right) > value) / len(AvgVal_right) * 100
                    for label, value in thresholds.items() if show_thresholds.get(label, False)
                }
                st.subheader(f"📊 PM Exceedance for {right_param}")
                for label, percentage in exceedance_results_right.items():
                    st.write(f"❌ **{label}** exceeded in **{percentage:.2f}%** of the time.")

    # ✅ This is the required 'except' to handle any issues during the main logic
    except Exception as e:
        st.error(f"❌ Unexpected error: {str(e)}")
