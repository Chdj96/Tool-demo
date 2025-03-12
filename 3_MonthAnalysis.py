import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import io

# Page configuration
st.set_page_config(page_title="Multi-Parameter Analysis Tool", layout="wide")
st.title("🌡️ Multi-Parameter Analysis Tool")
st.write("Upload your data files (Month) to analyze various parameters such as humidity, temperature, NOx, VOC, and PM.")

# Sidebar for user inputs
st.sidebar.header("User Inputs")
uploaded_files = st.sidebar.file_uploader("Upload CSV Files (One Month)", type=["csv"], accept_multiple_files=True)
period = st.sidebar.slider("Select Time Interval days", 1, 3, 6)

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
    length_of_segment = round(period * 60*60*12)
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

# Main program logic
if uploaded_files:
    data_list = []
    for file in uploaded_files:
        data = pd.read_csv(file)
        data_list.append(data)

    full_data = pd.concat(data_list, ignore_index=True)
    st.success("Files uploaded and combined successfully!")

    if 'ISO8601' in full_data.columns:
        full_data['ISO8601'] = pd.to_datetime(full_data['ISO8601'], errors='coerce')
        if full_data['ISO8601'].dt.tz is None:
            full_data['ISO8601'] = full_data['ISO8601'].dt.tz_localize(None)
        start_time_column = full_data['ISO8601']
    else:
        st.error("The dataset must contain a 'ISO8601' column.")
        st.stop()

    st.sidebar.header("Column Selection")
    all_columns = full_data.columns.tolist()
    left_param = st.sidebar.selectbox("Select Left Column", all_columns, index=0)
    right_column_optional = st.sidebar.checkbox("Compare with Right Column")

    if right_column_optional:
        right_param = st.sidebar.selectbox("Select Right Column", all_columns, index=1)

    left_unit = get_unit_for_column(left_param)
    right_unit = get_unit_for_column(right_param) if right_column_optional else None

    pm_type = st.sidebar.selectbox("Select PM Type", ["PM10.0", "PM2.5"])
    thresholds = threshold_values_pm10 if pm_type == "PM10.0" else threshold_values_pm25
    show_threshold_lines = st.sidebar.checkbox("Show Threshold Lines for PM")

    column_data_left = pd.to_numeric(full_data[left_param], errors="coerce")
    maxVal_left, AvgVal_left, minVal_left, number_of_points_left = analyze_data(column_data_left, period)

    if right_column_optional:
        column_data_right = pd.to_numeric(full_data[right_param], errors="coerce")
        maxVal_right, AvgVal_right, minVal_right, number_of_points_right = analyze_data(column_data_right, period)

    start_time = start_time_column.min()
    end_time = start_time_column.max()

    # Generate daily time labels
    time_range = pd.date_range(start=start_time, end=end_time, freq="D")
    time_labels = [t.strftime('%d %b') for t in time_range]

    # Determine tick positions dynamically based on days
    tick_indices = np.linspace(0, len(AvgVal_left) - 1, len(time_labels)).astype(int)

    st.subheader("Average Values")
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(AvgVal_left, label=f"{left_param}", linestyle="solid", color="green")
    ax.fill_between(range(len(AvgVal_left)), AvgVal_left, alpha=0.05, color="green")

    if right_column_optional:
        ax.plot(AvgVal_right, label=f"{right_param}", linestyle="solid", color="blue")
        ax.fill_between(range(len(AvgVal_right)), AvgVal_right, alpha=0.2, color="skyblue")

    if show_threshold_lines and thresholds:
        for label, value in thresholds.items():
            ax.axhline(y=value, color='red', linestyle='--', linewidth=1.5, label=f"{label}: {value} µg/m³")

    ax.set_xticks(tick_indices)
    ax.set_xticklabels(time_labels, rotation=45, ha='right')
    ax.set_xlabel("Time (Days)")
    ax.set_ylabel(f"Value ({left_unit})" if not right_unit else f"Value ({left_unit}, {right_unit})")
    ax.legend(title="Parameters", loc="best")
    st.pyplot(fig)

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=300)
    buf.seek(0)
    st.download_button(
        label="📥 Download Plot",
        data=buf,
        file_name=f"Average_Values.png",
        mime="image/png"
    )
    plt.close(fig)

# Option to calculate PM exceedance
    calculate_exceedance = st.sidebar.checkbox("Calculate PM Exceedance")

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
        exceedance_results = {label: sum(AvgVal_left > value) / len(AvgVal_left) * 100 for label, value in thresholds.items()}

        st.subheader(f"📊 PM Exceedance Calculation for {left_param}")
        for label, percentage in exceedance_results.items():
            st.write(f"❌ **{label}:** Exceeded in **{percentage:.2f}%** of the time.")

        if right_column_optional:
            exceedance_results_right = {label: sum(AvgVal_right > value) / len(AvgVal_right) * 100 for label, value in thresholds.items()}

            st.subheader(f"📊 PM Exceedance Calculation for {right_param}")
            for label, percentage in exceedance_results_right.items():
                st.write(f"❌ **{label}:** Exceeded in **{percentage:.2f}%** of the time.")

else:
    st.warning("Please upload CSV files to get started.")
