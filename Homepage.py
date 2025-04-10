import streamlit as st
from PIL import Image  # Import Pillow to handle images

# Set up page configuration
st.set_page_config(page_title="90 Green", layout="wide")

# Load and display the company logo at the top of the sidebar
logo_path = r"D:\chrf\pythonProject1\tool\Logo.jpg"  # Adjust the path accordingly

try:
    logo = Image.open(logo_path)  # Open logo image
    st.sidebar.image(logo, use_container_width=True)  # Ensure it stays at the top
except FileNotFoundError:
    st.sidebar.error("⚠️ Logo not found. Please check the file path.")
except Exception as e:
    st.sidebar.error(f"⚠️ An error occurred: {e}")

# Sidebar Navigation
st.sidebar.title("Navigation")

# Main Title
st.title("Welcome to 90green")
st.write(
    "### Data-Driven Urban Sustainability\n"
    "Mit innovativer Sensortechnologie und datenbasierten Insights treiben "
    "wir den Wandel hin zu sauberer Luft und klimafreundlichen Städten voran."
)

# 📌 Load and display the main image
image_path = r"D:\chrf\pythonProject1\tool\Kopie von clean16.jpg"

try:
    image = Image.open(image_path)
    st.image(image, caption="Nachhaltigkeit messbar machen")
except FileNotFoundError:
    st.error("⚠️ Image not found. Please check the file path.")
except Exception as e:
    st.error(f"⚠️ An error occurred: {e}")

# Footer
st.markdown("---")
st.write("Developed by **90green** | Data-Driven Urban Sustainability")
