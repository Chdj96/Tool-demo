import streamlit as st
from PIL import Image
import os

# Page configuration
st.set_page_config(page_title="90 Green", layout="wide")

# Base path to your image folder
base_path = r"D:\chrf\pythonProject1\tool\images"

def load_local_image(image_filename):
    """
    Load image from local file path.
    """
    image_path = os.path.join(base_path, image_filename)
    if os.path.exists(image_path):
        return Image.open(image_path)
    else:
        st.error(f"❌ Image not found: {image_path}")
        return None

# --- Logo Loading ---
logo = load_local_image("Logo.jpg")

if logo:
    st.sidebar.image(logo, use_container_width=True)
else:
    st.sidebar.warning("⚠️ Logo not available")

# --- Main Content ---
st.title("Welcome to 90green")
st.write(
    "### Data-Driven Urban Sustainability\n"
    "Mit innovativer Sensortechnologie und datenbasierten Insights treiben "
    "wir den Wandel hin zu sauberer Luft und klimafreundlichen Städten voran."
)

# --- Main Image ---
main_image = load_local_image("Kopie von clean16")

if main_image:
    st.image(main_image,
             caption="Nachhaltigkeit messbar machen",
             use_column_width=True)
else:
    st.warning("⚠️ Main image not available")

# --- Footer ---
st.markdown("---")
st.write("Developed by **90green** | Data-Driven Urban Sustainability")
