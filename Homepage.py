import streamlit as st
from PIL import Image
import os

# Set up page configuration
st.set_page_config(page_title="90 Green", layout="wide")

def load_image(image_path):
    """
    Loads an image from a local path only.
    """
    try:
        if os.path.exists(image_path):
            return Image.open(image_path)
        else:
            st.error(f"❌ Image not found at: {image_path}")
            return None
    except Exception as e:
        st.error(f"❌ Error loading image: {e}")
        return None

# --- Logo Loading ---
logo_path = os.path.join("images", "Logo.jpg")
logo = load_image(logo_path)

if logo:
    st.sidebar.image(logo, use_container_width=True)
else:
    st.sidebar.warning(f"⚠️ Logo not available at {logo_path}")

# --- Main Content ---
st.title("Welcome to 90green")
st.write(
    "### Data-Driven Urban Sustainability\n"
    "Mit innovativer Sensortechnologie und datenbasierten Insights treiben "
    "wir den Wandel hin zu sauberer Luft und klimafreundlichen Städten voran."
)

# --- Main Image Loading ---
main_image_path = os.path.join("images", "Kopie von clean16.jpg")
main_image = load_image(main_image_path)

if main_image:
    st.image(main_image,
             caption="Nachhaltigkeit messbar machen",
             use_column_width=True)
else:
    st.warning(f"⚠️ Main image not available at {main_image_path}")

# --- Footer ---
st.markdown("---")
st.write("Developed by **90green** | Data-Driven Urban Sustainability")

