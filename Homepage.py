import streamlit as st
from PIL import Image
import os
import requests
from io import BytesIO

# Set up page configuration
st.set_page_config(page_title="90 Green", layout="wide")

def load_image(image_path, image_url=None):
    """
    Loads an image from a local path or a remote URL as fallback.
    """
    try:
        # Try local image first
        if os.path.exists(image_path):
            return Image.open(image_path)
        # Fallback to URL if local fails
        elif image_url:
            response = requests.get(image_url)
            response.raise_for_status()
            return Image.open(BytesIO(response.content))
        else:
            st.error(f"❌ Image not found at: {image_path}")
            return None
    except Exception as e:
        st.error(f"❌ Error loading image: {e}")
        return None

# --- Logo Loading ---
logo = load_image(
    image_path=os.path.join("images", "Logo.jpg"),
    image_url="https://github.com/Chdj96/Tool-demo/blob/3dd98126e12a15127dc0f026c571e7cad5b76a86/images/Logo.jpg"
)

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

# --- Main Image Loading ---
main_image = load_image(
    image_path=os.path.join("images", " Kopie von clean16.jpg"),
    image_url="https://github.com/Chdj96/Tool-demo/blob/3dd98126e12a15127dc0f026c571e7cad5b76a86/images/Kopie%20von%20clean16.jpg"
)

if main_image:
    st.image(main_image,
             caption="Nachhaltigkeit messbar machen",
             use_column_width=True)
else:
    st.warning("⚠️ Main image not available")

# Footer
st.markdown("---")
st.write("Developed by **90green** | Data-Driven Urban Sustainability")
