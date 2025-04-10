import streamlit as st
from PIL import Image
import os
import requests
from io import BytesIO

# Set up page configuration
st.set_page_config(page_title="90 Green", layout="wide")

def load_image(image_path, image_url=None, caption=""):
    """
    Robust image loader with local and remote fallback
    """
    try:
        # First try local file
        if os.path.exists(image_path):
            img = Image.open(image_path)
            return img
        # Fallback to URL if provided
        elif image_url:
            response = requests.get(image_url)
            img = Image.open(BytesIO(response.content))
            return img
        else:
            st.error(f"Image not found at: {image_path}")
            return None
    except Exception as e:
        st.error(f"Error loading image: {str(e)}")
        return None

# --- Logo Loading ---
logo = load_image(
    image_path=os.path.join("Images", "Logo"),
    image_url="https://raw.githubusercontent.com/yourusername/yourrepo/main/Images/Logo.jpg"
)

if logo:
    st.sidebar.image(logo, use_container_width=True)
else:
    st.sidebar.warning("Logo not available")

# --- Main Content ---
st.title("Welcome to 90green")
st.write(
    "### Data-Driven Urban Sustainability\n"
    "Mit innovativer Sensortechnologie und datenbasierten Insights treiben "
    "wir den Wandel hin zu sauberer Luft und klimafreundlichen Städten voran."
)

# --- Main Image Loading ---
main_image = load_image(
    image_path=os.path.join("Images", "clean16"),  # Note: corrected from clean1s.jpg
    image_url="https://raw.githubusercontent.com/yourusername/yourrepo/main/Images/clean16.jpg"
)



# Footer
st.markdown("---")
st.write("Developed by **90green** | Data-Driven Urban Sustainability")
