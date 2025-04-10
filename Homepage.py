import streamlit as st
from PIL import Image
import os

# Set up page configuration
st.set_page_config(page_title="90 Green", layout="wide")

# Get current directory
current_dir = os.path.dirname(os.path.abspath(__file__))

# Correct paths - using os.path.join for cross-platform compatibility
logo_path = os.path.join(current_dir, "Images", "Logo.jpg")
image_path = os.path.join(current_dir, "Images", "clean16.jpg")

# Load logo with better error handling
try:
    logo = Image.open(logo_path)
    st.sidebar.image(logo, use_container_width=True)
except FileNotFoundError:
    st.sidebar.error(f"⚠️ Logo not found at: {logo_path}")
    # Fallback to URL if available
    try:
        logo_url = "https://raw.githubusercontent.com/yourusername/yourrepo/main/Images/Logo.jpg"
        logo = Image.open(requests.get(logo_url, stream=True).raw)
        st.sidebar.image(logo, use_container_width=True)
    except:
        st.sidebar.warning("Using placeholder instead of logo")
except Exception as e:
    st.sidebar.error(f"⚠️ Logo error: {str(e)}")

# Main content
st.title("Welcome to 90green")
st.write(
    "### Data-Driven Urban Sustainability\n"
    "Mit innovativer Sensortechnologie und datenbasierten Insights treiben "
    "wir den Wandel hin zu sauberer Luft und klimafreundlichen Städten voran."
)

# Load main image
try:
    image = Image.open(image_path)
    st.image(image, caption="Nachhaltigkeit messbar machen", use_column_width=True)
except FileNotFoundError:
    st.error(f"⚠️ Main image not found at: {image_path}")
    # Fallback to URL if available
    try:
        image_url = "https://raw.githubusercontent.com/yourusername/yourrepo/main/Images/clean16.jpg"
        image = Image.open(requests.get(image_url, stream=True).raw)
        st.image(image, caption="Nachhaltigkeit messbar machen", use_column_width=True)
    except:
        st.warning("Couldn't load main image")
except Exception as e:
    st.error(f"⚠️ Image error: {str(e)}")

# Footer
st.markdown("---")
st.write("Developed by **90green** | Data-Driven Urban Sustainability")
