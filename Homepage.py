import streamlit as st
from PIL import Image
import requests
from io import BytesIO

# Page config
st.set_page_config(page_title="90 Green", layout="wide")

# Load image from raw GitHub URL
def load_image_from_url(url):
    try:
        response = requests.get(url)
        response.raise_for_status()
        return Image.open(BytesIO(response.content))
    except Exception as e:
        st.warning(f"⚠️ Could not load image from URL: {e}")
        return None

# Logo URL (raw format)
logo_url = "https://raw.githubusercontent.com/Chdj96/Tool-demo/main/images/Logo.jpg"
logo = load_image_from_url(logo_url)

if logo is not None:
    st.sidebar.image(logo)
else:
    st.sidebar.warning("⚠️ Logo could not be loaded.")

# Title & Description
st.title("Welcome to 90green")
st.write(
    "### Data-Driven Urban Sustainability\n"
    "Mit innovativer Sensortechnologie und datenbasierten Insights treiben "
    "wir den Wandel hin zu sauberer Luft und klimafreundlichen Städten voran."
)

# Main image URL (ensure this exists in your repo!)
main_image_url = "https://raw.githubusercontent.com/Chdj96/Tool-demo/main/images/Kopie%20von%20clean16.jpg"
main_image = load_image_from_url(main_image_url)

if main_image is not None:
    st.image(main_image, caption="Nachhaltigkeit messbar machen")
else:
    st.warning("⚠️ Main image could not be loaded.")
# Footer
st.markdown("---")
st.write("Developed by **90green** | Data-Driven Urban Sustainability")
