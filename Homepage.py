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



#ChatBot
@st.cache_resource
def load_knowledge_faq():
     return [
        {
            "question": "How long does a typical air quality measurement take?",
            "answer": "Measurement campaigns usually range from a single day up to many months depending on project scope."
        },
        {
            "question": "Do you provide recommendations after the data analysis?",
            "answer": "Yes! Each client receives a tailored report with actionable measures and improvement strategies."
        },
        {
            "question": "What are your main services?",
            "answer": "We specialize in environmental data analytics, air quality monitoring with sensors, and sustainability workshops for cities and companies."
        },
        {
            "question": "How can I contact 90green?",
            "answer": "📧 Email: info@90green.com\n📞 Phone: +49 176 41 989 200"
        },
        {
            "question": "What are your office hours?",
            "answer": "🕒 Monday to Friday, 9:00 – 17:00 CET"
        }
    ]

def run_faq_assistant():
     st.markdown("## 🤖 90green Assistant")
     st.markdown("Welcome! Select a question below to learn more about our services.")

     faq_list = load_knowledge_faq()
     questions = [faq["question"] for faq in faq_list]

     selected_question = st.selectbox("Choose a question:", questions)

     if st.button("Get Answer"):
        for faq in faq_list:
            if faq["question"] == selected_question:
                st.markdown(f"**🟢 Question:** {faq['question']}")
                st.markdown(f"**💬 Answer:** {faq['answer']}")
                break

run_faq_assistant()

# Footer
st.markdown("---")
st.write("Developed by **90green** | Data-Driven Urban Sustainability")
