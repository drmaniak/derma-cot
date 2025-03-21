import base64
import logging
import os
import uuid
from io import BytesIO

import requests
import streamlit as st
from PIL import Image

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(), logging.FileHandler("ui_debug.log")],
)
logger = logging.getLogger("derma_cot_ui")

# API Configuration
API_URL = os.getenv("API_URL", "http://localhost:8000")

# Page configuration
st.set_page_config(
    page_title="DermaCOT - AI Dermatology Consultation",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for medical theme
st.markdown(
    """
<style>
    /* Custom styling for chat messages */
    .user-message {
        background-color: #e3f2fd;
        border-radius: 15px;
        padding: 10px 15px;
        margin: 5px 0;
        border-top-right-radius: 2px;
        box-shadow: 0 1px 2px rgba(0,0,0,0.1);
    }

    .doctor-message {
        background-color: #f1f8e9;
        border-radius: 15px;
        padding: 10px 15px;
        margin: 5px 0;
        border-top-left-radius: 2px;
        box-shadow: 0 1px 2px rgba(0,0,0,0.1);
    }
</style>
""",
    unsafe_allow_html=True,
)

# Initialize session state variables
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())
if "messages" not in st.session_state:
    st.session_state.messages = []
if "current_image" not in st.session_state:
    st.session_state.current_image = None
if "current_image_base64" not in st.session_state:
    st.session_state.current_image_base64 = None

# Main chat interface
st.markdown("## 🧑‍⚕️ AI Dermatology Consultation")
st.markdown(
    "Welcome to your AI dermatology consultation. Share an image of your skin condition and describe your concerns below."
)

# Display chat history
for message in st.session_state.messages:
    if message["role"] == "user":
        with st.chat_message("user", avatar="🧑"):
            st.markdown(
                f'<div class="user-message">{message["text"]}</div>',
                unsafe_allow_html=True,
            )
            if (
                "image_base64" in message and message["image_base64"] is not None
            ):  # Check if image_base64 is not None
                st.image(
                    base64.b64decode(message["image_base64"]),
                    caption="Uploaded Image",
                    width=300,
                )
    elif message["role"] == "assistant":
        with st.chat_message("assistant", avatar="🩺"):
            st.markdown(
                f'<div class="doctor-message">{message["text"]}</div>',
                unsafe_allow_html=True,
            )

# Image upload section
st.markdown("### 📷 Share an Image (Optional)")
image_source = st.radio(
    "Choose image source:", ["Upload Image", "Take Photo"], horizontal=True
)

if image_source == "Upload Image":
    uploaded_file = st.file_uploader("Upload a skin image", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.session_state.current_image = image
        st.image(image, caption="Uploaded Image", width=300)

elif image_source == "Take Photo":
    picture = st.camera_input("Take a picture")
    if picture is not None:
        image = Image.open(picture)
        st.session_state.current_image = image
        st.image(image, caption="Captured Photo", width=300)

# Message input
st.markdown("### 💬 Describe Your Concern")
user_input = st.text_area("Type your message here:", height=100)

if st.button("Send Message"):
    if not user_input.strip():
        st.warning("Please enter a message.")
    else:
        with st.spinner("Getting response..."):
            try:
                # Prepare the image for the API
                files = None
                if st.session_state.current_image:
                    buffered = BytesIO()
                    st.session_state.current_image.save(buffered, format="JPEG")
                    files = {"image": ("image.jpg", buffered.getvalue(), "image/jpeg")}

                # Prepare the form data
                data = {"text": user_input}

                # Send the request to the FastAPI backend
                response = requests.post(
                    f"{API_URL}/chat/",
                    data=data,
                    files=files if files else None,
                )

                # Log the raw response for debugging
                logger.info(f"Raw API response: {response.text}")

                if response.status_code == 200:
                    try:
                        response_data = response.json()
                        # Log the parsed JSON response for debugging
                        logger.info(f"Parsed API response: {response_data}")

                        # Extract the AI's response from the API response
                        if (
                            "choices" in response_data
                            and len(response_data["choices"]) > 0
                        ):
                            ai_response = response_data["choices"][0]["message"][
                                "content"
                            ]
                        else:
                            raise ValueError(
                                "Invalid response format: 'choices' field missing or empty"
                            )

                        # Add user message to chat history
                        st.session_state.messages.append(
                            {
                                "role": "user",
                                "text": user_input,
                                "image_base64": (
                                    base64.b64encode(buffered.getvalue()).decode(
                                        "utf-8"
                                    )
                                    if files
                                    else None
                                ),
                            }
                        )
                        # Add AI response to chat history
                        st.session_state.messages.append(
                            {
                                "role": "assistant",
                                "text": ai_response,
                            }
                        )
                        # Clear the current image
                        st.session_state.current_image = None
                        st.session_state.current_image_base64 = None
                        st.rerun()  # Use st.rerun() instead of st.experimental_rerun()
                    except Exception as e:
                        st.error(f"Failed to parse API response: {str(e)}")
                else:
                    st.error(f"API Error: {response.text}")

            except Exception as e:
                st.error(f"Communication error: {str(e)}")
