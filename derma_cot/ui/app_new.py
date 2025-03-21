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

# Custom CSS for modern and sleek UI
st.markdown(
    """
<style>
    /* Main theme colors */
    :root {
        --primary-color: #4A90E2;
        --secondary-color: #50E3C2;
        --background-color: #F5F7FA;
        --text-color: #2C3E50;
        --accent-color: #E74C3C;
        --light-gray: #ECF0F1;
        --dark-gray: #34495E;
    }

    /* General styling */
    .main {
        background-color: var(--background-color);
        color: var(--text-color);
    }

    /* Header styling */
    .css-1avcm0n {
        background-color: var(--dark-gray);
        border-bottom: 2px solid var(--primary-color);
    }

    /* Chat message containers */
    .user-message {
        background-color: var(--primary-color);
        color: white;
        border-radius: 15px;
        padding: 12px 16px;
        margin: 8px 0;
        border-top-right-radius: 2px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        max-width: 80%;
        align-self: flex-end;
    }

    .doctor-message {
        background-color: var(--light-gray);
        color: var(--text-color);
        border-radius: 15px;
        padding: 12px 16px;
        margin: 8px 0;
        border-top-left-radius: 2px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        max-width: 80%;
        align-self: flex-start;
    }

    /* Medical icons and badges */
    .medical-icon {
        color: var(--primary-color);
        margin-right: 5px;
    }

    .badge {
        background-color: var(--primary-color);
        color: white;
        padding: 4px 8px;
        border-radius: 10px;
        font-size: 0.8em;
        margin-right: 5px;
    }

    /* Image thumbnails */
    .thumbnail {
        border: 2px solid var(--light-gray);
        border-radius: 10px;
        padding: 4px;
    }

    /* Custom button styling */
    .stButton>button {
        background-color: var(--primary-color);
        color: white;
        border-radius: 20px;
        border: none;
        padding: 8px 20px;
        font-weight: 500;
        transition: background-color 0.3s ease;
    }

    .stButton>button:hover {
        background-color: #357ABD;
    }

    /* Sidebar styling */
    .css-1d391kg {
        background-color: var(--dark-gray);
        color: white;
        border-right: 1px solid var(--light-gray);
    }

    /* Input box styling */
    .stTextInput>div>div>input {
        border-radius: 20px;
        border: 1px solid var(--light-gray);
        padding: 10px 15px;
    }

    /* File uploader styling */
    .stFileUploader>div>button {
        background-color: var(--secondary-color);
        color: var(--text-color);
    }

    /* Progress bar */
    .stProgress>div>div>div {
        background-color: var(--primary-color);
    }

    /* Doctor avatar */
    .doctor-avatar {
        border-radius: 50%;
        border: 2px solid var(--primary-color);
        padding: 2px;
    }

    /* Patient avatar */
    .patient-avatar {
        border-radius: 50%;
        border: 2px solid var(--secondary-color);
        padding: 2px;
    }

    /* Medical record card */
    .medical-card {
        background-color: white;
        border-radius: 10px;
        padding: 15px;
        box-shadow: 0 2px 5px rgba(0,0,0,0.1);
        margin-bottom: 15px;
    }

    /* Consultation header */
    .consultation-header {
        background-color: var(--primary-color);
        color: white;
        padding: 10px 15px;
        border-radius: 10px;
        margin-bottom: 20px;
        text-align: center;
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
st.markdown(
    """
    <div class="consultation-header">
        <h1>🧑‍⚕️ DermaCOT - AI Dermatology Consultation</h1>
        <p>Welcome to your AI dermatology consultation. Share an image of your skin condition and describe your concerns below.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# Display chat history
for message in st.session_state.messages:
    if message["role"] == "user":
        with st.chat_message("user", avatar="🧑"):
            st.markdown(
                f'<div class="user-message">{message["text"]}</div>',
                unsafe_allow_html=True,
            )
            if "image_base64" in message and message["image_base64"] is not None:
                st.image(
                    base64.b64decode(message["image_base64"]),
                    caption="Uploaded Image",
                    width=300,
                    use_container_width=False,
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
        st.image(image, caption="Uploaded Image", width=300, use_container_width=False)

elif image_source == "Take Photo":
    picture = st.camera_input("Take a picture")
    if picture is not None:
        image = Image.open(picture)
        st.session_state.current_image = image
        st.image(image, caption="Captured Photo", width=300, use_container_width=False)

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
                        st.rerun()
                    except Exception as e:
                        st.error(f"Failed to parse API response: {str(e)}")
                else:
                    st.error(f"API Error: {response.text}")

            except Exception as e:
                st.error(f"Communication error: {str(e)}")
