import base64
import logging
import os
from typing import Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse
from openai import OpenAI

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI()

# Load API key from environment variables
NEBRIUS_API_KEY = os.environ.get("NEBIUS_API_KEY")
if not NEBRIUS_API_KEY:
    raise ValueError("NEBIUS_API_KEY environment variable is not set.")

# Initialize OpenAI client
client = OpenAI(base_url="https://api.studio.nebius.com/v1/", api_key=NEBRIUS_API_KEY)


@app.post("/chat")
async def chat(
    text: str = Form(...),  # Text is always required
    image: Optional[UploadFile] = File(None),  # Image is optional
):
    try:
        # Log the request
        logger.info(f"Received request with text: {text}, image: {image}")

        # Prepare the image URL if an image is uploaded
        image_url = None
        if image:
            image_content = await image.read()
            image_base64 = base64.b64encode(image_content).decode(
                "utf-8"
            )  # Properly encode to base64
            image_url = {"url": f"data:{image.content_type};base64,{image_base64}"}

        # Prepare messages for the AI model
        messages = [
            {
                "role": "system",
                "content": """You are an empatheti expert model trained at identifying medically significant features used for dermatological diagnosis of images of human skin.
                You need to process the text and/or image shared by the user and help them to understand what's going wrong with their skin.
                Use the background information provided by analysing the image and reading the text provided to assist in formulating a relevant and detailed answer.

                Follow these answer guidelines:
                1. Utilize the details observed in the image to comprehensively understand the physical condition of the human subject's skin in the image.
                2. Utilize the text content containing medically relevant information to provide a comprehensive and accurate answer.
                4. Please make sure to detail the provided patient demographic information, and how it bears a relation to the presented ailment.
                5. Ensure that you only generate a response that is a detailed description of the image and how it ties into the provided text context.
                .""",
            },
            {"role": "user", "content": [{"type": "text", "text": text}]},
        ]

        # Add image content if provided
        if image_url:
            messages[1]["content"].append({"type": "image_url", "image_url": image_url})

        # Call the AI model
        response = client.chat.completions.create(
            model="Qwen/Qwen2-VL-72B-Instruct", temperature=0, messages=messages
        )

        # Log the response
        logger.info(f"AI model response: {response}")

        # Return the response
        response_json = response.model_dump()
        return JSONResponse(content=response_json)

    except Exception as e:
        logger.error(f"An error occurred: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Run the FastAPI app
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
