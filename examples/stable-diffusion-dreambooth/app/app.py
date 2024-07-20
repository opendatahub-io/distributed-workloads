#!/usr/bin/env python

# Credit: https://medium.com/@dmitri.mahayana/create-chatbot-text-to-image-using-stable-diffusion-and-streamlit-fb88ffda97d2

import streamlit as st
import requests
import json
import os
from PIL import Image
import numpy as np

import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(message)s',
    handlers=[
        logging.StreamHandler()  # Also log to the console
    ]
)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

api_url = os.environ.get("MODEL_API_ENDPOINT")
model_name = os.environ.get("MODEL_NAME")

def call_api(prompt):

    payload = {
        "instances": [
            {
            "data": prompt
            }
        ]
    } 

    endpoint = f"{api_url}/v1/models/{model_name}:predict"

    logger.info(f"API endpoint: {endpoint}")

    response = requests.post(endpoint, json=payload, verify=False)

    return json.loads(response.text)['predictions'][0]

st.set_page_config(page_title="Chatbot - Text to Image")
st.title("💬 Chatbot - Text to Image")
st.caption("🚀 A Streamlit chatbot powered by Stable Diffusion")

if "messages" not in st.session_state:
    st.session_state["messages"] = [
        {"role": "assistant", "content": "What kind of image that I need to draw? (example: running cat)"}]

# Show previous prompts and results that saved in session
for message in st.session_state.messages:
    st.chat_message(message["role"]).write(message["content"])
    if "image" in message:
        st.chat_message("assistant").image(message["image"], caption=message["prompt"], use_column_width=True)

if prompt := st.chat_input():    
    # Input prompt
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)

    logger.info(f"User prompt: {prompt}")

    # Query Stable Diffusion
    image = Image.fromarray(np.array(call_api(prompt), dtype="uint8"))
    msg = f'here is your image related to "{prompt}"'

    # Show Result
    st.session_state.messages.append({"role": "assistant", "content": msg, "prompt": prompt, "image": image})
    st.chat_message("assistant").write(msg)
    st.chat_message("assistant").image(image, caption=prompt, use_column_width=True)
