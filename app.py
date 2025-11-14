import streamlit as st
from dotenv import load_dotenv
import os
from io import BytesIO
from PIL import Image as PILImage
import uuid
import httpx
import json

# Load environment variables
load_dotenv()

# API configuration
API_URL = os.getenv("API_URL", "http://localhost:8000")

# Streamlit page config
st.set_page_config(
    page_title="Image Query Assistant",
    page_icon="🖼️",
    layout="wide"
)

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "uploaded_image_bytes" not in st.session_state:
    st.session_state.uploaded_image_bytes = None
if "session_id" not in st.session_state:
    st.session_state.session_id = f"image_session_{uuid.uuid4().hex[:8]}"


def parse_sse_line(line: str):
    """Parse SSE line and extract event data"""
    if line.startswith("data: "):
        data_str = line[6:]  # Remove "data: " prefix
        if data_str.strip() == "[DONE]":
            return None, None
        try:
            data = json.loads(data_str)
            return data.get("event"), data
        except json.JSONDecodeError:
            return None, None
    elif line.startswith("event: "):
        return line[7:].strip(), None
    return None, None


def stream_image_query(query: str, session_id: str, image_bytes: bytes):
    """Call FastAPI endpoint and stream response"""
    url = f"{API_URL}/image-query"

    # Prepare multipart form data
    files = {
        "image": ("image.jpg", BytesIO(image_bytes), "image/jpeg")
    }
    data = {
        "query": query,
        "session_id": session_id
    }

    # Use httpx to stream the response
    with httpx.Client(timeout=120.0) as client:
        with client.stream("POST", url, files=files, data=data) as response:
            if response.status_code != 200:
                raise Exception("Service temporarily unavailable. Please try again later.")

            full_text = ""
            current_event = None

            for line in response.iter_lines():
                if not line:
                    continue

                event_type, event_data = parse_sse_line(line)

                if event_type:
                    current_event = event_type

                if event_data:
                    if current_event == "message.delta":
                        delta_content = event_data.get("content", "")
                        full_text += delta_content
                        yield delta_content
                    elif current_event == "message.completed":
                        # Final message received
                        final_content = event_data.get("content", "")
                        if final_content and not full_text:
                            # In case we didn't get deltas, use final content
                            yield final_content
                            full_text = final_content
                    elif current_event == "error":
                        raise Exception("Unable to process your request. Please try again.")

            return full_text


# Title and description
st.title("🖼️ Image Query Assistant")
st.markdown("Upload an image and ask questions about it. Get instant, intelligent answers!")

# Sidebar for image upload
with st.sidebar:
    st.header("📤 Upload Image")

    uploaded_file = st.file_uploader(
        "Choose an image file",
        type=["jpg", "jpeg", "png", "webp", "gif"],
        help="Upload an image to analyze"
    )

    if uploaded_file is not None:
        # Read and store image bytes
        image_bytes = uploaded_file.read()
        st.session_state.uploaded_image_bytes = image_bytes

        # Display the uploaded image
        pil_image = PILImage.open(BytesIO(image_bytes))
        st.image(pil_image, caption="Uploaded Image", use_container_width=True)

        # Image info
        st.info(f"**Size:** {pil_image.size[0]} x {pil_image.size[1]}\n\n**Format:** {pil_image.format}")

        # Clear chat button
        if st.button("🗑️ Clear Chat History", use_container_width=True):
            st.session_state.messages = []
            st.rerun()
    else:
        st.info("👆 Please upload an image to get started")

    # Reset session button
    st.markdown("---")
    if st.button("🔄 Start New Conversation", use_container_width=True):
        st.session_state.session_id = f"image_session_{uuid.uuid4().hex[:8]}"
        st.session_state.messages = []
        st.rerun()

# Main chat interface
if st.session_state.uploaded_image_bytes is not None:
    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Chat input
    if prompt := st.chat_input("Ask a question about the image..."):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})

        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)

        # Stream assistant response
        with st.chat_message("assistant"):
            try:
                # Use st.write_stream for proper streaming with typewriter effect
                full_response = st.write_stream(
                    stream_image_query(
                        prompt,
                        st.session_state.session_id,
                        st.session_state.uploaded_image_bytes
                    )
                )

                # Add assistant response to chat history
                if full_response:
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": full_response
                    })

            except Exception as e:
                st.error("❌ Sorry, something went wrong. Please try again.")
                # Log error for debugging (not shown to user)
                import logging
                logging.error(f"Error in image query: {e}")

else:
    # Empty state - No image uploaded
    st.info("👈 **Please upload an image from the sidebar to start chatting!**")

    # Example queries
    st.markdown("### 💡 Example Questions You Can Ask:")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        **Content Analysis:**
        - What objects are in this image?
        - Describe the scene in detail
        - What colors are dominant?
        - What's the mood or atmosphere?
        """)

    with col2:
        st.markdown("""
        **Detailed Queries:**
        - Are there any people in the image?
        - What's the setting or location?
        - What text can you see?
        - Identify any brands or logos
        """)

    # Features section
    st.markdown("---")
    st.markdown("### ✨ Features:")
    st.markdown("""
    - 🎯 **Real-time Responses:** Get instant answers as they're generated
    - 🧠 **Context Awareness:** Ask follow-up questions naturally
    - 💬 **Conversation Memory:** Your chat history is remembered
    - 🖼️ **Multi-format Support:** Works with JPG, PNG, WEBP, and GIF images
    """)
