import streamlit as st
import requests
import os
import zipfile
import torch
import clip
from PIL import Image
from io import BytesIO
from dotenv import load_dotenv

load_dotenv()
PEXELS_API_KEY = os.getenv("PEXELS_API_KEY")
PEXELS_URL = "https://api.pexels.com/v1/search"

# ✅ Load CLIP model for AI-based filtering
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

def fetch_image_urls(query, num_images, resolution):
    headers = {"Authorization": PEXELS_API_KEY}
    image_urls = []
    per_page = 80
    pages = (num_images // per_page) + 1

    for page in range(1, pages + 1):
        params = {"query": query, "per_page": min(per_page, num_images - len(image_urls)), "page": page}
        response = requests.get(PEXELS_URL, headers=headers, params=params)

        if response.status_code == 200:
            results = response.json()
            for img in results.get("photos", []):
                if resolution == "640x480":
                    image_urls.append(img["src"]["small"])
                elif resolution == "1280x720":
                    image_urls.append(img["src"]["medium"])
                elif resolution == "1920x1080":
                    image_urls.append(img["src"]["large"])
                else:
                    image_urls.append(img["src"]["original"])
        else:
            st.error(f"Failed to fetch images: {response.status_code}")
            return []

        if len(image_urls) >= num_images:
            break

    return image_urls[:num_images]

def filter_images(image_urls, query):
    filtered_urls = []
    query_embedding = model.encode_text(clip.tokenize([query]).to(device))
    query_embedding /= query_embedding.norm(dim=-1, keepdim=True)

    for url in image_urls:
        try:
            response = requests.get(url, stream=True)
            img = Image.open(BytesIO(response.content)).convert("RGB")
            img_tensor = preprocess(img).unsqueeze(0).to(device)

            with torch.no_grad():
                image_features = model.encode_image(img_tensor)
                image_features /= image_features.norm(dim=-1, keepdim=True)  # Normalize
                similarity = (query_embedding @ image_features.T).squeeze().item()

            if similarity > 0.2:
                filtered_urls.append(url)
        except Exception as e:
            st.error(f"Error processing image: {e}")

    return filtered_urls

def download_images_as_zip(image_urls):
    zip_buffer = BytesIO()

    with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zip_file:
        for i, url in enumerate(image_urls):
            try:
                response = requests.get(url, stream=True)
                if response.status_code == 200:
                    img_data = response.content
                    zip_file.writestr(f"image_{i+1}.jpg", img_data)
            except Exception as e:
                st.error(f"Error downloading image {i+1}: {e}")

    zip_buffer.seek(0)
    return zip_buffer

st.set_page_config(page_title="Image Downloader", page_icon="📸", layout="wide")
st.markdown("<h1 style='text-align: center;'>📷 Image Downloader</h1>", unsafe_allow_html=True)

# Sidebar Inputs
st.sidebar.title("🔍 Image Search Settings")
query = st.sidebar.text_input("Enter search keyword:")
num_images = st.sidebar.number_input("Number of images:", min_value=1, max_value=1000, value=10, step=10)
resolution = st.sidebar.selectbox("Select Resolution:", ["640x480", "1280x720", "1920x1080", "Original"])
ai_filter = st.sidebar.checkbox("Use AI Filtering (Recommended)", value=True)

# Fetch Images Button
if st.sidebar.button("Fetch Images"):
    if not query:
        st.sidebar.error("Please enter a keyword.")
    else:
        with st.spinner("Fetching images..."):
            image_urls = fetch_image_urls(query, num_images, resolution)

            if ai_filter:
                image_urls = filter_images(image_urls, query)

        if image_urls:
            st.success(f"Fetched {len(image_urls)} images successfully! 🎉")


            MAX_DISPLAY = 20
            cols = st.columns(min(5, len(image_urls)))
            for i, url in enumerate(image_urls[:MAX_DISPLAY]):
                with cols[i % len(cols)]:
                    st.image(url, use_container_width=True)

            st.write(f"Showing {min(len(image_urls), MAX_DISPLAY)} out of {len(image_urls)} images.")

            zip_buffer = download_images_as_zip(image_urls)
            st.download_button(
                label="📥 Download All Images as ZIP",
                data=zip_buffer,
                file_name=f"{query}_images.zip",
                mime="application/zip"
            )

if st.sidebar.button("Clear Images"):
    st.session_state.image_urls = []
    st.rerun()
# Footer
footer_style = """
    <style>
        .footer {
            position: fixed;
            bottom: 0;
            left: 0;
            width: 100%;
            background-color: #F0F2F6;
            text-align: center;
            padding: 12px;
            font-size: 16px;
            font-weight: bold;
            color: #333; /* Darker text for contrast */
            border-top: 1px solid #ddd; /* Subtle top border */
            box-shadow: 0px -2px 5px rgba(0, 0, 0, 0.1); /* Soft shadow */
            z-index: 1000;
        }
        .footer a {
            text-decoration: none;
            color: #007BFF; 
            font-weight: bold;
            margin-left: 6px;
        }
        /* Push content up so footer doesn't overlap */
        .stApp {
            padding-bottom: 60px;
        }
    </style>
    <div class="footer">
        Developed by  
        <a href="https://github.com/ARPurohit07" target="_blank">ARPurohit07</a>
    </div>
"""
st.markdown(footer_style, unsafe_allow_html=True)
