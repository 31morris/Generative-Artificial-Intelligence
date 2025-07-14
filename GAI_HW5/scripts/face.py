import os
import json
import time
import random
import requests
import cv2
import numpy as np
from urllib.parse import urlparse

from autocrop import Cropper

from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.chrome.options import Options

def setup_driver():
    options = Options()
    options.add_argument("--headless")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    return webdriver.Chrome(service = Service(ChromeDriverManager().install()), options = options)

def get_headers_from_driver(driver):
    driver.get("https://www.google.com")
    user_agent = driver.execute_script("return navigator.userAgent")
    headers = {
        "User-Agent": user_agent,
        "Referer": "https://d.img.vision/",
    }
    return headers

cropper = Cropper(face_percent = 80)

def crop_and_resize_face(image_bgr):
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    face_pil = cropper.crop(image_rgb)

    if face_pil is not None:
        face_rgb = np.array(face_pil)
        resized_face = cv2.resize(face_rgb, (64, 64), interpolation = cv2.INTER_AREA)
        return resized_face
    return None

def download_images(url_list, image_dir, face_dir):
    os.makedirs(image_dir, exist_ok=True)
    os.makedirs(face_dir, exist_ok=True)

    driver = setup_driver()
    headers = get_headers_from_driver(driver)

    START_INDEX = 1
    face_index = START_INDEX

    for index, url in enumerate(url_list, start = 1):
        ext = os.path.splitext(urlparse(url).path)[-1]
        if ext.lower() not in [".jpg", ".jpeg", ".png"]:
            ext = ".jpg"

        filename = f"{index}{ext}"
        filepath = os.path.join(image_dir, filename)

        try:
            response = requests.get(url, headers = headers, timeout = 10)
            response.raise_for_status()

            if "image" not in response.headers.get("Content-Type", ""):
                print(f"Not an image: {url}")

                continue

            with open(filepath, "wb") as f:
                f.write(response.content)
            print(f"image : {filename}")


            image = cv2.imdecode(np.frombuffer(response.content, np.uint8), cv2.IMREAD_COLOR)

            face = crop_and_resize_face(image)
            if face is not None:
                face_filename = f"{face_index}.png"
                face_path = os.path.join(face_dir, face_filename)
                cv2.imwrite(face_path, face)
                print(f"Saved Face : {face_filename}")
                face_index += 1
            else:
                print(f"No face")

        except Exception as e:
            print(f"Error: {e}")
            

        time.sleep(random.uniform(1, 2))

    driver.quit()

    print(f"\nfinish")

if __name__ == "__main__":
    json_path = 'outputs/article_images.jsonl'
    image_dir = 'images'
    face_dir = 'data'
    url_list = []
    with open(json_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            url_list.append(json.loads(line))

    download_images(url_list, image_dir, face_dir)
