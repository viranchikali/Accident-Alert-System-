import streamlit as st
import base64
import cv2
import numpy as np
import asyncio
from detection import AccidentDetectionModel
from concurrent.futures import ThreadPoolExecutor
import time
from notificationapi_python_server_sdk import notificationapi

headers ={
    "clientId": st.secrets["clientId"],
    "clientSecret": st.secrets["clientSecret"],
    "email": st.secrets["email"],
    "number": st.secrets["number"],
    "content-type": "application/json"
}


# Set the Streamlit page to run in wide mode by default
st.set_page_config(layout="wide")

# Path to the video file
video_path = "videoplayback.mp4"

st.title("Camera 1: Accident Detection")  # Adding "Camera 1" to the title

# Initialize the model
model = AccidentDetectionModel("model.json", 'model_weights.keras')
font = cv2.FONT_HERSHEY_SIMPLEX

# Initialize the last message time
last_message_time = 0

async def send_notification():
    notificationapi.init(
        headers["clientId"],  # clientId
        headers["clientSecret"] # clientSecret
    )

    await notificationapi.send({
        "notificationId": "accident_alert",
        "user": {
          "id": headers["email"],
          "number": headers["number"] # Replace with your phone number
        },
        "mergeTags": {
          "comment": "\nProbability of an accident at Kothanur, Bengaluru, Karnataka 560077.\n\nFor the exact location, click here: \nhttps://maps.app.goo.gl/YRGv6kR9SoTik5Sa7 ",
          "commentId": "testCommentId"
        }
    })

async def detect_accident(frame):
    global last_message_time
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    roi = cv2.resize(gray_frame, (250, 250))

    pred, prob = model.predict_accident(roi[np.newaxis, :, :])
    if pred == "Accident":
        prob = round(prob[0][0] * 100, 2)
        cv2.rectangle(frame, (0, 0), (280, 40), (0, 0, 0), -1)
        cv2.putText(frame, pred + " " + str(prob), (20, 30), font, 1, (255, 255, 0), 2)
        current_time = time.time()
        if prob > 99.50 and (current_time - last_message_time) > 240:
            asyncio.create_task(send_notification())  # Create a task to send the notification
            last_message_time = current_time
    return frame  # Return the frame with detection results

async def stream_video(video_path, placeholder, width, height):
    while True:
        video = cv2.VideoCapture(video_path)
        while video.isOpened():
            ret, frame = video.read()
            if not ret:
                break
            frame = await detect_accident(frame)  # Await the detection function
            _, buffer = cv2.imencode('.jpg', frame)
            frame_bytes = buffer.tobytes()
            encoded_frame = base64.b64encode(frame_bytes).decode()
            video_str = f'''
                <img width="{width}" height="{height}" src="data:image/jpeg;base64,{encoded_frame}">
            '''
            placeholder.markdown(video_str, unsafe_allow_html=True)
            await asyncio.sleep(0.0003)  # Control the frame rate

async def main():
    placeholder = st.empty()
    await stream_video(video_path, placeholder, 800, 600)

# Initialize a ThreadPoolExecutor
executor = ThreadPoolExecutor()

# Run the async main function
asyncio.run(main())

# Shut down the ThreadPoolExecutor
executor.shutdown(wait=True)
