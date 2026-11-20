import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
from deepface import DeepFace
import cv2

EMOTIONS = ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]

root = tk.Tk()
root.title("Facial Emotion Detection")
root.geometry("800x600")
root.configure(bg="#20232a")

image_label = tk.Label(root, bg="#20232a")
image_label.pack(pady=20)

result_label = tk.Label(
    root,
    text="Upload an image to detect emotion",
    font=("Arial", 18, "bold"),
    fg="white",
    bg="#20232a"
)
result_label.pack(pady=10)


def detect_emotion(path):
    try:
        results = DeepFace.analyze(
            img_path=path,
            actions=["emotion"],
            detector_backend="opencv",
            enforce_detection=True
        )

        emotion_data = results[0]["emotion"]
        dominant = results[0]["dominant_emotion"]
        confidence = emotion_data[dominant]

        result_label.config(
            text=f"Emotion: {dominant.upper()}\nConfidence: {confidence:.2f}%"
        )

    except Exception as e:
        result_label.config(text="No face detected or error occurred")


def upload_image():
    path = filedialog.askopenfilename(
        filetypes=[("Images", "*.jpg *.jpeg *.png")]
    )

    if not path:
        return

    img = Image.open(path)
    img.thumbnail((600, 400))

    photo = ImageTk.PhotoImage(img)
    image_label.config(image=photo)
    image_label.image = photo

    detect_emotion(path)


upload_button = tk.Button(
    root,
    text="Upload Image",
    command=upload_image,
    font=("Arial", 14),
    bg="#00adb5",
    fg="white",
    padx=20,
    pady=10
)
upload_button.pack(pady=20)

root.mainloop()