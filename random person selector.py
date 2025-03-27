import cv2
import random
import numpy as np
import tkinter as tk
from tkinter import messagebox

# Load the MobileNet SSD model (pre-trained on COCO dataset)
prototxt_path = r"C:\Users\jessi\OneDrive\Documents\vscode\random person selector\deploy.prototxt"
model_path = r"C:\Users\jessi\OneDrive\Documents\vscode\random person selector\mobilenet_iter_73000.caffemodel"

# Load OpenCV's deep learning model
try:
    net = cv2.dnn.readNetFromCaffe(prototxt_path, model_path)
    print("✅ Model loaded successfully!")
except cv2.error as e:
    print("❌ Error loading model:", e)
    exit()

# Class labels (COCO dataset)
CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat", "bottle",
           "bus", "car", "cat", "chair", "cow", "diningtable", "dog", "horse",
           "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"]

# Assign colors for different labels
COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))

# Function to ask for camera access permission
def ask_permission():
    root = tk.Tk()
    root.withdraw()  # Hide the main window
    response = messagebox.askyesno("Camera Access", "Do you allow this app to access your camera?")
    root.destroy()
    return response

# Function to detect humans and objects
def detect_humans_objects(frame):
    h, w = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, scalefactor=0.007843, size=(300, 300), mean=127.5)
    net.setInput(blob)
    detections = net.forward()

    human_list = []
    object_list = []

    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.6:  # Increased confidence threshold for better accuracy
            idx = int(detections[0, 0, i, 1])
            label = CLASSES[idx]

            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (x, y, x2, y2) = box.astype("int")

            # Ensure detection is within frame bounds
            x, y, x2, y2 = max(0, x), max(0, y), min(w, x2), min(h, y2)

            if label == "person":  
                human_list.append((x, y, x2 - x, y2 - y))
                color = (0, 255, 0)  # Green for humans
                text_label = "Human"
            else:
                object_list.append((label, x, y, x2 - x, y2 - y))
                color = (255, 0, 0)  # Blue for objects
                text_label = "Object"

            # Draw bounding box and label
            cv2.rectangle(frame, (x, y), (x2, y2), color, 2)
            cv2.putText(frame, text_label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    return human_list, object_list

# Function to run the camera
def run_camera():
    cap = cv2.VideoCapture(0)  # Open webcam

    # Set high resolution
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    if not cap.isOpened():
        messagebox.showerror("Error", "Unable to access the camera!")
        return

    cv2.namedWindow("Human & Object Detector", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Human & Object Detector", 900, 600)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        humans, objects = detect_humans_objects(frame)  # Detect humans & objects

        # Show video feed with labels
        cv2.imshow("Human & Object Detector", frame)

        key = cv2.waitKey(1)
        
        # Select a random person when 's' is pressed
        if key == ord('s') and humans:
            chosen_person = random.choice(humans)
            (x, y, w, h) = chosen_person

            # Crop and zoom selected person
            zoom_factor = 2
            new_w, new_h = w * zoom_factor, h * zoom_factor

            # Ensure zoomed-in image does not go out of bounds
            x_start = max(0, x - (new_w - w) // 2)
            y_start = max(0, y - (new_h - h) // 2)
            x_end = min(frame.shape[1], x_start + new_w)
            y_end = min(frame.shape[0], y_start + new_h)

            # Extract and resize person
            person_roi = frame[y_start:y_end, x_start:x_end]
            if person_roi.size > 0:
                zoomed_person = cv2.resize(person_roi, (300, 300), interpolation=cv2.INTER_CUBIC)

                # Show zoomed-in person
                cv2.imshow("Selected Person", zoomed_person)
                cv2.waitKey(1000)  # Show for 1 second
                cv2.destroyWindow("Selected Person")

        # Exit on 'q'
        if key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# Main Execution
if __name__ == "__main__":
    if ask_permission():
        run_camera()
    else:
        messagebox.showinfo("Permission Denied", "Camera access was denied.")
