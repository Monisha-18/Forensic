# Import libraries
import os
import cv2
import imutils
import time
import pickle
import numpy as np
from imutils.video import FPS
from imutils.video import VideoStream

# Get the absolute path of the project directory
BASE_DIR = os.path.abspath(os.path.dirname(__file__))

# Load serialized face detector model
print(" Loading Face Detector...")
protoPath = os.path.join(BASE_DIR, "face-recognition-using-deep-learning/face_detection_model/deploy.prototxt")
modelPath = os.path.join(BASE_DIR, "face-recognition-using-deep-learning/face_detection_model/res10_300x300_ssd_iter_140000.caffemodel")

if not os.path.exists(protoPath) or not os.path.exists(modelPath):
    raise FileNotFoundError(" Face detection model files not found!")

detector = cv2.dnn.readNetFromCaffe(protoPath, modelPath)

# Load serialized face embedding model
print(" Loading Face Recognizer...")
embedder_path = os.path.join(BASE_DIR, "face-recognition-using-deep-learning/openface_nn4.small2.v1.t7")

if not os.path.exists(embedder_path):
    raise FileNotFoundError(" Face embedding model file not found!")

embedder = cv2.dnn.readNetFromTorch(embedder_path)

# Load the trained face recognition model and label encoder
recognizer_path = os.path.join(BASE_DIR, "face-recognition-using-deep-learning/output/recognizer.pickle")
label_encoder_path = os.path.join(BASE_DIR, "face-recognition-using-deep-learning/output/le.pickle")

if not os.path.exists(recognizer_path) or not os.path.exists(label_encoder_path):
    raise FileNotFoundError(" Face recognition model or label encoder not found!")

recognizer = pickle.loads(open(recognizer_path, "rb").read())
le = pickle.loads(open(label_encoder_path, "rb").read())

# Start the video stream
print(" Starting Video Stream...")
vs = VideoStream(src=0).start()
time.sleep(2.0)  # Allow camera sensor to warm up

# Start FPS counter
fps = FPS().start()

# Loop over frames from the video stream
while True:
    # Grab the frame and resize it
    frame = vs.read()
    frame = imutils.resize(frame, width=600)
    (h, w) = frame.shape[:2]

    # Create a blob from the frame
    imageBlob = cv2.dnn.blobFromImage(
        cv2.resize(frame, (300, 300)), 1.0, (300, 300),
        (104.0, 177.0, 123.0), swapRB=False, crop=False)

    # Detect faces in the frame
    detector.setInput(imageBlob)
    detections = detector.forward()

    # Loop over detected faces
    for i in range(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]

        # Filter weak detections
        if confidence > 0.5:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            # Extract face ROI
            face = frame[startY:endY, startX:endX]
            if face.shape[0] < 20 or face.shape[1] < 20:
                continue

            # Create a blob from the face and get embeddings
            faceBlob = cv2.dnn.blobFromImage(face, 1.0 / 255,
                (96, 96), (0, 0, 0), swapRB=True, crop=False)
            embedder.setInput(faceBlob)
            vec = embedder.forward()

            # Recognize the face
            preds = recognizer.predict_proba(vec)[0]
            j = np.argmax(preds)
            proba = preds[j]
            name = le.classes_[j]

            # Display name & probability
            text = f"{name}: {proba * 100:.2f}%"
            y = startY - 10 if startY - 10 > 10 else startY + 10
            cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)
            cv2.putText(frame, text, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 2)

    # Update FPS counter
    fps.update()

    # Show output frame
    cv2.imshow("Face Recognition", frame)
    key = cv2.waitKey(1) & 0xFF

    # Quit if 'q' key is pressed
    if key == ord("q"):
        break

# Stop FPS counter and cleanup
fps.stop()
print(f" Elapsed Time: {fps.elapsed():.2f} sec")
print(f" Approx. FPS: {fps.fps():.2f}")

cv2.destroyAllWindows()
vs.stop()
