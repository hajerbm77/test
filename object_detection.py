# Import libraries
import os
import cv2
import numpy as np

from zipfile import ZipFile
from urllib.request import urlretrieve

classFile  = "coco_class_labels.txt"

with open(classFile) as fp:
    labels = fp.read().split("\n")

print(labels)

modelFile  = os.path.join("models", "ssd_mobilenet_v2_coco_2018_03_29", "frozen_inference_graph.pb")
configFile = os.path.join("models", "ssd_mobilenet_v2_coco_2018_03_29.pbtxt")

# Read the Tensorflow network
net = cv2.dnn.readNetFromTensorflow(modelFile, configFile)

FONTFACE = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 0.7
THICKNESS = 1

# For ach file in the directory
def detect_objects(net, im, dim = 300):

    # Create a blob from the image
    try:
        blob = cv2.dnn.blobFromImage(im, 1.0, size=(dim, dim), mean=(0, 0, 0), swapRB=True, crop=False)
        # Pass blob to the network
        net.setInput(blob)
        # Peform Prediction
        objects = net.forward()
        return objects
    except Exception as e:
        print(str(e))

def display_text(im, text, x, y):
    # Get text size
    textSize = cv2.getTextSize(text, FONTFACE, FONT_SCALE, THICKNESS)
    dim = textSize[0]
    baseline = textSize[1]

    # Use text size to create a black rectangle
    cv2.rectangle(
        im,
        (x, y - dim[1] - baseline),
        (x + dim[0], y + baseline),
        (0, 0, 0),
        cv2.FILLED,
    )

    # Display text inside the rectangle
    cv2.putText(
        im,
        text,
        (x, y - 5),
        FONTFACE,
        FONT_SCALE,
        (0, 255, 255),
        THICKNESS,
        cv2.LINE_AA,
    )

def display_objects(im, objects, threshold=0.25):
    rows = im.shape[0]
    cols = im.shape[1]

    # For every Detected Object
    for i in range(objects.shape[2]):
        # Find the class and confidence
        classId = int(objects[0, 0, i, 1])
        score = float(objects[0, 0, i, 2])

        # Recoveroriginal cordinates from normalized coordinates
        x = int(objects[0, 0, i, 3] * cols)
        y = int(objects[0, 0, i, 4] * rows)
        w = int(objects[0, 0, i, 5] * cols - x)
        h = int(objects[0, 0, i, 6] * rows - y)

        global bbox
        bbox = (x, y, w, h)
        print(bbox)

        # Check if the detection is of good quality
        if score > threshold and labels[classId] in ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'street sign', 'stop sign']:
            display_text(im, "{}".format(labels[classId]), x, y)
            cv2.rectangle(im, (x, y), (x + w, y + h), (255, 255, 255), 2)

    cv2.imshow("Objects detection",im)

def drawRectangle(frame, bbox):
    p1 = (int(bbox[0]), int(bbox[1]))
    p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
    cv2.rectangle(frame, p1, p2, (255, 0, 0), 2, 1)

def displayRectangle(frame, bbox):
    frameCopy = frame.copy()
    drawRectangle(frameCopy, bbox)
    frameCopy = cv2.cvtColor(frameCopy, cv2.COLOR_RGB2BGR)

def drawText(frame, txt, location, color=(50, 170, 50)):
    cv2.putText(frame, txt, location, cv2.FONT_HERSHEY_SIMPLEX, 1, color, 3)

def track_objects(frame, ok):
	# Set up tracker
    tracker_types = [
        "BOOSTING",
        "MIL",
        "KCF",
        "CSRT",
        "TLD",
        "MEDIANFLOW",
        "GOTURN",
        "MOSSE",
    ]

    # Change the index to change the tracker type
    tracker_type = tracker_types[6]

    if tracker_type == "BOOSTING":
        tracker = cv2.legacy.TrackerBoosting.create()
    elif tracker_type == "MIL":
        tracker = cv2.legacy.TrackerMIL.create()
    elif tracker_type == "KCF":
        tracker = cv2.TrackerKCF.create()
    elif tracker_type == "CSRT":
        tracker = cv2.TrackerCSRT.create()
    elif tracker_type == "TLD":
        tracker = cv2.legacy.TrackerTLD.create()
    elif tracker_type == "MEDIANFLOW":
        tracker = cv2.legacy.TrackerMedianFlow.create()
    elif tracker_type == "GOTURN":
        tracker = cv2.TrackerGOTURN.create()
    else:
        tracker = cv2.legacy.TrackerMOSSE.create()
    
    # Initialize tracker with first frame and bounding box
    global bbox
    try:
        ok = tracker.init(frame, bbox)
        while True:
            # Start timer
            timer = cv2.getTickCount()

            # Update tracker
            ok, bbox = tracker.update(frame)

            # Calculate Frames per second (FPS)
            fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer)

            # Draw bounding box
            if ok:
                drawRectangle(frame, bbox)
            else:
                drawText(frame, "Tracking failure detected", (80, 140), (0, 0, 255))

            # Display Info
            drawText(frame, tracker_type + " Tracker", (80, 60))
            drawText(frame, "FPS : " + str(int(fps)), (80, 100))  
    except Exception as e:
        print(str(e))   

video_capture = cv2.VideoCapture(0)

while video_capture.isOpened(): 
# Captures video_capture frame by frame 
    _, frame = video_capture.read() 

    # calls the detect() function	 
    canvas = detect_objects(net, frame)
    display_objects(frame, canvas)
    # track_objects(frame, canvas)

    # The control breaks once q key is pressed						 
    if cv2.waitKey(1) & 0xff == ord('q'):
        break

# Release the capture once all the processing is done. 
video_capture.release()								 
cv2.destroyAllWindows() 