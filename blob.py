import cv2
import numpy as np

params = cv2.SimpleBlobDetector_Params()

# Set the threshold for detecting blobs
# params.minThreshold = 10
# params.maxThreshold = 200
# Filter by area
params.filterByArea = True
params.minArea = 1500
# Filter by circularity
# params.filterByCircularity = True
# params.minCircularity = 0.1
# Filter by convexity
# params.filterByConvexity = True
# params.minConvexity = 0.87
# Filter by inertia
# params.filterByInertia = True
# params.minInertiaRatio = 0.01

# Create the detector object
detector = cv2.SimpleBlobDetector_create(params)

# Read video from file
video = cv2.VideoCapture("../videos/Japan_USA_Friendly.mp4")

# Check if video was successfully opened
if not video.isOpened():
    print("Error opening video file")

# Read first frame
success, frame = video.read()

# Check if frame was successfully read
if not success:
    print("Error reading video file")

# Set up the background subtractor
fgbg = cv2.createBackgroundSubtractorMOG2()

# Set the lower and upper bounds for the yellow color range (HSV color space)
yellow_lower = (50, 200, 200)
yellow_upper = (0, 250, 250)

while success:

    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Blur the image to reduce noise
    blur = cv2.GaussianBlur(gray, (5,5), 0)

    # Perform blob detection on the image
    detector = cv2.SimpleBlobDetector_create(params)
    keypoints = detector.detect(blur)

    # Create an empty image to draw the contours on
    blank = np.zeros((1,1))
    blank = cv2.drawKeypoints(blank, keypoints, np.array([]), (0,0,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    _, contours, _ = cv2.findContours(blank, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    
    if contours:
        c = max(contours, key=cv2.contourArea)

        # Get the bounding box of the largest contour
        x, y, w, h = cv2.boundingRect(c)
        
        # Draw a yellow circle around the volleyball
        cv2.circle(frame, (x + w//2, y + h//2), 15, (200, 70, 250), 2)

        # Draw the contours on the black image
        cv2.drawContours(frame, contours, -1, (0,255,0), 3)


    # Draw the contours on the frame
    cv2.drawContours(frame, contours, -1, (0, 255, 0), 3)

    # Display the frame
    cv2.imshow("Volleyball Tracker", frame)

    # Check if user pressed 'q' to quit
    if cv2.waitKey(1) == ord("q"):
        break
    
    # Read next frame
    success, frame = video.read()

# When everything done, release the capture
video.release()
cv2.destroyAllWindows()
