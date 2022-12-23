import cv2
import numpy as np

# Read video from file
video = cv2.VideoCapture("../videos/USA_Canada_Highlights.mp4")

# Check if video was successfully opened
if not video.isOpened():
    print("Error opening video file")

# Read first frame
success, frame = video.read()

# Check if frame was successfully read
if not success:
    print("Error reading video file")

# Define range of yellow color in BGR color space
yellow_lower = (30, 150, 100) #0,100,100
yellow_upper = (60, 200, 200)

# Loop over video frames
while success:
    contour_image = np.zeros((frame.shape[0], frame.shape[1], 3), dtype=np.uint8)
    # Convert frame from BGR to HSV color space
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Threshold the HSV image to get only yellow colors
    mask = cv2.inRange(hsv, yellow_lower, yellow_upper)

    # Find contours in the binary image
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if contours:
        c = min(contours, key=cv2.contourArea)

        # Get the bounding box of the largest contour
        x, y, w, h = cv2.boundingRect(c)
        
        # Draw a yellow circle around the volleyball
        cv2.circle(frame, (x + w//2, y + h//2), 30, (200, 70, 250), 2)

        # Draw the contours on the black image
        cv2.drawContours(frame, contours, -1, (0,255,0), 3)

    # Display the frame
    cv2.imshow("Volleyball Tracker", frame)

    # Check if user pressed 'q' to quit
    if cv2.waitKey(1) == ord("q"):
        break

    # Read next frame
    success, frame = video.read()

# Release video file and destroy all windows
video.release()
cv2.destroyAllWindows()