import cv2

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

    # Use the background subtractor to create a foreground mask
    fgmask = fgbg.apply(frame)

    # Convert the frame to the HSV color space
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Threshold the HSV image to get only yellow colors
    mask = cv2.inRange(hsv, yellow_lower, yellow_upper)

    # Combine the mask created by the background subtractor with the color threshold mask
    mask = cv2.bitwise_and(mask, fgmask)

    # Find contours in the mask
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
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
