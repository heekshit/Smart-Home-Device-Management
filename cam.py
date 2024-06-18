import cv2
from imutils.video import VideoStream
import winsound
import numpy as np
import pandas as pd
import streamlit as st
import time

def is_motion_detected(frame, background_subtractor):
    """Check if significant motion is detected in the frame."""
    # Apply the background subtractor to get the foreground mask
    fg_mask = background_subtractor.apply(frame)
    # Threshold the mask to clean up the foreground
    _, thresh = cv2.threshold(fg_mask, 25, 255, cv2.THRESH_BINARY)
    # Find contours in the thresholded mask
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # Return True if any contour is found (indicating motion)
    return len(contours) > 0, thresh

# Initialize background subtractors for each video stream
background_subtractor1 = cv2.createBackgroundSubtractorMOG2()
background_subtractor2 = cv2.createBackgroundSubtractorMOG2()

# Initialize and warm up video streams
print("[INFO] Starting video streams...")
vs1 = VideoStream(src=0).start()
vs2 = VideoStream(src=1).start()

cv2.namedWindow("Camera 1", cv2.WINDOW_NORMAL)
cv2.namedWindow("Camera 2", cv2.WINDOW_NORMAL)
cv2.namedWindow("Motion 1", cv2.WINDOW_NORMAL)
cv2.namedWindow("Motion 2", cv2.WINDOW_NORMAL)

while True:
    # Read frames from both video streams
    data = pd.read_csv('device_data.csv')
    
    
    frame1 = vs1.read()
    frame2 = vs2.read()
    
    # Resize frames for consistency
    frame1 = cv2.resize(frame1, (400, 400))
    frame2 = cv2.resize(frame2, (400, 400))
    
    # Detect motion in both frames
    motion1_detected, thresh1 = is_motion_detected(frame1, background_subtractor1)
    motion2_detected, thresh2 = is_motion_detected(frame2, background_subtractor2)

    # Update frames with text indicating motion status
    text1 = "Motion Detected" if motion1_detected else "No Motion"
    text2 = "Motion Detected" if motion2_detected else "No Motion"
    cv2.putText(frame1, text1, (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    cv2.putText(frame2, text2, (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    # Display the updated frames
    cv2.imshow("Camera 1", frame1)
    cv2.imshow("Camera 2", frame2)
    cv2.imshow("Motion 1", thresh1)
    cv2.imshow("Motion 2", thresh2)

    for i, device in enumerate(data.columns):
        st.table(data)
        if data[device].sum() > 10000:
            
    # If motion is not detected in either frame, play a beep sound
            if not motion1_detected or not motion2_detected:
                winsound.Beep(1000, 1000)  # Frequency 1000Hz, Duration 1000ms
        

    key = cv2.waitKey(1) & 0xFF
    # If the `q` key was pressed, break from the loop
    if key == ord('q'):
        break
    time.sleep(3)
    st.experimental_rerun()

# Cleanup
cv2.destroyAllWindows()
vs1.stop()
vs2.stop()
