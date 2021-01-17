# Python code for Multiple Color Detection 

import numpy as np 
import cv2 


# Capturing video 
camCapture = cv2.VideoCapture(0) 

# while loop 
while(1): 
    
    # Reading from the video 
    
    _, imgFrame = camCapture.read() 

    # Convert the imgFrame from BGR(RGB color space) to 
    # HSV(hue-saturation-value) color space 
    hsvFrame = cv2.cvtColor(imgFrame, cv2.COLOR_BGR2HSV) 

    # Set range for red color for defining mask
    red_lower = np.array([155,25,0], np.uint8) 
    red_upper = np.array([180, 255, 255], np.uint8) 
    red_mask = cv2.inRange(hsvFrame, red_lower, red_upper) 

    # Set range for green for defining mask 
    green_lower = np.array([36,0,0], np.uint8) 
    green_upper = np.array([80, 255, 255], np.uint8) 
    green_mask = cv2.inRange(hsvFrame, green_lower, green_upper) 

    # Set range for blue color for defining mask
    blue_lower = np.array([94, 80, 2], np.uint8) 
    blue_upper = np.array([120, 255, 255], np.uint8) 
    blue_mask = cv2.inRange(hsvFrame, blue_lower, blue_upper) 
    
    # mask to detect only that particular color 
    kernel = np.ones((5, 5), "uint8") 
    
    # For red color 
    red_mask = cv2.dilate(red_mask, kernel) 
    res_red = cv2.bitwise_and(imgFrame, imgFrame, 
                            mask = red_mask) 
    
    # For green color 
    green_mask = cv2.dilate(green_mask, kernel) 
    res_green = cv2.bitwise_and(imgFrame, imgFrame, 
                                mask = green_mask) 
    
    # For blue color 
    blue_mask = cv2.dilate(blue_mask, kernel) 
    res_blue = cv2.bitwise_and(imgFrame, imgFrame, 
                            mask = blue_mask) 

    # Creating tracker for red color 
    contours, hierarchy = cv2.findContours(red_mask, 
                                        cv2.RETR_TREE, 
                                        cv2.CHAIN_APPROX_SIMPLE) 
    
    for pic, contour in enumerate(contours): 
        area = cv2.contourArea(contour) 
        if(area > 300): 
            x, y, w, h = cv2.boundingRect(contour) 
            imgFrame = cv2.rectangle(imgFrame, (x, y), 
                                    (x + w, y + h), 
                                    (0, 0, 255), 2) 
            
            cv2.putText(imgFrame, "Red Colour", (x, y), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, 
                        (0, 0, 255))     

    # Creating tracker for green color 
    contours, hierarchy = cv2.findContours(green_mask, 
                                        cv2.RETR_TREE, 
                                        cv2.CHAIN_APPROX_SIMPLE) 
    
    for pic, contour in enumerate(contours): 
        area = cv2.contourArea(contour) 
        if(area > 300): 
            x, y, w, h = cv2.boundingRect(contour) 
            imgFrame = cv2.rectangle(imgFrame, (x, y), 
                                    (x + w, y + h), 
                                    (0, 255, 0), 2) 
            
            cv2.putText(imgFrame, "Green Colour", (x, y), 
                        cv2.FONT_HERSHEY_SIMPLEX, 
                        1.0, (0, 255, 0)) 

    # Creating tracker for blue color 
    contours, hierarchy = cv2.findContours(blue_mask, 
                                        cv2.RETR_TREE, 
                                        cv2.CHAIN_APPROX_SIMPLE) 
    for pic, contour in enumerate(contours): 
        area = cv2.contourArea(contour) 
        if(area > 300): 
            x, y, w, h = cv2.boundingRect(contour) 
            imgFrame = cv2.rectangle(imgFrame, (x, y), 
                                    (x + w, y + h), 
                                    (255, 0, 0), 2) 
            
            cv2.putText(imgFrame, "Blue Colour", (x, y), 
                        cv2.FONT_HERSHEY_SIMPLEX, 
                        1.0, (255, 0, 0)) 
            
    # Hold until termination 
    cv2.imshow("Multiple Color Detection in Real-TIme", imgFrame) 
    if cv2.waitKey(10) & 0xFF == ord('q'):  
        cv2.destroyAllWindows() 
        break