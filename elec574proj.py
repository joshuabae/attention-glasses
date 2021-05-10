#!/usr/local/bin/env python3
"""
Josh and Justin ELEC 574 Final Project: Portable Computer Vision Eye-Tracking Anti-Distracted-Driving Device
"""
import numpy as np
import cv2
import time

#TURN THIS TO "True" FOR USE ON OUR RASPBERRY PI SETUP
#IF YOU DO NOT HAVE ACCESS TO THE PI AND WOULD LIKE TO TEST BASIC FUNCTIONALITY ON A COMPUTER, THEN SET THIS TO "False"
onPi = True

if onPi:
    import RPi.GPIO as GPIO
    GPIO.setwarnings(False)
    GPIO.setmode(GPIO.BCM)
    buzzer = 4
    GPIO.setup(buzzer,GPIO.OUT)
    flag = 0 
    



#Haar Cascade classifiers for finding face and eyes
face_cascade = cv2.CascadeClassifier('./haarcascades/haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('./haarcascades/haarcascade_eye.xml')

detector_params = cv2.SimpleBlobDetector_Params()
detector_params.filterByArea = True
detector_params.maxArea = 2000
blob_detector = cv2.SimpleBlobDetector_create(detector_params)

colors = [(255,0,0), (0,0,255)]

cv2.namedWindow('Left Eye')
cv2.moveWindow('Left Eye',600,0)
cv2.namedWindow('Right Eye')
cv2.moveWindow('Right Eye',900,0)
cv2.namedWindow('Gaze Tracker')
cv2.moveWindow('Gaze Tracker',300,150)

kx = -99999999
ky = -99999999
pupil_present = True
last_time_pupil_present = 0
time_since_pupil_present = 0
pupil_centered = True
last_time_pupil_centered = 0
time_since_pupil_centered = 0

distractedness_threshold_time = 1.5

warning_flag = False


def trackbar_value_write(position):
    """
    Returns position of threshold-choosing trackbar
    """
    print(position)

cv2.createTrackbar('Threshold', 'Gaze Tracker', 60, 255, trackbar_value_write)

def pick_best_face(image_set):
    """
    Picks the biggest image out of a set of face coordinates image_set
    """
    #If there's more than one candidate for the face, pick the biggest one.
    if len(image_set) > 1:
        biggest = np.array([0,0,0,0])
        for potential_img in image_set:
            if potential_img[3] > biggest[3]:
                biggest = potential_img
        biggest_img = np.array([biggest])
    #If there's just one candidate, then just pick it.
    elif len(image_set) == 1:
        biggest_img = image_set
    #If there are no viable candidates for the face, then return an empty tuple.
    else:
        biggest_img = ()
    return biggest_img

def pick_best_eye(image_set):
    """
    Picks the biggest image out of a set of eye coordinates image_set. This image also needs to be less than a certain threshold size.
    """
    # If there's more than one candidate for the eye, pick the biggest one. 
    # However, if the biggest one is bigger than the threshold of width * (5/24), 
    # then it is probably a false positive (not an eye), so instead choose the next biggest one.
    if len(image_set) > 1:
        biggest = np.array([0,0,0,0])
        for potential_img in image_set:
            #second condition prevents eyebrow + eye to be considered as an eye
            if (potential_img[3] > biggest[3]) & (potential_img[3] < (w2*2*(5/24))):
                biggest = potential_img
        if biggest.all() == np.array([0,0,0,0]).all():
            biggest_img = ()
        else:
            biggest_img = np.array([biggest])
    # If there's just one candidate for the eye, pick it. The eye must meet the same threshold condition as before.
    elif len(image_set) == 1:
        if image_set[0][3] < (w2*2*(5/24)):
            biggest_img = image_set
        else:
            biggest_img = ()
    #If there are no candidates, return an empty set.
    else:
        biggest_img = ()
    return biggest_img

def crop_around_eye(eye_img, side, eye_size):
    """
    Cut out the top bit of the eye image, where the eyebrows are
    """
    #Whichever side (left = 0, or right = 1) the eye in question is on, cut out the top brow_height number of pixels. 
    #Also, cut out a few pixels from the side to keep the eye centered and get rid of some of the corner of the eye.
    if side == 0:
        eye_img = eye_img[brow_height:eye_size, (eye_side):eye_size]
    else:
        eye_img = eye_img[brow_height:eye_size, 0:(eye_size-eye_side)]
    return eye_img

def blob_process(eye_img):
    """
    Process the eye_img so that the iris/pupil is more distinct
    """
    #Erosion to get rid of eyebrows
    eye_img = cv2.erode(eye_img, None, iterations=1)
    #Dilation to enhance round iris/pupil
    eye_img = cv2.dilate(eye_img, None, iterations=3)
    #Median Blur did not help performance, so it was removed
    #eye_img = cv2.medianBlur(eye_img, 3)

    return eye_img

def detect_blobs(eye_img, detector):
    """
    Find keypoints from eye blob image using blob detector algorithm from OpenCV
    """
    keypts = detector.detect(eye_img)
    return keypts

#for testing: 
#capture_object = cv2.VideoCapture("test_video.mov")
#capture_object = cv2.VideoCapture("test_drive.mov")
#for live video:
capture_object = cv2.VideoCapture(0)

while True:
    #Capture each frame of the video for analysis. One run of this while loop will analyze one frame.
    _, frame = capture_object.read()
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    face = pick_best_face(face_cascade.detectMultiScale(frame_gray, 1.3, 5))

    #Restrict search for eyes to only the face in question. If there's no face, skip analysis.
    #Using a for loop on a zero- or one-item list allows me to skip execution of the included code
    #if the list has no items.
    for (x,y,w,h) in face:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(255,255,0),2)
        h2 = int(h/2)
        w2 = int(w/2)

        #Separate the left and right halves of the face, and find an eye in each
        face_frames = [frame[y:y+h2, x:x+w2], frame[y:y+h2, x+w2:x+w]]
        face_frames_gray = [frame_gray[y:y+h2, x:x+w2], frame_gray[y:y+h2, x+w2:x+w]]
        eyes = [pick_best_eye(eye_cascade.detectMultiScale(face_frames_gray[0])), pick_best_eye(eye_cascade.detectMultiScale(face_frames_gray[1]))]

        #At the moment, the user defines a detection threshold for binary thresholding using a trackbar on a GUI. 
        #Later work could replace this with automatic threshold-setting, perhaps by adjusting based on the overall 
        #brightness of the scene (for example, the average brightness of the frame)
        threshold_val = cv2.getTrackbarPos('Threshold', 'Gaze Tracker')

        #idx represents which eye is being examined; left is 0, right is 1
        for idx in [0,1]:
            for (ex,ey,ew,eh) in eyes[idx]:
                
                cv2.rectangle(face_frames[idx],(ex,ey),(ex+ew,ey+eh),colors[idx],2)

                #From here on, measure positions in reference to eye_frame_gray, 
                #which is just the part of the face frame that contains the eye.
                eye_frame_gray = face_frames_gray[idx][ey:ey+eh, ex:ex+ew]
                eye_sz = eye_frame_gray.shape[0]
                brow_height = int(eye_sz * (4/24))
                eye_side = int(eye_sz * (1/60))

                eye_frame_gray = crop_around_eye(eye_frame_gray,idx, eye_sz)

                #This defines the mean filter used to blur the eye_frame_gray image so that specular reflections are removed
                kernel_unnormalized = np.ones([15,15],np.float32)
                kernel = kernel_unnormalized/np.sum(kernel_unnormalized)
                eye_frame_gray = cv2.filter2D(eye_frame_gray,-1,kernel)

                #Binary thresholding and other pre-processing for blob detection are performed here
                _, img = cv2.threshold(eye_frame_gray, threshold_val, 255, cv2.THRESH_BINARY_INV)
                eye = blob_process(img)
                _, eye = cv2.threshold(eye, 128, 255, cv2.THRESH_BINARY_INV)

                keypts = detect_blobs(eye, blob_detector)

                if keypts != []:
                    pupil_present = True

                    kx, ky = keypts[0].pt
                    kx = int(kx)
                    ky = int(ky)
                    last_time_pupil_present = time.time()
                    time_since_pupil_present = 0

                    #These lines draw markers on Raspberry Pi GUI; these are not necessary for sound
                    eye = cv2.drawKeypoints(eye, keypts, eye, (255, 255, 0), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
                    cv2.circle(eye, (kx, ky), 5, colors[idx], -1)
                    
                else:
                    pupil_present = False

                    time_since_pupil_present = time.time() - last_time_pupil_present


                x_acceptable_range = int(eye_sz*(7/60))
                y_acceptable_range = int(eye_sz*(60/60))

                eye_center = eye_sz/2

                if idx == 0:
                    if (kx > eye_center + x_acceptable_range) | (kx < eye_center - x_acceptable_range) | (ky > eye_center + y_acceptable_range) | (ky < eye_center - y_acceptable_range):
                        pupil_centered = False
                        time_since_pupil_centered = time.time() - last_time_pupil_centered
                    else:
                        pupil_centered = True
                        last_time_pupil_centered = time.time()
                        time_since_pupil_centered = 0

                if (time_since_pupil_present > distractedness_threshold_time) | (time_since_pupil_centered > distractedness_threshold_time):
                    warning_flag = True
                else:
                    warning_flag = False

                #print("pupil_centered:", pupil_centered)
                #print("pupil_present:", pupil_present)
                #print("     warning_flag:", warning_flag)

                if idx == 0:
                    #cv2.imshow('Right Eye', eye_frame_gray)
                    cv2.imshow('Left Eye', eye)
                else:
                    cv2.imshow('Right Eye', eye)
                    pass

    cv2.imshow('Gaze Tracker',frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    #These are the lines that produce sound

    if onPi:
        if warning_flag == True:
            if flag == 0:
                GPIO.output(buzzer,GPIO.HIGH)
                time.sleep(0.00001)
                flag = 1
            else:
                GPIO.output(buzzer,GPIO.LOW)
                time.sleep(0.00001)
                flag = 0
    if warning_flag == True:
        print("DISTRACTION DETECTED")
    else:
        print("No distraction detected")

            
capture_object.release()
cv2.destroyAllWindows()
