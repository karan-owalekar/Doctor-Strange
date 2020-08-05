import numpy as np
import cv2
import dlib
from imutils import face_utils

#Using haar-cascades to detect palms and fists in the image...
palm_cascade = cv2.CascadeClassifier("Models/haarcascade_palm.xml")
fist_cascade = cv2.CascadeClassifier("Models/haarcascade_fist.xml")

#Using shape predictor to get eye-landmarks
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("Models/shape_predictor_68_face_landmarks.dat")

#Eye landmarks will be used to make eyes glowing...
(lStart,lEnd)= face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rStart, rEnd)= face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

#Capturing the video from webcam...
cap = cv2.VideoCapture(0)

#Loading all the image files...
#Here we are using .jpg images with black background...
#As our image dosen't contain any black area so we can affort to use jpg images...
#As we are not using png images so removing background (black color) is quiet easy...
green_aura_0 = cv2.imread("Aura_Img/green_aura_0.jpg")
green_aura_1 = cv2.imread("Aura_Img/green_aura_1.jpg")
red_aura = cv2.imread("Aura_Img/red_aura.jpg")
#Green aura will rotate based on rotation_number...
rotation_number = 0


while True:
    #Reading and flipping the frame for user convinence...
    ret, frame = cap.read()
    frame = cv2.flip(frame,1)
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)

    #Setting up haar cascades to detect hand and palm...
    palm = palm_cascade.detectMultiScale(gray,scaleFactor=1.5,minNeighbors=5)
    fist = fist_cascade.detectMultiScale(gray,scaleFactor=1.5,minNeighbors=5)
 
    for (x,y,w,h) in palm:
        #If palm is detected, we display green aura...
        rotation_number += 1
        text = "Palm"
        #cv2.rectangle(frame, (x,y), (x+w,y+h), (0,0,128), 2)
        #cv2.putText(frame, text, (x,y-10), cv2.FONT_HERSHEY_TRIPLEX, 1, (128,128,0), 2, cv2.LINE_AA)

        #Finding out centroid to place image perfectly in the center
        centre_x = x+w // 2
        centre_y = y+h // 2

        #image is of 300x300 pixels hence ... roi is sepersted 150 px from centre

        #Getting the region of the palm...
        roi = frame[centre_y-150:centre_y+150,centre_x-150:centre_x+150]

        #Alternatively displaying one of the green auras... 
        #This alternative displaying make it feel like it is moving...
        if (rotation_number % 3) % 2  == 0:
            try:
                img = cv2.addWeighted(green_aura_0, 1, roi, 1, 0)
                frame[centre_y-150:centre_y+150,centre_x-150:centre_x+150] = img
            except Exception:
                pass
        else:
            try:
                img = cv2.addWeighted(green_aura_1, 1, roi, 1, 0)
                frame[centre_y-150:centre_y+150,centre_x-150:centre_x+150] = img
            except Exception:
                pass

        faces = detector(gray, 0)
        for face in faces:
            #Making eyes glow...
            shape = predictor(gray , face)
            shape = face_utils.shape_to_np(shape)

            leftEye = shape[lStart:lEnd]
            rightEye = shape[rStart:rEnd]

            leftEyeHull = cv2.convexHull(leftEye)
            rightEyeHull = cv2.convexHull(rightEye)

            cv2.drawContours(frame, [leftEyeHull], -1, (0,255,0),1)  #-1 because we want to draw all the counter and we dont have any index for it
            cv2.drawContours(frame, [rightEyeHull], -1, (0,255,0),1)
     
    for (x,y,w,h) in fist:
        #Red aura if you close the fist...
        text = "Fist"
        #cv2.rectangle(frame, (x,y), (x+w,y+h), (0,0,128), 2)
        #cv2.putText(frame, text, (x,y-10), cv2.FONT_HERSHEY_TRIPLEX, 1, (128,128,0), 2, cv2.LINE_AA)

        centre_x = x+w // 2
        centre_y = y+h // 2

        #image is of 300x300 pixels hence ... roi is sepersted 150 px from centre

        roi = frame[centre_y-150:centre_y+150,centre_x-150:centre_x+150]
        try:
            img = cv2.addWeighted(red_aura, 1, roi, 1, 0)
            frame[centre_y-150:centre_y+150,centre_x-150:centre_x+150] = img
        except Exception:
            pass

    cv2.imshow("frame",frame)
    #Displaying the frames...
    key = cv2.waitKey(1)
    if key == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()