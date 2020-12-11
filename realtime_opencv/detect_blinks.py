	# USAGE
# python detect_blinks.py --shape-predictor shape_predictor_68_face_landmarks.dat --video blink_detection_demo.mp4
# python detect_blinks.py --shape-predictor shape_predictor_68_face_landmarks.dat

# import the necessary packages
from scipy.spatial import distance as dist
from imutils.video import FileVideoStream
from imutils.video import VideoStream
from imutils import face_utils
import numpy as np
import argparse
import imutils
import time
import dlib
import cv2
import serial

#아두이노 파트
# arduinoData=serial.Serial(port='COM7',baudrate=9600,timeout=1)
# time.sleep(1)



# while True:
# 	if arduinoData.readable():
# 		data = arduinoData.readline()[:-2] #the last bit gets rid of the new-line chars
# 		print (data)


def led_on():
	# arduinoData.write(b'1')
	time.sleep(1./120)



def led_off():
	# arduinoData.write(b'0')
	time.sleep(1./120)


def eye_aspect_ratio(eye):
	# compute the euclidean distances between the two sets of
	# vertical eye landmarks (x, y)-coordinates
	A = dist.euclidean(eye[1], eye[5])
	B = dist.euclidean(eye[2], eye[4])

	# compute the euclidean distance between the horizontal
	# eye landmark (x, y)-coordinates
	C = dist.euclidean(eye[0], eye[3])

	# compute the eye aspect ratio
	ear = (A + B) / (2.0 * C)

	# return the eye aspect ratio
	return ear


# define two constants, one for the eye aspect ratio to indicate
# blink and then a second constant for the number of consecutive
# frames the eye must be below the threshold
EYE_AR_THRESH = 0.2
EYE_AR_CONSEC_FRAMES = 3


# initialize the frame counters and the total number of blinks
COUNTER = 0
TOTAL = 0

mode=0
before_status = 0
after_status=0
fileDir='C:/Users/marshmallow/Desktop/shape_predictor_68_face_landmarks.dat'
start=False


# initialize dlib's face detector (HOG-based) and then create
# the facial landmark predictor
print("[INFO] loading facial landmark predictor...")
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(fileDir)

# grab the indexes of the facial landmarks for the left and
# right eye, respectively
(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

# start the video stream thread
print("[INFO] starting video stream thread...")
# vs = FileVideoStream(args["video"]).start()
# fileStream = True
vs = VideoStream(src=0).start()
# vs = VideoStream(usePiCamera=True).start()
fileStream = False
time.sleep(1.0)

# loop over frames from the video stream
while True:
	# if this is a file video stream, then we need to check if
	# there any more frames left in the buffer to process
	if fileStream and not vs.more():
		break

	# grab the frame from the threaded video file stream, resize
	# it, and convert it to grayscale
	# channels)
	frame = vs.read()
	frame = imutils.resize(frame, width=450)
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

	# detect faces in the grayscale frame
	rects = detector(gray, 0)

	# loop over the face detections
	for rect in rects:
		# determine the facial landmarks for the face region, then
		# convert the facial landmark (x, y)-coordinates to a NumPy
		# array
		shape = predictor(gray, rect)
		shape = face_utils.shape_to_np(shape)

		# extract the left and right eye coordinates, then use the
		# coordinates to compute the eye aspect ratio for both eyes
		leftEye = shape[lStart:lEnd]
		rightEye = shape[rStart:rEnd]
		leftEAR = eye_aspect_ratio(leftEye)
		rightEAR = eye_aspect_ratio(rightEye)

		# average the eye aspect ratio together for both eyes
		ear = (leftEAR + rightEAR) / 2.0

		# compute the convex hull for the left and right eye, then
		# visualize each of the eyes
		leftEyeHull = cv2.convexHull(leftEye)
		rightEyeHull = cv2.convexHull(rightEye)
		cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
		cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)

		# check to see if the eye aspect ratio is below the blink
		# threshold, and if so, increment the blink frame counter

		if ear < EYE_AR_THRESH:
			COUNTER += 1
			if  TOTAL==2 and start==False:
				
				led_off()
				
 
		elif ear> 0.25 :		
			 COUNTER =0 
			 if mode==1:
				 before_status = 0
			 if TOTAL==1 and start==True:
				 led_on()
				 start=False
				 print(led_on())
				
		# otherwise, the eye aspect ratio is not below the blink
		# threshold

		if mode==0:
			if COUNTER >= 50:
				TOTAL=0
				mode=1
				
		if mode==1:
			if COUNTER >= 3 and before_status == 0:
				TOTAL +=1
				
				before_status = 2


		if TOTAL==1 and after_status==0 and ear> 0.25:
			# led_on()
			start=True
	
			after_status=1
		
		
		if TOTAL==2 and after_status==1:
			# led_off()
		
			
			# cv2.putText(frame, "BLINKED!", (200, 100),
			# cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
			after_status=0
		

		if TOTAL==3:
			TOTAL=0
			COUNTER=0
			mode=0
			
			
		
	
	



		# print(COUNTER, "and" ,mode, send)

		# draw the total number of blinks on the frame along with
		# the computed eye aspect ratio for the frame
		cv2.putText(frame, "Blinks: {}".format(TOTAL), (10, 30),
			cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
		cv2.putText(frame, "EAR: {:.2f}".format(ear), (300, 30),
			cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
 
	# show the frame
	cv2.imshow("Frame", frame)
	# if the `q` key was pressed, break from the loop
	key = cv2.waitKey(1) & 0xFF
	if key == ord("q"):
		break

# do a bit of cleanup
cv2.destroyAllWindows()
vs.stop()
