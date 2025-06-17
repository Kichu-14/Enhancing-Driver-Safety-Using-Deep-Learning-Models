import cv2
import dlib
from scipy.spatial import distance
from time import time
import numpy as np
import mediapipe as mp
import keras
import tensorflow as tf

cap = cv2.VideoCapture(0)

face_detector = dlib.get_frontal_face_detector()
mpHands = mp.solutions.hands
hands = mpHands.Hands(static_image_mode=False,
                      max_num_hands=2,
                      min_detection_confidence=0.5,
                      min_tracking_confidence=0.5)
mpDraw = mp.solutions.drawing_utils


dlib_facelandmark = dlib.shape_predictor("C:\\Users\\Poonam\\OneDrive - Ishaan\\Desktop\\Ishaan\\Assignments\\sem4\\AI_MINOR\\Project\\CODE\\shape_predictor_68_face_landmarks.dat\\shape_predictor_68_face_landmarks.dat")
time_delay = 3
start_time = time()  

def Detect_Eye(eye):
	poi_A = distance.euclidean(eye[1], eye[5])
	poi_B = distance.euclidean(eye[2], eye[4])
	poi_C = distance.euclidean(eye[0], eye[3])
	aspect_ratio_Eye = (poi_A+poi_B)/(2*poi_C)
	return aspect_ratio_Eye

def Detect_Mouth(mouth):
	poi_D = distance.euclidean(mouth[1], mouth[5])
	poi_E = distance.euclidean(mouth[2], mouth[4])
	poi_F = distance.euclidean(mouth[0], mouth[3])
	aspect_ratio_Mouth = (poi_D+poi_E)/(2*poi_F)
	return aspect_ratio_Mouth




while True:
	null, frame = cap.read()
	#print(frame) 
	gray_scale = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	faces = face_detector(gray_scale)
	imgRGB = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
	results = hands.process(imgRGB)
	if results.multi_hand_landmarks:
		for handLms in results.multi_hand_landmarks:
			for id, lm in enumerate(handLms.landmark):
				h, w, c = frame.shape
				cx, cy = int(lm.x *w), int(lm.y*h)
				cv2.circle(frame, (cx,cy), 3, (255,0,255), cv2.FILLED)
				mpDraw.draw_landmarks(frame, handLms, mpHands.HAND_CONNECTIONS)

	for face in faces:
		face_landmarks = dlib_facelandmark(gray_scale, face)
		leftEye = [] 
		rightEye = [] 
		mouth = []
	
		for n in range(42, 48):
			x = face_landmarks.part(n).x
			y = face_landmarks.part(n).y
			rightEye.append((x, y))
			next_point = n+1
			if n == 47:
				next_point = 42
			x2 = face_landmarks.part(next_point).x
			y2 = face_landmarks.part(next_point).y
			cv2.line(frame, (x, y), (x2, y2), (102, 255, 255), 1)

		for n in range(36, 42):
			x = face_landmarks.part(n).x
			y = face_landmarks.part(n).y
			leftEye.append((x, y))
			next_point = n+1
			if n == 41:
				next_point = 36
			x2 = face_landmarks.part(next_point).x
			y2 = face_landmarks.part(next_point).y
			cv2.line(frame, (x, y), (x2, y2), (255, 255, 0), 1)

		for n in range(48,68):
			x = face_landmarks.part(n).x
			y = face_landmarks.part(n).y
			mouth.append((x,y))
			next_point = n+1
			if n == 67:
				next_point = 48
			x2 = face_landmarks.part(next_point).x
			y2 = face_landmarks.part(next_point).y
			cv2.line(frame, (x, y), (x2, y2), (255, 255, 255), 1)


	
	
	model=tf.keras.models.load_model('CODE\\shape_predictor_68_face_landmarks.dat\\Brain.keras')
	resultant = model(frame)
	resultant = resultant>0.5
	if(resultant==0):
		pred = 'Closed'
	else:
		pred ='Open'
	font = cv2.FONT_HERSHEY_SIMPLEX 
	org = (50, 50) 
	fontScale = 1
	color = (255, 0, 0)
	thickness = 2
	cv2.putText(frame, pred , org, font,  
                   fontScale, color, thickness, cv2.LINE_AA) 
	cv2.imshow("Drowsiness DETECTOR IN OPENCV2", frame)
	key = cv2.waitKey(9)
	if key == 20:
		break


cap.release()
cv2.destroyAllWindows()