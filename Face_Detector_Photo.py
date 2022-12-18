import cv2

#create classifier using opencv haar cascade algorithm pre-tained frontal face default data
face_data = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

#choose image
img = cv2.imread('obama.png')

#grayscale image
grayscaled_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

#detect faces
face_coordinates = face_data.detectMultiScale(grayscaled_img)

#draw rectangle around face
for (x, y, w, h) in face_coordinates:
    cv2.rectangle(img, (x,y), (x+w, y+h), (0, 255, 0), 2)

#display image and pause
cv2.imshow('Face Detector', img)
cv2.waitKey()