import cv2

#create classifier using opencv haar cascade algorithm pre-tained frontal face default data
face_data = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

#choose video or capture from webcam
camera = cv2.VideoCapture(0)

#loop over frames of the video
while True:
    #read current frame
    succesful_frame_read, frame = camera.read()

    #grayscale frame
    grayscaled_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    #detect faces
    face_coordinates = face_data.detectMultiScale(grayscaled_img)

    #draw rectangle around face
    for (x, y, w, h) in face_coordinates:
        cv2.rectangle(frame, (x,y), (x+w, y+h), (0, 255, 0), 2)

    #display images and move to next frame after 1 ms
    cv2.imshow('Face Detector', frame)
    key = cv2.waitKey(1)

    #stop if escape or space bar key was pressed
    if key == 27 or key==32:
        break

camera.release()