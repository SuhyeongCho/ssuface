import cv2

filepath = '/Users/suhyeongcho/Desktop/Github/ssuface/q1w2.mp4'

movie = cv2.VideoCapture(filepath)


if movie.isOpened() == False:
    print('Can\'t open the File',FilePath)
    exit()

cv2.namedWindow('Face')

face_cascade = cv2.CascadeClassifier()
face_cascade.load('/usr/local/share/OpenCV/haarcascades/haarcascade_frontalface_default.xml')

while(True):
    ret, frame = movie.read()
    
    if frame is None:
        break
    frame = cv2.transpose(frame)
    
    grayframe = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(grayframe, 1.1, 3, 0, (10, 10))

    for (x,y,w,h) in faces:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),3, 4, 0)

    cv2.imshow('Face',frame)

    if cv2.waitKey(1) == 27:
        break


movie.release()
cv2.destroyWindow('Face')


