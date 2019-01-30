import cv2
import numpy as np

filepath = '/Users/suhyeongcho/Desktop/Github/ssuface/q1w2.mp4'

movie = cv2.VideoCapture(filepath)

if movie.isOpened() == False:
    print('Can\'t open the File',FilePath)
    exit()

width = int(movie.get(cv2.CAP_PROP_FRAME_WIDTH) + 0.5)
height = int(movie.get(cv2.CAP_PROP_FRAME_HEIGHT) + 0.5)
fps = movie.get(cv2.CAP_PROP_FPS)

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('qq.mov',fourcc,fps,(height,width))
print('width {0}, height {1}, fps {2}'.format(width, height, fps))

cv2.namedWindow('Face')

face_cascade = cv2.CascadeClassifier()
face_cascade.load('/usr/local/share/OpenCV/haarcascades/haarcascade_frontalface_default.xml')

count = 0
count2 = 0
while(True):
    ret, frame = movie.read()
    
    if ret == False:
        break
    faces =[]
    frame = cv2.transpose(frame)
    frame = cv2.flip(frame,1)


    height, width = frame.shape[:2]
#    print('({0},{1})'.format(height,width))

#    for rad in range(0,1):
    rads = [0,-15,15,-30,30,-45,45]
    for rad in rads:

        count = count + 1

        matrix = cv2.getRotationMatrix2D((width/2,height/2),rad,1)
        
        frame1 = cv2.warpAffine(frame,matrix,(width,height))

        grayframe = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
        grayframe = cv2.GaussianBlur(grayframe,(5,5), 0)

        scaleFactor = 1.1
        minNeighbors = 10
        flags = 0
        minSize = 10
        maxSize = 10

#        faces = face_cascade.detectMultiScale(grayframe, scaleFactor, minNeighbors, flags, (minSize, maxSize))
        face = face_cascade.detectMultiScale(grayframe, scaleFactor, minNeighbors)


        if len(face):
            count2 = count2 + 1
#        print(type(face))

#        matrix = cv2.getRotationMatrix2D((width/2,height/2),-rad,1)
#
#        frame = cv2.warpAffine(frame,matrix,(width,height))

        if type(face) is np.ndarray:
            faces.append(face)

#    print(faces)
    newface = []
    for face in faces:
        for (x,y,w,h) in face:
            if [x,y,w,h] in newface:
                continue
            for (x1,y1,w1,h1) in face:
                if [x1,y1,w1,h1] in newface:
                    continue
                if (x == x1) and (y == y1) and (w == w1) and (h == h1):
                    continue
                elif (x <= x1) and (x1 <= x+w/2):
                    newface.append([x1,y1,w1,h1])
                elif (x+w/2 <= x1+w1) and (x1+w1 <= x+w):
                    newface.append([x1,y1,w1,h1])
                elif (y <= y1) and (y1 <= y+h/2):
                    newface.append([x1,y1,w1,h1])
                elif (y+h/2 <= y1+h1) and (y1+h1 <= y+h):
                    newface.append([x1,y1,w1,h1])

    for face in faces:
        for (x,y,w,h) in face:
            if [x,y,w,h] in newface:
                continue
            cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),3, 4, 0)




    out.write(frame)
    cv2.imshow('Face',frame)


    if cv2.waitKey(1000) & 0xFF  == 27:
        break

print(count)
print(count2)
print(float(count2)/count)
out.release()
movie.release()
cv2.destroyWindow('Face')

print('Finish')
