import cv2
import dlib
import subprocess
import os

openfilename = 'suhyeong_movie.mp4'
directory = '/Users/suhyeongcho/Desktop/Github/ssuface/'

movie = cv2.VideoCapture(openfilename)


detector = dlib.get_frontal_face_detector()
count = 0;
while(True):
    ret,frame = movie.read()
    if ret == True:
        frame = cv2.transpose(frame)
        frame = cv2.flip(frame,1)
        grayframe = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        
        dets = detector(grayframe,1)
        
        for i,d in enumerate(dets):
            count = count + 1
            l = d.left()
            r = d.right()
            t = d.top()
            b = d.bottom()
            face = frame[t:b,l:r]
            face = cv2.resize(face,(96,96))
            cv2.imwrite("./suhyeong/"+str(count)+".jpg",face)

    else:
        break

movie.release()

