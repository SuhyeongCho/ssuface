import cv2
import dlib
import subprocess
import os

openfilename = 'q1w2e3r4.mp4'
savefilename = 'output.mov'
audiofilename = 'audio.wav'
directory = '/Users/suhyeongcho/Desktop/Github/ssuface/'

movie = cv2.VideoCapture(openfilename)


width = int(movie.get(cv2.CAP_PROP_FRAME_WIDTH) + 0.5)
height = int(movie.get(cv2.CAP_PROP_FRAME_HEIGHT) + 0.5)
fps = movie.get(cv2.CAP_PROP_FPS)

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('tmp.mov',fourcc,fps,(height,width))

command = 'ffmpeg -i '+openfilename+' -ab 160k -ac 2 -ar 44100 -vn '+audiofilename
subprocess.call(command,shell=True)

detector = dlib.get_frontal_face_detector()

while(True):
    ret,frame = movie.read()
    if ret == True:
        frame = cv2.transpose(frame)
        frame = cv2.flip(frame,1)
        grayframe = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        
        dets = detector(grayframe,1)

        for i,d in enumerate(dets):
            l = d.left()
            r = d.right()
            t = d.top()
            b = d.bottom()
            cv2.rectangle(frame,(l,t),(r,b),(0,255,0),3, 4, 0)

        out.write(frame)
    else:
        break

out.release()
movie.release()

command = 'ffmpeg -i tmp.mov -i '+audiofilename+' -shortest -c:v copy -c:a aac -b:a 256k '+savefilename
subprocess.call(command,shell=True)

os.remove(audiofilename)
os.remove('tmp.mov')
