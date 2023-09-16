

#import libraries
import cv2
import numpy as np
import tensorflow as tf

#Load the cnn model for cricket shot prediction
cnn_model = tf.keras.models.load_model(r'D:\Data Science\DL Deep Learning\cricket_shots\shot_model.h5')

#create a dictionary for different cricket shits
shot_dict={0:"Drive",1:"Flick",2:"Sweep",3:"Pullshot"}

#load the video
video=cv2.VideoCapture(r'D:\Data Science\DL Deep Learning\cricket_shots\kholi2.mp4')

while True:
    suc,frame=video.read()
    if not suc:
        break

    #Load the cascadeclassifier for fullbody detection
    body_detector=cv2.CascadeClassifier(r'D:\Data Science\DL Deep Learning\cricket_shots\haarcascade_fullbody.xml')

    #change color from BGR to RGB
    image_rgb=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)

    #Find the body  using cascadeclassifier
    body=body_detector.detectMultiScale(image_rgb,scaleFactor=1.1,minNeighbors=5)

    for (x,y,w,h) in body:
        #Draw rectangle around the person
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,120),2)
        cricket_shot=image_rgb[y:y+h,x:x+w]

        #resize and reshape the image
        image_resized=cv2.resize(cricket_shot,(200,200))
        image=image_resized.reshape(1,200,200,3)
        
        #predict using cnn model
        cricketshot_prediction = cnn_model.predict(image)
        max_index = int(np.argmax(cricketshot_prediction))

        #write text on the frame which shot is playing
        cv2.putText(frame,shot_dict[max_index],(x+5, y-20),cv2.FONT_HERSHEY_SIMPLEX,1,(255,100,0),2)

    cv2.imshow('cricket shot', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
video.release()
cv2.destroyAllWindows()
