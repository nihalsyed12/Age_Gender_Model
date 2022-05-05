import cv2 
import numpy as np 
import tensorflow as tf 


#store prediction Label data 
MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)
age_list = ['(0, 2)', '(4, 6)', '(8, 12)', '(15, 20)', '(25, 32)', '(38, 43)', '(48, 53)', '(60, 100)']
gender_list = ['Male', 'Female']

#captures video using webcam 
webcam = cv2.VideoCapture(0, cv2.CAP_DSHOW)

#loops through webcam frame by frame as 
def webcam_detector(age_model, gender_model):

    while True:
        #get frame from webcam input
        #convert captured frame to gray and detect faces using haarcascade
        ret, frame = webcam.read()
        face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')
        grayImg = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(grayImg, 1.1, 5)



        #loops throug detected face box and create a rectangle around detected face
        for(x, y, w, h) in faces:
            cv2.rectangle(frame, (x,y), (x+w, y+h), (255, 255, 0), 2)

        #crop face 
            face_img = frame[y:y+h, x:x+w].copy()
        #preprocess img by creating a blob
        #blobFromImage() performs mean subtraction, scaling, schannel swapping is set to false
        #resize to (277, 277)
            blob = cv2.dnn.blobFromImage(face_img, 1, (227, 227), MODEL_MEAN_VALUES, swapRB=False)
            
        #predict age 
            age_model.setInput(blob)
            age_preds = age_model.forward()
            age = age_list[age_preds[0].argmax()]
        #predict gender 
            gender_model.setInput(blob)
            gender_preds = gender_model.forward()
            gender = gender_list[gender_preds[0].argmax()]
        #create text displaying age and gender 
            overlay_text = "%s %s" % (gender, age)
            cv2.putText(frame, overlay_text, (x,y), cv2.FONT_HERSHEY_SCRIPT_COMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

        cv2.imshow('Webcam Detection', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

if __name__ == "__main__":
    #import pretrained models
    age_net = cv2.dnn.readNetFromCaffe('age_deploy.prototxt', 'age_net.caffemodel')
    gender_net = cv2.dnn.readNetFromCaffe('gender_deploy.prototxt', 'gender_net.caffemodel')
    webcam_detector(age_net, gender_net)
        

webcam.release()
cv2.destroyAllWindows()
