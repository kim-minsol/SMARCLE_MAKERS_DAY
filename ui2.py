import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array



upperbody_cascade = cv2.CascadeClassifier('./opencv/haarcascade_upperbody.xml')                                   


cap = cv2.VideoCapture(0)

red_color = (0,0,255)

#모델 불러오기
load_model = tf.keras.models.load_model('C:\\Users\\rlaal\\OneDrive\\바탕 화면\\model\\acc_07099.h5')

while True:
    ret, cam = cap.read()

    if ret :
        cam = cv2.rectangle(cam, (0,0),(640,480),red_color,3)
        grayframe = cv2.cvtColor(cam, cv2.COLOR_BGR2GRAY)
        dst = cam.copy()
        cv2.imshow('camera', cam)
        
        #upper
        upper_body = upperbody_cascade.detectMultiScale(grayframe)


        for (x,y,w,h) in upper_body:
            cv2.rectangle(cam, (x,y), (x+w, y+h+100), (255,0,0),3)
            roi = cam[y+160:y+h+100, x:x+w]
            '''
            dst=roi
            '''
            dst = cv2.resize(roi, dsize=(224, 224), interpolation=cv2.INTER_AREA)
            cv2.imshow('camera', cam)
            cv2.imshow("dst", dst)
            
            '''
            #이미지 저장
            img_save= cv2.imwrite('test.jpg', roi)
            '''

            #predict
            b = np.expand_dims(dst,axis=0)
            predict = load_model.predict(x=b)
            
            print(np.argmax(predict))

            
                
        '''
        dst = cam.copy()
        roi = cam[a2:a2+a4+100, a1:a1+a3]
        dst = roi

        cv2.imshow("dst", dst)
        

        #이미지 저장
        img_save= cv2.imwrite('test.jpg', roi)
        '''
        
            
        #모델 불러오기
        '''
        load_model = tf.keras.models.load_model('./model/acc_07099.h5')

        
        image = load_img("", target_size=(224, 224))
        image = img_to_array(image)
        image = image.reshape(1, 224, 224, 3)
        predict = model.predict(x=image)
            
        print(np.argmax(predict))
        '''
        
        
        if cv2.waitKey(1) & 0xFF == 27:
            break #esc 키 누르면 닫힘
    else:
        break

cap.release()
cv2.destroyAllWindows()
