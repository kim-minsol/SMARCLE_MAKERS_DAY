import cv2
import numpy as np
import tensorflow as tf

cap = cv2.VideoCapture(0)

red_color = (0,0,255)

#모델 불러오기
load_model = tf.keras.models.load_model('C:\\Users\\rlaal\\OneDrive\\바탕 화면\\model\\acc_07099.h5')

while True:
    ret, cam = cap.read()

    if ret :
        cam = cv2.rectangle(cam, (200,120),(440,380),red_color,3)
        cv2.imshow('camera', cam)
        dst = cam.copy()
        roi = cam[123:377, 203:437]
        dst = cv2.resize(roi, dsize=(224, 224), interpolation=cv2.INTER_AREA)

        cv2.imshow("dst", dst)

        
        #이미지 저장
        img_save= cv2.imwrite('test.jpg', dst)
        
        
        #predict
        b = np.expand_dims(dst,axis=0)
        predict = load_model.predict(x=b)
            
        print(np.argmax(predict))
        

        if cv2.waitKey(1) & 0xFF == 27:
            break #esc 키 누르면 닫힘

cap.release()
cv2.destroyAllWindows()
