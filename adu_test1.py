import serial
import time

ard = serial.Serial(
    port = 'COM4',
    baudrate = 9600
    )

time.sleep(3)


while True:
    listen = ard.readline()
    decoded  = listen.decode('utf-8')
    decoded  = decoded[:-2]

    print(decoded)
    
    distance = float(decoded[:-2])
    

    if (distance > 20 and distance<=30):
        shirt=1
        print("GOOD T-shirt: ",end='')
        print(shirt)
        

    elif (distance > 10 and distance <= 20):
        shirt=2
        print("OKAY T-shirt: ",end='')
        print(shirt)
            

    elif (distance > 0 and distance <= 10):
        shirt=3
        print("FULL T-shirt: ",end='')
        print(shirt)
           
       
    else:
        shirt=0
        print("EMPTY")

       
