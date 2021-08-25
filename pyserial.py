import serial
import time

ard = serial.Serial(
    port = 'COM4',
    baudrate = 9600
    )

time.sleep(3)

while True:
    print(" - receive : receive distance")
    print(" - send : send LED command")

    selection = input()

    if selection == "receive" :
        ard.write("let me know".encode())
        listen = ard.readline()
        decoded  = listen.decode('utf-8')
        decoded  = decoded[:-2]

        print(decoded)

    if selection == "send":
        print("0: off")
        print("1: on")

        x= int(input())

        if (x==0):
            ard.write("0".encode())

        if (x==1):
            ard.write("1".encode())


'''
while True:
    listen = ard.readline()
    decoded  = listen.decode('utf-8')
    decoded  = decoded[:-2]

    print(decoded)
'''
    

