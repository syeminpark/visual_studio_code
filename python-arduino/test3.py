import serial
import time

arduinoData=serial.Serial(port='COM4',baudrate=9600)

def led_on():
    arduinoData.write(b'1')

def led_off():
    arduinoData.write(b'0')

time.sleep(2)

led_on()

print("done")