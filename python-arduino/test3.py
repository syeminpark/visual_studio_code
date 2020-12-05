import serial
import time

arduinoData=serial.Serial('/dev/cu.usbserial-14420',9600)

def led_on():
    arduinoData.write(b'1')

def led_off():
    arduinoData.write(b'0')

time.sleep(2)

led_on()

print("done")