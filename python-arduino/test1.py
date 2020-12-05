from pyfirmata2 import Arduino #import library from pyfirmata2 to detect Arduino
import time #time library to be able setup lenght of led lighting

board = Arduino(Arduino.AUTODETECT) #detect Arduino with Autodetect

print("...Arduino detected") #print statement #1
print("...Blink test started") #print statement #2

while True: #while this statement is true execute script hereunder
    print("...Blink LED 1st time") #print statement blink 1st time
    board.digital[13].write(1)
    time.sleep(2) #light up led number 13 for 2 seconds
    board.digital[13].write(0)
    time.sleep(1) #light off led number 13 for 1 seconds
    print("...Blink LED 2nd time") #print statement blink 2nd time
    board.digital[13].write(1)
    time.sleep(2) #light up led number 13 for 2 seconds
    board.digital[13].write(0)
    time.sleep(1) #light off led number 13 for 1 seconds
    print("...Blink LED 3nd time") #print statement blink 3nd time
    board.digital[13].write(1)
    time.sleep(2) #light up led number 13 for 2 seconds
    board.digital[13].write(0)
    time.sleep(1) #light off led number 13 for 1 seconds
    print("...Blink LED 4nd time") #print statement blink 4nd time
    board.digital[13].write(1)
    time.sleep(2) #light up led number 13 for 2 seconds
    board.digital[13].write(0)
    time.sleep(1) #light off led number 13 for 1 seconds
    print("...Blink test successfull!") #print statement #3
    exit() #exit script