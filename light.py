import serial
import time

ser = serial.Serial('/dev/ttyACM0', 115200)

# Exemple : allumer la LED
ser.write(b'{"cmd":"light","value":1}\n')
time.sleep(1)

# Exemple : éteindre la LED
ser.write(b'{"cmd":"light","value":0}\n')
