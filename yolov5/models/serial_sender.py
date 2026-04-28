# serial_sender.py
import serial

class SerialSender:
    def __init__(self, port="/dev/ttyUSB0", baud=115200):
        self.ser = serial.Serial(port, baud, timeout=0.1)

    def send_zones(self, zones):
        """
        zones = [0,1,0]
        """
        msg = ",".join(str(int(z)) for z in zones) + "\n"
        self.ser.write(msg.encode())
