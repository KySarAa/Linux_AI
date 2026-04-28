# -*- coding: utf-8 -*-
import paho.mqtt.client as mqtt
import subprocess
import os
import signal

# -------------------------
# CONFIG MQTT
# -------------------------
#MQTT_BROKER = "172.16.151.161"
#MQTT_PORT = 1883
#MQTT_TOPIC = "raspberry/cmd"
MQTT_BROKER = "0e6eab887f084d39be5a2e37743aa9bc.s1.eu.hivemq.cloud"
MQTT_PORT = 8883
MQTT_TOPIC = "raspberry/cmd"

MQTT_USER = "pc-linux"
MQTT_PASS = "Neeko3****" 

# -------------------------
# PROCESSUS EN COURS
# -------------------------
processes = {}

# -------------------------
# CHEMINS PYTHON
# -------------------------
PYTHON_VENV = "/home/ajpi/yolov5/venv/bin/python3"
PYTHON_SYS  = "python3"

# -------------------------
# CHEMINS IA
# -------------------------
YOLOV5_PATH     = "/home/ajpi/yolov5/models/yolov5.py"
YOLOBESTPT_PATH = "/home/ajpi/yolov5/models/yolobestpt.py"
RTK_PATH        = "/home/ajpi/RTKfinal.py"


# -------------------------
# MQTT CONNECT
# -------------------------
def on_connect(client, userdata, flags, rc):
    print(f"Connecte au broker MQTT (code {rc})")
    client.subscribe(MQTT_TOPIC)


# -------------------------
# MQTT MESSAGE
# -------------------------
def on_message(client, userdata, msg):
    command = msg.payload.decode('utf-8', errors='ignore')
    print(f"Commande recue : {command}")

    # ============================================================
    # RTK FINAL
    # ============================================================
    if command == "run:RTKfinal":
        if "RTKfinal" not in processes or processes["RTKfinal"].poll() is not None:
            print("Demarrage de RTKfinal.py...")
            processes["RTKfinal"] = subprocess.Popen(
                [PYTHON_SYS, RTK_PATH],
                cwd="/home/ajpi"
            )
        else:
            print("RTKfinal.py deja en cours.")

    elif command == "stop:RTKfinal":
        if "RTKfinal" in processes and processes["RTKfinal"].poll() is None:
            print("Arret de RTKfinal.py...")
            os.kill(processes["RTKfinal"].pid, signal.SIGTERM)
            processes["RTKfinal"].wait()
            print("RTKfinal.py arrete.")
        else:
            print("RTKfinal.py n est pas actif.")


    # ============================================================
    # YOLOv5
    # ============================================================
    elif command == "run:yolov5":
        if "yolov5" not in processes or processes["yolov5"].poll() is not None:
            print("Demarrage de yolov5.py...")
            processes["yolov5"] = subprocess.Popen(
                [PYTHON_VENV, YOLOV5_PATH],
                cwd="/home/ajpi/yolov5/models"
            )
        else:
            print("yolov5.py deja en cours.")

    elif command == "stop:yolov5":
        if "yolov5" in processes and processes["yolov5"].poll() is None:
            print("Arret de yolov5.py...")
            os.kill(processes["yolov5"].pid, signal.SIGTERM)
            processes["yolov5"].wait()
            print("yolov5.py arrete.")
        else:
            print("yolov5.py n est pas actif.")


    # ============================================================
    # YOLO BEST PT
    # ============================================================
    elif command == "run:yolobestpt":
        if "yolobestpt" not in processes or processes["yolobestpt"].poll() is not None:
            print("Demarrage de yolobestpt.py...")
            processes["yolobestpt"] = subprocess.Popen(
                [PYTHON_VENV, YOLOBESTPT_PATH],
                cwd="/home/ajpi/yolov5/models"
            )
        else:
            print("yolobestpt.py deja en cours.")

    elif command == "stop:yolobestpt":
        if "yolobestpt" in processes and processes["yolobestpt"].poll() is None:
            print("Arret de yolobestpt.py...")
            os.kill(processes["yolobestpt"].pid, signal.SIGTERM)
            processes["yolobestpt"].wait()
            print("yolobestpt.py arrete.")
        else:
            print("yolobestpt.py n est pas actif.")


# -------------------------
# LANCEMENT MQTT
# -------------------------
client = mqtt.Client()
client.on_connect = on_connect
client.on_message = on_message

print(f"Connexion au broker MQTT {MQTT_BROKER}...")

client.username_pw_set(MQTT_USER, MQTT_PASS)

import ssl
client.tls_set(tls_version=ssl.PROTOCOL_TLS)
client.tls_insecure_set(False)

client.connect(MQTT_BROKER, MQTT_PORT, 60)

client.loop_forever()
