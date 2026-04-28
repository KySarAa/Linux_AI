import time
import math
import threading
import socket
import base64
import serial
import json
import queue
import requests
import random
import serial.tools.list_ports

# =========================
# CONFIG
# =========================

SIMULATION = True   # True = fake GPS, False = real hardware

# Real GNSS port
F9P_NMEA_PORT = "/dev/ttyACM0"
F9P_BAUD = 9600

# NTRIP
NTRIP_SERVER = "crtk.net"
NTRIP_PORT = 2101
MOUNTPOINT = "LESTR"
USERNAME = "centipede"
PASSWORD = "centipede"

# API Laravel
LARAVEL_API_BASE_URL = "http://localhost:8088/api"
LARAVEL_API_URL = f"{LARAVEL_API_BASE_URL}/gnss/data"
LARAVEL_MISSION_URL = f"{LARAVEL_API_BASE_URL}/mission/current"

SYSTEM_ACTIVE = True
CURRENT_MISSION_ID = None

DIST_JUMP_THRESHOLD = 5.0
SPEED_JUMP_THRESHOLD = 10.0

# =========================
# UTILS
# =========================

def nmea_to_deg(raw, direction):
    if not raw or len(raw) < 4:
        return None
    deg_len = 2 if direction in ("N", "S") else 3
    try:
        deg = float(raw[:deg_len])
        minutes = float(raw[deg_len:])
        val = deg + minutes / 60.0
        return -val if direction in ("S", "W") else val
    except:
        return None

def parse_gga(line):
    parts = line.split(",")
    if len(parts) < 15:
        return None
    try:
        lat = nmea_to_deg(parts[2], parts[3])
        lon = nmea_to_deg(parts[4], parts[5])

        # --- FIX QUALITY FORCÉ À 5 ---
        fix_q = 5

        if lat is None or lon is None:
            return None
        return {"lat": lat, "lon": lon, "fix_quality": fix_q}
    except:
        return None

# =========================
# ANOMALIES
# =========================

def detect_anomalies(prev_pos, curr_pos, prev_time, curr_time, prev_fix, curr_fix):
    alerts = []
    if prev_pos and curr_pos:
        lat1, lon1 = prev_pos
        lat2, lon2 = curr_pos
        dt = curr_time - prev_time
        if dt > 0:
            R = 6371000.0
            dlat = math.radians(lat2 - lat1)
            dlon = math.radians(lon2 - lon1)
            a = math.sin(dlat/2)**2 + math.cos(math.radians(lat1))*math.cos(math.radians(lat2))*math.sin(dlon/2)**2
            c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
            dist = R * c
            speed = dist / dt
            if dist > DIST_JUMP_THRESHOLD and speed > SPEED_JUMP_THRESHOLD:
                alerts.append(f"Jump detected: {dist:.1f} m in {dt:.1f} s")

    # RTK loss impossible car fix forcé à 5
    return alerts

# =========================
# SIMULATION
# =========================

def simulation_thread(pos_queue):
    print("[SIM] Generating fake positions...")
    lat, lon = 48.8, 2.3
    while True:
        lat += random.uniform(-0.0001, 0.0001)
        lon += random.uniform(-0.0001, 0.0001)
        pos_queue.put((time.time(), {"lat": lat, "lon": lon, "fix_quality": 5}))
        time.sleep(1)

# =========================
# REAL GNSS READER
# =========================

def hardware_thread(pos_queue):
    print("[GNSS] Opening serial port COM9...")
    try:
        ser = serial.Serial(F9P_NMEA_PORT, F9P_BAUD, timeout=1)
        print("[GNSS] Serial OK")
    except Exception as e:
        print("[GNSS] ERROR: cannot open port:", e)
        return

    buffer = b""

    while True:
        try:
            if ser.in_waiting:
                buffer += ser.read(ser.in_waiting)

                while b"\n" in buffer:
                    line, buffer = buffer.split(b"\n", 1)
                    line = line.decode(errors="ignore").strip()

                    if "GGA" in line:
                        info = parse_gga(line)
                        if info:
                            pos_queue.put((time.time(), info))
                            print("[GNSS] GGA:", line[:60])
        except Exception as e:
            print("[GNSS] Error:", e)
            time.sleep(1)

# =========================
# API
# =========================

def send_to_laravel(lat, lon, fix_quality, alerts):
    payload = {
        "mission_id": CURRENT_MISSION_ID,
        "latitude": lat,
        "longitude": lon,
        "altitude": 0.0,
        "speed": 0.0,
        "pressure": 0.0,
        "timestamp": int(time.time())
    }

    print("[DEBUG API] Payload:", payload)

    try:
        r = requests.post(LARAVEL_API_URL, json=payload, timeout=2)
        if r.status_code >= 400:
            print("[API] Error", r.status_code, r.text[:80])
    except Exception as e:
        print("[API] Failed:", e)

# =========================
# POLLING
# =========================

def command_polling_thread():
    global SYSTEM_ACTIVE, CURRENT_MISSION_ID
    print("[CONTROL] Polling mission...")
    while True:
        try:
            r = requests.get(LARAVEL_MISSION_URL, timeout=3)
            if r.status_code == 200:
                data = r.json()
                CURRENT_MISSION_ID = data.get("data", {}).get("id")
                SYSTEM_ACTIVE = (data.get("data", {}).get("statut") == "ongoing")
        except:
            pass
        time.sleep(2)

# =========================
# MONITORING
# =========================

def monitoring_loop(pos_queue):
    prev_pos = None
    prev_time = None
    prev_fix = None

    print("=== START ===")
    print("SIMULATION =", SIMULATION)

    while True:
        ts, info = pos_queue.get()

        if not SYSTEM_ACTIVE:
            while not pos_queue.empty():
                pos_queue.get()
            time.sleep(0.5)
            continue

        lat, lon, fix_q = info["lat"], info["lon"], info["fix_quality"]
        curr_pos = (lat, lon)
        curr_time = ts

        alerts = detect_anomalies(prev_pos, curr_pos, prev_time or curr_time, curr_time, prev_fix, fix_q)

        print(f"[POS] {lat:.7f}, {lon:.7f}, Fix={fix_q}, Alerts={len(alerts)}")

        send_to_laravel(lat, lon, fix_q, alerts)

        prev_pos = curr_pos
        prev_time = curr_time
        prev_fix = fix_q

# =========================
# MAIN
# =========================

def main():
    pos_queue = queue.Queue()

    threading.Thread(target=command_polling_thread, daemon=True).start()

    if SIMULATION:
        threading.Thread(target=simulation_thread, args=(pos_queue,), daemon=True).start()
    else:
        threading.Thread(target=hardware_thread, args=(pos_queue,), daemon=True).start()

    monitoring_loop(pos_queue)

if __name__ == "__main__":
    main()

