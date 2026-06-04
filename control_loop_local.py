import time
from machine import Pin, ADC

from electrovanne import Electrovanne
from pressure_sensor import PressureSensor
from flow_sensor import FlowSensor
from ai_module import AIModule


class ControlLoop:
    """
    Modes :
    - MANUAL : contrôle direct
    - AUTO   : asservissement pression
    - IA     : ouverture selon zones IA
    """

    def __init__(self, valve_pins=[2], pressure_pins=[3], flow_pin=None):

        # --- Matériel ---
        self.valves = [Electrovanne(pin) for pin in valve_pins]
        self.pressure_sensors = [PressureSensor(pin) for pin in pressure_pins]
        self.flow = FlowSensor(flow_pin) if flow_pin else None

        # --- IA ---
        self.ai = AIModule(nb_zones=len(self.valves))
        self.ai_zones = [0] * len(self.valves)
        self.last_ai_time = 0
        self.ai_timeout = 2

        # --- Modes ---
        self.mode = "manual"

        # --- MANUAL ---
        self.manual_states = [0] * len(self.valves)

        # --- AUTO ---
        self.auto_period     = 1.0
        self.auto_open_ratio = 1.0
        self.dynamic_ratio   = 1.0
        self.target_pressure = 3.0
        self._auto_last_time = time.time()

        # --- Sécurité pression ---
        self.pressure_max = 4.0
        self.pressure_min = 0.05

        # --- Sécurité communication ---
        self.last_cmd_time = time.time()
        self.comm_timeout = None

        # --- Détection d'événements pression ---
        self.last_pressure = None
        self.last_event_time = None
        self.reaction_time = None

        # --- EAU ON/OFF ---
        self.water_available = True

        # --- Sécurité absence d’eau ---
        self.flow_protection = True
        self.no_flow_timeout = 3
        self.last_flow_time = time.time()

        print("[CTRL] Boucle d'asservissement initialisée")
        print(f"[CTRL] Cible={self.target_pressure} bar | Sécurité max={self.pressure_max} bar")


    # -----------------------------------------------------
    # RECEPTION COMMANDES JSON
    # -----------------------------------------------------
    def receive_command(self, json_str):
        import json
        data = json.loads(json_str)
        cmd = data["cmd"]
        self.last_cmd_time = time.time()

        if cmd == "set_mode":
            new_mode = data["value"]
            if new_mode in ["manual", "auto", "ia"]:
                self.mode = new_mode
                print(f"[MODE] Passage en mode {new_mode.upper()}")
            return

        if cmd == "set_valve":
            if self.mode != "manual":
                print("[WARN] Pas en mode MANUAL")
                return
            vid   = data["value"]["id"]
            state = data["value"]["state"]
            self.manual_states[vid] = state
            self.valves[vid].set_state(state)
            return

        if cmd == "ai_zones":
            zones = data["value"]
            self.ai.update_zones(zones)
            self.ai_zones = zones
            self.last_ai_time = time.time()
            self.mode = "ia"
            print("[AI] Zones mises à jour :", zones)
            return

        if cmd == "stop":
            self.emergency_stop()
            return

        if cmd == "water":
            if data["value"] == "on":
                self.water_available = True
                print("[WATER] Eau disponible")
            else:
                self.water_available = False
                print("[WATER] Eau coupée → STOP d'urgence")
                self.emergency_stop()
            return

        if cmd == "flow_protect":
            if data["value"] == "on":
                self.flow_protection = True
                print("[FLOW] Protection débit ACTIVÉE")
            else:
                self.flow_protection = False
                print("[FLOW] Protection débit DÉSACTIVÉE")
            return


    # -----------------------------------------------------
    # SECURITE
    # -----------------------------------------------------
    def emergency_stop(self):
        print("[CRITICAL] STOP d'urgence")
        for v in self.valves:
            v.set_state(0)
        self.mode = "manual"


    # -----------------------------------------------------
    # MODE AUTO
    # -----------------------------------------------------
    def compute_auto_states(self):
        now     = time.time()
        elapsed = now - self._auto_last_time

        if elapsed > self.auto_period:
            self._auto_last_time = now
            elapsed = 0

        open_time = self.auto_period * self.auto_open_ratio
        return [1 if elapsed < open_time else 0 for _ in self.valves]


    # -----------------------------------------------------
    # BOUCLE PRINCIPALE
    # -----------------------------------------------------
    def update(self):

        pressures        = [p.read() for p in self.pressure_sensors]
        current_pressure = pressures[0]

        if self.last_pressure is None:
            self.last_pressure = current_pressure

        delta = abs(current_pressure - self.last_pressure)
        if delta > 0.05:
            self.last_event_time = time.ticks_ms()
            print(f"[EVENT] ΔP={round(delta,3)} bar")

        self.last_pressure = current_pressure

        for i, pr in enumerate(pressures):
            if pr > self.pressure_max:
                print(f"[SECURITY] Pression trop haute capteur {i} : {pr:.2f} bar")
                self.emergency_stop()
                return
            if pr < self.pressure_min:
                print(f"[SECURITY] Pression trop basse capteur {i} : {pr:.2f} bar")
                self.emergency_stop()
                return

        if self.flow:
            debit, pulses = self.flow.read_flow()
            vitesse = self.flow.compute_speed(debit)

            if debit > 0.01:
                self.last_flow_time = time.time()

            if self.flow_protection and self.water_available:
                if time.time() - self.last_flow_time > self.no_flow_timeout:
                    print("[SECURITY] Absence d'eau détectée → STOP d'urgence")
                    self.emergency_stop()
                    return

        else:
            debit = vitesse = 0

        if self.mode == "manual":
            states = self.manual_states

        elif self.mode == "auto":

            error_percent = ((current_pressure - self.target_pressure) / self.target_pressure) * 100

            if abs(error_percent) > 5:
                if abs(error_percent) <= 10:
                    step = 0.02
                elif abs(error_percent) <= 20:
                    step = 0.05
                else:
                    step = 0.10

                if error_percent > 5:
                    self.dynamic_ratio -= step
                else:
                    self.dynamic_ratio += step

                self.dynamic_ratio   = min(max(self.dynamic_ratio, 0.0), 1.0)
                self.auto_open_ratio = self.dynamic_ratio

            states = self.compute_auto_states()

        elif self.mode == "ia":

            if time.time() - self.last_ai_time > self.ai_timeout:
                print("[IA] Timeout → EMERGENCY STOP")
                self.emergency_stop()
                return

            states = self.ai_zones

        else:
            states = [0] * len(self.valves)

        for i, v in enumerate(self.valves):
            v.set_state(states[i])

        # --- Logs + stockage état ---
        self.last_state = {
            "mode"      : self.mode,
            "pressures" : [round(p, 3) for p in pressures],
            "states"    : states,
            "ratio"     : round(self.dynamic_ratio, 3),
            "debit"     : round(debit, 3),
            "vitesse"   : round(vitesse, 3),
        }

        print(self.last_state)


    # -----------------------------------------------------
    # TICK NON BLOQUANT
    # -----------------------------------------------------
    def tick(self, interval=1):
        current = time.time()
        if not hasattr(self, "_last_tick"):
            self._last_tick = current

        if current - self._last_tick >= interval:
            self.update()
            self._last_tick = current
