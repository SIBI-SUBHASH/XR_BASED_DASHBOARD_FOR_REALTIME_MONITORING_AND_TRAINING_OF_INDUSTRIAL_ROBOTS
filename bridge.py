import os
import sys
import warnings

# Suppress ALL warnings before any imports
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
warnings.filterwarnings('ignore')

import asyncio
import json
import random
from datetime import datetime
import websockets
import sqlite3
import requests
import numpy as np
from collections import deque
from sklearn.ensemble import IsolationForest
import threading
import time
import logging

# Kill all loggers
logging.getLogger('tensorflow').setLevel(logging.FATAL)
logging.getLogger('keras').setLevel(logging.FATAL)
logging.disable(logging.WARNING)

# TensorFlow import
try:
    import tensorflow as tf
    tf.get_logger().setLevel('ERROR')
    tf.autograph.set_verbosity(0)
    import absl.logging
    absl.logging.set_verbosity(absl.logging.ERROR)
    from tensorflow import keras
    LSTM_AVAILABLE = True
except ImportError:
    LSTM_AVAILABLE = False

# ----------------------------
# CONFIGURATION
# ----------------------------
NUM_MOTORS = 6
TRAINING_SAMPLES = 20
LSTM_SEQ_LEN = 20
LSTM_TRAIN_MIN = 40
CONTAMINATION = 0.05

WS_HOST = "localhost"
WS_PORT = 8765
DB_FILE = "telemetry_log.db"
N8N_WEBHOOK = "http://localhost:5678/webhook-test/robot/telemetry"

# ----------------------------
# STORAGE
# ----------------------------
motor_models = {}
motor_training_data = {}
motor_trained = {}
motor_score_history = {}

lstm_models = {}
lstm_sequences = {}
lstm_train_buffer = {}
lstm_trained = {}
lstm_training_started = {}
lstm_score_history = {}

for i in range(1, NUM_MOTORS + 1):
    name = f"Motor{i}"
    motor_models[name] = IsolationForest(contamination=CONTAMINATION)
    motor_training_data[name] = []
    motor_trained[name] = False
    motor_score_history[name] = deque(maxlen=200)

    lstm_models[name] = None
    lstm_sequences[name] = deque(maxlen=LSTM_SEQ_LEN)
    lstm_train_buffer[name] = []
    lstm_trained[name] = False
    lstm_training_started[name] = False
    lstm_score_history[name] = deque(maxlen=200)

# ----------------------------
# DATABASE
# ----------------------------
def init_db():
    conn = sqlite3.connect(DB_FILE)
    conn.cursor().execute("""
        CREATE TABLE IF NOT EXISTS telemetry_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            payload TEXT NOT NULL
        )
    """)
    conn.commit()
    conn.close()
    print("✅ Database ready")

def log_to_db(payload):
    conn = sqlite3.connect(DB_FILE)
    conn.cursor().execute(
        "INSERT INTO telemetry_logs (timestamp, payload) VALUES (?, ?)",
        (datetime.now().isoformat(), json.dumps(payload))
    )
    conn.commit()
    conn.close()

# ----------------------------
# TELEMETRY GENERATOR
# ----------------------------
def generate_dummy_data():
    data = {}
    for i in range(1, NUM_MOTORS + 1):
        p = f"Motor{i}_"
        data[p + "Torque"]    = round(random.uniform(10, 15), 2)
        data[p + "Temp"]      = round(random.uniform(40, 50), 2)
        data[p + "RPM"]       = random.randint(1350, 1450)
        data[p + "Vibration"] = round(random.uniform(4, 6), 3) if i == 3 else round(random.uniform(0.5, 1.5), 3)
    return data

def extract_features(packet, motor):
    p = motor + "_"
    return [packet[p+"Torque"], packet[p+"Temp"], packet[p+"RPM"], packet[p+"Vibration"]]

# ----------------------------
# HEALTH SCORING
# ----------------------------
def normalize_score(score, history):
    history.append(score)
    mn, mx = min(history), max(history)
    if mx - mn == 0:
        return 100.0
    return round((score - mn) / (mx - mn) * 100, 2)

def health_to_severity(h):
    if h >= 80: return "NORMAL"
    if h >= 60: return "LOW"
    if h >= 40: return "MEDIUM"
    if h >= 20: return "HIGH"
    return "CRITICAL"

# ----------------------------
# LSTM
# ----------------------------
def build_lstm(seq_len, n_features):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        inp = keras.Input(shape=(seq_len, n_features))
        x   = keras.layers.LSTM(32)(inp)
        out = keras.layers.Dense(1, activation='sigmoid')(x)
        model = keras.Model(inp, out)
        model.compile(optimizer='adam', loss='binary_crossentropy')
    return model

def prepare_lstm_data(motor):
    buf = np.array(lstm_train_buffer[motor])
    X, y = [], []
    for i in range(len(buf) - LSTM_SEQ_LEN):
        X.append(buf[i:i+LSTM_SEQ_LEN])
        y.append(1 if buf[i+LSTM_SEQ_LEN][3] > 2.5 else 0)
    return np.array(X), np.array(y)

def train_lstm_thread(motor, X, y):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            model = build_lstm(LSTM_SEQ_LEN, X.shape[2])
            model.fit(X, y, epochs=5, batch_size=8, verbose=0)
            lstm_models[motor] = model
            lstm_trained[motor] = True
            print(f"✅ LSTM trained for {motor}")
        except Exception as e:
            print(f"⚠️  LSTM failed for {motor}: {e}")

# ----------------------------
# ANOMALY DETECTION
# ----------------------------
def detect_anomalies(packet):
    alerts, health_report = [], {}

    for i in range(1, NUM_MOTORS + 1):
        motor    = f"Motor{i}"
        features = extract_features(packet, motor)

        # IsolationForest training phase
        if not motor_trained[motor]:
            motor_training_data[motor].append(features)
            if len(motor_training_data[motor]) >= TRAINING_SAMPLES:
                motor_models[motor].fit(motor_training_data[motor])
                motor_trained[motor] = True
                print(f"✅ IsolationForest trained for {motor}")
            if LSTM_AVAILABLE:
                lstm_train_buffer[motor].append(features)
                lstm_sequences[motor].append(features)
            health_report[motor] = {"health_percent": 100, "anomaly_probability": 0,
                                    "status": "TRAINING", "if_health": 100, "lstm_health": None}
            continue

        # IsolationForest inference
        if_score  = motor_models[motor].decision_function([features])[0]
        if_health = normalize_score(if_score, motor_score_history[motor])

        # LSTM feed + training trigger
        lstm_health = None
        if LSTM_AVAILABLE:
            lstm_train_buffer[motor].append(features)
            lstm_sequences[motor].append(features)

            if (not lstm_training_started[motor]
                    and len(lstm_train_buffer[motor]) >= LSTM_TRAIN_MIN):
                lstm_training_started[motor] = True
                X, y = prepare_lstm_data(motor)
                if len(X) > 0:
                    print(f"🔄 LSTM training started for {motor}...")
                    t = threading.Thread(target=train_lstm_thread, args=(motor, X, y), daemon=True)
                    t.start()

            # LSTM inference
            if lstm_trained[motor] and len(lstm_sequences[motor]) == LSTM_SEQ_LEN:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    seq  = np.array(list(lstm_sequences[motor])).reshape(1, LSTM_SEQ_LEN, 4)
                    prob = float(lstm_models[motor].predict(seq, verbose=0)[0][0])
                lstm_health = normalize_score(1.0 - prob, lstm_score_history[motor])

        # Combine scores
        combined = round(0.6 * if_health + 0.4 * lstm_health, 2) if lstm_health is not None else if_health
        anomaly_prob = round(100 - combined, 2)
        severity     = health_to_severity(combined)

        health_report[motor] = {
            "health_percent":    combined,
            "anomaly_probability": anomaly_prob,
            "status":            severity,
            "if_health":         if_health,
            "lstm_health":       lstm_health
        }

        if severity in ["HIGH", "CRITICAL"]:
            alerts.append({
                "motor":              motor,
                "issue":              "AI Pattern Deviation Detected",
                "severity":           severity,
                "health_percent":     combined,
                "anomaly_probability": anomaly_prob
            })

    return alerts, health_report

# ----------------------------
# N8N WATCHER
# ----------------------------
def n8n_watcher():
    was_listening = False
    while True:
        time.sleep(1)
        packet        = generate_dummy_data()
        alerts, health = detect_anomalies(packet)
        payload = {
            "robot_id":     "fanuc_r2000i",
            "timestamp":    datetime.now().isoformat(),
            "telemetry":    packet,
            "health_report": health,
            "alerts":       alerts
        }
        try:
            r = requests.post(N8N_WEBHOOK, json=payload, timeout=1)
            if r.status_code == 200 and not was_listening:
                was_listening = True
                print("✅ Real-time snapshot sent to n8n")
            elif r.status_code != 200:
                was_listening = False
        except:
            was_listening = False

# ----------------------------
# WEBSOCKET SERVER
# ----------------------------
async def telemetry_server(websocket):
    print("🔌 Unity connected")
    try:
        while True:
            packet        = generate_dummy_data()
            alerts, health = detect_anomalies(packet)
            payload = {
                "robot_id":     "fanuc_r2000i",
                "timestamp":    datetime.now().isoformat(),
                "telemetry":    packet,
                "health_report": health,
                "alerts":       alerts
            }
            await websocket.send(json.dumps(payload))
            log_to_db(payload)

            print("------ MOTOR STATUS ------")
            for m, d in health.items():
                lstm_str = f"{d['lstm_health']}%" if d['lstm_health'] is not None else "training..."
                print(f"{m} | Health: {d['health_percent']}% | "
                      f"IF: {d['if_health']}% | LSTM: {lstm_str} | "
                      f"Anomaly: {d['anomaly_probability']}% | Status: {d['status']}")
            print(f"🚨 Alerts: {len(alerts)}\n")

            await asyncio.sleep(0.5)
    except websockets.exceptions.ConnectionClosed:
        print("⚠️  Unity disconnected safely")

# ----------------------------
# MAIN
# ----------------------------
async def main():
    init_db()
    threading.Thread(target=n8n_watcher, daemon=True).start()
    print(f"🚀 WebSocket running at ws://{WS_HOST}:{WS_PORT}")
    async with websockets.serve(telemetry_server, WS_HOST, WS_PORT):
        await asyncio.Future()

if __name__ == "__main__":
    asyncio.run(main())