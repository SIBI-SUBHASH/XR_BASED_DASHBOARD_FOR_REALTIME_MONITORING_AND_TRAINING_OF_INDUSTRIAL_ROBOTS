# 🤖 XR_Dash — Real-Time Robot Telemetry & Predictive Maintenance System

An AI-powered digital twin and predictive maintenance pipeline for the **FANUC M-20iB/25 industrial robot**. The system streams live motor telemetry into Blender for 3D visualization, runs dual anomaly detection (IsolationForest + LSTM), and triggers automated maintenance email alerts via an n8n workflow.

---

## 🧠 System Architecture

```
bridge.py (WebSocket Server + AI Engine)
    │
    ├──► Blender (Python_script_for_blender.txt)
    │       └── 3D Digital Twin | Color-coded health | Motion mirroring | Click-to-inspect panels
    │
    ├──► SQLite (telemetry_log.db)
    │       └── Persistent telemetry archive
    │
    └──► n8n Webhook (XR.json workflow)
            └── Alert filter → JS scoring → Groq LLM → Gmail alert
```

---

## 📁 File Overview

| File | Purpose |
|------|---------|
| `bridge.py` | Core backend — WebSocket server, AI anomaly detection, DB logging, n8n forwarding |
| `Python_script_for_blender.txt` | Blender script — 3D robot visualization, motion mirroring, health panels |
| `fanuc.blend` | Blender scene file — FANUC M-20iB/25 robot 3D model |
| `telemetry_log.db` | SQLite database — archived telemetry snapshots |
| `XR.json` | n8n workflow — alert processing, LLM analysis, Gmail notifications |

---

## ⚙️ Features

- **Dual AI Anomaly Detection**: IsolationForest (real-time) + LSTM neural network (sequential patterns)
- **6-Axis Digital Twin**: Live color-coded joint health in Blender (Green → Yellow → Orange → Red → Blinking Red)
- **Motion Mirroring**: Robot joints animate based on live RPM and vibration data
- **Click-to-Inspect Panels**: Click any joint in Blender to see a live telemetry overlay
- **Automated Email Alerts**: n8n + Groq LLaMA generates and sends professional maintenance emails
- **Persistent Logging**: All telemetry snapshots stored in SQLite

---

## 🚀 Getting Started

### Prerequisites

```bash
Python 3.9+
Blender 3.x or 4.x
n8n (self-hosted or cloud)
```

### Python Dependencies

```bash
pip install websockets scikit-learn numpy requests tensorflow
```

> **Note:** TensorFlow is optional. If not installed, the system runs with IsolationForest only (LSTM is skipped automatically).

---

## 🏃 Running the System

### Step 1 — Start the Bridge Server

```bash
python bridge.py
```

You should see:
```
✅ Database ready
🚀 WebSocket running at ws://localhost:8765
```

### Step 2 — Run the Blender Script

1. Open `fanuc.blend` in Blender
2. Go to the **Scripting** workspace
3. Open `Python_script_for_blender.txt`
4. Click **▶ Run Script**

The robot will connect to the WebSocket and start animating with live health colors.

### Step 3 — Import the n8n Workflow

1. Open your n8n instance
2. Go to **Workflows → Import**
3. Upload `XR.json`
4. Configure your **Groq API** and **Gmail OAuth2** credentials
5. Activate the workflow

---

## 🎨 Health Status Color Codes

| Color | Status | Health % |
|-------|--------|----------|
| 🟢 Green | NORMAL | ≥ 80% |
| 🟡 Yellow | LOW | 60–79% |
| 🟠 Orange | MEDIUM | 40–59% |
| 🔴 Dark Red | HIGH | 20–39% |
| 🚨 Blinking Red | CRITICAL | < 20% |
| 🔵 Blue | TRAINING | Warming up |

---

## 🔧 Configuration

Edit the top of `bridge.py` to adjust system parameters:

```python
NUM_MOTORS       = 6       # Number of robot axes
TRAINING_SAMPLES = 20      # Samples before IsolationForest activates
LSTM_SEQ_LEN     = 20      # Sequence length for LSTM input
LSTM_TRAIN_MIN   = 40      # Samples before LSTM training begins
CONTAMINATION    = 0.05    # IsolationForest anomaly sensitivity (5%)

WS_PORT          = 8765    # WebSocket port
DB_FILE          = "telemetry_log.db"
N8N_WEBHOOK      = "http://localhost:5678/webhook-test/robot/telemetry"
```

---

## 📧 n8n Alert Email Workflow

The `XR.json` workflow does the following automatically:

1. **Webhook** — Receives POST from `bridge.py` every second
2. **If Node** — Filters payloads that contain alerts
3. **JavaScript Code** — Enriches alert data with severity scoring
4. **AI Agent (Groq LLaMA 3.1)** — Generates a professional maintenance analysis
5. **Merge** — Combines alert data + AI output
6. **Gmail** — Sends a styled HTML email to the maintenance team

> Update the `sendTo` email address in the Gmail node before activating.

---

## 🗄️ Database Schema

```sql
CREATE TABLE telemetry_logs (
    id        INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT NOT NULL,
    payload   TEXT NOT NULL   -- Full JSON snapshot
);
```

---

## 📡 WebSocket Payload Format

```json
{
  "robot_id": "fanuc_r2000i",
  "timestamp": "2026-03-13T10:30:00.000Z",
  "telemetry": {
    "Motor1_Torque": 12.5,
    "Motor1_Temp": 45.2,
    "Motor1_RPM": 1400,
    "Motor1_Vibration": 0.9
  },
  "health_report": {
    "Motor1": {
      "health_percent": 87.3,
      "anomaly_probability": 12.7,
      "status": "NORMAL",
      "if_health": 87.3,
      "lstm_health": null
    }
  },
  "alerts": []
}
```

---

## 🛠️ Troubleshooting

| Problem | Solution |
|---------|----------|
| `Connection refused` on Blender | Make sure `bridge.py` is running before the Blender script |
| LSTM not training | Need 40+ telemetry samples first; watch console for `🔄 LSTM training started` |
| n8n webhook not receiving | Confirm `N8N_WEBHOOK` URL in `bridge.py` matches your n8n instance |
| Gmail not sending | Re-authorize Gmail OAuth2 credentials in n8n |
| Blender joints not moving | Verify object names match `motor_map` in the Blender script |

---

## 📄 License

Copyright (c) 2026 Sibi Subhash. All Rights Reserved.

This repository and its contents are proprietary. No permission is granted to use, 
copy, modify, distribute, or reproduce any part of this project without the 
explicit written consent of the author.

## 🙏 Acknowledgements

- [FANUC Robotics](https://www.fanucamerica.com/) — Robot platform
- [Blender](https://www.blender.org/) — 3D visualization engine
- [n8n](https://n8n.io/) — Workflow automation
- [Groq](https://groq.com/) — LLM inference (LLaMA 3.1)
- [scikit-learn](https://scikit-learn.org/) — IsolationForest
- [TensorFlow/Keras](https://www.tensorflow.org/) — LSTM model
