# 🩺 Predictive Health Monitoring System: Respiratory & Glucose Tracking

![Python](https://img.shields.io/badge/Python-3.x-blue?style=flat&logo=python)
![Google Cloud](https://img.shields.io/badge/GCP-Hosted-1a73e8?style=flat&logo=google-cloud)
![ESP32](https://img.shields.io/badge/Hardware-ESP32-black?style=flat&logo=espressif)
![MySQL](https://img.shields.io/badge/Database-SQL-orange?style=flat&logo=mysql)

## 📖 Overview
This project is a comprehensive, microservice-based health monitoring system. It leverages non-invasive Infrared (IR) sensor readings to accurately predict continuous Respiratory Rates and perform active glucose monitoring. By bridging edge computing hardware with cloud-based data processing, the system provides real-time, actionable biometric insights via a dedicated web interface.

## 🏗️ Architecture
The system is built on a scalable microservice architecture to ensure fast data transmission and reliable processing:

1.  **Edge Hardware (ESP32):** Collects raw IR readings and handles initial signal filtering before transmitting the data via Wi-Fi.
2.  **Cloud Server (Google Cloud Platform):** Acts as the central hub, receiving high-frequency telemetry data from the ESP32 and routing it to the processing modules.
3.  **Data Processing Pipeline (Python):** Utilizes advanced Python libraries (e.g., Pandas, NumPy, SciPy) and custom predictive algorithms to transform raw IR signals into accurate respiratory and glucose metrics.
4.  **Database (SQL):** Securely stores historical user data, system logs, and processed health metrics for retrieval and trend analysis.
5.  **Web Frontend:** A responsive user interface that queries the cloud server to display real-time analytics and historical health trends to the user.

## ✨ Key Features
*   **Non-Invasive Tracking:** Utilizes IR sensor data for pain-free, continuous monitoring.
*   **Real-Time Data Pipeline:** Low-latency transmission from ESP32 to GCP.
*   **Advanced Signal Processing:** Robust Python-based algorithms to filter noise and predict biometric values.
*   **Secure User Profiles:** SQL-backed infrastructure to manage and protect user data and session history.

## 🛠️ Tech Stack
*   **Hardware:** ESP32, IR Sensors
*   **Backend & Cloud:** Google Cloud Platform (GCP), Python (FastAPI)
*   **Data Science:** Python, NumPy, SciPy, Pandas
*   **Database:** SQL (MySQL)
*   **Frontend:** Gradio


---
*Note: This project is currently in development. Please refer to the specific branches for experimental features and algorithms.*
