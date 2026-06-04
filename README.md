# 🔬 Skin Disease Detector — MLOps with Kubernetes

![Docker](https://img.shields.io/badge/Docker-2496ED?style=for-the-badge&logo=docker&logoColor=white)
![Kubernetes](https://img.shields.io/badge/Kubernetes-326CE5?style=for-the-badge&logo=kubernetes&logoColor=white)
![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Flask](https://img.shields.io/badge/Flask-000000?style=for-the-badge&logo=flask&logoColor=white)
![Prometheus](https://img.shields.io/badge/Prometheus-E6522C?style=for-the-badge&logo=prometheus&logoColor=white)
![Grafana](https://img.shields.io/badge/Grafana-F46800?style=for-the-badge&logo=grafana&logoColor=white)

> A production-grade MLOps project demonstrating how to deploy, scale, and monitor a Machine Learning model using Docker and Kubernetes.

---

## 🎯 Project Overview

This project deploys a **skin disease detection ML model** (DINOv2 fine-tuned) as a containerized microservice with full **auto-scaling** and **monitoring** capabilities.

**The ML model can detect 19 types of skin diseases** including Melanoma, Impetigo, Psoriasis, and more — with confidence scores for top 3 predictions.

---

## 🏗️ Architecture

```
User Request
     ↓
Kubernetes Service (Load Balancer)
     ↓
┌─────────┐  ┌─────────┐  ┌─────────┐
│  Pod 1  │  │  Pod 2  │  │  Pod N  │  ← Auto-scaled by HPA
│ [Flask] │  │ [Flask] │  │ [Flask] │
│ [DINOv2]│  │ [DINOv2]│  │ [DINOv2]│
└─────────┘  └─────────┘  └─────────┘
     ↓
Prometheus (Metrics) → Grafana (Dashboard)
```

---

## ✨ Key Features

- 🤖 **ML Model**: DINOv2 fine-tuned for skin disease classification (19 classes)
- 🐳 **Dockerized**: Fully containerized with optimized multi-layer caching
- ☸️ **Kubernetes**: Deployed with Deployment, Service, and HPA
- 📈 **Auto-Scaling**: HPA scales pods from 2 → 10 based on CPU usage (threshold: 50%)
- 📊 **Monitoring**: Real-time metrics with Prometheus + Grafana dashboard
- 🏥 **Health Checks**: Readiness and Liveness probes for zero-downtime

---

## 🛠️ Tech Stack

| Category | Technology |
|---|---|
| **ML Model** | DINOv2 (Facebook), HuggingFace Transformers |
| **Backend** | Python, Flask |
| **Containerization** | Docker, Docker Hub |
| **Orchestration** | Kubernetes, Minikube |
| **Auto-Scaling** | Horizontal Pod Autoscaler (HPA) |
| **Monitoring** | Prometheus, Grafana |

---

## 🚀 Quick Start

### Prerequisites
- Docker Desktop
- Minikube
- kubectl
- Python 3.11+

### 1. Clone Repository
```bash
git clone https://github.com/astrogurldev/skin-disease-k8s.git
cd skin-disease-k8s
```

### 2. Build & Push Docker Image
```bash
docker build -f docker/Dockerfile -t astrogurldev/skin-disease-detector:v1 .
docker push astrogurldev/skin-disease-detector:v1
```

### 3. Deploy to Kubernetes
```bash
minikube start
minikube addons enable metrics-server
kubectl apply -f k8s/deployment.yaml
kubectl apply -f k8s/service.yaml
kubectl apply -f k8s/hpa.yaml
```

### 4. Access the App
```bash
kubectl port-forward service/skin-disease-detector-service 8080:80
```
Open: http://localhost:8080

### 5. Start Monitoring
```bash
cd monitoring
docker-compose up -d
```
- Prometheus: http://localhost:9090
- Grafana: http://localhost:3000 (admin/admin123)

---

## 📊 Auto-Scaling Demo

HPA automatically scales pods based on CPU usage:

```bash
# Watch pods scale in real-time
kubectl get pods -w

# Check HPA status
kubectl get hpa
```

When CPU exceeds 50%, Kubernetes automatically adds new pods:
```
TARGETS       MINPODS   MAXPODS   REPLICAS
cpu: 94%/50%  2         10        4          ← Scaled up automatically!
```

---

## 📁 Project Structure

```
skin-disease-k8s/
├── app/
│   ├── app.py              # Flask API + ML model
│   ├── requirements.txt    # Python dependencies
│   └── templates/
│       └── index.html      # Web UI
├── docker/
│   └── Dockerfile          # Docker image definition
├── k8s/
│   ├── deployment.yaml     # Kubernetes Deployment
│   ├── service.yaml        # Kubernetes Service
│   └── hpa.yaml            # Horizontal Pod Autoscaler
├── monitoring/
│   ├── docker-compose.yml  # Prometheus + Grafana stack
│   └── prometheus.yml      # Prometheus config
└── README.md
```

---

## 🤝 Connect

**Aisha** — DevOps Engineer
- GitHub: [@astrogurldev](https://github.com/astrogurldev)
- Docker Hub: [astrogurldev](https://hub.docker.com/u/astrogurldev)