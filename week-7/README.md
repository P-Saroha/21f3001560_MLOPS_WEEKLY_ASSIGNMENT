# Week 7 – Stress Testing, Observability & Scaling the IRIS Pipeline

## Overview

This week's assignment focuses on validating the deployed IRIS Classification API under high concurrency, configuring Kubernetes Horizontal Pod Autoscaling (HPA), monitoring application behavior using Google Cloud Monitoring and Cloud Logging, and analyzing performance bottlenecks under constrained scaling.

The IRIS API is deployed on Google Kubernetes Engine (GKE) and tested using the `wrk` benchmarking tool to simulate production-like traffic.

---

## Objectives

- Perform stress testing using `wrk`.
- Configure Kubernetes Horizontal Pod Autoscaler (HPA).
- Observe pod scaling during high traffic.
- Monitor CPU and memory usage using Google Cloud Monitoring.
- Inspect application logs using Google Cloud Logging.
- Analyze performance bottlenecks under constrained scaling.
- Integrate stress testing into the existing CI/CD pipeline.

---

## Project Structure

```
week-7/
│
├── Dockerfile
├── deployment.yaml
├── service.yaml
├── hpa.yaml
├── iris_fastapi.py
├── iris_model.pkl
├── label_encoder.pkl
├── post.lua
├── requirements.txt
├── workflow.txt
└── README.md
```

---

## Technologies Used

- Python 3.12
- FastAPI
- Scikit-learn
- Docker
- Kubernetes (GKE)
- Google Artifact Registry
- Google Cloud Monitoring
- Google Cloud Logging
- GitHub Actions
- wrk

---

## Task 1 – CI/CD Workflow Enhancement

The existing GitHub Actions workflow was extended to include automated stress testing after deployment to Google Kubernetes Engine.

### Workflow Steps

- Checkout Repository
- Install Dependencies
- Authenticate with Google Cloud
- Build Docker Image
- Push Image to Artifact Registry
- Deploy to GKE
- Restart Deployment
- Install `wrk`
- Execute Stress Test

---

## Task 2 – Stress Testing

The deployed API was tested using `wrk`.

### Command

```bash
wrk -t4 -c1000 -d30s -s post.lua http://<EXTERNAL_IP>/predict
```

The following metrics were observed:

- Requests per Second
- Average Latency
- Socket Timeouts
- Throughput

---

## Task 3 – Horizontal Pod Autoscaler

Configured HPA using CPU utilization.

### Configuration

- Min Replicas: 1
- Max Replicas: 3
- CPU Target Utilization: 5%

The deployment was updated with CPU and memory requests/limits to enable autoscaling.

During the stress test:

- HPA metrics were monitored using:

```bash
kubectl get hpa
```

- Running Pods were observed using:

```bash
kubectl get pods
```

Autoscaling behavior was successfully demonstrated.

---

## Task 4 – Observability

### Google Cloud Monitoring

Metrics Explorer was used to monitor:

- CPU Usage
- Memory Usage

during stress testing.

### Google Cloud Logging

Logs Explorer was used to inspect:

- Application Logs
- HTTP Requests
- Container Logs

This helped verify application behavior during high load.

---

## Task 5 – Bottleneck Analysis

The HPA configuration was modified:

- Min Replicas: 1
- Max Replicas: 1

Stress testing was repeated using:

```bash
wrk -t4 -c2000 -d30s -s post.lua http://<EXTERNAL_IP>/predict
```

The results were compared against the autoscaling scenario.

### Comparison

| Scenario | Connections | Max Replicas |
|-----------|------------:|-------------:|
| Scenario 1 | 1000 | 3 |
| Scenario 2 | 2000 | 1 |

The constrained deployment experienced:

- Higher request latency
- Increased socket timeouts
- Lower throughput

This demonstrated the importance of autoscaling for handling increased traffic.

---

## Useful Commands

### Deploy

```bash
kubectl apply -f deployment.yaml
kubectl apply -f service.yaml
```

### Configure HPA

```bash
kubectl apply -f hpa.yaml
```

### View HPA

```bash
kubectl get hpa
```

### View Pods

```bash
kubectl get pods
```

### Stress Test

```bash
wrk -t4 -c1000 -d30s -s post.lua http://<EXTERNAL_IP>/predict
```

### Constrained Stress Test

```bash
wrk -t4 -c2000 -d30s -s post.lua http://<EXTERNAL_IP>/predict
```

---

## Learning Outcomes

- Performed high-concurrency stress testing using `wrk`.
- Configured Kubernetes Horizontal Pod Autoscaler.
- Observed pod scaling behavior under load.
- Monitored application metrics using Google Cloud Monitoring.
- Analyzed logs using Google Cloud Logging.
- Evaluated system performance under constrained autoscaling.
- Extended the CI/CD pipeline to include stress testing.

---

## Author

**Parveen Saroha**

**IIT Madras BS in Data Science and Applications**

**Roll Number:** 21f3001560