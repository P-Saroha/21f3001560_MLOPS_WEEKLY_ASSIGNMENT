# Week 6 - Deploying the IRIS Prediction API on Google Kubernetes Engine (GKE)

## Problem Statement

The objective of this assignment is to deploy the IRIS Machine Learning prediction API on Google Kubernetes Engine (GKE) using Docker and Kubernetes.

The API developed in previous assignments is containerized using Docker, stored in Google Artifact Registry, and deployed to a Kubernetes cluster. GitHub Actions is extended to automate the build, push, and deployment process, enabling Continuous Deployment (CD).

---

# Objectives

- Containerize the FastAPI application using Docker.
- Push the Docker image to Google Artifact Registry.
- Create a Google Kubernetes Engine (GKE) cluster.
- Deploy the application using Kubernetes Deployment and Service.
- Expose the API using a LoadBalancer.
- Verify that the deployed API serves predictions correctly.
- Automate deployment through GitHub Actions.

---

# Project Structure

```
week-6/
│
├── Dockerfile
├── iris_fastapi.py
├── deployment.yaml
├── service.yaml
├── iris_model.pkl
├── label_encoder.pkl
├── requirements.txt
└── README.md
```

---

# Technology Stack

- Python 3.12
- FastAPI
- Docker
- Google Cloud Platform (GCP)
- Google Artifact Registry
- Google Kubernetes Engine (GKE)
- Kubernetes
- GitHub Actions
- Joblib
- Scikit-Learn

---

# Workflow

```
FastAPI
      │
      ▼
Docker Image
      │
      ▼
Artifact Registry
      │
      ▼
Google Kubernetes Engine
      │
      ▼
Deployment + Service
      │
      ▼
LoadBalancer
      │
      ▼
Prediction API
```

---

# Docker Build

Build the Docker image

```bash
docker build -t iris-api:v1 .
```

Run locally

```bash
docker run -d -p 8200:8200 iris-api:v1
```

Test

```bash
curl http://localhost:8200/
```

---

# Push Image to Artifact Registry

Configure Docker

```bash
gcloud auth configure-docker us-central1-docker.pkg.dev
```

Tag Image

```bash
docker tag iris-api:v1 \
us-central1-docker.pkg.dev/<PROJECT_ID>/iris-repo/iris-api:latest
```

Push Image

```bash
docker push \
us-central1-docker.pkg.dev/<PROJECT_ID>/iris-repo/iris-api:latest
```

---

# Create GKE Cluster

```bash
gcloud container clusters create iris-cluster \
--zone us-central1-a \
--num-nodes 1
```

Configure kubectl

```bash
gcloud container clusters get-credentials iris-cluster \
--zone us-central1-a
```

---

# Kubernetes Deployment

Deploy

```bash
kubectl apply -f deployment.yaml
```

Create Service

```bash
kubectl apply -f service.yaml
```

Check Pods

```bash
kubectl get pods
```

Check Services

```bash
kubectl get services
```

---

# API Verification

Health Endpoint

```bash
curl http://<EXTERNAL_IP>/
```

Output

```json
{
  "message":"IRIS Classification API is running successfully!"
}
```

Prediction Endpoint

```bash
curl -X POST http://<EXTERNAL_IP>/predict \
-H "Content-Type: application/json" \
-d '{
"sepal_length":5.1,
"sepal_width":3.5,
"petal_length":1.4,
"petal_width":0.2
}'
```

Output

```json
{
  "prediction":"setosa"
}
```

---

# GitHub Actions

The CI/CD pipeline performs the following tasks automatically:

- Checkout Repository
- Install Dependencies
- Authenticate with Google Cloud
- Build Docker Image
- Push Image to Artifact Registry
- Get GKE Credentials
- Deploy to Kubernetes
- Restart Deployment
- Verify Rollout

---

# Results

Successfully deployed the FastAPI IRIS Prediction API on Google Kubernetes Engine.

Verified:

- Docker Image Build
- Artifact Registry Push
- Kubernetes Deployment
- LoadBalancer Service
- External API Access
- Prediction Endpoint

---

# Conclusion

The Week-6 assignment extends the MLOps pipeline by introducing container orchestration using Kubernetes and Continuous Deployment using GitHub Actions. The deployed API is publicly accessible through a LoadBalancer and successfully serves machine learning predictions.