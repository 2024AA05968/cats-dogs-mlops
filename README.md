# Cats vs Dogs MLOps

End-to-end MLOps pipeline for binary image classification (Cats vs Dogs), covering
model development, packaging, CI/CD, deployment, and validation.

---

## Project Overview

This project demonstrates a complete MLOps workflow, starting from model training
and experiment tracking, through containerization and CI, and finally deployment
and post-deployment validation on a virtual machine.

The pipeline is structured into clearly defined modules (M1–M4) aligned with the
assignment requirements.

---

## Modules Overview

### M1: Model Development & Experiment Tracking
- Implemented a baseline Convolutional Neural Network (CNN) using PyTorch for
  Cats vs Dogs image classification.
- Performed data preprocessing, train/validation/test splitting, and data leakage checks.
- Trained the model and tracked experiments (parameters, metrics, and artifacts)
  using MLflow.
- Saved the trained model artifact (`baseline_cnn.pt`) for downstream inference
  and deployment.

---

### M2: Packaging & Containerization
- Developed a FastAPI-based inference service exposing `/health` and `/predict` endpoints.
- Packaged the inference service into a Docker container.
- Created a Dockerfile and `.dockerignore` for reproducible container builds.
- Defined deployment manifests using Docker Compose.

---

### M3: Continuous Integration (CI)
- Configured GitHub Actions to run automated unit tests (pytest) on every push.
- Added unit tests for preprocessing and inference utilities.
- Implemented a CI workflow to build and publish the Docker image to
  GitHub Container Registry (GHCR).
- Managed large artifacts using Git LFS and ensured the trained model artifact
  is correctly included in the container image during CI builds.

---

### M4: Continuous Deployment (CD) & Deployment
- Deployed the inference service on the OSHA lab virtual machine using Docker Compose.
- Pulled the latest container image from GHCR and started the service on the VM.
- Performed post-deployment smoke tests by invoking the `/health` and `/predict`
  endpoints to validate service availability and inference correctness.
- Verified the deployed service is healthy and returns predictions successfully.

---

## Docker Execution Note (Local & Deployment Environment)

Docker Desktop installation and local container execution could not be performed
on the local development machine due to endpoint security restrictions.

The inference service was deployed and validated on the OSHA lab virtual machine
using Docker Compose. The Docker image was built and published via GitHub Actions
to GitHub Container Registry (GHCR), then pulled and executed on the OSHA VM.

Post-deployment smoke tests were performed using the `/health` and `/predict`
endpoints to confirm successful deployment and model inference.

---

## Final Submission

The final submission includes:
- Source code for all modules (M1–M4)
- CI/CD workflows (GitHub Actions)
- Docker artifacts (Dockerfile, docker-compose.yml)
- Unit tests and test configuration
- Screen recording (< 5 minutes) demonstrating:
  - CI pipeline execution
  - Container image publishing to GHCR
  - Deployment on OSHA VM
  - Live inference using `/health` and `/predict`