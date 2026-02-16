# Cats vs Dogs MLOps

End-to-end MLOps pipeline for binary image classification (Cats vs Dogs).

## Modules
- M1: Model Development & Experiment Tracking
- M2: Packaging & Containerization
- M3: CI for Build/Test/Image
- M4: CD & Deployment
- M5: Monitoring
- Final Submission

## Docker Execution Note
Docker Desktop installation and local container execution could not be performed due to endpoint security restrictions on the development machine.

The Dockerfile and .dockerignore were implemented and validated for correctness. The FastAPI inference service (/health, /predict) was tested successfully outside the container.

Container image build and execution are handled in the CI pipeline (GitHub Actions), which does not have these local restrictions.