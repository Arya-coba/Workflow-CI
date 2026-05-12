# Workflow-CI
## Credit Card Fraud Detection — CI/CD Pipeline

---

## 📁 Struktur Repository

```
Workflow-CI/
├── .github/
│   └── workflows/
│       └── ci.yml                      ← GitHub Actions CI workflow
└── MLProject/
    ├── MLProject                        ← Konfigurasi MLflow Project
    ├── conda.yaml                       ← Environment dependencies
    ├── modelling.py                     ← Script training
    ├── Dockerfile                       ← Docker image definition
    ├── requirements.txt                 ← Python dependencies
    └── creditcard_preprocessing/        ← Dataset hasil preprocessing
        ├── train.csv
        └── test.csv
```

---

## ⚙️ CI/CD Pipeline

Workflow otomatis berjalan saat ada push ke `main` pada folder `MLProject/`:

1. ✅ Setup Python 3.12
2. ✅ Install dependencies
3. ✅ Jalankan MLflow Project (training model)
4. ✅ Upload model artifact ke GitHub
5. ✅ Build Docker image
6. ✅ Push Docker image ke Docker Hub

---

## 🐳 Docker Image

```bash
docker pull arya9696/creditcard-fraud-detection:latest
docker run -p 5001:5001 arya9696/creditcard-fraud-detection:latest
```

---

## 🔗 Links

- **DagsHub MLflow:** https://dagshub.com/Arya-coba/Workflow-CI.mlflow
- **Docker Hub:** https://hub.docker.com/r/arya9696/creditcard-fraud-detection
