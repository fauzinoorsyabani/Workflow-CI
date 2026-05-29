
# Menggunakan base image Python
FROM python:3.12-slim

# Menentukan direktori kerja di dalam container
WORKDIR /app

# Meng-copy semua file dari folder MLProject ke dalam container
COPY MLProject/ /app/

# Menginstal semua library yang dibutuhkan
RUN pip install setuptools pandas scikit-learn mlflow dagshub

# Perintah yang akan dijalankan saat container dihidupkan
CMD ["python", "modelling.py"]
