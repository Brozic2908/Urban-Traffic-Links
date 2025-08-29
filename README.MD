# Phân tích Mối liên hệ Giao thông TP.HCM sử dụng Big Data và Deep Learning

**Dự án chuyên ngành Kỹ thuật Máy tính - HK251**

## 📜 Giới thiệu

Dự án này tập trung vào việc nghiên cứu và xây dựng một hệ thống phân tích tình trạng giao thông (TTGT) tại các nút giao thông quan trọng ở Thành phố Hồ Chí Minh. Mục tiêu chính là khám phá và mô hình hóa mối liên hệ, tương quan về TTGT giữa các nút giao thông khác nhau trong một mạng lưới phức tạp, từ đó đưa ra những dự báo và đánh giá có giá trị.

Hệ thống sử dụng các công nghệ Big Data và Deep Learning hiện đại để xử lý lượng dữ liệu lớn và xây dựng các mô hình dự báo thông minh, với mục tiêu tích hợp vào hệ thống UTraffic.

## 🎯 Mục tiêu chính

- **Khảo sát**: Nghiên cứu các giải pháp phân tích TTGT và các nghiên cứu hiện có về sự tương quan trong mạng lưới giao thông đô thị.
- **Thu thập & Xử lý Dữ liệu**: Xây dựng cơ chế thu thập dữ liệu TTGT và sử dụng Apache Spark để xử lý, làm sạch và trích xuất đặc trưng từ dữ liệu lớn.
- **Mô hình hóa**: Ứng dụng PyTorch để xây dựng các mô hình Deep Learning (ví dụ: Graph Neural Networks, LSTM) nhằm phân tích sự tương quan và dự báo TTGT.
- **Xây dựng Demo**: Triển khai mô hình thành một dịch vụ API sử dụng FastAPI và đóng gói bằng Docker để dễ dàng triển khai và tích hợp.
- **Đánh giá**: Thử nghiệm và đánh giá giải pháp trên dữ liệu thực tế tại TP.HCM.

## 🛠️ Ngăn xếp Công nghệ (Tech Stack)

| Lĩnh vực            | Công nghệ                                                                                                  |
| ------------------- | ---------------------------------------------------------------------------------------------------------- |
| Big Data Processing | 🐘 Apache Spark (PySpark) - Dùng để xử lý, biến đổi và phân tích các tập dữ liệu giao thông quy mô lớn.    |
| Deep Learning       | 🔥 PyTorch - Framework chính để xây dựng, huấn luyện và đánh giá các mô hình mạng nơ-ron.                  |
| API Development     | 🚀 FastAPI - Xây dựng API hiệu suất cao để cung cấp các dự đoán từ mô hình.                                |
| Deployment          | 🐳 Docker & Docker Compose - Đóng gói và triển khai ứng dụng một cách nhất quán và độc lập với môi trường. |
| Ngôn ngữ            | 🐍 Python 3.9+                                                                                             |
| Quản lý Dữ liệu     | Pandas, NumPy, Parquet                                                                                     |

## 🗂️ Cấu trúc Thư mục

```
HCM_TNA/
│
├── 📂 data/                  # Chứa dữ liệu thô, đã xử lý và features
├── 📂 notebooks/             # Jupyter notebooks cho phân tích khám phá
├── 📂 src/                   # Mã nguồn chính của dự án
│   ├── 📦 data_processing/   # Scripts xử lý dữ liệu với PySpark
│   ├── 📦 model/             # Mã nguồn xây dựng và huấn luyện mô hình PyTorch
│   └── 📦 visualization/     # Scripts trực quan hóa
│
├── 📂 models/                # Lưu các file model .pth đã huấn luyện
├── 📂 deployment/            # Cấu hình và scripts để triển khai (Docker, FastAPI)
├── 📜 .gitignore
├── 📜 README.md
├── 📜 requirements.txt       # Danh sách các thư viện Python cần thiết
└── 📜 config.yaml            # File cấu hình tập trung
```

## ⚙️ Hướng dẫn Cài đặt & Sử dụng

### Yêu cầu tiên quyết

- Git
- Python 3.9+
- Docker và Docker Compose

### Các bước thực hiện

**Clone repository về máy:**

```bash
git clone <URL_CUA_REPOSITORY>
cd hcmc-traffic-spark-pytorch
```

**Cài đặt các thư viện Python cần thiết:**

```bash
pip install -r requirements.txt
```

**(Tùy chọn) Chạy tác vụ xử lý dữ liệu với Spark:**

```bash
# Ví dụ lệnh để chạy một Spark job
python src/data_processing/spark_job.py
```

**Huấn luyện mô hình Deep Learning:**

```bash
python src/model/train.py
```

Quá trình này sẽ huấn luyện mô hình PyTorch và lưu file trọng số vào thư mục `models/`.

**Khởi chạy hệ thống API bằng Docker:**

```bash
docker-compose up --build
```

API sẽ chạy tại địa chỉ: [http://localhost:8000](http://localhost:8000)  
Tài liệu API: [http://localhost:8000/docs](http://localhost:8000/docs)

## 🧩 Workflow Luồng làm việc của Dự án

1. **Thu thập Dữ liệu**: Lấy dữ liệu thô về TTGT từ hệ thống UTraffic và các nguồn khác.
2. **Xử lý Dữ liệu Lớn**: Sử dụng PySpark để làm sạch, tổng hợp dữ liệu và tạo các features (ví dụ: lưu lượng xe, vận tốc trung bình theo từng khung giờ).
3. **Huấn luyện Mô hình**: Dữ liệu features được đưa vào mô hình PyTorch để huấn luyện, tìm ra các quy luật và mối tương quan.
4. **Triển khai API**: Mô hình đã huấn luyện được đóng gói cùng với FastAPI trong một Docker container.
5. **Tích hợp**: API có thể được gọi bởi hệ thống UTraffic hoặc các ứng dụng khác để lấy thông tin phân tích.

## 📚 Tài liệu tham khảo

- [UTraffic System](https://bktraffic.com/home/)
