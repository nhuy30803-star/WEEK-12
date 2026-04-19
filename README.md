Iris Flower Classification: A Machine Learning Approach

Dự án này thực hiện bài toán phân loại (Classification) trên bộ dữ liệu kinh điển Iris Flower Dataset. Mục tiêu là xây dựng các mô hình học máy để dự đoán chính xác loài hoa Iris dựa trên các đặc tính về kích thước đài hoa (sepal) và cánh hoa (petal).
🚀 Quy trình thực hiện (Project Workflow)

Dự án được thực hiện qua các bước bài bản:

    Exploratory Data Analysis (EDA): Sử dụng seaborn và matplotlib để trực quan hóa mối quan hệ giữa các đặc tính thông qua biểu đồ Pairplot.

    Data Preprocessing: Chuẩn hóa dữ liệu bằng StandardScaler và chia tập dữ liệu (Training/Testing) để đảm bảo tính khách quan.

    Model Training: Triển khai đồng thời 3 thuật toán phổ biến:

        Logistic Regression

        Support Vector Machine (SVM)

        Random Forest

    Evaluation: Đánh giá và so sánh hiệu suất các mô hình để tìm ra thuật toán tối ưu nhất cho bài toán.

📊 Kết quả và Đánh giá (Results & Evaluation)

Mỗi mô hình được đánh giá dựa trên các chỉ số chuẩn trong Classification:

    Accuracy Score: Tỷ lệ dự đoán đúng trên tổng số mẫu.

    Classification Report: Chi tiết về Precision, Recall và F1-score cho từng loại hoa (Setosa, Versicolor, Virginica).

    Best Model Selection: Tự động xác định mô hình có độ chính xác cao nhất.

🛠 Yêu cầu hệ thống (Dependencies)

Các thư viện chính được sử dụng trong dự án:

    scikit-learn: Xây dựng mô hình học máy.

    pandas & numpy: Xử lý và tính toán dữ liệu.

    seaborn & matplotlib: Trực quan hóa dữ liệu.

📂 Cách chạy dự án

    Cài đặt thư viện:
    Bash

    pip install -r requirements.txt

    Chạy script phân loại:
    Bash

    python iris_classification.py**
    
    **
