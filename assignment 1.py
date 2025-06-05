import numpy as np
from skimage.io import imread
from skimage.transform import resize
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
#pip install scikit-learn scikit-image matplotlib
#pip install numpy
# ==== 1. Đọc ảnh và gán nhãn ====
duong_dan_anh = ["D:/c/c1.png", "D:/c/c2.png", "D:/c/c3.png", "D:/c/c4.png"]  # Đặt ảnh cùng thư mục với file này
nhan = [0, 1, 1, 1]  # 0 = Nhà, 1 = Hoa

# ==== 2. Trích đặc trưng màu bằng histogram RGB ====
def trich_dac_trung_mau(anh, bins=(8, 8, 8)):
    anh_doi_kich_thuoc = resize(anh, (128, 128), anti_aliasing=True)
    if anh_doi_kich_thuoc.shape[-1] == 4:
        anh_doi_kich_thuoc = anh_doi_kich_thuoc[:, :, :3]  # Bỏ kênh alpha nếu có
    hist, _ = np.histogramdd(
        anh_doi_kich_thuoc.reshape(-1, 3),
        bins=bins,
        range=[(0, 1), (0, 1), (0, 1)]
    )
    hist = hist.flatten()
    hist /= hist.sum()  # Chuẩn hóa histogram
    return hist

# Trích đặc trưng cho tất cả ảnh
dac_trung = []
for duong_dan in duong_dan_anh:
    anh = imread(duong_dan)
    dac_trung_mau = trich_dac_trung_mau(anh)
    dac_trung.append(dac_trung_mau)

X = np.array(dac_trung)
y = np.array(nhan)

# ==== 3. Huấn luyện bộ phân loại KNN ====
knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(X, y)
du_doan = knn.predict(X)

# ==== 4. Đánh giá độ chính xác ====
print(" Độ chính xác:", accuracy_score(y, du_doan))
print(" Báo cáo phân loại:")
print(classification_report(y, du_doan, target_names=["Nhà", "Hoa"]))

# ==== 5. Phân đoạn ảnh bằng thuật toán K-means ====
anh_mau = imread(duong_dan_anh[0])
anh_mau = resize(anh_mau, (128, 128), anti_aliasing=True)
if anh_mau.shape[-1] == 4:
    anh_mau = anh_mau[:, :, :3]

cac_pixel = anh_mau.reshape((-1, 3))
kmeans = KMeans(n_clusters=3, random_state=42)
kmeans.fit(cac_pixel)
anh_phan_cum = kmeans.labels_.reshape((128, 128))

# ==== 6. Hiển thị ảnh gốc và ảnh sau phân đoạn ====
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.title("Ảnh gốc ")
plt.imshow(anh_mau)
plt.axis('off')

plt.subplot(1, 2, 2)
plt.title("Phân đoạn ảnh bằng K-means")
plt.imshow(anh_phan_cum, cmap='viridis')
plt.axis('off')

plt.tight_layout()
plt.show()
