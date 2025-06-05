import cv2
import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# === Hàm tính histogram RGB ===
def tinh_histogram(anh, bins=64, khoang=([0, 256])):
    hist = np.zeros((bins*3), dtype=np.float32)
    so_pixel = 0
    b, g, r = cv2.split(anh)
    for kenh in (b, g, r):
        kenh_hist, _ = np.histogram(kenh, bins=bins, range=khoang)
        hist[so_pixel:so_pixel + bins] = kenh_hist
        so_pixel += bins
    return hist

# === Đọc ảnh kiểm tra ===
anh_kiem_tra = cv2.imread('D:\\a\\a1.png')
anh_kiem_tra = cv2.resize(anh_kiem_tra, (300, 300))

# === Danh sách ảnh dữ liệu và histogram ===
anh_du_lieu = []
hist_du_lieu = []

# Đọc các ảnh từ thư mục 'c'
for i in range(1, 4):
    du_lieu = cv2.imread(f'D:\\c\\c{i}.png')
    du_lieu = cv2.resize(du_lieu, (300, 300))
    hist = tinh_histogram(du_lieu).flatten()
    anh_du_lieu.append(du_lieu)
    hist_du_lieu.append(hist)

# === Tính histogram cho ảnh kiểm tra ===
hist_kiem_tra = tinh_histogram(anh_kiem_tra).flatten()

# === Áp dụng KNN tìm ảnh gần giống ===
K = 3
tim_lan_can = NearestNeighbors(n_neighbors=K)
tim_lan_can.fit(hist_du_lieu)
_, chi_so = tim_lan_can.kneighbors([hist_kiem_tra])

# === Hiển thị ảnh kiểm tra và ảnh gần giống ===
cv2.imshow("Ảnh kiểm tra", anh_kiem_tra)
for i in range(K):
    cv2.imshow(f"Ảnh gần giống {i+1}", anh_du_lieu[chi_so[0][i]])

# === PHÂN VÙNG ẢNH KIỂM TRA BẰNG KMEANS ===
# Chuyển ảnh sang không gian RGB
anh_rgb = cv2.cvtColor(anh_kiem_tra, cv2.COLOR_BGR2RGB)
pixels = anh_rgb.reshape((-1, 3))

# Số cụm (có thể chỉnh)
so_cum = 5
kmeans = KMeans(n_clusters=so_cum, random_state=0)
nhan = kmeans.fit_predict(pixels)

# Tạo ảnh mới từ trung tâm cụm
anh_phan_cum = kmeans.cluster_centers_[nhan].astype(np.uint8)
anh_phan_cum = anh_phan_cum.reshape(anh_rgb.shape)

# Hiển thị bằng matplotlib
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.title("Ảnh gốc (RGB)")
plt.imshow(anh_rgb)
plt.axis('off')

plt.subplot(1, 2, 2)
plt.title(f"Phân vùng KMeans (K={so_cum})")
plt.imshow(anh_phan_cum)
plt.axis('off')
plt.tight_layout()
plt.show()

cv2.waitKey(0)
cv2.destroyAllWindows()
