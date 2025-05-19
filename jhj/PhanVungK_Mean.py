"""
import matplotlib.pyplot as plt
import numpy as np
import cv2
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist

# Đọc ảnh
anh = cv2.imread('D:\\6.png')
anh = cv2.cvtColor(anh, cv2.COLOR_BGR2RGB)
anh = np.array(anh)

# Chuyển ảnh sang dạng ma trận 2D cho KMeans
X = anh.reshape((anh.shape[0] * anh.shape[1], anh.shape[2]))

# Hàm KMeans viết từ đầu
def kmeans_tu_dong(X, so_cum: int):
    def khoi_tao_tam(X, so_cum):
        return X[np.random.choice(X.shape[0], so_cum, replace=False)]

    def gan_nhan(X, tam):
        khoang_cach = cdist(X, tam)
        return np.argmin(khoang_cach, axis=1)

    def cap_nhat_tam(X, nhan, so_cum):
        tam_moi = np.zeros((so_cum, X.shape[1]))
        for k in range(so_cum):
            diem = X[nhan == k]
            tam_moi[k] = np.mean(diem, axis=0)
        return tam_moi

    def kiem_tra_hoi_tu(tam_cu, tam_moi):
        return set([tuple(a) for a in tam_cu]) == set([tuple(a) for a in tam_moi])

    tam = [khoi_tao_tam(X, so_cum)]
    nhan = []
    lan_lap = 0

    while True:
        nhan.append(gan_nhan(X, tam[-1]))
        tam_moi = cap_nhat_tam(X, nhan[-1], so_cum)
        if kiem_tra_hoi_tu(tam[-1], tam_moi):
            break
        tam.append(tam_moi)
        lan_lap += 1

    return tam[-1], nhan[-1], lan_lap

# Hàm thực hiện phân cụm và hiển thị
def chay_kmeans(X, danh_sach_so_cum: list):
    fig, ax = plt.subplots(3, len(danh_sach_so_cum), figsize=(8*len(danh_sach_so_cum), 20))

    for i, so_cum in enumerate(danh_sach_so_cum):
        anh_trung_binh = np.zeros_like(X)
        anh_cluster = np.zeros_like(X)
        anh_trung_binh_tu_dong = np.zeros_like(X)

        # KMeans dùng sklearn
        kmeans = KMeans(n_clusters=so_cum).fit(X)
        nhan = kmeans.predict(X)

        # KMeans tự cài
        tam_tu_dong, nhan_tu_dong, _ = kmeans_tu_dong(X, so_cum)

        for k in range(so_cum):
            anh_cluster[nhan == k] = kmeans.cluster_centers_[k]
            anh_trung_binh[nhan == k] = X[nhan == k].mean(axis=0)
            anh_trung_binh_tu_dong[nhan_tu_dong == k] = X[nhan_tu_dong == k].mean(axis=0)

        # Hiển thị ảnh
        reshape = lambda X: X.reshape(anh.shape[0], anh.shape[1], anh.shape[2])
        img1 = reshape(anh_cluster)
        img2 = reshape(anh_trung_binh)
        img3 = reshape(anh_trung_binh_tu_dong)

        ax[0][i].imshow(img1, interpolation='nearest')
        ax[0][i].set_title(f'Ảnh phân cụm theo màu, K={so_cum}')
        ax[0][i].axis('off')

        ax[1][i].imshow(img2, interpolation='nearest')
        ax[1][i].set_title(f'Ảnh trung bình màu, K={so_cum}')
        ax[1][i].axis('off')

        ax[2][i].imshow(img3, interpolation='nearest')
        ax[2][i].set_title(f'Ảnh trung bình màu (thuật toán tự viết), K={so_cum}')
        ax[2][i].axis('off')

    plt.show()

# Gọi hàm thực thi với các giá trị K
chay_kmeans(X, [2, 5, 7])
"""

import numpy as np
import cv2
from sklearn.cluster import KMeans

# Đọc ảnh
anh = cv2.imread('D:\\6.png')
cv2.imshow("anh goc", anh)

# Đưa ảnh về dạng ma trận 2 chiều để KMeans xử lý
X = anh.reshape((anh.shape[0] * anh.shape[1], anh.shape[2]))

# Số cụm
K = 11

# Huấn luyện mô hình KMeans
kmeans = KMeans(n_clusters=K).fit(X)
nhan = kmeans.predict(X)

# Tạo ảnh mới với màu trung tâm của cụm
anh_moi = np.zeros_like(X)
for i in range(K):
    anh_moi[nhan == i] = kmeans.cluster_centers_[i]

# Chuyển lại về dạng ảnh gốc
anh_moi = anh_moi.reshape((anh.shape[0], anh.shape[1], anh.shape[2]))

# Hiển thị ảnh sau khi phân cụm
cv2.imshow("anh sau khi phan cum", anh_moi)
cv2.waitKey(0)
cv2.destroyAllWindows()
