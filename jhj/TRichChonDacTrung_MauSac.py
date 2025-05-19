# Đọc ảnh kiểm tra
import cv2
import numpy as np
from sklearn.neighbors import NearestNeighbors

# Hàm tính histogram của ảnh
def tinh_histogram(anh, bins=64, khoang=([0, 256])):
    hist = np.zeros((bins**3), dtype=np.float32)
    so_pixel = 0
    b, g, r = cv2.split(anh)

    for kenh in (b, g, r):
        kenh_hist, _ = np.histogram(kenh, bins=bins, range=khoang)
        hist[so_pixel:so_pixel + bins] = kenh_hist
        so_pixel += bins

    return hist


anh_kiem_tra = cv2.imread('D:\\a\\a1.png')
anh_kiem_tra = cv2.resize(anh_kiem_tra, (300, 300))

# Danh sách lưu ảnh và histogram
anh_du_lieu = []
hist_du_lieu = []

# Đọc các ảnh nhóm 'a'     chon a hay c deu duoc
# for i in range(1, 4):     voi truong hop trong thu muc moi tao co 4 anh
#     du_lieu = cv2.imread('D:\\a\\a{}.png'.format(i))      tao thu muc ten la a, luu 4 anh ten a1,a2,a3,a4
#     du_lieu = cv2.resize(du_lieu, (300, 300))
#     hist = tinh_histogram(du_lieu).flatten()
#     anh_du_lieu.append(du_lieu)
#     hist_du_lieu.append(hist)

# Đọc các ảnh nhóm 'c'
for i in range(1, 4):
    du_lieu = cv2.imread('D:\\c\\c{}.png'.format(i))
    du_lieu = cv2.resize(du_lieu, (300, 300))
    hist = tinh_histogram(du_lieu).flatten()
    anh_du_lieu.append(du_lieu)
    hist_du_lieu.append(hist)

# Histogram của ảnh kiểm tra
hist_kiem_tra = tinh_histogram(anh_kiem_tra).flatten()
K = 3  # số ảnh gần giống muốn lấy ra

# Tạo mô hình và huấn luyện
tim_lan_can = NearestNeighbors(n_neighbors=K)
tim_lan_can.fit(hist_du_lieu)

# Dự đoán các ảnh gần nhất
khoang_cach, chi_so = tim_lan_can.kneighbors([hist_kiem_tra])

# Hiển thị ảnh kiểm tra
cv2.imshow("anh kiem tra", anh_kiem_tra)

# Hiển thị các ảnh gần giống nhất
for i in range(K):
    anh = anh_du_lieu[chi_so[0][i]]
    cv2.imshow(f"anh gan giong {i+1}", anh)

cv2.waitKey(0)
cv2.destroyAllWindows()
