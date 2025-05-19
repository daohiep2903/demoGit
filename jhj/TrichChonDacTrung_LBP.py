import cv2
import numpy as np
from sklearn.neighbors import NearestNeighbors

# Hàm thay đổi kích thước ảnh
def thay_doi_kich_thuoc(anh, kich_thuoc_moi):
    return cv2.resize(anh, kich_thuoc_moi, interpolation=cv2.INTER_AREA)

# Hàm tính toán LBP cho ảnh
def tinh_LBP(anh):
    if len(anh.shape) > 2 and anh.shape[2] == 3:
        xam = cv2.cvtColor(anh, cv2.COLOR_BGR2GRAY)
    else:
        xam = anh

    lbp = np.zeros_like(xam)
    cao, rong = xam.shape

    for i in range(1, cao - 1):
        for j in range(1, rong - 1):
            trungtam = xam[i, j]
            ma = 0
            ma |= (xam[i - 1, j - 1] > trungtam) << 7
            ma |= (xam[i - 1, j    ] > trungtam) << 6
            ma |= (xam[i - 1, j + 1] > trungtam) << 5
            ma |= (xam[i,     j + 1] > trungtam) << 4
            ma |= (xam[i + 1, j + 1] > trungtam) << 3
            ma |= (xam[i + 1, j    ] > trungtam) << 2
            ma |= (xam[i + 1, j - 1] > trungtam) << 1
            ma |= (xam[i,     j - 1] > trungtam) << 0
            lbp[i, j] = ma
    return lbp

# Hàm tìm ảnh tương tự
def tim_anh_tuong_tu(anh_kiem_tra, danh_sach_anh_train, k):
    lbp_kiem_tra = tinh_LBP(anh_kiem_tra)
    lbps_train = [tinh_LBP(anh) for anh in danh_sach_anh_train]

    lbps_train_array = np.array(lbps_train).reshape(len(lbps_train), -1)
    lbp_kiem_tra_array = lbp_kiem_tra.reshape(1, -1)

    model = NearestNeighbors(n_neighbors=k, metric='hamming')
    model.fit(lbps_train_array)

    khoang_cach, chi_so = model.kneighbors(lbp_kiem_tra_array)
    return chi_so[0]

# ==== Bắt đầu chương trình chính ====
# Danh sách ảnh train
duongdan_anh_train = [
    # tao thu muc moi ten anhTrain, luu cac anh lan luot la train1,2,3,4 roi gan duong link vao day
    "D:\\anhTrain\\train1.png", "D:\\anhTrain\\train2.png", "D:\\anhTrain\\train3.png", "D:\\anhTrain\\train4.png"

]

# Đọc ảnh train
anh_train = []
for duongdan in duongdan_anh_train:
    anh = cv2.imread(duongdan)
    if anh is not None:
        anh_train.append(anh)

# Đọc ảnh test
duongdan_anh_test = "D:\\anhTest\\test.png"
anh_test = cv2.imread(duongdan_anh_test)
if anh_test is None:
    print("Khong the đoc anh test.")
    exit()

# Chuẩn hóa kích thước
kich_thuoc_chuan = (200, 200)
anh_train = [thay_doi_kich_thuoc(anh, kich_thuoc_chuan) for anh in anh_train]
anh_test = thay_doi_kich_thuoc(anh_test, kich_thuoc_chuan)

# Tìm k ảnh giống nhất
k = 3
chi_so_anh_giong = tim_anh_tuong_tu(anh_test, anh_train, k)

# Hiển thị và lưu kết quả
cv2.imshow("anh Test", anh_test)
cv2.imwrite("output/anh_test.jpg", anh_test)  # lưu ảnh test

for i, idx in enumerate(chi_so_anh_giong):
    anh_giong = anh_train[idx]
    ten_cua_so = f"anh Giong {i+1}"
    cv2.imshow(ten_cua_so, anh_giong)
    cv2.imwrite(f"output/anh_giong_{i+1}.jpg", anh_giong)  # lưu ảnh giống

cv2.waitKey(0)
cv2.destroyAllWindows()
