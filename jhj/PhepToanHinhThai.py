import numpy as np
import cv2

def phep_co(gray):
    """
    Phep co (logical AND) theo chieu ngang.
    """
    ket_qua = np.zeros((gray.shape[0], gray.shape[1]-1), dtype=np.uint8)
    for i in range(gray.shape[0]):
        for j in range(gray.shape[1]-1):
            if gray[i][j] == 0 or gray[i][j+1] == 0:
                ket_qua[i][j] = 0
            else:
                ket_qua[i][j] = 255
    return ket_qua

def phep_gian(gray):
    """
    Phep gian (logical OR) theo chieu ngang.
    """
    ket_qua = np.zeros((gray.shape[0], gray.shape[1]-1), dtype=np.uint8)
    for i in range(gray.shape[0]):
        for j in range(gray.shape[1]-1):
            if gray[i][j] == 255 or gray[i][j+1] == 255:
                ket_qua[i][j] = 255
            else:
                ket_qua[i][j] = 0
    return ket_qua

# Doc anh
anh = cv2.imread("D:\\3.png", cv2.IMREAD_GRAYSCALE)

# Kiem tra neu doc anh that bai
if anh is None:
    raise ValueError("Khong the doc anh. Kiem tra lai duong dan.")

# Thuc hien phep co va gian
anh_co = phep_co(anh)
anh_gian = phep_gian(anh)

# Hien thi anh
cv2.imshow("Anh goc", anh)
cv2.imshow("Phep co", anh_co)
cv2.imshow("Phep gian", anh_gian)
cv2.waitKey(0)
cv2.destroyAllWindows()
