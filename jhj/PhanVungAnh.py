#Phan vung anh bang nguong tu dong
import numpy as np
import  cv2

def Tichchap(img, matrix):
    h,w = img.shape
    expand_img = np.zeros((h+2,w+2),dtype=np.float32)
    expand_img[1:h+1,1:w+1] = img[:,:]
    new_img = img.copy()
    for i in range(h):
        for j in range(w):
            new_img[i,j] = np.sum(expand_img[i:i+3,j:j+3]*matrix)
    return new_img

def Prewitt(img):
    matrix = img.copy().astype(np.float32)
    gx = Tichchap(matrix, np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=np.float32))
    gy = Tichchap(matrix, np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float32))

    mag = np.sqrt(gx ** 2 + gy ** 2)
    mag = np.clip(mag, 0, 255).astype(np.uint8)

    # Chuyển sang ảnh nhị phân (biên trắng, nền đen)
    _, binary = cv2.threshold(mag, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return binary




img =cv2.imread("D:\\5.png")
gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
matrix = np.array(gray_img)

# Hiển thị ảnh gốc
cv2.imshow("anh ban dau", matrix)
# Phân vùng bằng biên (Prewitt)
anh_bien = Prewitt(gray_img)
cv2.imshow("Phan vung bang bien (Prewitt)", anh_bien)
# Phân vùng bằng ngưỡng tự động (Otsu Thresholding)
ret, anh_nguong = cv2.threshold(gray_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
cv2.imshow("Phan vung bang nguong tu dong (Otsu)", anh_nguong)

# Chờ và đóng các cửa sổ
cv2.waitKey(0)
cv2.destroyAllWindows()


