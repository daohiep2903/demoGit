import numpy as np
import cv2
# tich chap, loc trung vi, loc trung binh
def tich_chap_2d(anh, mat_na):
    """
    Ham thuc hien phep tich chap 2D giua anh va mat na.
    :param anh: Anh dau vao dang numpy array (grayscale)
    :param mat_na: Mat na tich chap (numpy array)
    :return: Anh da loc
    """
    cao, rong = anh.shape
    kcao, krong = mat_na.shape

    dem = kcao // 2
    anh_pad = np.pad(anh, ((dem, dem), (dem, dem)), mode='constant', constant_values=0)

    ket_qua = np.zeros((cao, rong), dtype=np.float32)

    for y in range(cao):
        for x in range(rong):
            vung_con = anh_pad[y:y + kcao, x:x + krong]
            ket_qua[y, x] = np.sum(vung_con * mat_na)

    ket_qua = np.clip(ket_qua, 0, 255)
    return ket_qua.astype(np.uint8)


def loc_trung_vi(anh, kich_thuoc):
    """
    Ham thuc hien loc trung vi.
    :param anh: Anh grayscale
    :param kich_thuoc: Kich thuoc kernel (so le)
    :return: Anh da loc
    """
    cao, rong = anh.shape
    dem = kich_thuoc // 2
    anh_pad = np.pad(anh, ((dem, dem), (dem, dem)), mode='constant', constant_values=0)
    ket_qua = np.zeros((cao, rong), dtype=np.uint8)

    for y in range(cao):
        for x in range(rong):
            vung_con = anh_pad[y:y + kich_thuoc, x:x + kich_thuoc]
            ket_qua[y, x] = np.median(vung_con)

    return ket_qua


def loc_trung_binh(anh, kich_thuoc):
    """
    Ham thuc hien loc trung binh.
    :param anh: Anh grayscale
    :param kich_thuoc: Kich thuoc kernel (so le)
    :return: Anh da loc
    """
    mat_na = np.ones((kich_thuoc, kich_thuoc), dtype=np.float32) / (kich_thuoc * kich_thuoc)
    return tich_chap_2d(anh, mat_na)


def ap_dung_bo_loc(duong_dan_anh, mat_na, loai_loc="trung_vi"):
    anh = cv2.imread(duong_dan_anh, cv2.IMREAD_GRAYSCALE)
    if anh is None:
        raise ValueError("Khong the doc anh. Kiem tra duong dan.")

    if loai_loc == "tich_chap":
        anh_loc = tich_chap_2d(anh, mat_na)
    elif loai_loc == "trung_vi":
        anh_loc = loc_trung_vi(anh, mat_na.shape[0])
    elif loai_loc == "trung_binh":
        anh_loc = loc_trung_binh(anh, mat_na.shape[0])
    else:
        raise ValueError("Loai bo loc khong hop le.")

    cv2.namedWindow("Anh goc", cv2.WINDOW_NORMAL)
    cv2.namedWindow("Anh da loc", cv2.WINDOW_NORMAL)

    cv2.imshow("Anh goc", anh)
    cv2.imshow("Anh da loc", anh_loc)
    cv2.waitKey(0)
    cv2.destroyAllWindows()



mat_na_trung_binh = np.ones((3, 3), dtype=np.float32) / 9
ap_dung_bo_loc("D:\\1.png", mat_na_trung_binh, loai_loc="tich_chap")
