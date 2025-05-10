def Hang():
    n = int(input("nhap so luong hang: "))
    tu_dien = dict()
    for i in range(n):
        MaHang = input("nhap ma hang: ")
        SoLuong = int(input("nhap so luong: "))
        tu_dien[MaHang] = SoLuong
    return tu_dien

def NCC():
    m = int(input("nhap so luong nha cung cap:"))
    tu_dien = dict()
    for i in range(m):
        MaNCC = input("nhap ma nha cung cap: ")
        TenNCC = input("nhap ten nha cugn cap: ")
        tu_dien[MaNCC] = TenNCC
    return tu_dien

def in_thong_tin(tu_dien):
    for i in tu_dien:
        print("|{}|{}|".format(i,tu_dien[i]))

def kiem_tra(tu_dien):
    if "H001" in tu_dien.keys():
        tu_dien["H001"] = 200
    else:
        tu_dien["H001"] = int(input("nhap so luong cua H001:"))

def xoa(tu_dien):
    a=[]
    for hang,solg in tu_dien.items():
        if solg == 0:
            a.append(hang)
    for hang in a:
        del tu_dien[hang]
    return tu_dien

def chuyen_tu_dien(tu_dien):
    lish1 = list(tu_dien.keys())
    lish2 = list(tu_dien.values())
    print("3 phan tu dau tien cua list 1 la: ",lish1[:3])
    print("3 phan tu cuoi cua list 2 la: ",lish2[-3:])
    return lish1,lish2
tudien_hang = Hang()
tudien_ncc = NCC()
print("\tTHONG TIN HANG ")
in_thong_tin(tudien_hang)
print("\tTHONG TIN NHA CUNG CAP")
in_thong_tin(tudien_ncc)
print("\tKIEM TRA TU DIEN")
kiem_tra(tudien_hang)
in_thong_tin(tudien_hang)

print("\tXOA CAC MAT HANG CO SO LUONG BANG 0")
xoa(tudien_hang)
in_thong_tin(tudien_hang)

print("\tCHUYEN DU LIEU THANH CONG")
chuyen_tu_dien(tudien_hang)
