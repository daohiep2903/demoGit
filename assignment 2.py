from math import sqrt
import numpy as np

# Đọc toàn bộ dữ liệu của 1 tệp lên mảng, trả về mảng đọc được
def read(filename):
    result = []
    with open(filename, mode='r') as f:
        while True:
            line = f.readline().split()
            if len(line) == 0 : break
            result.append(line)
    return np.array(result)

# Tính khoảng cách Euclidean giữa hai dòng bất kỳ của x
def distance(row1, row2):
    dis = 0
    for i in range(len(row1)):
        dis += (row1[i] - row2[i])**2
    return sqrt(dis)

# Tính khoảng cách Euclidean từ 1 dòng test tới tất cả các dòng của x
def distancetoall(test_row, x):
    dis = []
    for row in x:
        dis.append(distance(test_row, row))
    return dis

def getRow(data):
    num = len(data)
    row = int(input('Nhập dòng: '))
    while (row <= 0 or row >= num):
        print('Dòng không hợp lệ, nhập lại')
        row = int(input('Nhập dòng: '))
    return data[row]

def predict(test, data):
    label = np.array([])
    distance_array = distancetoall(test, data)
    dis = distance_array[0]
    for i in range(len(distance_array)):
        if dis > distance_array[i]:
            label = data[i]
    return label


# Đọc toàn bộ tệp kddcup.data lên mảng x_train
x_train = read('D:/kddcup.data')
# Cắt bỏ 4 cột đầu tiên và cột cuối cùng của x_train
x_train = x_train[::, 4:len(x_train[0])-1:]
# Cắt cột cuối của x_train thành mảng label
label = x_train[::, len(x_train[0])-1]
# Chuyển x_train về kiểu float
x_train = x_train.astype(float)

# Đọc toàn bộ tệp kddcup.test lên mảng x_test
x_test = read("D:/kddcup.test")
# Cắt 4 cột đầu và cột cuối của x_test rồi chuyển về kiểu thực
x_test = x_test[::,4:len(x_test)-1:]
x_test = x_test.astype(float)

# Lấy 1 dòng (đặt tên là test) bất kỳ từ mảng x_test hãy dự đoán label của nó bằng cách:
# tính khoảng cách từ nó tới tất cả các dòng của x_train
# Label của test là label của dòng gần với nó nhất
# (sinh viên có thể tổ chức thành hàm predict
# với đầu vào là mảng các khoảng cách - dis và mảng label)
test = getRow(x_test)
label = predict(test, x_train)
print('Dòng chọn từ x_test:')
print(test)
print('Label của nó được dự đoán là:')
print(label)



