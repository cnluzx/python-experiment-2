import numpy as np

def gauss_elimination_with_partial_pivoting(A, b):
    """
    使用列主元的高斯消元法
    :param A: 系数矩阵
    :param b: 右侧向量
    :return: 解向量x
    """
    n = len(b)
    Ab = np.c_[A, b].astype(float)  # 增广矩阵
    
    # 前向消元
    for k in range(n-1):
        # 列主元选择：在当前列中找出绝对值最大的元素
        max_row = np.argmax(np.abs(Ab[k:, k])) + k
        
        # 交换当前行与主元行
        if max_row != k:
            Ab[[k, max_row]] = Ab[[max_row, k]]
        
        # 消元
        for i in range(k+1, n):
            factor = Ab[i, k] / Ab[k, k]
            Ab[i, k:] -= factor * Ab[k, k:]
    
    # 回代
    x = np.zeros(n)
    for i in range(n-1, -1, -1):
        if Ab[i, i] == 0:
            raise ValueError("矩阵奇异，无法求解")
        x[i] = (Ab[i, -1] - np.dot(Ab[i, i+1:n], x[i+1:n])) / Ab[i, i]
    
    return x

# 测试数据
A = np.array([[6, 3, 2],
              [10, 5, 6],
              [8, 5, 3]])
b = np.array([1/3, 0, 0])

try:
    x = gauss_elimination_with_partial_pivoting(A, b)
    print("解:", x)

except ValueError as e:
    print("错误:", e)