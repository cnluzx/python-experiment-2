import numpy as np
from fractions import Fraction

A = np.array([[2, 1, 2], [1, 2, 3], [4, 1, 2]])


def lu_decomposition(A):
    n = A.shape[0]
    L = np.eye(n)
    U = np.zeros((n, n))

    for k in range(n):
        # 计算 U 的第 k 行
        for j in range(k, n):
            U[k, j] = A[k, j] - np.dot(L[k, :k], U[:k, j])
        # 计算 L 的第 k 列
        for i in range(k + 1, n):
            L[i, k] = (A[i, k] - np.dot(L[i, :k], U[:k, k])) / U[k, k]

    return L, U


L, U = lu_decomposition(A)
print("下三角矩阵 L:")
L_fractions = [[Fraction(x).limit_denominator() for x in row] for row in L]
for row in L_fractions:
    print([str(x) for x in row])

print("上三角矩阵 U:")
U_fractions = [[Fraction(x).limit_denominator() for x in row] for row in U]
for row in U_fractions:
    print([str(x) for x in row])

print("验证 A = LU:")
print(np.allclose(A, np.dot(L, U)))

# 计算矩阵 A 的逆矩阵
A_inv = np.linalg.inv(A)
print("矩阵 A 的逆矩阵:")
A_inv_fractions = [[Fraction(x).limit_denominator() for x in row] for row in A_inv]
for row in A_inv_fractions:
    print([str(x) for x in row])

# 计算矩阵 A 的行列式值
A_det = np.linalg.det(A)
A_det_fraction = Fraction(A_det).limit_denominator()
print("矩阵 A 的行列式值:")
print(str(A_det_fraction))