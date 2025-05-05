import numpy as np


def hilbert_matrix_cond_num():
    # 遍历前10阶Hilbert矩阵
    for n in range(1, 11):
        # 生成n阶Hilbert矩阵
        hilbert_matrix = np.array([[1 / (i + j + 1) for j in range(n)] for i in range(n)])
        # 计算2 - 条件数
        cond_num = np.linalg.cond(hilbert_matrix, 2)
        print(f"{n}阶Hilbert矩阵的2 - 条件数: {cond_num}")

def Jacobi_Gauss_Seidel_iterative_method():
    jacobi_spectral_radii = []
    gs_spectral_radii = []

# 遍历前 10 阶矩阵
    for n in range(1, 11):
        # 生成 Hilbert 矩阵
        A = np.array([[1 / (i + j + 1) for j in range(n)] for i in range(n)])

        # 分解矩阵 A 为 D, L, U
        D = np.diag(np.diag(A))
        L = -np.tril(A, k=-1)
        U = -np.triu(A, k=1)

        # 计算 Jacobi 迭代矩阵
        J = np.linalg.inv(D) @ (L + U)
        # 计算 Jacobi 迭代矩阵的谱半径
        jacobi_eigenvalues = np.linalg.eigvals(J)
        jacobi_spectral_radius = np.max(np.abs(jacobi_eigenvalues))
        jacobi_spectral_radii.append(jacobi_spectral_radius)

        # 计算 Gauss - Seidel 迭代矩阵
        G = np.linalg.inv(D - L) @ U
        # 计算 Gauss - Seidel 迭代矩阵的谱半径
        gs_eigenvalues = np.linalg.eigvals(G)
        gs_spectral_radius = np.max(np.abs(gs_eigenvalues))
        gs_spectral_radii.append(gs_spectral_radius)

        # 打印表头
        print("阶数n\tJacobi迭代矩阵谱半径\tGauss - Seidel迭代矩阵谱半径")
        # 打印每一行数据

        print(f"{n}\t{jacobi_spectral_radii[n - 1]:.6f}\t\t{gs_spectral_radii[n - 1]:.6f}")

####################################################################################
# 以下为 Gauss - Seidel 迭代法的实现

def gauss_seidel_method(A, b, max_iter=1000, tol=1e-6):
    n = len(A)
    x = np.zeros(n, dtype=np.float64)
    L = np.tril(A)
    U = A - L

    for _ in range(max_iter):
        try:
            x_new = np.linalg.solve(L, b - np.dot(U, x))
        except np.linalg.LinAlgError:
            return np.full(n, np.nan)
        
        if np.linalg.norm(x_new - x, ord=np.inf) < tol:
            return x_new
        x = x_new
    return x

# 真实解
true_solution = np.ones(10, dtype=np.float64)

print("表 3 前10阶直接法与迭代法求解结果对比")
print("阶数n\t变量\t直接法\t迭代法\t实际解")

for n in range(2, 11):
    # 生成 Hilbert 矩阵
    A = 1 / (np.arange(1, n + 1).reshape(-1, 1) + np.arange(n))
    # 计算右侧向量
    b = np.dot(A, true_solution[:n])

    # 直接法求解
    try:
        direct = np.linalg.solve(A, b)
    except np.linalg.LinAlgError:
        direct = np.full(n, np.nan)

    # 高斯-赛德尔迭代法求解
    gs = gauss_seidel_method(A, b)

    # 格式化输出
    for i in range(n):
        var_name = f"x{i+1}"
        direct_val = f"{direct[i]:.4f}" if not np.isnan(direct[i]) else "nan"
        gs_val = f"{gs[i]:.4f}" if not np.isnan(gs[i]) else "nan"
        
        # 特殊处理n=10的第6个变量
        if n == 10 and i == 5:
            direct_val = f"{direct[i]:.4f}" if not np.isnan(direct[i]) else "nan"
        
        print(f"{n}\t{var_name}\t{direct_val}\t{gs_val}\t1.0")
    
    # 添加空行分隔不同阶数
    print("\n", end="")



Jacobi_Gauss_Seidel_iterative_method()
hilbert_matrix_cond_num()