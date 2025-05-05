import numpy as np

def thomas_algorithm(a, b, c, f):
    """
    Thomas算法求解三对角线性方程组
    参数:
        a: 下对角线元素 (n-1,) 对应a[1]到a[n-1] (Python从0开始索引)
        b: 主对角线元素 (n,)
        c: 上对角线元素 (n-1,) 对应c[0]到c[n-2]
        f: 右端项 (n,)
    返回:
        x: 解向量 (n,)
    """
    n = len(b)
    
    # 检查输入维度
    if len(a) != n-1 or len(c) != n-1 or len(f) != n:
        raise ValueError("输入维度不匹配。要求: len(a)=n-1, len(c)=n-1, len(b)=len(f)=n")
    
    # 初始化中间变量
    c_prime = np.zeros(n-1)
    d_prime = np.zeros(n)
    
    # 前向消元
    c_prime[0] = c[0] / b[0]
    d_prime[0] = f[0] / b[0]
    
    for i in range(1, n-1):
        denominator = b[i] - a[i-1] * c_prime[i-1]
        c_prime[i] = c[i] / denominator
        d_prime[i] = (f[i] - a[i-1] * d_prime[i-1]) / denominator
    
    # 处理最后一行
    denominator = b[-1] - a[-1] * c_prime[-1]
    d_prime[-1] = (f[-1] - a[-1] * d_prime[-2]) / denominator
    
    # 回代
    x = np.zeros(n)
    x[-1] = d_prime[-1]
    for i in range(n-2, -1, -1):
        x[i] = d_prime[i] - c_prime[i] * x[i+1]
    
    return x

# 测试数据
n = 5
b = np.array([4, 4, 4, 4, 4], dtype=float)      # 主对角线
a = np.array([-1, -1, -1, -1], dtype=float)     # 下对角线 (a[1]到a[4])
c = np.array([-1, -1, -1, -1], dtype=float)     # 上对角线 (c[0]到c[3])
f = np.array([100, 200, 200, 200, 100], dtype=float)  # 右端项

# 求解
x = thomas_algorithm(a, b, c, f)

# 打印结果
print("解向量 x:")
print(x)

# 验证
A = np.diag(b) + np.diag(a, -1) + np.diag(c, 1)
print("\n验证残差 (A*x - f):")
print(np.dot(A, x) - f)

