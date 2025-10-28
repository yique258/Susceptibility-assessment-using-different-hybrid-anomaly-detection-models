import numpy as np
import matplotlib.pyplot as plt
from scipy import linalg
import warnings
plt.rcParams['font.sans-serif'] = ['SimHei']


def lowess(y, x,num,frac=0.666, it=3, alpha=1e-6):
    """
    Locally Weighted Scatterplot Smoothing (LOWESS)
    
    参数:
    y : array-like
        因变量 (y值)
    x : array-like
        自变量 (x值)
    num:array-like
        x[i]size
    frac : float, 可选 (默认: 2/3)
        局部窗口大小比例 (0-1)
    it : int, 可选 (默认: 3)
        鲁棒迭代次数

    
    返回:
    array or tuple
        如果return_conf_int=False: 返回平滑后的y值
    """
    # 数据预处理
    y = np.asarray(y).flatten()
    x = np.asarray(x).flatten()
    num=np.asarray(num).flatten()
    
    if len(x) != len(y):
        raise ValueError("x 和 y 必须具有相同的长度")
    
    n = len(x)
    k = int(np.ceil(frac * n))
    
    if k < 2:
        k = 2
        warnings.warn("frac参数设置过小，已自动调整为最小窗口大小")
    
    # 三立方核函数
    def tricube(u):
        return np.where(np.abs(u) < 1, (1 - np.abs(u)**3)**3, 0)
    
    # 双权重函数（用于鲁棒迭代）
    def bisquare(u):
        return np.where(np.abs(u) < 1, (1 - u**2)**2, 0)
    
    weights = np.ones(n)
    y_smoothed = np.zeros(n)
    total= np.sum(num)

    for iter_count in range(it + 1):
        for i in range(n):

            dist = np.abs(x - x[i])
            max_dist = np.partition(dist, k)[k]
            windowtotal_num=np.sum(num[i:i+k])
            window_num=np.zeros(n)
            window_num[i:i+k]=num[i:i+k]

            # 计算权重 如果window的数据量极小，直接让该点的y——smoothed为0的
            # if windowtotal_num<(frac*total*0.2):
            #      y_smoothed[i]=0
            #      continue
            
            u = (dist / max_dist+1-(window_num/windowtotal_num))*0.5
            kernel_weights = tricube(u)
            final_weights = weights * kernel_weights
            


            X = np.column_stack([np.ones_like(x), x])
            W = np.diag(final_weights)
            
            # 解加权最小二乘问题
            try:
                XWX = X.T @ W @ X
                XWY = X.T @ W @ y
                XWX += alpha * np.eye(XWX.shape[0])  # 添加正则化项
                beta = linalg.solve(XWX, XWY, assume_a='sym')
                y_smoothed[i] = beta[0] + beta[1] * x[i]
            except linalg.LinAlgError:
                # 如果矩阵奇异，使用加权平均值
                y_smoothed[i] = np.average(y, weights=final_weights)

            y_smoothed[i]=max(-1,y_smoothed[i])
            y_smoothed[i]=min(1,y_smoothed[i])
        # 第一次迭代后，计算残差并更新鲁棒权重
        if iter_count < it:
            residuals = y - y_smoothed
            mad = np.median(np.abs(residuals - np.median(residuals)))
            if mad < 1e-10:
                mad = 1e-10
            s = residuals / (6 * mad)
            weights = bisquare(s)

    return y_smoothed

