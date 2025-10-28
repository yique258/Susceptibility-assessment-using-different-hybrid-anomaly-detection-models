import csv
import os
from utils import geodata,metrics
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats

factorsfolder = r"F:\resampled data from yousef"
debris_area = r"F:\data from xiaobo\debris flow.shp"
# low_area=r'F:\Susceptibility Assessment\low_area\low_area.shp'
nondata=-10000
# 因子编号与文件名及中文含义映射
factor_map = {      
    0:  ("Aspect", "坡向"),
    1:  ("Curvature", "曲率"),
    2:  ("DTG", "distance to glacier"),
    3:  ("DTS", "distance to stream"),
    4:  ("DTW", "distance to water bodies"),
    5:  ("Elevation", "高程"),
    6:  ("FD", "fault density"),
    7:  ("Geology", "地质"),
    8:  ("LULC", "land use & land cover"),
    9:  ("NDVI", "归一化植被指数"),
    10: ("prec", "precipitation 降水"),
    11: ("Slope", "坡度"),
    12: ("SR", "solar radiation"),
    13: ("Tavg_annual", "1-6月平均温度"),
    14: ("Tavg_diff", "温度差异"),
    15: ("TRI", "?"),
    16: ("TWI", "topographic wetness index")
}


color_map = {
    "Low": (81/255, 137/255, 61/255),       # #51893D
    "Moderate": (186/255, 191/255, 44/255), # #BABF2C
    "High": (242/255, 191/255, 39/255),     # #F2BF27
    "Very High": (242/255, 65/255, 65/255)  # #F24141
}
def pre_dataprocess():
    data, profile = geodata.read_tifs(factorsfolder)
    print(f'data shape: {data.shape}')
    # 17 1 4909 8050
    # -1 是flatten
    # ndvi/10000  ->[-1,1]
    data[9][0][data[9][0]!=nondata]/=1e4
    #DTS数值过小
    data[3][0][data[3][0]!=nondata]*=1e5
    # FD的数值过于紧密，需要处理,无用
    # data[6][0][data[6][0]!=nondata]*=1e4

    np.save('factors.npy', data) 
    # np.save('profile.npy', profile)
    print('data shape:', data.shape)
    print(profile)
    geodata.shp2tif(debris_area,'./debris_area.tif',factorsfolder+'\Aspect.tif')
    data,profile=geodata.read_tifs('./debris_area')
    np.save('debris_area.npy',data)

    # 转为单分类的相似性问题
    # geodata.shp2tif(low_area,'./low_area.tif',factorsfolder+'\Aspect.tif')
    # data,profile=geodata.read_tifs('./low_area')
    # np.save('low_area.npy',data)

def factor_stastics(factors):
    csv_file = './sts_data.csv'
    print('因子统计信息：')
    for i in range(factors.shape[0]):
        arr = factors[i].reshape(-1)
        arr = arr[~np.isnan(arr)] 
        arr = arr[arr != nondata]
        mean = np.mean(arr)
        std = np.std(arr)
        minv = np.min(arr)
        maxv = np.max(arr)
        name, cname = factor_map.get(i, (f'Factor{i}', f'因子{i}'))
        sts=[]
        sts.append([i,name,cname,mean,std,minv,maxv])
        # # 正态分布拟合
        # mu, sigma = stat.norm.fit(arr)
        # residuals = arr - mu
        print(f'{name}的有效数据量：{len(arr)}')
        print(f'{i}: {name} ({cname}) -> 均值: {mean:.4f}, 标准差: {std:.4f}, 最小值: {minv:.4f}, 最大值: {maxv:.4f}')
        file_exists = os.path.isfile(csv_file)

        with open(csv_file, 'a', newline='', encoding='utf-8-sig') as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(['序号', '名称', '中文名称', '均值', '标准差', '最小值', '最大值'])
            for i, name, cname, mean, std, minv, maxv in sts:
                writer.writerow([i, name, cname, f"{mean:.2f}", f"{std:.2f}", f"{minv:.2f}", f"{maxv:.2f}"])

        print(f"\n数据已成功导出到 {csv_file}")
        # print(f'   拟合正态分布参数: mu={mu:.4f}, sigma={sigma:.4f}')
        # print(f'   残差均值: {np.mean(residuals):.4f}, 残差标准差: {np.std(residuals):.4f}')
        # stat, p_value = stats.normaltest(arr)   
        # skew = stats.skew(arr)
        # kurt = stats.kurtosis(arr)
        # if p_value >= 0.05:
        #     print("数据无明显偏离正态分布")
        # elif abs(skew) < 0.5 and abs(kurt) < 1.0:
        #     print("虽有统计显著偏离，但实际偏离程度小，可近似为正态")
        # else:
        #     print("数据显著偏离正态分布")
        # plot_pp(arr,name)
        # plot_qq(arr,name)

        # for n in range(3,6):
        #     create_quantile_groups(arr,name,n)


def create_quantile_groups(data,name, n_groups=4):
    percentiles = np.linspace(0, 100, n_groups+1)
    boundaries = np.percentile(data, percentiles)
    groups = []
    for i in range(n_groups):
        group = data[(data >= boundaries[i]) & (data <= boundaries[i+1])]
        groups.append(group)
    print(f'{name}{n_groups}组:{boundaries}')
    y=[len(g) for g in groups]
    x = [f'{boundaries[i]:.2f}~{boundaries[i+1]:.2f}' for i in range(n_groups)]
    plt.bar(x, y, color='skyblue', edgecolor='black')
    plt.xlabel('区间范围')
    plt.ylabel('样本数量')
    plt.title(f'{name}分位分组直方图（{n_groups}组）')
    plt.tight_layout()
    plt.savefig(f'./quantile/{name}{n_groups}组直方图.png')
    plt.close() 
    
def plot_qq(data,name):
    # 计算理论分位数和样本分位数
    quantiles = stats.probplot(data, dist="norm")
    theoretical = quantiles[0][0]
    sample = quantiles[0][1]
    
    plt.figure(figsize=(10, 6))
    plt.scatter(theoretical, sample, alpha=0.5)
    
    # 添加参考线
    min_val = min(theoretical.min(), sample.min())
    max_val = max(theoretical.max(), sample.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--')
    
    plt.xlabel('Theoretical Quantiles')
    plt.ylabel('Sample Quantiles')
    plt.title(f'{name} QQ Plot')
    plt.grid(True)
    # plt.show()
    plt.savefig(f'./qq_plot/{name} QQ Plot')

def plot_pp(data,name):
    # 计算经验累积分布和理论累积分布
    n = len(data)
    p = np.arange(1, n+1) / n - 0.5/n  # 避免0和1
    theo_quantiles = stats.norm.ppf(p)
    
    sorted_data = np.sort(data)
    data_mean, data_std = np.mean(sorted_data), np.std(sorted_data)
    sample_quantiles = (sorted_data - data_mean) / data_std
    
    plt.figure(figsize=(10, 6))
    plt.scatter(p, stats.norm.cdf(sample_quantiles), alpha=0.5)
    plt.plot([0, 1], [0, 1], 'r--')  # 对角线参考线
    
    plt.xlabel('Theoretical Cumulative Probability')
    plt.ylabel('Sample Cumulative Probability')
    plt.title(f'{name} PP Plot')
    plt.grid(True)
    # plt.show()
    plt.savefig(f'./pp_plot/{name} PP Plot')

if __name__ == '__main__':
    pre_dataprocess()
    factors=np.load('factors.npy')
    debris_area=np.load('debris_area.npy')
    factor_stastics(factors)

    
#     categories = ['Low', 'Moderate', 'High', 'Very High']
#     values = [15, 25, 35, 50]

# # 使用归一化颜色
#     colors = [color_map[cat] for cat in categories]
# # 绘制柱状图
#     plt.figure(figsize=(10, 6))
#     bars = plt.bar(categories, values, color=colors)
#     for bar in bars:
#         height = bar.get_height()
#         plt.text(bar.get_x() + bar.get_width()/2., height,
#             f'{height}%', ha='center', va='bottom')

#     plt.title('Landslide Susceptibility Distribution')
#     plt.ylabel('Percentage (%)')
#     plt.grid(axis='y', linestyle='--', alpha=0.7)
#     plt.show()
    