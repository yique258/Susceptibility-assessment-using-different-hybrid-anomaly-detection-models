import numpy as np
import pandas as pd
from Lowess import lowess
from utils import geodata, metrics
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import os
import seaborn as sns

plt.rcParams.update({
    'font.family': 'serif',        # 使用衬线字体
    'font.serif': 'Times New Roman', # 指定Times New Roman
    'figure.dpi': 300,             # 高分辨率输出
    'figure.autolayout': True,     # 自动调整布局
    'mathtext.fontset': 'stix',    # 数学公式字体(与Times兼容)
})


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
    14: ("Tavg_diff", "年温度差异"),
    15: ("TRI", "topographic roughness index"),
    16: ("TWI", "topographic wetness index")
}

color_map = {
    "Low": (81/255, 137/255, 61/255),       # #51893D
    "Moderate": (186/255, 191/255, 44/255), # #BABF2C
    "High": (242/255, 191/255, 39/255),     # #F2BF27
    "Very High": (242/255, 65/255, 65/255)  # #F24141
}


# 把数据中nondata都设置为-10000
nondata=-10000
class CertaintyFactor:
    def __init__(self, total_area, debris_area,factor_data,debris_data):
        """
        初始化CF模型
        :param total_area: 研究区总面积
        :param debris_area: 总泥石流面积
        """
        self.PPs = debris_area / total_area  # 先验概率
        self.factors_data=factor_data
        self.debris_data=debris_data

    def cal_class_area(self,idx,boundry):
        data = self.factors_data[idx][0].flatten()
        valid_mask = (data != nondata)
        group_mask = (data >= boundry[0]) & (data <= boundry[1])
        return np.sum(valid_mask & group_mask)
      

    def cal_debris_in_class(self, idx, boundry):
        """
        计算某因子某区间内的泥石流像元数量（排除无效值）
        :param idx: 因子编号
        :param boundry: 区间 [下界, 上界]
        :return: 泥石流像元数量
        """
        data = self.factors_data[idx][0].flatten()
        debris = self.debris_data[0].flatten()
        valid_mask = (data != nondata)
        group_mask = (data >= boundry[0]) & (data <= boundry[1])
        debris_mask = (debris == 1)  # 泥石流区像元值为1
        return np.sum(valid_mask & group_mask & debris_mask)
        

    def calculate_CF(self,idx,boundry):
        """
        计算单个类别的CF值
        :param class_area: 该类别的面积
        :param debris_in_class: 该类别的泥石流面积
        :return: CF值
        """
        class_area=self.cal_class_area(idx,boundry)
        if class_area==0:
            return 0
        debris_in_class=self.cal_debris_in_class(idx,boundry)
        PPa = debris_in_class / class_area  # 条件概率
       
        if PPa >= self.PPs:
            CF = (PPa - self.PPs) / (PPa * (1 - self.PPs))
        else:
            CF = (PPa - self.PPs) / (self.PPs * (1 - PPa))
        
        return np.round(CF,3)
    
    def combine_CF(self, CF1, CF2):
        """
        组合两个CF值
        :param CF1: 第一个CF值
        :param CF2: 第二个CF值
        :return: 组合后的CF值
        """
        if CF1 >= 0 and CF2 >= 0:
            return CF1 + CF2 - CF1*CF2
        elif CF1 < 0 and CF2 < 0:
            return CF1 + CF2 + CF1*CF2
        else:
            return (CF1 + CF2) / (1 - min(abs(CF1), abs(CF2)))


def plot_lowess():
        # loess_cf_distribution
    frac_values = [0.1, 0.2,0.3,0.4,0.5, 0.7]
    it_values = [0, 1, 3, 5]
    for frac in frac_values:
        for it in it_values:
            df = pd.read_csv("filter_factors_cf.csv")
            folder_name='lowess_cf_distribution_'+str(frac)+'_'+str(it)
            os.makedirs(folder_name,exist_ok=True)
            res=[]
            for idx in range(17):
                sub_df = df[df["factor_index"] == idx]
                name=factor_map[idx][0]
                boundaries=sub_df["boundary"].to_numpy()
                x = sub_df["boundary"].apply(lambda f: [round(float(f.strip('[]').split(',')[0]), 2),round(float(f.strip('[]').split(',')[1]), 2)])
                x=np.array(x)
                x_left = np.array([a[0] for a in x])
                x_mid = np.array([round((a[0] + a[1]) / 2,2) for a in x])
                x_mid.astype(str)
                x_str = np.array([f"[{a},{b}]" for a, b in x])
                y = sub_df["CF"].to_numpy()
                num=sub_df['num'].to_numpy()
                
                lowess_y=lowess(y,x_mid,num,frac,it)
                
                res.append(pd.DataFrame({
                    'idx':idx,
                    'lowess_y':lowess_y,
                    'x':x_mid,
                    'num':num}))
                fig, ax1 = plt.subplots(figsize=(12, 6))
                ax2 = ax1.twinx()

                ax1.axhspan(0.4,1, alpha=0.3, color=color_map['Very High'])
                ax1.axhspan(0,0.4, alpha=0.3, color=color_map["High"])
                ax1.axhspan(-0.4, 0, alpha=0.3, color=color_map["Moderate"])
                ax1.axhspan(-1, -0.4, alpha=0.3, color=color_map['Low'])
                for a in [-0.4, 0, 0.4]:
                    ax1.axhline(a, color='gray', linestyle='--', alpha=0.7)

    
                raw_color = '#6baed6'  # 浅钴蓝
                smooth_color = '#fb6a4a'  # 浅朱红
                if idx in [7,8]:
                    if idx == 7:
                        l1=ax1.scatter(x_left, y, label='Raw CF',color=raw_color)
                    else:
                        l1 = ax1.scatter(x_left, y, label='Raw CF', color=raw_color)
                    # l2=ax1.plot(x_left, lowess_y, '-', linewidth=2,alpha=0.9, label='LOWESS Smoothed CF',color=smooth_color)
                    l3=ax2.bar(x_left, num, alpha=0.4, color='purple', label='Data Size')
                    ax1.legend(handles=[l1, l3], loc='upper left')
                else:
                    l1=ax1.plot(x_str, y, '-', linewidth=1.8, alpha=0.85, label='Raw CF',color=raw_color)
                    l2=ax1.plot(x_str, lowess_y, '-', linewidth=2,alpha=0.9, label='LOWESS Smoothed CF',color=smooth_color)
                    l3=ax2.bar(x_str, num, alpha=0.4, color='purple', label='Data Size')
                    ax1.legend(handles=[l1[0], l2[0], l3], loc='upper left')

                ax1.set_xlabel('Data Value')
                ax1.set_ylabel('CF Value')
                ax1.set_ylim(-1, 1)
                ax2.set_ylabel('Data Size')
                # ax1.xaxis.set_major_locator(MaxNLocator(nbins="auto"))

                if idx!=12:
                    ax1.xaxis.set_major_locator(MaxNLocator(nbins="auto"))
                else:
                    ax1.xaxis.set_major_locator(MaxNLocator(nbins=6))

                plt.title(f"CF Distribution of {name}")
                plt.savefig(f"./{folder_name}/{idx}{name}_cf.png")
                plt.close()
            a = pd.concat(res, ignore_index=True)
            a.to_csv(folder_name+'/sts.csv',index=False)



    

if __name__ == "__main__":

    # total_area = 39517450    包含了空白的区域
    # 这边选取最大的有效数据量
    total_area=19901941
    debris_area=107230
    # factors_data(17,1,4909,8050) debris_data(1,1,4909,8050)  nodata：-10000

    # 计算 factors_cf.csv
    debris_data=np.load('debris_area.npy')
    factors_data=np.load('factors.npy')
    cf_model = CertaintyFactor(total_area, debris_area,factors_data,debris_data)
    results = []
    bins = [200, 100, 200, 100, 100, 100, 100, 36, 8, 75, 50, 100, 100, 100, 100, 100, 100]
    for idx in range(factors_data.shape[0]):  # 17个因子
        data = factors_data[idx][0].flatten()
        data=data[data!=nondata]
        # data=np.round(data,2)
        if idx==7:
            steps=range(1,38)
            print(steps)
        elif idx==8:
            steps=range(1,10)
        else:
            mn,mx=min(data),max(data)
            # steps = np.round(np.linspace(mn, mx, 201),4)
            steps = np.linspace(mn, mx, bins[idx]+1)    #quan_cf_factors
        for i in range(len(steps) - 1):
            if idx not in (7,8):
                boundry = [steps[i], steps[i + 1]]
            else:
                boundry=[steps[i],steps[i]]

            cf = cf_model.calculate_CF(idx, boundry)
            num=np.sum((data >= boundry[0]) & (data <= boundry[1]))
            print(f'idx:{idx},boundry:{boundry},CF:{cf},num:{num}')
            # 此处筛选数据量较大的方便作图
            # if num>1e4:
            #     results.append({
            #     "factor_index": idx,
            #     "boundary": boundry,
            #     "num":num,
            #     "CF": cf
            #     })
            results.append({
                "factor_index": idx,
                "boundary": boundry,
                "num": num,
                "CF": cf
            })
    df = pd.DataFrame(results)
    df.to_csv("quan_factors_cf.csv", index=False)    #完整且无四舍五入
    # df.to_csv("filter_factors_cf.csv", index=False)    #num>1e4
    print("已保存所有因子不同值的CF值到 filter_factors_cf.csv")
    # print("已保存所有因子不同值的CF值到 factors_cf.csv")
    exit()
    plot_lowess()

    # df = pd.read_csv("factors_cf.csv")
    # for idx in range(17):
    #     name=factor_map[idx][0]
    #     sub_df = df[df["factor_index"] == idx]
    #     # x轴为区间
    #     boundaries=sub_df["boundary"].to_numpy()
    #     x = sub_df["boundary"].apply(lambda f: [round(float(f.strip('[]').split(',')[0]), 2),round(float(f.strip('[]').split(',')[1]), 2)])


    #     x=np.array(x)
    #     x_left = np.array([a[0] for a in x])
    #     x_right = np.array([a[1] for a in x])
    #     x_mid=(x_left+x_right)/2
    #     y = sub_df["CF"].to_numpy()       # y轴为CF值
    #     num=sub_df['num'].to_numpy()

    #     # 常规的cf_distribution
    #     fig, ax1 = plt.subplots(figsize=(12, 6))
    #     ax2 = ax1.twinx()
    #             # 获取数据边界范围用于背景区域
    #     ax1.axhspan(0.4,1, alpha=0.3, color=color_map['Very High'], label='very high')
    #     ax1.axhspan(0,0.4, alpha=0.3, color=color_map["High"], label='High')
    #     ax1.axhspan(-0.4, 0, alpha=0.3, color=color_map["Moderate"], label='Moderate')
    #     ax1.axhspan(-1, -0.4, alpha=0.3, color=color_map['Low'], label='Low')
    #     for a in [-0.4, 0, 0.4]:
    #         ax1.axhline(a, color='gray', linestyle='--', alpha=0.7)


    #     ax1.plot(boundaries, y, 'b-', linewidth=2, label='CF Values')
    #     ax1.set_xlabel('Boundary Range')
    #     ax1.set_ylabel('CF Values', color='b')
    #     ax1.set_ylim(-1, 1)
    #     ax2.bar(boundaries, num, alpha=0.4, color='purple', label='Data Size')
    #     ax2.set_ylabel('Data Size', color='purple')
    #     ax1.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=10))

    #     lines1, labels1 = ax1.get_legend_handles_labels()
    #     lines2, labels2 = ax2.get_legend_handles_labels()
    #     ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')

    #     plt.title(f"Factor {name} CF Distribution")
    #     plt.savefig(f"./cf_distribution/{name}_cf.png")
    #     plt.close()



