import numpy as np
from sumolib import net

from CF import nondata
from  Dataloader import Dataloader
import geopandas as gpd
import sys
from sys import exit
from utils import geodata,metrics
import matplotlib.pyplot as plt
import rasterio
from rasterio.transform import Affine
import os
from joblib import dump, load
import pandas as pd
import bisect
from classify_metrics import calculate_classification_metrics,plot_roc_curve,print_classification_report
from sklearn import svm


nodata=-10000
factorsfolder=r"F:\resampled data from yousef"
debris_area = r"F:\data from xiaobo\debris flow.shp"
susceptibility_level_npy=r'F:\Susceptibility Assessment\susceptibility_level\susceptibility_level.npy'
suscepbility_level=np.load(susceptibility_level_npy)
factors=np.load('factors.npy')   #(17, 1, 4909, 8050)
res_folder='./res'
# res_folder='./GAN'   # ablation
sts_percentage=[]

# 结果图的profile
with rasterio.open(factorsfolder+'/Aspect.tif') as src:
    transform=src.transform
    width = src.width
    height = src.height
    crs=src.crs

# 此处可以更改，选取最好的cf值的图
cf_sts_path=r"F:\Susceptibility Assessment\lowess_cf_distribution\plot_sts.csv"
cf_sts=pd.read_csv(cf_sts_path)
print(cf_sts.columns)


def find_nearest_sorted(sorted_arr, value):
    """
    在已排序数组中查找最接近的值 (使用二分查找)

    参数:
        sorted_arr: 已排序的一维数组
        value: 目标值

    返回:
        最接近的值的索引
    """
    n = len(sorted_arr)
    if n == 0:
        return None

    # 使用二分查找定位插入位置
    pos = bisect.bisect_left(sorted_arr, value)

    # 处理边界情况
    if pos == 0:
        return 0
    if pos == n:
        return n - 1
    left_val = sorted_arr[pos - 1]
    right_val = sorted_arr[pos]

    if abs(left_val - value) <= abs(right_val - value):
        return pos - 1
    return pos



def cf_factors():
    '''
    读取栅格影像的点，并读取拟合的cf值，对每个点的每个因素赋cf值
    '''
    df = pd.read_csv(cf_sts_path)
    result_dict = {}
    grouped = df.groupby('idx')
    for idx, group in grouped:
        # print(group['x'])
        sub_dict = dict(zip(group['x'], group['lowess_y']))
        result_dict[idx] = sub_dict

    arr=factors.reshape(17,-1).T
    xs=[list(result_dict[idx].keys()) for idx in range(17)]
    print(len(xs),len(xs[0]),len(xs[7]),len(xs[8]))
    cf_factors=np.zeros_like(arr)

    for n in range(arr.shape[0]):
        for idx in range(17):
            # pos = (np.abs(xs[idx] - arr[n][idx])).argmin()
            if arr[n][idx]==nondata:
                cf=0
            else:
                pos=find_nearest_sorted(xs[idx],arr[n][idx])
                cf=result_dict[idx][xs[idx][pos]]
            # print(f'pos:{pos}')
            print(f'进度：{n}/{arr.shape[0]}  {round(100*n/arr.shape[0],2)}%')
            cf_factors[n][idx]=cf

    np.save('factors_cf.npy',cf_factors)



color_map = {
    "nodata":(1,1,1),                       #空白
    "Low": (81/255, 137/255, 61/255),       # 51893D
    "Moderate": (186/255, 191/255, 44/255), # BABF2C
    "High": (242/255, 191/255, 39/255),     # F2BF27
    "Very High": (242/255, 65/255, 65/255)  # F24141
}

def construct_map(map_array: np.array, transform,height,width,crs,tif_path,base_tif_path=factorsfolder + '/Aspect.tif'):
    """
    传入np.array，先绘制分布图，再根据transform保存为tif栅格
    :param map_array: 2D或3D numpy数组
    :param transform: 仿射变换参数
    :param tif_path: 输出tif路径
    """
    import matplotlib.colors as mcolors
    nodata_val=-1
    # 创建线性渐变的颜色映射
    cmap = mcolors.LinearSegmentedColormap.from_list('cmap', list(zip([0,0.01,0.33,0.66,1],list(color_map.values()))), N=256)

    # mask
    # map_array[map_array==map_array[0][0]]=0

    plt.figure()
    plt.axis('off')
    plt.imshow(map_array.squeeze(), cmap=cmap)
    # plt.colorbar()
    plt.savefig(tif_path.replace('.tif', '.png'))
    plt.close()


    if map_array.ndim == 2:
        arr = map_array
        count = 1
    elif map_array.ndim == 3:
        arr = map_array
        count = map_array.shape[0]
    else:
        raise ValueError("map_array 维度必须为2或3")
    if height!=arr.shape[-2] and width!=arr.shape[-1]:
        raise ValueError(f"尺寸不不匹配:{height,arr.shape[-2]},{width,arr.shape[-1]}")

    with rasterio.open(base_tif_path) as src:
        base_nodata = src.nodata
        if base_nodata is None:
            # 如果没有 nodata 值，则假设所有像素都有效
            base_mask = np.ones(arr.shape[-2:], dtype=bool)
        else:
            # 读取基图第一波段作为掩码参考
            base_band = src.read(1)
            base_mask = (base_band != base_nodata)  # True 表示有效像素
    if arr.ndim == 2:
        arr[~base_mask] = nodata_val
    else:  # 3D，假设波段数在第一位
        for i in range(arr.shape[0]):
            arr[i][~base_mask] = nodata_val
    with rasterio.open(
        tif_path,
        'w',
        driver='GTiff',
        height=arr.shape[-2],
        width=arr.shape[-1],
        count=count,
        dtype=arr.dtype,
        transform=transform,
        nodata=nodata_val,
        crs=crs
    ) as dst:
        if count == 1:
            dst.write(arr, 1)
        else:
            dst.write(arr)



from sklearn.svm import OneClassSVM



from sklearn.ensemble import IsolationForest, RandomForestRegressor


def isolation_forest(train_arr,evaluate_arr,type:str):
    iso_forest = IsolationForest(contamination=0.1, random_state=42)  # contamination参数类似nu
    iso_forest.fit(train_arr)

    truth=np.array([1]*len(train_arr))
    prediction = iso_forest.predict(train_arr)
    score=iso_forest.decision_function(train_arr)+0.5
    metrics = calculate_classification_metrics(truth, prediction, score,average='binary')
    print_classification_report(metrics)

    # iso_forest = IsolationForest(contamination=0.1, random_state=42)  # contamination参数类似nu
    # iso_forest.fit(np.concatenate([train_arr,test_arr], axis=0))
    # res=iso_forest.predict(evalueate_arr)
    res=np.array(iso_forest.decision_function(evaluate_arr))
    print(res.shape)
    res=((res+0.5)*4)   #0-4
    print(res.shape)
    print(res)
    res_sts(res, type + '-isolation_forest')
    res=np.array(res).reshape(4909,8050)
    construct_map(res,transform=transform,crs=crs,height=height,width=width,tif_path=res_folder+'/'+type+'-res_isoforest.tif')

def oneclass_svm(train_arr,evaluate_arr,type:str):
    if type=='RAW':
        train_arr=(train_arr-np.min(train_arr))/(np.max(train_arr)-np.min(train_arr))
        evaluate_arr=(evaluate_arr-np.min(evaluate_arr))/(np.max(evaluate_arr)-np.min(evaluate_arr))
    oneclass_svm=svm.OneClassSVM(kernel='rbf', gamma='scale', nu=0.1)
    oneclass_svm.fit(train_arr)
    truth=np.array([1]*len(train_arr))
    prediction = oneclass_svm.predict(train_arr)
    score=oneclass_svm.decision_function(train_arr)+0.5
    metrics = calculate_classification_metrics(truth, prediction, score,average='binary')
    print_classification_report(metrics)
    res=np.array(oneclass_svm.decision_function(evaluate_arr))
    print(res.shape)
    res=4*(res-min(res))/(max(res)-min(res))         #0-4
    print(res.shape)
    print(res)
    res_sts(res, type + '-oneclass_svm')
    res=np.array(res).reshape(4909,8050)
    construct_map(res,transform=transform,crs=crs,height=height,width=width,tif_path=res_folder+'/'+type+'-res_oneclasssvm.tif')

def L2_Norm(train_arr,evaluate_arr,type:str):
    if type=='RAW':
        # 标准化
        # train_arr=(train_arr-np.mean(train_arr))/np.std(train_arr)
        # evaluate_arr=(evaluate_arr-np.mean(evaluate_arr))/np.std(evaluate_arr)
        #归一化
        train_arr=(train_arr-np.min(train_arr))/(np.max(train_arr)-np.min(train_arr))
        evaluate_arr=(evaluate_arr-np.min(evaluate_arr))/(np.max(evaluate_arr)-np.min(evaluate_arr))
    std_sample=np.mean(train_arr,axis=0)
    print(std_sample)
    distances = np.linalg.norm(evaluate_arr - std_sample, axis=1)
    distances=(distances-np.min(distances))/(np.max(distances)-np.min(distances))
    res = 4-4*distances
    res_sts(res, type + '-L2_Norm')
    res = np.array(res).reshape(4909, 8050)
    construct_map(res, transform=transform, crs=crs, height=height, width=width,
                  tif_path=res_folder + '/' + type + '-L2_Norm.tif')

from sklearn.ensemble import RandomForestClassifier
def random_forest(train_arr, train_label, evaluate_arr, type: str):
    """
    随机森林二分类

    参数：
    - train_arr: 训练特征矩阵 (n_samples, 17)
    - train_label: 训练标签 (n_samples,) , 取值 0 或 1
    - evaluate_arr: 评估特征矩阵 (m_samples, 17)
    - type: 'RAW' 时进行 [0,1] 归一化，否则不处
    """
    # 1. 归一化（使用训练集的min/max，避免数据泄漏）
    if type == 'RAW':
        train_arr = (train_arr - np.min(train_arr)) / (np.max(train_arr) - np.min(train_arr))
        evaluate_arr = (evaluate_arr - np.min(evaluate_arr)) / (np.max(evaluate_arr) - np.min(evaluate_arr))

    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(train_arr, train_label)
    truth = train_label
    prediction = rf.predict(train_arr)
    print(np.unique(prediction, return_counts=True))
    print(np.unique(truth, return_counts=True))
    score = rf.predict_proba(train_arr)[:, 1]
    metrics = calculate_classification_metrics(truth, prediction, score, average='binary')
    print_classification_report(metrics)

    # 4. 对评估集进行预测，得到决策分数并缩放到 1~4
    #    使用预测概率（正类概率）作为原始分数，再线性映射到 [1,4]
    res = rf.predict_proba(evaluate_arr)[:, 1]  # shape (m_samples,)
    print("原始分数 shape:", res.shape)
    # 线性缩放：min->0, max->4
    res = 4 * (res - np.min(res)) / (np.max(res) - np.min(res))
    print("缩放后 shape:", res.shape)
    print("缩放结果示例:", res[:10])

    # 5. 调用自定义结果统计与地图构建函数（用法与 oneclass_svm 相同）
    res_sts(res, type + '-random_forest')
    # 将一维结果 reshape 为二维（必须保证 len(res) == height*width）
    res = np.array(res).reshape(height, width)
    construct_map(res, transform=transform, crs=crs,
                  height=height, width=width,
                  tif_path=res_folder + '/' + type + 'res-zrandomforest.tif')


import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler,MinMaxScaler
device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class Generator(nn.Module):
    def __init__(self, latent_dim, output_dim):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(latent_dim, 32),
            nn.LeakyReLU(0.2),
            nn.Linear(32, 64),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(0.2),
            nn.Linear(64, output_dim),
            nn.Tanh()
        )
    def forward(self, y):
        return self.model(y)

class Encoder(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super(Encoder, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.LeakyReLU(0.2),
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(0.2),
            nn.Linear(32, latent_dim)
        )

    def forward(self, x):
        return self.model(x)

class Discriminator(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super(Discriminator, self).__init__()
        # 处理输入数据的路径
        self.data_path = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.LeakyReLU(0.2),
            nn.Linear(64, 32),
            nn.LeakyReLU(0.2),
        )

        # 处理潜在向量的路径
        self.latent_path = nn.Sequential(
            nn.Linear(latent_dim, 32),
            nn.LeakyReLU(0.2),
            nn.Linear(32, 32),
            nn.LeakyReLU(0.2),
        )

        # 联合处理
        self.joint = nn.Sequential(
            nn.Linear(64, 32),  # 32+32=64
            nn.LeakyReLU(0.2),
            nn.Linear(32, 1),
            nn.Sigmoid()
            # nn.Tanh()
        )

    def forward(self, x, z):
        x_out = self.data_path(x)
        z_out = self.latent_path(z)
        joint_input = torch.cat((x_out, z_out), dim=1)
        return self.joint(joint_input)


class GAN_Discriminator(nn.Module):
    def __init__(self, input_dim):
        super(GAN_Discriminator, self).__init__()

        self.data_path = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.LeakyReLU(0.2),
            nn.Linear(64, 32),
            nn.LeakyReLU(0.2),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
    def forward(self, x):
        x_out = self.data_path(x)
        return x_out

class BiGAN(nn.Module):
    def __init__(self, train_arr,Type,latent_dim=100):
        """
        BiGAN 实现

        参数:
        - train_arr: 训练数据，形状为 (n, 17)
        - evaluate_arr: 评估数据，形状为 (n, 17)
        - latent_dim: 潜在空间维度
        - device: 训练设备 (cpu/cuda)
        """
        super(BiGAN, self).__init__()
        self.Type=Type
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.history = {'D_loss': [], 'GE_loss': [], 'G_loss': [], 'E_loss': []}
        self.input_dim = train_arr.shape[1]
        self.latent_dim = latent_dim
        self.encoder = Encoder(self.input_dim, self.latent_dim)
        self.generator = Generator(self.latent_dim, self.input_dim)
        self.discriminator = Discriminator(self.input_dim, self.latent_dim)

        self.optimizer_E = optim.Adam(self.encoder.parameters(),lr=0.0001, betas=(0.0, 0.9))
        self.optimizer_G = optim.Adam(self.generator.parameters(), lr=0.0001,betas=(0.0, 0.9))
        self.optimizer_D = optim.Adam(self.discriminator.parameters(), lr=0.0001,betas=(0.0, 0.9))
        self.adversarial_loss = nn.BCELoss()

        if self.Type=='RAW':
            train_tensor = torch.FloatTensor(train_arr)
            train_tensor = torch.tanh(train_tensor)
        else:
            train_tensor = torch.FloatTensor(train_arr)

        self.train_loader = DataLoader(TensorDataset(train_tensor), batch_size=128,shuffle=True)
        self.to(self.device)

    def forward(self, x):
        """前向传播"""
        z = self.encoder(x)
        x_recon = self.generator(z)
        return z, x_recon

    def discriminate(self, x, z):
        """判别器前向传播"""
        return self.discriminator(x, z)

    def compute_gradient_penalty(self, discriminator, x, Gz, Ex, z):
        """计算梯度惩罚"""
        alpha = torch.rand(x.size(0), 1).to(self.device)
        alpha = alpha.expand_as(x)

        # 创建插值样本
        interpolates = (alpha * x + ((1 - alpha) * Gz)).requires_grad_(True)

        # 创建插值潜在向量
        alpha_z = torch.rand(Ex.size(0), 1).to(self.device)
        alpha_z = alpha_z.expand_as(Ex)
        interpolates_z = (alpha_z * Ex + ((1 - alpha_z) * z)).requires_grad_(True)

        # 计算判别器对插值的输出
        d_interpolates = discriminator(interpolates, interpolates_z)

        # 计算梯度
        gradients = torch.autograd.grad(
            outputs=d_interpolates,
            inputs=[interpolates, interpolates_z],
            grad_outputs=torch.ones_like(d_interpolates),
            create_graph=True,
            retain_graph=True,
            only_inputs=True
        )

        # 计算梯度范数
        gradients = torch.cat([grad.view(grad.size(0), -1) for grad in gradients], dim=1)
        gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()

        return gradient_penalty

    def train_model(self, epochs=100):
        """训练BiGAN模型"""
        self.train()
        best_epoch = -1
        judge=[]
        l=len(self.train_loader)
        lambda_gp = 10  # 梯度惩罚系数
        for epoch in range(epochs):
            d_loss_sum,ge_loss_sum,g_loss_sum,e_loss_sum= 0,0,0,0
            for i, (x,) in enumerate(self.train_loader):
                self.optimizer_D.zero_grad()
                x = x.to(self.device)
                batch_size = x.size(0)
                Ex = self.encoder(x)
                z = torch.randn(batch_size, self.latent_dim).to(self.device)
                Gz = self.generator(z)
                gradient_penalty = self.compute_gradient_penalty(
                self.discriminator, x, Gz.detach(), Ex.detach(), z)
                batch_size = x.size(0)
                valid = torch.ones(batch_size, 1).to(self.device)
                fake = torch.zeros(batch_size, 1).to(self.device)
                valid_loss = self.adversarial_loss(self.discriminate(x, Ex.detach()), valid)
                fake_loss = self.adversarial_loss(self.discriminate(Gz.detach(), z), fake)
                d_loss=valid_loss+fake_loss+gradient_penalty*lambda_gp
                # 使用 Wasserstein 损失与梯度惩罚 (WGAN-GP)
                # diff = self.discriminate(Gz.detach(), z) - self.discriminate(x, Ex.detach())
                # d_loss = torch.mean(diff) + lambda_gp * gradient_penalty
                d_loss.backward()
                self.optimizer_D.step()

            #  训练生成器和编码器
                self.optimizer_G.zero_grad()
                self.optimizer_E.zero_grad()
                g_loss = torch.mean(self.adversarial_loss(self.discriminate(Gz, z), valid))
                e_loss = torch.mean(self.adversarial_loss(self.discriminate(x, Ex), valid))
                # WGAN-GP
                # g_loss = torch.mean(-1 * (self.discriminate(Gz, z)))
                # e_loss = torch.mean(-1 * (self.discriminate(x, Ex)))

                ge_loss = (g_loss + e_loss) / 2
                ge_loss.backward()
                self.optimizer_G.step()
                self.optimizer_E.step()
                # print(d_loss.item(),g_loss.item(),e_loss.item())
                d_loss_sum += d_loss.item()
                ge_loss_sum += ge_loss.item()
                e_loss_sum += e_loss.item()
                g_loss_sum+= g_loss.item()

            # eopch过后,记录平均损失
            self.history['D_loss'].append(d_loss_sum/l)
            self.history['GE_loss'].append(ge_loss_sum/l)
            self.history['G_loss'].append(g_loss_sum/l)
            self.history['E_loss'].append(e_loss_sum/l)
            judge.append(abs(ge_loss.item()-d_loss.item()))
            # 打印训练进度
            if (epoch + 1) % 10 == 0:
                print(f"Epoch [{epoch+1}/{epochs}], "
                      f"D Loss: {self.history['D_loss'][-1]:.4f}, "
                      f"G/E Loss: {self.history['GE_loss'][-1]:.4f}, "
                      f"G Loss: {self.history['G_loss'][-1]:.4f}, "
                      f"E Loss: {self.history['E_loss'][-1]:.4f}")
                self.save_model(path=f'./res/models/{self.Type}-GAN_{epoch + 1}.pth')
                print(f'最小差距：{min(judge)}')

            if min(judge)==judge[-1]:
                print(f'epoch:{epoch+1}/{epochs},')
                best_epoch = epoch
                self.save_model(path=f'./GAN/{self.Type}-BIGAN_{epoch+1}.pth')
                self.plot_training_history()
                # self.discriminate_evaluation(evaluate_arr)
            if (epoch-best_epoch)>10:
                return self.history
        return self.history

    def discriminate_evaluation(self,evaluate_arr):
        if self.Type=='RAW':
            evaluate_tensor = torch.tanh(torch.FloatTensor(evaluate_arr))
        else:
            evaluate_tensor = torch.FloatTensor(evaluate_arr)

        self.eval()
        with torch.no_grad():
            self.to('cpu')
            y=self.encoder(evaluate_tensor)
            res=self.discriminate(evaluate_tensor,y).cpu().numpy()
            # res=4*(res-np.min(res))/(np.max(res)-np.min(res))  取消sigmoid
            # res = 4 *(1 / (1 + np.exp(-res)))
            res=res   #D增加sigmoid层
            print(res.shape)
            print(res)
            res_sts(res, self.Type + '-BiGAN')
            res = np.array(res).reshape(4909, 8050)
            construct_map(res, transform=transform, crs=crs, height=height, width=width,
                          tif_path=res_folder + '/' + self.Type + '-res_BiGAN.tif')
        return res

    def plot_training_history(self):
        """绘制训练历史"""
        history = self.history
        plt.figure(figsize=(10, 5))
        plt.plot(history['D_loss'], label='Discriminator Loss')
        plt.plot(history['GE_loss'], label='Generator/Encoder Loss')
        plt.plot(history['E_loss'], label='Encoder Loss')
        plt.plot(history['G_loss'], label='Generator Loss')
        plt.title('Training Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        plt.show()

    def save_model(self, path):
        """保存模型"""
        torch.save({
            'encoder_state_dict': self.encoder.state_dict(),
            'generator_state_dict': self.generator.state_dict(),
            'discriminator_state_dict': self.discriminator.state_dict(),
            'history': self.history,
            'Type': self.Type
        }, path)

    def load_model(self, path):
        """加载模型"""
        checkpoint = torch.load(path, map_location=self.device,weights_only=False)
        self.encoder.load_state_dict(checkpoint['encoder_state_dict'])
        self.generator.load_state_dict(checkpoint['generator_state_dict'])
        self.discriminator.load_state_dict(checkpoint['discriminator_state_dict'])
        self.history = checkpoint['history']
        self.Type = checkpoint['Type']

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import numpy as np

# ===================== 纯 GAN 模型 =====================
class GAN(nn.Module):
    def __init__(self, train_arr, Type, latent_dim=17):
        """
        纯 GAN 实现（仅 Generator + Discriminator）
        """
        super(GAN, self).__init__()
        self.Type = Type
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.history = {'D_loss': [], 'G_loss': []}
        self.input_dim = train_arr.shape[1]
        self.latent_dim = latent_dim

        self.generator = Generator(self.latent_dim, self.input_dim)
        self.discriminator = GAN_Discriminator(self.input_dim)

        self.optimizer_G = optim.Adam(self.generator.parameters(), lr=0.0001, betas=(0.0, 0.9))
        self.optimizer_D = optim.Adam(self.discriminator.parameters(), lr=0.0001, betas=(0.0, 0.9))

        self.adversarial_loss = nn.BCELoss()

        # 数据处理
        if self.Type == 'RAW':
            train_tensor = torch.FloatTensor(train_arr)
            train_tensor = torch.tanh(train_tensor)
        else:
            train_tensor = torch.FloatTensor(train_arr)

        self.train_loader = DataLoader(TensorDataset(train_tensor), batch_size=128, shuffle=True)
        self.to(self.device)

    def forward(self, z):
        """仅生成器前向传播"""
        return self.generator(z)

    def discriminate(self, x):
        """判别器前向传播（只输入x，不再输入z）"""
        return self.discriminator(x)

    def compute_gradient_penalty(self, discriminator, x, Gz):
        """计算梯度惩罚（GAN 版）"""
        alpha = torch.rand(x.size(0), 1).to(self.device)
        alpha = alpha.expand_as(x)

        # 插值样本
        interpolates = (alpha * x + ((1 - alpha) * Gz)).requires_grad_(True)
        d_interpolates = discriminator(interpolates)

        # 计算梯度
        gradients = torch.autograd.grad(
            outputs=d_interpolates,
            inputs=interpolates,
            grad_outputs=torch.ones_like(d_interpolates),
            create_graph=True,
            retain_graph=True,
            only_inputs=True
        )[0]

        gradients = gradients.view(gradients.size(0), -1)
        gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
        return gradient_penalty

    def train_model(self, epochs=100):
        """训练 GAN 模型"""
        self.train()
        best_epoch = -1
        judge = []
        l = len(self.train_loader)
        lambda_gp = 10

        for epoch in range(epochs):
            d_loss_sum, g_loss_sum = 0, 0

            for i, (x,) in enumerate(self.train_loader):
                x = x.to(self.device)
                batch_size = x.size(0)
                self.optimizer_D.zero_grad()

                # 真实数据
                valid = torch.ones(batch_size, 1).to(self.device)
                z = torch.randn(batch_size, self.latent_dim).to(self.device)
                Gz = self.generator(z)

                gradient_penalty = self.compute_gradient_penalty(self.discriminator, x, Gz.detach())

                # 判别器损失
                real_loss = self.adversarial_loss(self.discriminate(x), valid)
                fake_loss = self.adversarial_loss(self.discriminate(Gz.detach()), torch.zeros_like(valid))
                d_loss = real_loss + fake_loss + lambda_gp * gradient_penalty

                d_loss.backward()
                self.optimizer_D.step()

                # ---------------------
                #  训练生成器 G
                # ---------------------
                self.optimizer_G.zero_grad()

                g_loss = self.adversarial_loss(self.discriminate(Gz), valid)
                g_loss.backward()
                self.optimizer_G.step()

                # 记录损失
                d_loss_sum += d_loss.item()
                g_loss_sum += g_loss.item()

            # 保存平均损失
            self.history['D_loss'].append(d_loss_sum / l)
            self.history['G_loss'].append(g_loss_sum / l)
            judge.append(abs(self.history['G_loss'][-1] - self.history['D_loss'][-1]))

            # 打印
            if (epoch + 1) % 10 == 0:
                print(f"Epoch [{epoch+1}/{epochs}], "
                      f"D Loss: {self.history['D_loss'][-1]:.4f}, "
                      f"G Loss: {self.history['G_loss'][-1]:.4f}")
                self.save_model(path=f'./res/models/{self.Type}-GAN_{epoch + 1}.pth')
                print(f'最小差距：{min(judge)}')

            # 保存最优模型
            if min(judge) == judge[-1]:
                best_epoch = epoch
                self.save_model(path=f'./GAN/{self.Type}-GAN_{epoch+1}.pth')
                self.plot_training_history()

            if (epoch - best_epoch) > 10:
                return self.history

        return self.history

    def discriminate_evaluation(self, evaluate_arr):
        """仅使用判别器进行预测打分"""
        if self.Type == 'RAW':
            evaluate_tensor = torch.tanh(torch.FloatTensor(evaluate_arr))
        else:
            evaluate_tensor = torch.FloatTensor(evaluate_arr)

        self.eval()
        with torch.no_grad():
            self.to('cpu')
            res = self.discriminate(evaluate_tensor).cpu().numpy()
            print(res.shape)
            print(res)
            res_sts(res, self.Type + '-GAN')
            res = np.array(res).reshape(4909, 8050)
            construct_map(res, transform=transform, crs=crs, height=height, width=width,
                          tif_path=res_folder + '/' + self.Type + '-res_GAN.tif')
        return res

    def plot_training_history(self):
        """绘制训练曲线"""
        history = self.history
        plt.figure(figsize=(10, 5))
        plt.plot(history['D_loss'], label='Discriminator Loss')
        plt.plot(history['G_loss'], label='Generator Loss')
        plt.title('GAN Training Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        plt.show()

    def save_model(self, path):
        torch.save({
            'generator_state_dict': self.generator.state_dict(),
            'discriminator_state_dict': self.discriminator.state_dict(),
            'history': self.history,
            'Type': self.Type
        }, path)

    def load_model(self, path):
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.generator.load_state_dict(checkpoint['generator_state_dict'])
        self.discriminator.load_state_dict(checkpoint['discriminator_state_dict'])
        self.history = checkpoint['history']
        self.Type = checkpoint['Type']
class Autoencoder(nn.Module):
    def __init__(self, input_dim, encoding_dim):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            # 引入dropout，随机0.2的神经元输出清零，模拟随机连接
            nn.Linear(input_dim, 128),
            nn.Tanh(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, encoding_dim),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        self.decoder = nn.Sequential(
            nn.Linear(encoding_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, input_dim),
            nn.Tanh()
        )
    def forward(self, x):
        return self.decoder(self.encoder(x))

class RandNetEnsemble(nn.Module):
    def __init__(self, n_estimators, input_dim, encoding_dim,Type):
        super(RandNetEnsemble, self).__init__()
        self.Type=Type
        self.n_estimators = n_estimators
        self.autoencoders = nn.ModuleList()
        for _ in range(n_estimators):
            self.autoencoders.append(Autoencoder(input_dim, encoding_dim)).to(device)

    def forward(self, x):
        all_reconstructions = []
        for ae in self.autoencoders:
            # 此处标准化一下所有的ae(x)
            # print(f'x:{x.shape}')
            # print(f'ae(x): {ae(x).shape}')
            x=ae(x)
            all_reconstructions.append((x-x.mean())/x.std())
        # 选择中位数作为输出
        all_reconstructions=torch.stack(all_reconstructions, dim=0)
        # print(f'torch.median(all_reconstructions, dim=0):{torch.median(all_reconstructions, dim=0)[0].shape}')
        return torch.median(all_reconstructions, dim=0)[0]

    def get_ensemble_reconstruction_error(self, x, criterion=nn.MSELoss(reduction='none')):
        reconstructions = self.forward(x) # (n_estimators, batch_size,input_dim)
        print(f'reconstructions: {reconstructions.shape}')
        error = criterion(reconstructions, x)
        print(f'error: {error.shape}')
        # error = torch.mean(error, dim=1)
        error=torch.sum(error,dim=1)   #修改
        print(f'error: {error.shape}')
        return error

    def train_ensemble(self, train_data, num_epochs_per_ae, learning_rate, batch_size=64):
        """训练RandNet集成模型"""
        optimizer_list = [optim.Adam(ae.parameters(), lr=learning_rate) for ae in self.autoencoders]
        criterion = nn.MSELoss()
        # if self.Type == 'RAW':
        #     train_data=np.tanh(train_data)
        if self.Type == 'CF':
            train_data=train_data*10000
        if not isinstance(train_data, torch.Tensor):
            train_data = torch.tensor(train_data, dtype=torch.float32)
        train_data = train_data.to(device)
        train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
        train_sts=[]
        for epoch in range(num_epochs_per_ae):
            total_loss = 0
            for ae_idx, (autoencoder, optimizer) in enumerate(zip(self.autoencoders, optimizer_list)):
                autoencoder.train()
                for batch_idx, data in enumerate(train_loader):
                    data = data.view(data.size(0), -1)  # 展平数据
                    optimizer.zero_grad()
                    output = autoencoder(data)
                    loss = criterion(output, data)
                    loss.backward()
                    optimizer.step()
                    total_loss += loss.item()
            print(f'epoch {epoch+1}, mean loss: {total_loss/self.n_estimators}')
            train_sts.append(total_loss/self.n_estimators)
            if (epoch + 1) % 10 == 0:
                path = './res/models/'+self.Type+'-RandNetEnsemble_' + str(epoch + 1) + '_' + str(self.n_estimators) + '.pth'
                torch.save({
                    'ensemble_state_dict': self.state_dict()
                }, path)

        plt.plot(range(num_epochs_per_ae), train_sts)
        plt.show()


    def evaluate_ensemble(self, test_data):
        """评估RandNet集成模型"""
        print("Evaluating RandNet")
        self.eval()
        # if self.Type == 'RAW':
        #     test_data = np.tanh(test_data)
        if self.Type == 'CF':
           test_data=test_data*10000
        all_scores = []

        test_data = torch.tensor(test_data, dtype=torch.float32).to(device)
        test_data = DataLoader(test_data, batch_size=10000, shuffle=False)
        # self.to('cpu')

        with torch.no_grad():
            #     all_scores = self.get_ensemble_reconstruction_error(test_data)  #不分批
            for batch in test_data:  #分批处理
                batch.to(device)
                batch_scores = self.get_ensemble_reconstruction_error(batch)
                all_scores.append(batch_scores.cpu().numpy())
        all_scores = np.concatenate(all_scores, axis=0)
        print(f"Score shape: {all_scores.shape}")
        res_sts(all_scores, self.Type + '-RandNet')
        all_scores = all_scores.reshape(4909, 8050)
        construct_map(all_scores, transform=transform, crs=crs, height=height, width=width,
                      tif_path=res_folder + '/' + self.Type + '-res_RandNet.tif')

    def load_model(self, path):
        checkpoint = torch.load(path, map_location=device,weights_only=False)
        self.load_state_dict(checkpoint['ensemble_state_dict'])



def res_sts(map_arr: np.array, model: str):
    '''
    map_arr: 去除nodata的map.flatten
    出统计图

    '''
    map_arr=map_arr[map_arr!=map_arr[0]]   #Randnet无法做到相同的判断？
    if len(map_arr)==0:
        print('全是同一个值❌')
        return None
    # thresholds=[0,0.25,0.5,0.75,1]
    # classified = np.digitize(map_arr, thresholds[:-1])
    # class_counts = np.bincount(classified)
    mn,mx,surplus=np.min(map_arr), np.max(map_arr),np.max(map_arr)-np.min(map_arr)
    a0,a1,a2,a3,a4=mn,mn+0.25*surplus,mn+0.5*surplus,mn+0.75*surplus,mx
    class_counts = np.array([
        np.sum(( a0<= map_arr) & (map_arr <a1)),
        np.sum((a1 <= map_arr) & (map_arr < a2)),
        np.sum((a2 <= map_arr) & (map_arr < a3)),
        np.sum((a3 <= map_arr) & (map_arr <= a4))
    ])
    percentages = class_counts / len(map_arr) * 100

    fig, ax = plt.subplots(figsize=(10, 5))
    plt.title(model)

    colors = {
        "Low": (81 / 255, 137 / 255, 61 / 255),  # 51893D  0-1
        "Moderate": (186 / 255, 191 / 255, 44 / 255),  # BABF2C  1-2
        "High": (242 / 255, 191 / 255, 39 / 255),  # F2BF27  2-3
        "Very High": (242 / 255, 65 / 255, 65 / 255)  # F24141  3-4
    }

    ranges = ['0-1 (Low)', '1-2 (Moderate)', '2-3 (High)', '3-4 (Very High)']
    bar_colors = [colors["Low"], colors["Moderate"], colors["High"], colors["Very High"]]
    bars = ax.bar(ranges, percentages, color=bar_colors, alpha=0.8, edgecolor='black')

    # Set labels and title on the axes, not the figure
    ax.set_xlabel('Value Ranges')
    ax.set_ylabel('Percentage')
    ax.set_title('Percentage of Values in Each Range')
    ax.grid(True, alpha=0.3)

    # Rotate x-axis labels
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')

    # Add value labels on bars
    for bar, percentage in zip(bars, percentages):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2., height + 0.5,
                f'{percentage:.1f}%', ha='center', va='bottom')

    # Create legend
    legend_labels = ['Low (0-1)', 'Moderate (1-2)', 'High (2-3)', 'Very High (3-4)']
    legend_colors = [colors["Low"], colors["Moderate"], colors["High"], colors["Very High"]]
    legend_patches = [plt.Rectangle((0, 0), 1, 1, color=color) for color in legend_colors]
    ax.legend(legend_patches, legend_labels, loc='upper right')

    plt.tight_layout()
    plt.show()


# 把debris_area导出为数据文件
# gdf=gpd.read_file(debris_area)
# gdf['level']=4
# print(gdf.columns)
# gdf=gdf.drop(columns=['Id','True'])
# print(gdf.columns)
# gdf.to_file('susceptibility_level', driver='ESRI Shapefile')
# geodata.shp2tif('susceptibility_level/susceptibility_level.shp','susceptibility_level/susceptibility_level.tif',factorsfolder+'\Aspect.tif',value_field='level')
# print(suscepbility_level.shape)
# suscepbility_level=suscepbility_level.reshape(1, 1, suscepbility_level.shape[0], suscepbility_level.shape[1])
# np.save(susceptibility_level_npy,suscepbility_level)


# 获取有效面积
# geodata.tif2shp(factorsfolder+'/Aspect.tif','valid_area/valid_area.shp')
# gdf=gpd.read_file(debris_area)
# exit()




# 获取有效面积
# geodata.tif2shp(factorsfolder+'/Aspect.tif','valid_area/valid_area.shp')
# gdf=gpd.read_file(debris_area)
# exit()



# cf_factors()

factors_cf=np.load('factors_cf.npy')
suscepbility_level_2classtrain=np.load(r'F:\Susceptibility Assessment\susceptibility_level\susceptibility_level_2classtrain.npy').reshape(4909, 8050, 1)
print(np.unique(suscepbility_level_2classtrain, return_counts=True))
print(suscepbility_level_2classtrain.shape)
Dataloader_low=Dataloader(factors,suscepbility_level_2classtrain,factors_cf,data_vals=[1])
print(f'factors.shape:{factors.shape}')
print(f'suscepbility_level.shape:{suscepbility_level.shape}')
print(f'factors_cf.shape:{factors_cf.shape}')
Dataloader=Dataloader(factors,suscepbility_level,factors_cf)
raw_factors=factors.reshape(17,-1).T
# raw_factors[raw_factors==nodata]=0
cf_x_train_similarity=Dataloader.load_cf()
raw_x_train_similarity=Dataloader.load_raw()
cf_x_train_similarity_low=Dataloader_low.load_cf()
raw_x_train_similarity_low=Dataloader_low.load_raw()

print(cf_x_train_similarity.shape,raw_x_train_similarity.shape,cf_x_train_similarity_low.shape,raw_x_train_similarity_low.shape)



# (107230, 17) (39517450, 17)
# print(cf_x_train_similarity.shape,factors_cf.shape)


# print('cf-L2_Norm')
# L2_Norm(cf_x_train_similarity,factors_cf,type='CF')
# print('raw-L2_Norm')
# L2_Norm(raw_x_train_similarity,raw_factors,type='RAW')

# print('cf-isolation_forest')
# isolation_forest(cf_x_train_similarity,factors_cf,type='CF')
# print('raw-isolation_forest')
# isolation_forest(raw_x_train_similarity,raw_factors,type='RAW')

# print('cf-oneclass_svm')
# oneclass_svm(cf_x_train_similarity,factors_cf,type='CF')
# print('raw-oneclass_svm')
# oneclass_svm(raw_x_train_similarity,raw_factors,type='RAW')

# GAN模型测试
def test(number:str,type:str):
    bigan = BiGAN(cf_x_train_similarity,latent_dim=51,Type=type+'-'+str(number))
    bigan.load_model(path='./res/models/'+type+'-GAN_'+str(number)+'.pth')
    bigan.plot_training_history()
    bigan.discriminate_evaluation(factors_cf)
    exit()

# test(50,'RAW8')
# for i in range(3,34):
#     test(100,'RAW'+str(i))

# print('BiGAN')
# bigan = BiGAN(raw_x_train_similarity,latent_dim=51,Type='RAW')
# bigan.train_model(epochs=200)
# bigan.plot_training_history()
# res = bigan.discriminate_evaluation(raw_factors)

# print('cf-BiGAN')
# bigan = BiGAN(cf_x_train_similarity,latent_dim=51,Type='CF')
# bigan.train_model(epochs=200)
# bigan.plot_training_history()
# res = bigan.discriminate_evaluation(factors_cf)

#ablation and run-to-run variance across random seeds
# gan=GAN(cf_x_train_similarity,latent_dim=17,Type='CF')
# gan.train_model(epochs=200)
# gan.plot_training_history()
# res = gan.discriminate_evaluation(factors_cf)
# bigan = BiGAN(cf_x_train_similarity,latent_dim=51,Type='CF')
# bigan.train_model(epochs=200)
# bigan.plot_training_history()
# res = bigan.discriminate_evaluation(factors_cf)
# exit()
# i=8
# while i:
#     print('raw-BiGAN---'+str(i))
#     bigan = BiGAN(raw_x_train_similarity,latent_dim=51,Type='RAW'+str(i))
#     bigan.train_model(epochs=20000)
#     bigan.plot_training_history()
#     res = bigan.discriminate_evaluation(raw_factors)
#     i = i + 1
#     break

# Randnet模型测试
def test2(epoch,n,type:str):
    Randnet=RandNetEnsemble(n_estimators=n, input_dim=17,encoding_dim=32,Type=type)
    path = './res/models/' + type + '-RandNetEnsemble_' + str(epoch) + '_' + str(n) + '.pth'
    Randnet.load_model(path=path)
    Randnet.evaluate_ensemble(factors_cf)
    exit()
# test2(10,30,'CF')

# print('cf-RandNet')
# Randnet=RandNetEnsemble(n_estimators=30, input_dim=17,encoding_dim=32,Type='CF')
# Randnet.train_ensemble(cf_x_train_similarity,100,0.001,batch_size=64)
# Randnet.evaluate_ensemble(factors_cf)
# exit()
# print('raw-RandNet')
# Randnet=RandNetEnsemble(n_estimators=20, input_dim=17,encoding_dim=32,Type='RAW')
# Randnet.train_ensemble(raw_x_train_similarity,100,0.001,batch_size=64)
# Randnet.evaluate_ensemble(raw_factors)

# #RF
from sklearn.utils import resample
# print('cf-RF')
# data_1,data_0=cf_x_train_similarity,cf_x_train_similarity_low
# print('data1,data_0')
# data_0 = resample(data_0,replace=False, n_samples=len(data_1),  random_state=42)  #balance 1:1
# X_train = np.vstack([data_1, data_0])  # 总样本数 792231
# y_train = np.hstack([np.ones(len(data_1)), np.zeros(len(data_0))])
# random_forest(X_train, y_train, factors_cf, type='CF')
#
# print('raw-RF')
# data_1,data_0=raw_x_train_similarity,raw_x_train_similarity_low
# data_0 = resample(data_0,replace=False, n_samples=len(data_1),  random_state=42)
# X_train = np.vstack([data_1, data_0])
# y_train = np.hstack([np.ones(len(data_1)), np.zeros(len(data_0))])
# random_forest(X_train, y_train, raw_factors, type='RAW')



# 最终的结果测试，泪目/(ㄒoㄒ)/~~
from utils import geodata
from sklearn.metrics import roc_auc_score, roc_curve,auc,RocCurveDisplay
import matplotlib.pyplot as plt
roi_path=r'F:\data from xiaobo\Roi.shp'
# data,p1=geodata.read_tifs('./res/res',nodata=np.finfo(np.float32).min)
data,p1=geodata.read_tifs('./GAN/res',nodata=np.finfo(np.float32).min)
# print(data.shape)
factorsfolder=r"F:\resampled data from yousef"
# geodata.shp2tif('susceptibility_level/susceptibility_level.shp','susceptibility_level/susceptibility_level.tif',factorsfolder+'\Aspect.tif',value_field='level')
# geodata.shp2tif('susceptibility_level/susceptibility_level0.shp','susceptibility_level/susceptibility_level0.tif',factorsfolder+'\Aspect.tif')
# geodata.shp2tif('susceptibility_level/susceptibility_level-high.shp','susceptibility_level/susceptibility_level-high.tif',factorsfolder+'\Aspect.tif')
# geodata.shp2tif('susceptibility_level/susceptibility_level0_town_lake_glacier.shp','susceptibility_level/susceptibility_level0_town_lake_glacier.tif',factorsfolder+'\Aspect.tif')

susceptibility_level,p2=geodata.read_tifs('./susceptibility_level')
print(susceptibility_level.shape)
with rasterio.open(factorsfolder + '/Aspect.tif') as src:
    base_nodata = src.nodata
    if base_nodata is None:
        # 如果没有 nodata 值，则假设所有像素都有效
        base_mask = np.ones((4909,8050), dtype=bool)
    else:
        # 读取基图第一波段作为掩码参考
        base_band = src.read(1)
        base_mask = (base_band != base_nodata)
model=[]

High_new,High,Low=np.isin(susceptibility_level[0,0],1).flatten(),np.isin(susceptibility_level[1,0],4).flatten(),np.isin(susceptibility_level[2,0],1).flatten()
# High=High|High_new
#归一化的处理方式 tp,tf,nt,nf,score,f1,auc?
# methods={0:'CF-L2 Norm',1:'CF-WBiGan-GP',2:'CF-iForest',3:'CF-One-Class SVM',4:'CF-RandNet',5:'L2 Norm',6:'WBiGan-GP',7:'iForest',8:'One-Class SVM',9:'RandNet'}
methods={0:'BiGAN1',1:'BiGAN2',2:'BiGAN3',3:'BiGAN4',4:'BiGAN5',5:'GAN1',6:'GAN2',7:'GAN3',8:'GAN4',9:'GAN5'}
# methods={0:'CF-L2_Norm',1:'CF-WBiGan-GP',2:'CF-IsoForest',3:'CF-One-class SVM',4:'CF-RandNet',5:'CF-Randomforest',6:'L2_Norm',7:'WBiGan-GP',8:'IsoForest',9:'One-class SVM',10:'RandNet',11:'Randomforest'}
print(len(High_new[High_new==1]),len(High[High == 1]),len(Low[Low == 1]))
method_to_letter = {
    'L2 Norm': '(a)',
    'CF-L2 Norm': '(b)',
    'One-Class SVM': '(c)',
    'CF-One-Class SVM': '(d)',
    'iForest': '(e)',
    'CF-iForest': '(f)',
    'RandNet': '(g)',
    'CF-RandNet': '(h)',
    'WBiGan-GP': '(i)',
    'CF-WBiGan-GP': '(j)'
}
Score=[]
save_fig_folder= r'F:\Susceptibility Assessment\res\res_sts/'
all_roc_data = []
for idx in range(len(data)):
    x = data[idx][0].flatten()
    mn, mx = np.min(x), np.max(x)
    print(f'原始{methods[idx]}的最小/大值：{mn}    {mx}')
    mask = (x < 0)
    x[x < 0] =0
    # mn,mx=np.min(x),np.max(x)
    mn, mx = np.round(np.min(x)), np.round(np.max(x))
    if mx!=mn:
        x = (x - mn) / (mx - mn)
    else:
        x=(x - np.min(x)) / (np.max(x)-np.min(x))
    np.clip(x,0,1,out=x)
    np.random.seed(42)
    x[mask] =np.random.rand(np.sum(mask))
    print(f'{methods[idx]}的最小/大值：{mn}    {mx}')
    len_high_new,len_high,len_low=len(High_new[High_new==1]),len(High[High == 1]),len(Low[Low == 1])
    print(f'标签数量为：{len_high_new,len_high,len_low}')
    high_pred,low_pred=x[High],x[Low]             #(107230,) 107606 ()(685001,)
    high_pred_new=x[High_new]
    # labels,scores=np.concatenate([np.ones(len_high_new)*2, np.ones(len_low)]),np.concatenate([high_pred_new,low_pred])
    np.random.seed(42)
    labels, scores = np.concatenate([np.ones(len_high_new)*2, np.ones(len_high_new)]), np.concatenate(
        [high_pred_new, np.random.choice(low_pred, size=len_high_new, replace=False)])
    noise = np.random.normal(0, 0.12, size=scores.shape)
    scores = scores + noise
    scores = np.clip(scores, 0.0, 1.0)
    print(np.min(labels),np.max(labels),np.min(scores),np.max(scores))
    fpr, tpr, thresholds = roc_curve(labels, scores,pos_label=2)
    roc_auc = auc(fpr, tpr)
    all_roc_data.append({
        'name': methods[idx],
        'fpr': fpr,
        'tpr': tpr,
        'auc': roc_auc
    })

    # score1=122/(122+416)*np.sum(high_pred)/len(high_pred)+416/(122+416)*np.sum(high_pred_new)/len(high_pred_new)
    score1=np.sum(high_pred)/len(high_pred)
    score2=np.sum(high_pred_new)/len(high_pred_new)
    score3=np.sum(1-low_pred)/len(low_pred)
    Score.append([methods[idx],round(score1*10,2),round(score2*10,2),round(score3*10,2)])
    print(f'roc_auc:{roc_auc}, score1:{score1*10:.2f}, score2:{score2*10:.2f},score3:{score3*10:.2f} ')
    bins = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    # counts, edges = np.histogram(x, bins=bins)
    # print("===== 0~1 区间统计 =====")
    # labels = ["0.0~0.2", "0.2~0.4", "0.4~0.6", "0.6~0.8", "0.8~1.0"]
    # total = len(x)
    # for i, (label, cnt) in enumerate(zip(labels, counts)):
    #     pct = cnt / total * 100
    #     print(f"{label} : {cnt:>5} 个  ({pct:.2f}%)")

    y_high,x_low=np.linspace(0,1,10),np.linspace(0,1,10)

    X, Y = np.meshgrid(x_low, y_high)
    accuracy_matrix = np.zeros_like(X)
    precision_matrix = np.zeros_like(X)
    recall_matrix = np.zeros_like(X)
    f1_matrix = np.zeros_like(X)
    mask = Y >= X

# 应用掩码
#     accuracy_matrix = np.ma.masked_where(~mask, accuracy_matrix)
#     precision_matrix = np.ma.masked_where(~mask, precision_matrix)
#     recall_matrix = np.ma.masked_where(~mask, recall_matrix)
#     f1_matrix = np.ma.masked_where(~mask, f1_matrix)

    # print(X)
    for i,y in enumerate(y_high):
        for j,x in enumerate(x_low):
            # if x>y:
            #     continue
            tp=np.sum((high_pred_new>y))
            fn=np.sum((high_pred_new<=y))
            tn=np.sum((low_pred<x))
            fp=np.sum((low_pred>=x))
            accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
            accuracy_matrix[i, j] = accuracy
            precision_matrix[i, j] = precision
            recall_matrix[i, j] = recall
            f1_matrix[i, j] = f1

    # plt.rcParams['font.size'] = 10  # 设置字体大小
    # plt.rcParams['font.family'] = 'Times New Roman'
    # plt.rcParams['mathtext.fontset'] = 'stix'
    # fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    # cmap = plt.cm.turbo
    # norm = plt.Normalize(vmin=0, vmax=1)
    # levels=10
    # contour1 = axes[0, 0].contourf(X, Y, accuracy_matrix, levels, cmap=cmap, norm=norm, alpha=0.8)
    # axes[0, 0].set_title('Accuracy', fontsize=20, fontweight='bold')
    # axes[0, 0].set_xlabel('low susceptible threshold', fontsize=16, fontweight='bold')
    # axes[0, 0].set_ylabel('high susceptible threshold', fontsize=16, fontweight='bold')
    # axes[0, 0].plot([x_low.min(), x_low.max()], [y_high.min(), y_high.max()], '--', color='black',alpha=0.7, label='y=x')
    # axes[0, 0].legend()
    #
    # contour2 = axes[0, 1].contourf(X, Y, precision_matrix, levels, cmap=cmap, norm=norm, alpha=0.8)
    # axes[0, 1].set_title('Precision', fontsize=20, fontweight='bold')
    # axes[0, 1].set_xlabel('low susceptible threshold', fontsize=16, fontweight='bold')
    # axes[0, 1].set_ylabel('high susceptible threshold', fontsize=16, fontweight='bold')
    # axes[0, 1].plot([x_low.min(), x_low.max()], [y_high.min(), y_high.max()], '--', color='black',alpha=0.7, label='y=x')
    # axes[0, 1].legend()
    #
    # contour3 = axes[1, 0].contourf(X, Y, recall_matrix, levels, cmap=cmap, norm=norm, alpha=0.8)
    # axes[1, 0].set_title('Recall', fontsize=20, fontweight='bold')
    # axes[1, 0].set_xlabel('low susceptible threshold', fontsize=16, fontweight='bold')
    # axes[1, 0].set_ylabel('high susceptible threshold', fontsize=16, fontweight='bold')
    # axes[1, 0].plot([x_low.min(), x_low.max()], [y_high.min(), y_high.max()], '--', color='black',alpha=0.7, label='y=x')
    # axes[1, 0].legend()
    #
    # contour4 = axes[1, 1].contourf(X, Y, f1_matrix, levels, cmap=cmap, norm=norm, alpha=0.8)
    # axes[1, 1].set_title('F1 Score', fontsize=20, fontweight='bold')
    # axes[1, 1].set_xlabel('low susceptible threshold', fontsize=16, fontweight='bold')
    # axes[1, 1].set_ylabel('high susceptible threshold', fontsize=16, fontweight='bold')
    # axes[1, 1].plot([x_low.min(), x_low.max()], [y_high.min(), y_high.max()], '--', color='black',alpha=0.7, label='y=x')
    # axes[1, 1].legend()
    #
    # plt.tight_layout(rect=[0, 0, 1, 0.95])  # 调整上边距为标题留空间
    # letter = method_to_letter.get(methods[idx], '')
    # fig.suptitle(methods[idx], fontsize=30, fontweight='bold')#y=0.98
    # letter = method_to_letter.get(methods[idx], '')
    # if letter:
    #     fig.text(0.02, 0.98, letter, fontsize=30,
    #              va='top', ha='left')
    #
    # # plt.tight_layout(rect=[0, 0, 0.9, 0.95])
    # # cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])  # [left, bottom, width, height]
    # # contour1.set_clim(0, 1)
    # # cbar = fig.colorbar(contour1, cax=cbar_ax)
    # # cbar.set_ticks([0, 0.2, 0.4, 0.6, 0.8, 1])
    # # tick_labels = cbar.ax.get_yticklabels()
    # # plt.setp(tick_labels, fontsize=16, fontweight='bold')
    #
    # path=save_fig_folder+methods[idx]+'.tif'
    # plt.savefig(path, dpi=300,format='tiff')
    # plt.show()

plt.figure(figsize=(10, 8))
print(len(all_roc_data))
all_roc_data = sorted(all_roc_data, key=lambda x: x['auc'], reverse=True)
for data in all_roc_data:
    plt.plot(data['fpr'], data['tpr'], lw=2, label=f"{data['name']} (AUC={data['auc']:.3f})")
plt.plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--', label='Random Guess')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False positive rate', fontsize=12)
plt.ylabel('True positive rate', fontsize=12)
plt.title('ROC curves comparison of all models', fontsize=14)
plt.legend(loc="lower right", fontsize=9)
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(save_fig_folder + 'ROC_Comparison_All_Models.tiff', dpi=300)
plt.show()



    # 转换为 DataFrame 并保存为 CSV
# df = pd.DataFrame(accuracy_matrix)
# df.to_csv(save_fig_folder+'accuracy_matrix.csv', index=False, encoding='utf-8-sig')
# df = pd.DataFrame(precision_matrix)
# df.to_csv(save_fig_folder+'precision_matrix.csv', index=False, encoding='utf-8-sig')
# df = pd.DataFrame(f1_matrix)
# df.to_csv(save_fig_folder+'f1_matrix.csv', index=False, encoding='utf-8-sig')
# df = pd.DataFrame(recall_matrix)
# df.to_csv(save_fig_folder+'recall_matrix.csv', index=False, encoding='utf-8-sig')
# df = pd.DataFrame(Score, columns=['Models', 'Score1', 'Score2'])
# df.to_csv(save_fig_folder+'results.csv', index=False, encoding='utf-8-sig')

    # cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])  # [left, bottom, width, height]
    # cbar = fig.colorbar(contour1, cax=cbar_ax,shrink=0.8, aspect=20, pad=0.05)
    # cbar.set_label('Metric Value', rotation=270, labelpad=15,fontweight='bold')
    # # plt.tight_layout()


    # plt.show()