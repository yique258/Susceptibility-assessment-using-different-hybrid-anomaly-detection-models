import numpy as np
from sklearn.model_selection import train_test_split

class Dataloader:
    def __init__(self, x: np.ndarray, y: np.ndarray,cf_x: np.ndarray):
        y=y.reshape(1, 1, 4909, 8050)
        self.mapx = x
        self.mapy = y
        self.height,self.width=y.shape[2],y.shape[3]
        self.cf_x=cf_x
        np.squeeze(x)
        np.squeeze(y)
        print(self.height,self.width)
        print(f'x.shape{x.shape} y.shape{y.shape}')
    def load_raw(self,data_vals=[4],test_size=0.3,random_state=42):
        """
        Args:
            data_vals (list): 包含有效y值的列表
            test_size (float): 测试集比例,默认0.2
            random_state (int): 随机种子,默认42
        
        Returns:
            x_train, x_test, y_train, y_test: 分割后的数据集
        """
        valid_mask = np.isin(self.mapy[0, 0],data_vals)
        channels = self.mapx.shape[0]
        x_flat = self.mapx[:, 0].reshape(channels, -1).T
        x = x_flat[valid_mask.flatten()]
        if len(x) == 0:
            raise ValueError("未找到匹配data_vals的有效像素点")

        return x

    def load_cf(self, data_vals=[4]):
            """
            Args:
                cf_factors:全部cf_factors

            Returns:
                 x_train, x_test, y_train, y_test: 分割后的数据集
            """
            valid_mask = np.isin(self.mapy[0, 0],data_vals)
            x = self.cf_x[valid_mask.flatten()]
            return  x