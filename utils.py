import torch
from collections import Counter
import os
import geopandas as gpd
import rasterio
from rasterio.features import rasterize
from matplotlib import pyplot as plt
from pathlib import Path
from rasterio.features import shapes
from collections import Counter
from matplotlib import pyplot as plt
import rasterio
import spectral
from pathlib import Path
import numpy as np
from rasterio.windows import Window
from rasterio import mask
from rasterio.merge import merge
from shapely.geometry import box
from rasterio.warp import calculate_default_transform, reproject, Resampling
from collections import Counter
import torch
import torch.nn as nn
import torch.nn.functional as F
import rasterio
from rasterio.transform import from_origin
import matplotlib.pyplot as plt
from torch.utils.data import TensorDataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
import sys

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
plt.rcParams['font.sans-serif'] = ['SimHei']  # 设置中文字体
plt.rcParams['axes.unicode_minus'] = False  # 显示负号

class metrics:
    '''
    计算指标类
    输入:preds: 预测值  masks: 真实值   tesnor类型(batch, channel, height, width)
    输出:iou, accuracy, precision, recall, f1_score
    '''
    def __init__(self,preds, masks,epoch):
        self.intersection = (preds * masks).sum((1, 2, 3))  
        self.union = (preds + masks).clamp(0, 1).sum((1, 2, 3))  
        self.TP = (preds * masks).sum((1, 2, 3))  
        self.FP = ((preds - masks) > 0).float().sum((1, 2, 3))
        self.FN = ((masks - preds) > 0).float().sum((1, 2, 3))
        self.TN = ((preds - masks) < 0).float().sum((1, 2, 3))
        self.epoch=epoch
    def iou(self):
        iou = ((self.intersection / (self.union + 1e-6))).mean().item()
        return iou
    def accuracy(self):
        acc = ((self.TP + self.TN) / (self.TP + self.FP + self.FN + self.TN)).mean().item()
        return acc
    def precision(self):
        precision = (self.TP / (self.TP + self.FP)).mean().item()
        return precision
    def recall(self):
        recall = (self.TP / (self.TP + self.FN )).mean().item()
        return recall
    def f1_score(self):   
        f1= 2 * (self.precision() * self.recall()) / (self.precision() + self.recall()+1e-6)
        return f1
    def display_all_metrics(self):
        iou = self.iou()
        acc = self.accuracy()
        precision = self.precision()
        recall = self.recall()
        f1 = self.f1_score()
        print(f'epoch: {self.epoch} mIoU: {iou:.4f}, Accuracy: {acc:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1 Score: {f1:.4f}')


class geodata:
    '''
    '''
    def read_tifs(directory,rglob=False,channel=-1,nodata=-10000):
        """
        读取指定目录下所有的 TIFF 文件，并将它们的数据和[crs,Transform]存储在两个个列表中。
    Args:
        directory (str): 包含 TIFF 文件的目录路径、rglob是否包含子文件夹、想要读取的波段(可选，不填读取全部波段)。
        rglob=False
        nodata=-10000
        roi_path=None裁剪的shp路径
    Returns:
        list: 一个包含 NumPy 数组的列表，每个数组代表一个 TIFF 文件的数据。float32格式,原始数据未归一化!!!
        list: 一个包含 profiles的列表。
            如果目录中没有找到任何 TIFF 文件，则返回两个空列表。
        """

        data = []
        profiles=[]
        tif_files = Path(directory).glob("*.tif") if rglob==False else Path(directory).rglob("*.tif")

        if not tif_files:
            print(f"在目录 {directory} 中没有找到任何 TIFF 文件。")
            return data,profiles
        crs_set =set()
        transform_set = set()
        for tif_file in tif_files:
            with rasterio.open(tif_file) as src:

                if channel != -1:
                    arr = src.read(channel)
                else:
                    arr = src.read()  # 读取所有波段的数据
                nodata_val = src.nodata
                if nodata_val is not None:
                    arr = np.where(arr == nodata_val, nodata, arr)
                data.append(arr)
                crs_set.add(src.crs.to_epsg())
                transform_set.add(src.transform)
                profiles.append(src.profile.copy())  # 复制文件的元数据
                print(f"成功读取文件: {tif_file} EPSG Code: {src.crs}  Affine Transform: {src.transform}")
        if len(crs_set) > 1 or len(transform_set) > 1:
            print(f'crs_set:{len(crs_set)}')
            print(crs_set)
            print(f'transform:{len(transform_set)}')
            print(transform_set)
            print("⚠️ 警告：部分tif文件的投影坐标系或仿射变换不一致，请检查数据！")
            # return  data,profiles
            # sys.exit(1)
        data = np.array(data).astype(np.float32)
        print(f'nodata={nodata}')
        return data, profiles 

    def load_data(image_path, mask_path):    
        images,profiles = geodata.read_tifs(image_path)     
        masks=np.load(mask_path)
        masks=np.expand_dims(masks, axis=1)
        images.astype(np.float32)
        masks.astype(np.float32)
        print(f'影像：{images.shape}----标签：{masks.shape}')
        print(f'影像：{images.dtype}----标签：{masks.dtype}')
        return torch.tensor(images)/255,profiles, torch.tensor(masks)
    

    def shp2tif(shp_path,tif_path,origin_tiff_image,value_field=None,data_val=1,nodata=0):
        """
        将 Shapefile的标签转换为 TIFF 格式的栅格数据。
        参数:
            shp_path (str): 输入的 标签Shapefile 文件路径。
            tif_path (str): 输出的 变迁TIFF 文件路径。
            origin_tiff_image (str): 原始tiff影像的路径。
        """
        os.environ['SHAPE_RESTORE_SHX'] = 'YES'
        os.makedirs(os.path.dirname(tif_path), exist_ok=True) 
        with rasterio.open(origin_tiff_image) as src:
            origin_width = src.width
            origin_height = src.height
            origin_transform = src.transform
            origin_crs = src.crs
            origin_cell_size=src.res[0]  # 获取原始影像的分辨率
            print(f"origin_width: {origin_width}, origin_height: {origin_height}, origin_cell_size: {origin_cell_size}")


        gdf = gpd.read_file(shp_path,SHAPE_RESTORE_SHX=True)  # 读取 SHP 文件
        if gdf.crs != origin_crs:
            print(f"❌ 输入 SHP 坐标系与原始影像坐标系不一致，正在转换为原始影像坐标系...")
            gdf = gdf.to_crs(origin_crs)
        print(gdf.crs)  # 查看数据对应的投影信息
        print(gdf.head())  # 查看数据的前几行
        print(gdf.columns)  # 查看数据的字段信息
        gdf.plot()
        plt.show()
        

    # 边界不用设置，直接和原始影像对齐，方便后面的save_index
        cell_size = origin_cell_size
        cols = origin_width
        rows = origin_height
        transform = origin_transform

        # gdf[value_field] = gdf[value_field].fillna(0)
        # shape = ((geom, value) for geom, value in zip(gdf.geometry, gdf[value_field]))  # 多边形 + 值
        shape = (
        ((geom, data_val) for geom in gdf.geometry) if value_field is None 
        else ((geom, value) for geom, value in zip(gdf.geometry, gdf[value_field])))

        raster_data = rasterize(
            shapes=shape,
            out_shape=(rows,cols),
            transform=transform,
            fill=nodata,  # 默认值填充 0
            all_touched=True,  # 让每个像元包含的多边形都参与计算
            dtype=np.float32  # 数据类型
    )
        # 此处多值有问题
        with rasterio.open(
            tif_path, 
            "w",
            driver="GTiff",
            height=rows,
            width=cols,
            count=1,
            dtype=raster_data.dtype,
            crs=gdf.crs,  # 使用投影坐标系
            transform=transform,
            nodata=nodata
        ) as dst:
            dst.write(raster_data,1)  # 写入栅格数据,band=1

        print(f"✅ 成功生成 {cell_size}m 栅格数据: {tif_path}")

# **10. 可视化栅格数据**
        plt.imshow(raster_data, cmap='gray')
        plt.show()
        # print(Counter(raster_data.flatten()))
        raster_data = np.expand_dims(raster_data, axis=2)  # 扩展维度   # 使其成为 (rows, cols, 1) 的形状 1是band
        print(raster_data.shape)
        np.save(tif_path.replace('.tif','.npy'),raster_data)  

    def tif2shp(tif_path, shp_path, target_value=None,channel=1):

        """
        将TIFF文件中指定值的区域转换为SHP文件
        参数:
        tif_path: 输入的TIFF文件路径
        shp_path: 输出的SHP文件路径
        target_value: 要提取的值,默认为none提取整个边界
        channel: 要提取的波段,默认为1
        """
        with rasterio.open(tif_path) as src:
            image = src.read(channel)  # 读取第一个波段
            transform = src.transform
            crs = src.crs
        
        # 创建一个掩膜，只保留目标值
        if target_value is None:
            mask = image != src.nodata if src.nodata else None
        else:
            mask = (image == target_value)
        results = (
            {'properties': {'value': v}, 'geometry': s}
            for i, (s, v) in enumerate(shapes(image, mask=mask, transform=transform))
        )
        gdf = gpd.GeoDataFrame.from_features(list(results), crs=crs)
        gdf.to_file(shp_path)
        print(f"✅ {tif_path}-->{shp_path}")
    def masknpy_segment_return_save_index(mask_path:str,tile_size:int,output_dir:str):
        """
        将掩膜分割为小图块，并保存索引
        参数:
            mask_path: 整块的掩膜文件路径,npy格式 
            output_dir: 输出目录
            tile_size: 小图块的大小

            return: 分割后的掩膜数据tiles.npy和索引save_index.npy
        """
        os.makedirs(output_dir, exist_ok=True) 
        data=np.load(mask_path)
        print(f'data.shape: {data.shape}')
        width, height,bands=data.shape
        save_index=[]
        percent_cnt=[0 for i in range(101)]   #percent_cnt[0]表示有效面积为0%~0.49%的样本的个数
        cols = (width + tile_size - 1) // tile_size
        rows = (height + tile_size - 1) // tile_size
        tiles=np.array([])
        for row in range(rows):
            for col in range(cols):
                xoff = col * tile_size
                yoff = row * tile_size
                # 如果是最后一行/列，调整偏移量确保不会超出边界
                xoff = max(0, min(xoff, width - tile_size))
                yoff = max(0, min(yoff, height - tile_size))  

                tile=data[yoff:yoff+tile_size,xoff:xoff+tile_size,:]

                cnt=Counter(tile.flatten())
                percent_index = int(round(cnt[1] / (tile_size * tile_size) * 100))
                percent_cnt[percent_index] += 1
                if cnt[1]!=0: # 筛选数据，过滤掉数据严重不平衡的样本
                    save_index.append((row,col))
                    tile=np.transpose(tile,(2,0,1))
                    if tiles.size==0:
                        tiles=tile
                    else:
                        tiles=np.concatenate((tiles,tile),axis=0)
        print(tiles.shape)
        print(percent_cnt)
        plt.bar([i for i in range(101)], percent_cnt)
        plt.ylim(0, 100)  
        plt.xlabel('有效面积百分比 (%)')
        plt.ylabel('样本数量')
        plt.xticks([i for i in range(0, 101, 10)])  # 设置 x 轴刻度
        plt.title('分布柱状图')
        for i, v in enumerate(percent_cnt):
            plt.text(i, v / 2, f"{v}", ha='center', va='center', color='white')
        plt.grid(axis='y', linestyle='--', alpha=0.7)  # 添加网格线
        plt.show()
        plt.savefig(output_dir+'/percent_cnt'+str(tile_size)+'.png', bbox_inches='tight')


        save_index=np.array(save_index)
        tiles_filename = output_dir+'/tiles.npy'
        save_index_filename = output_dir+'/save_index.npy'
        print(save_index)
        print(save_index.shape)
        np.save(tiles_filename, tiles)
        np.save(save_index_filename, save_index)
        print(f"✅ mask_segement数据: {tiles_filename} 位置标志数据: {save_index_filename}")
    
    def tiff_segemnt(input_path, output_dir, tile_size=256, save_index=None):
        """
        切割大TIFF图像为小图块,边角不足时与前面图块重叠
    
        参数:
            input_path: 输入TIFF文件路径
            output_dir: 输出目录
            tile_size: 目标图块大小(宽度和高度相同)
        """
        os.makedirs(output_dir, exist_ok=True)

        with rasterio.open(input_path) as src:
            profile = src.profile.copy()
            width, height = src.width, src.height
            if save_index is not None:
                save_index = np.load(save_index)
                save_index=save_index.tolist()
                print(f'save_index: {save_index}')
            # 计算行列数
            cols = (width + tile_size - 1) // tile_size
            rows = (height + tile_size - 1) // tile_size
        
            for row in range(rows):
                for col in range(cols):
                    if save_index and [row,col] not in save_index:
                        continue
                    xoff = col * tile_size
                    yoff = row * tile_size
                
                # 如果是最后一行/列，调整偏移量确保不会超出边界
                    xoff = max(0, min(xoff, width - tile_size))
                    yoff = max(0, min(yoff, height - tile_size))
                
                    window = Window(xoff, yoff, tile_size, tile_size)
                    data = src.read(window=window)
                    win_transform = src.window_transform(window)
                
                # 更新元数据
                    profile.update({
                        'width': tile_size,
                        'height': tile_size,
                        'transform': win_transform,
                        'driver': 'GTiff',
                        'dtype': data.dtype,
                        'crs': src.crs,
                        'count': src.count,
                        'nodata': src.nodata,
                    })
                
                    output_path = os.path.join(output_dir, f"tile_{row}_{col}_{tile_size}.tif")
                
                # 写入文件
                    with rasterio.open(output_path, 'w', **profile) as dst:
                        dst.write(data)
                    print(f"生成: {output_path} (窗口: {window})")
    def clip_tif_by_shp(tif_path, shp_path, output_path):
        """
        使用 Shapefile 裁剪 TIFF 文件
        参数:
            tif_path: 输入的 TIFF 文件路径(输出crs与此一致)
            shp_path: 提供裁剪范围的 TIFF 文件路径
            output_path: 输出的裁剪后 TIFF 文件路径
        """
      
        with rasterio.open(tif_path) as src:
            gdf=gpd.read_file(shp_path)  
            gdf = gdf.to_crs(src.crs)  # 确保坐标系一致
            shapes=gdf.geometry.to_list()
            out_image, out_transform = mask.mask(src, shapes, crop=True,all_touched=True,indexes=None)
            out_meta = src.meta.copy()
        
        out_meta.update({
            "driver": "GTiff",
            "height": out_image.shape[1],
            "width": out_image.shape[2],
            "transform": out_transform,
            'count': src.count
        })
        
        with rasterio.open(output_path, "w", **out_meta) as dest:
            dest.write(out_image)

    def shp2kml(shp_path,kml_path):
        gdf = gpd.read_file(shp_path)  
        gdf.to_crs(epsg=4326).to_file(kml_path, driver="KML")
        print(f"✅ shp转kml成功: {kml_path}")
    def mergetif(in_dir, out_path=None):
        if out_path is None:
            output_dir = Path(in_dir) / "output"
            output_dir.mkdir(exist_ok=True)
            out_path = str(output_dir / "merged.tif")
    # 读取所有的 tif 文件
        src_files_to_mosaic = []
        files = Path(in_dir).glob("*.tif") 
        for file in files:
            src = rasterio.open(file)
            src_files_to_mosaic.append(src)

        merged_data, merged_transform = merge(src_files_to_mosaic,)
    # 更新元数据
        out_meta =src.profile.copy()
        out_meta.update({
            "driver": "GTiff",
            "height": merged_data.shape[1],
            "width": merged_data.shape[2],
            "transform": merged_transform,
            'count': merged_data.shape[0],  
            'dtype': np.uint8,
            'nodata': 0,  # 可选：设置NoData值
        })

        with rasterio.open(out_path, "w+", **out_meta) as dest:
            dest.write(merged_data)
        print(f"✅ tif合并成功: {out_path}")
    
    def reproject_tif(tif_path, output_path,target_tif_path=None,resampling=Resampling.bilinear):
        '''
            重投影tif,保证拥有相同的crs、res、bounds
            参数：
                tif_path: 输入的 TIFF 文件路径
                output_path: 输出的重投影后 TIFF 文件路径
                target_tif_path: 投影到此图像上
                resampling: 重投影时的采样方法
        '''
    
        with rasterio.open(target_tif_path) as a:
                dst_crs=a.crs
                dst_transform=a.transform
                dst_shp=a.shape
                profile = a.profile.copy()
                print(f'a.shape: {a.shape}')
        print(f'dst_crs: {dst_crs}, dst_transform: {dst_transform}, dst_shp: {dst_shp}')
        with rasterio.open(tif_path) as src:
            profile.update({
                'nodata': 0,
                'count': src.count,
                'dtype': src.dtypes[0]
            })
        with rasterio.open(tif_path) as src,rasterio.open(output_path,'w', **profile) as dst:
            dst_array = np.zeros((src.count,*dst_shp), dtype=profile['dtype'])
            reproject(
                source=rasterio.band(src, range(1, src.count + 1)),
                src_crs=src.crs,
                src_transform=src.transform,
                src_shape=src.shape,

                destination=dst_array,
                dst_transform=dst_transform,
                dst_crs=dst_crs,
                dst_shape=dst_shp,
                resampling=resampling
            )
            dst.write(dst_array)

        print(f"✅ tif重投影成功: {output_path}")
        with rasterio.open(output_path) as src:
            print(f'src.crs: {src.crs}')


    def mosic_two_tifs(tif_path1, tif_path2, output_path, mosic_method='stack'):

        """
        将两幅相同特性的TIFF图像嵌合在一起
    参数:
            tif_path1: 第一幅TIFF路径
            tif_path2: 第二幅TIFF路径
            output_path: 输出文件路径
            mosic_method: 嵌合方法 ('stack'波段叠加, 'average'平均值, 'max'最大值等)
        """
        with rasterio.open(tif_path1) as src1, rasterio.open(tif_path2) as src2:
        # 验证两幅图像是否具有相同的CRS、transform和尺寸
            if (src1.crs != src2.crs or 
                src1.transform != src2.transform or 
                src1.width != src2.width or 
                src1.height != src2.height):
                raise ValueError("输入的TIFF文件不具有相同的CRS、transform或尺寸")
            data1 = src1.read()
            data2 = src2.read()
        
            if mosic_method == 'stack':
                mosic_data = np.vstack((data1, data2))
            elif mosic_method == 'average':
                mosic_data = np.mean(np.array([data1, data2]), axis=0)
            elif mosic_method == 'max':
                mosic_data = np.maximum(data1, data2)
            elif mosic_method == 'min':
                mosic_data = np.minimum(data1, data2)
            elif mosic_method == 'sum':
                mosic_data = data1 + data2
            else:
                raise ValueError(f"不支持的合并方法: {mosic_method}")
        
            print(f"嵌合后的数据形状：{mosic_data.shape}")
            out_meta = src1.meta.copy()
            if mosic_method == 'stack':
                out_meta.update(count=src1.count + src2.count)
        
        # 写入输出文件
            with rasterio.open(output_path, 'w', **out_meta) as dst:
                dst.write(mosic_data)
    
        print(f"成功嵌合图像并保存到: {output_path}")


def train(model,criterion=None,optimizer=None,dataloader=None,epochs=200,writer=None,save_dir=None):
    
    model.train()
    for epoch in range(epochs):
        for batch_x,batch_y in dataloader:
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        with torch.no_grad():
            outputs = (torch.sigmoid(outputs) > 0.5).float()
            m=metrics(outputs, batch_y,epoch)
            m.display_all_metrics()
            writer.add_scalar('Loss/train', loss.item(), epoch)
            writer.add_scalar('IoU/train', m.iou(), epoch)
            writer.add_scalar('Accuracy/train', m.accuracy(), epoch)
            writer.add_scalar('Precision/train', m.precision(), epoch)
            writer.add_scalar('Recall/train', m.recall(), epoch)
            writer.add_scalar('F1/train', m.f1_score(), epoch)
    os.makedirs(save_dir, exist_ok=True)
    torch.save(model, f"{save_dir}/{model.__class__.__name__}.pth")
    print(f"Saved PyTorch Model State to {save_dir}{model.__class__.__name__}.pth")

def test(model_path,x_test,x_test_profiles,output_dir):
    os.makedirs(output_dir, exist_ok=True)
    model=torch.load(model_path,weights_only=False)
    model.eval()
    with torch.no_grad():
        for i in range(x_test.shape[0]):
            x=x_test[i]
            x = np.expand_dims(x, axis=0)  # 添加批次维度
            x=torch.tensor(x).to(device)
            outputs = model(x)
            outputs = np.where(torch.sigmoid(outputs).cpu().numpy() > 0.5, 1, 0)
            output_path = os.path.join(output_dir, f'prediction_{i}.tif')
            x_test_profiles[i].update({
                    'dtype': outputs.dtype,
                    'count': 1,
                    'nodata': 0,
                })
            with rasterio.open(output_path,'w',**x_test_profiles[i]) as dst:
                dst.write(outputs[0, 0], 1)
            print(f"Saved prediction{i} to {output_path}")





if __name__ == '__main__':

    # preds = torch.tensor([[[[1, 0], [0, 1]], [[0, 1], [1, 0]]],[[[1, 0], [0, 1]], [[0, 1], [1, 0]]]]).to(torch.float32)
    # masks = torch.tensor([[[[1, 0], [1, 1]], [[0, 1], [1, 0]]],[[[1, 0], [1, 1]], [[0, 1], [1, 0]]]]).to(torch.float32)
    # metric = metrics(preds, masks)
    # metric.all_metrics()
    # input_tif = 'res256.tif'  # 输入的TIFF文件路径
    # output_shp = 'res256.shp'  # 输出的SHP文件路径
    # geodata.tif2shp(input_tif, output_shp)
    # geodata.mask_segment_with_save_index('./raster_data.npy',512,'./tiles512')
    # tile_size=64
    # input_tiff = r'E:\glacier_extract\DATA_LIUYING\24_summer_roi.tif'
    # geodata.tiff_segemnt(input_tiff,'./image128',128,save_index=r'E:\glacier_extract\u_net\tiles128\save_index.npy')
    # geodata.tiff_segemnt(input_tiff,'full_image_test128',tile_size=128)
    print('utils.py')
    

    # in_dir=r'E:\Sardata\demdata'
    # geodata.mergetif(in_dir)

