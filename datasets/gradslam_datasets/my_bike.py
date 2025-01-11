import os
import glob
import yaml
import numpy as np
import cv2
import torch


# 定义默认设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class MyBikeDataset:
    def __init__(self, basedir, sequence, gradslam_data_cfg, n=None, device=device):
        self.dataset = []
        self.device = device
        self._load_camera_matrix(gradslam_data_cfg)
        self._load_images(basedir, sequence, n)

    def _load_camera_matrix(self, file_path):
        with open(file_path, 'r') as file:
            data = yaml.safe_load(file)
            # 假设内参矩阵存储在 'array' 键下
            camera_matrix = data.get('array', None)
            if camera_matrix is None:
                raise ValueError("YAML 文件中没有找到 'array' 键")
            # 将列表转换为 PyTorch 张量，并指定设备
            self.k = torch.tensor(np.array(camera_matrix), dtype=torch.float32).to(self.device)

    def _load_images(self, basedir, sequence, n=None):
        rgb_paths = []
        depth_paths = []

        # 读取RGB图像路径
        rgb_path = os.path.join(basedir, sequence, 'RGB')
        for extension in ["jpg", "png", "jpeg"]:
            rgb_paths += glob.glob(os.path.join(rgb_path, "*.{}".format(extension)))

        # 读取深度图像路径
        depth_path = os.path.join(basedir, sequence, 'depth')
        for extension in ["png", "jpg", "jpeg"]:
            depth_paths += glob.glob(os.path.join(depth_path, "*.{}".format(extension)))

        # 确保路径按自然排序
        rgb_paths.sort(key=lambda f: int(''.join(filter(str.isdigit, os.path.basename(f)))))
        depth_paths.sort(key=lambda f: int(''.join(filter(str.isdigit, os.path.basename(f)))))

        # 如果指定了 n，则限制加载的图像数量
        if n is not None:
            rgb_paths = rgb_paths[:n]
            depth_paths = depth_paths[:n]

        # 加载图像数据到dataset
        for rgb_file, depth_file in zip(rgb_paths, depth_paths):
            color = cv2.imread(rgb_file)
            depth = cv2.imread(depth_file, cv2.IMREAD_GRAYSCALE)  # 读取深度图时不改变其原始格式

            # 将颜色图像从 BGR 转换为 RGB，并转换为 PyTorch 张量 (C, H, W)，并指定设备
            color_tensor = torch.tensor(cv2.cvtColor(color, cv2.COLOR_BGR2RGB)).to(self.device)

            # 处理深度图像，假设深度值为 uint16 并转换为 float32，并指定设备
            depth_tensor = torch.tensor(depth, dtype=torch.float32).to(self.device)
            depth_tensor = depth_tensor.unsqueeze(2)
            # 添加到数据集
            self.dataset.append((color_tensor, depth_tensor, self.k))

    def __getitem__(self, idx):
        return self.dataset[idx]

    def __len__(self):
        return len(self.dataset)

if __name__ == "__main__":
    basedir = '/path/to/dataset'
    sequence = 'your_sequence_name'
    gradslam_data_cfg = 'camera_params.yaml'

    dataset = MyBikeDataset(basedir, sequence, gradslam_data_cfg)

    # 获取第一帧的数据
    color, depth, k = dataset[0]

    print("Color image shape:", color.shape)
    print("Depth image shape:", depth.shape)
    print("Camera intrinsics matrix:\n", k)