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
    def load_stereo_coefficients(self,path):
        """ Loads stereo matrix coefficients. """
        # FILE_STORAGE_READ
        cv_file = cv2.FileStorage(path, cv2.FILE_STORAGE_READ)

        # note we also have to specify the type to retrieve other wise we only get a
        # FileNode object back instead of a matrix
        K1 = cv_file.getNode("K1").mat()
        D1 = cv_file.getNode("D1").mat()
        K2 = cv_file.getNode("K2").mat()
        D2 = cv_file.getNode("D2").mat()
        R = cv_file.getNode("R").mat()
        T = cv_file.getNode("T").mat()
        E = cv_file.getNode("E").mat()
        F = cv_file.getNode("F").mat()
        R1 = cv_file.getNode("R1").mat()
        R2 = cv_file.getNode("R2").mat()
        P1 = cv_file.getNode("P1").mat()
        P2 = cv_file.getNode("P2").mat()
        Q = cv_file.getNode("Q").mat()

        cv_file.release()
        return [K1, D1, K2, D2, R, T, E, F, R1, R2, P1, P2, Q]
    def _load_camera_matrix(self, file_path):
            # 载入相机的参数
        K1, D1, K2, D2, R, T, E, F, R1, R2, P1, P2, Q = self.load_stereo_coefficients(file_path)

        self.k = torch.tensor(np.array(K1), dtype=torch.float32).to(self.device)
        self.k_r = torch.tensor(np.array(K2), dtype=torch.float32).to(self.device)


    def _load_images(self, basedir, sequence, n=None):
        rgb_paths = []
        depth_paths = []

        # 读取RGB图像路径
        rgb_path = os.path.join(basedir, sequence, 'RGB')
        for extension in ["jpg", "png", "jpeg"]:
            rgb_paths += glob.glob(os.path.join(rgb_path, "*.{}".format(extension)))

        # 读取深度图像路径
        baseline = 0.07
        k = self.k.cpu() if self.k.is_cuda else self.k
        k_r = self.k_r.cpu() if self.k_r.is_cuda else self.k_r
        # 提取元素并转换为 NumPy 标量
        fx = k[0, 0].item()
        fy = k[1, 1].item()
        cx = k[0, 2].item()
        cy = k[1, 2].item()
        cx2= k_r[0][2].item()
        depth_path = os.path.join(basedir, sequence, 'depth')
        for extension in ["npy"]:
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
            disp = np.load(depth_file)
            depth = (fx * baseline) / (-disp + (cx2 - cx))
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