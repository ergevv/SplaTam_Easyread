import torch
from diff_gaussian_rasterization import GaussianRasterizationSettings as Camera

def setup_camera(w, h, k, w2c, near=0.01, far=100):
    fx, fy, cx, cy = k[0][0], k[1][1], k[0][2], k[1][2]
    w2c = torch.tensor(w2c).cuda().float()
    cam_center = torch.inverse(w2c)[:3, 3]
    w2c = w2c.unsqueeze(0).transpose(1, 2)
    opengl_proj = torch.tensor([[2 * fx / w, 0.0, -(w - 2 * cx) / w, 0.0],
                                [0.0, 2 * fy / h, -(h - 2 * cy) / h, 0.0],
                                [0.0, 0.0, far / (far - near), -(far * near) / (far - near)],
                                [0.0, 0.0, 1.0, 0.0]]).cuda().float().unsqueeze(0).transpose(1, 2)
    full_proj = w2c.bmm(opengl_proj)  
    # 历时4天，终于看懂w2c矩阵的作用了，w2c的作用就是把全部点转到第一帧坐标下观察，即是以第一帧为中心，可以查看最后渲染完的三维物体，第一帧对应的相机就在中心位置，因为是整体移动，所以颜色该是什么颜色就是什么颜色
    # opengl_proj将相机坐标投影到裁剪空间，而w2ca将裁剪空间转到第一帧观察视角
    # 错误原因：
    # 1、一开始以为w2c是世界到相机的变换矩阵，但是应该更具体，w2c是世界到第一帧相机位姿的变换矩阵
    # 2、我把点云分开考虑，代码是合在一起的，使用w2c变换点是变换所有相机提取出来的点的
    # 3、mean3D，第一帧相机的点云已经转到世界坐标系了，所以第一帧相机坐标跟世界坐标系原点重合了，应该是单元值。后续相机坐标是根据第一帧相机预测的，所以也是世界坐标系下的值
    cam = Camera(
        image_height=h,
        image_width=w,
        tanfovx=w / (2 * fx),
        tanfovy=h / (2 * fy),
        bg=torch.tensor([0, 0, 0], dtype=torch.float32, device="cuda"),
        scale_modifier=1.0,
        viewmatrix=w2c,
        projmatrix=full_proj,
        sh_degree=0,
        campos=cam_center,
        prefiltered=False
    )
    return cam
