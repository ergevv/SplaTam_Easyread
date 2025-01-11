#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

from setuptools import setup
from torch.utils.cpp_extension import CUDAExtension, BuildExtension
import os
os.path.dirname(os.path.abspath(__file__))

setup(
    name="diff_gaussian_rasterization",
    packages=['diff_gaussian_rasterization'],
    ext_modules=[
        CUDAExtension(
            name="diff_gaussian_rasterization._C",
            sources=[
            "cuda_rasterizer/rasterizer_impl.cu",
            "cuda_rasterizer/forward.cu",
            "cuda_rasterizer/backward.cu",
            "rasterize_points.cu",
            "ext.cpp"],
            extra_compile_args={"nvcc": ["-I" + os.path.join(os.path.dirname(os.path.abspath(__file__)), "third_party/glm/")]})
        ],
    cmdclass={
        'build_ext': BuildExtension
    }
)


# from setuptools import setup
# from torch.utils.cpp_extension import CUDAExtension, BuildExtension
# import os

# # 获取当前文件所在目录
# current_dir = os.path.dirname(os.path.abspath(__file__))

# setup(
#     name="diff_gaussian_rasterization",
#     packages=['diff_gaussian_rasterization'],
#     ext_modules=[
#         CUDAExtension(
#             name="diff_gaussian_rasterization._C",
#             sources=[
#                 "cuda_rasterizer/rasterizer_impl.cu",
#                 "cuda_rasterizer/forward.cu",
#                 "cuda_rasterizer/backward.cu",
#                 "rasterize_points.cu",
#                 "ext.cpp"
#             ],
#             extra_compile_args={
#                 "cxx": ["-g", "-O0"],  # 对于 C++ 文件启用调试模式
#                 "nvcc": [
#                     "-I" + os.path.join(current_dir, "third_party/glm/"),
#                     "-g",              # 启用调试信息
#                     "-G",              # 启用完整的调试符号（可选）
#                     "-O0"              # 禁用优化
#                 ]
#             }
#         )
#     ],
#     cmdclass={
#         'build_ext': BuildExtension
#     }
# )
