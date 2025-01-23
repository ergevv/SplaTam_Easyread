## 一、安装
1、下载源码:
```bash
git clone https://github.com/ergevv/SplaTam_Easyread.git
```
2、安装cuda11.3
3、在原版上修改了requirements.txt，并把diff-gaussian-rasterization-w-depth直接下载到本地，并修改setup.py方便安装调试
```bash
pip install -r venv_requirements.txt
```
4、增加argparse参数，方便调试
5、修改launch.json文件，支持cuda调试，注意cuda调试不支持中文路径
6、在data文件夹增加20张Replica Room的图片，并修改configs/replica/splatam.py文件
7、安装diff-gaussian-rasterization-w-depth：
```bash
cd diff-gaussian-rasterization-w-depth
python setup.py develop
```
8、显示结果:
```bash
python viz_scripts/final_recon.py configs/replica/splatam.py
python viz_scripts/online_recon.py configs/replica/splatam.py
```

## 二、公式
1、多维高斯分布公式:
高斯分布，也称为正态分布（Normal Distribution），是一种在统计学和概率论中非常常见的连续概率分布。对于一维（单变量）的情况，其概率密度函数（PDF）由以下公式给出：

\[ f(x | \mu, \sigma^2) = \frac{1}{\sqrt{2\pi\sigma^2}} e^{-\frac{(x - \mu)^2}{2\sigma^2}} \]

其中：
- \( x \) 是随机变量。
- \( \mu \) (mu) 是分布的均值（期望值），决定了分布的位置。
- \( \sigma^2 \) (sigma squared) 是方差，它衡量了数据点围绕均值的散布程度；\( \sigma \) (sigma) 是标准差。
- \( e \) 是自然对数的底，大约等于 2.71828。
- \( \pi \) 是圆周率，大约等于 3.14159。

对于多维（多变量）高斯分布，概率密度函数的形式稍微复杂一些，可以表示为：

\[ f(\mathbf{x}|\boldsymbol{\mu},\boldsymbol{\Sigma}) = \frac{1}{(2\pi)^{k/2}|\boldsymbol{\Sigma}|^{1/2}} \exp\left(-\frac{1}{2}(\mathbf{x}-\boldsymbol{\mu})^\top\boldsymbol{\Sigma}^{-1}(\mathbf{x}-\boldsymbol{\mu})\right) \]

这里：
- \( \mathbf{x} \) 是一个 k 维随机向量。
- \( \boldsymbol{\mu} \) 是 k 维的均值向量。
- \( \boldsymbol{\Sigma} \) 是 k×k 的协方差矩阵，描述了各个维度之间的方差和协方差。
- \( |\boldsymbol{\Sigma}| \) 表示协方差矩阵的行列式。
- \( \boldsymbol{\Sigma}^{-1} \) 是协方差矩阵的逆矩阵。
- \( (\mathbf{x}-\boldsymbol{\mu})^\top \) 表示 \( \mathbf{x}-\boldsymbol{\mu} \) 的转置。

2、协方差矩阵的表达形式：
协方差矩阵可以描述高斯分布的影响范围，可以想象其影响范围可能是圆或者椭圆形，而椭圆可以用缩放尺度和旋转角度来表示：
（1）特征值：它们代表了椭圆的半轴长度的平方（即沿着主成分方向上的方差）。较大的特征值对应于椭圆的长轴，较小的特征值对应于短轴。因此，特征值可以被视为缩放因子，因为它们决定了椭圆的大小。
（2）特征向量：它们定义了椭圆的主轴方向，即数据的主要变异方向。特征向量给出了旋转角度，说明了椭圆相对于坐标轴的旋转程度。
（3）所以 $\Sigma = RSS^TR^T$，满足协方差的半正定性质
注：
椭圆可以通过缩放尺度和旋转角度来表示，主要是因为这些参数能够简洁地描述椭圆的几何特性。具体来说：

1. **缩放尺度（Scale Factors）**：椭圆的形状由其两个半轴长度决定，分别是长轴（major axis）和短轴（minor axis）。这两个长度可以被视为在各自方向上的缩放因子。如果我们将一个单位圆（所有点到中心的距离都是1的圆）沿两个正交方向分别缩放不同的比例，我们就可以得到一个椭圆。长轴和短轴的长度对应于在相应方向上的缩放比例。

2. **旋转角度（Rotation Angle）**：标准位置的椭圆其主轴（长轴和短轴）是与坐标系的x轴和y轴平行的。然而，实际中的椭圆可能被旋转了一定的角度。这个旋转角度定义了椭圆相对于原坐标系的倾斜程度。通过应用一个旋转矩阵，我们可以将标准位置的椭圆转换为任意方向上的椭圆。旋转矩阵是一个特殊的正交矩阵，它保持了形状和大小不变，只改变了位置和方向。

综上所述，一个二维空间中的椭圆可以用三个主要参数来完全描述：长轴长度、短轴长度以及旋转角度。长轴和短轴长度给出了椭圆的尺寸信息，而旋转角度则说明了椭圆的方向。这种表示方法不仅直观而且数学上非常有用，因为它允许我们使用简单的变换（如缩放和平移）来构造复杂的形状，并且易于进行计算和分析。

在数学表达式中，一个经过缩放和旋转的椭圆可以表示为：
\[ \frac{(x\cos(\theta) + y\sin(\theta))^2}{a^2} + \frac{(-x\sin(\theta) + y\cos(\theta))^2}{b^2} = 1 \]
其中 \(a\) 和 \(b\) 分别是椭圆的半长轴和半短轴长度，\(\theta\) 是椭圆的旋转角度。

3、高斯分布的性质：
不证明给出结论（可参考https://zhuanlan.zhihu.com/p/666465701）：
(1)高斯分布经仿射变换后仍是高斯分布
(2)多元高斯分布的边缘分布仍是高斯分布
(3)高斯分布经过线性变换仍是高斯分布

4、透视投影矩阵



5、透视投影矩阵线性化：
