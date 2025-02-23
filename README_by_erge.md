## 一、安装
1、下载源码:
```bash
git clone https://github.com/ergevv/SplaTam_Easyread.git
```
2、安装cuda11.3
3、在原版上修改了requirements.txt，并把diff-gaussian-rasterization-w-depth直接下载到本地，并修改setup.py方便安装调试
```bash
pip install -r requirements.txt
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




**(1) 关于二维高斯分布和椭圆的关系，我们可以这么考虑：**

二维高斯分布的概率密度函数为

\[
p(\boldsymbol{x}) = \frac{1}{(2\pi)^{\frac{n}{2}}|\Sigma|^{\frac{1}{2}}} \exp\left\{-\frac{1}{2}(\boldsymbol{x}-\boldsymbol{\mu})^T\Sigma^{-1}(\boldsymbol{x}-\boldsymbol{\mu})\right\}
\]

其中 \(\boldsymbol{x}=(x,y)^T\)，\(\Sigma\) 为协方差矩阵，\(\boldsymbol{\mu}\) 为均值。考虑令 \(p(\boldsymbol{x})\) 等于一个常数，并令 \(\boldsymbol{u}=\boldsymbol{x}-\boldsymbol{\mu}\)，即

\[
\boldsymbol{u}^T\Sigma^{-1}\boldsymbol{u}=R^2
\]

其中 \(R^2\) 为常数。由于 \(\Sigma\) 是对称矩阵，一定存在正交矩阵 \(P\) 使得

\[
\Sigma=P^T\Lambda P
\]

其中 \(\Lambda=\text{diag}(\lambda_1,\lambda_2)\) 是由 \(\Sigma\) 的特征值组成的对角矩阵。带入概率密度函数，得

\[
\boldsymbol{u}^TP^T\Lambda^{-1}P\boldsymbol{u}=R^2
\]

令 \(\boldsymbol{v}=P\boldsymbol{u}\)（也就是相对 \(\boldsymbol{u}\) 坐标系旋转了一个角度），则

\[
\boldsymbol{v}^T\Lambda^{-1}\boldsymbol{v}=R^2
\]

即

\[
\frac{v_1^2}{\lambda_1 R^2} + \frac{v_2^2}{\lambda_2 R^2} = 1
\]

正好是一个长短轴分别为 \(\sqrt{\lambda_1}R\)、\(\sqrt{\lambda_2}R\) 的椭圆。令 \(R=3\) 就得到了代码中算 `my_radius` 的公式。






3、高斯分布的性质：
不证明给出结论（可参考https://zhuanlan.zhihu.com/p/666465701）：
(1)高斯分布经仿射变换后仍是高斯分布
(2)多元高斯分布的边缘分布仍是高斯分布
(3)高斯分布经过线性变换仍是高斯分布

4、透视投影矩阵
基于相机内参推导透视投影矩阵（splatam）：

$$
M_{cam}= \begin{bmatrix}
\frac{2 \cdot fx}{w} & 0.0 & \frac{(w - 2 \cdot cx)}{w} & 0.0 \\
0.0 & \frac{2 \cdot fy}{h} & \frac{(h - 2 \cdot cy)}{h} & 0.0 \\
0 & 0 & \frac{far + near}{near - far} & \frac{2far \cdot near}{near - far} \\
0.0 & 0.0 & -1.0 & 0.0
\end{bmatrix}$$

- `fx`: 相机内参中的焦距在x方向上的分量。
- `fy`: 相机内参中的焦距在y方向上的分量。
- `cx`: 图像中心在x方向上的坐标。
- `cy`: 图像中心在y方向上的坐标。
- `w`: 图像宽度。
- `h`: 图像高度。
- `near`: 视锥体（frustum）的近裁剪平面距离。
- `far`: 视锥体（frustum）的远裁剪平面距离。
注：此处深度为负

参考[透射投影矩阵的数学推导](https://zhuanlan.zhihu.com/p/421962223)的推导得到透视投影矩阵另一种形式：
$$
M_p = 
\begin{bmatrix}
\frac{2n}{r - l} & 0 & \frac{r + l}{r - l} & 0 \\
0 & \frac{2n}{t - b} & \frac{t + b}{t - b} & 0 \\
0 & 0 & \frac{f + n}{n - f} & \frac{2fn}{n - f} \\
0 & 0 & -1 & 0
\end{bmatrix}
$$
- `n`: 近裁剪平面距离。
- `f`: 远裁剪平面距离。
- `r`: 近裁剪平面的右边界。
- `l`: 近裁剪平面的左边界。
- `t`: 近裁剪平面的上边界。
- `b`: 近裁剪平面的下边界。


可以先看这两个参考再往下看：
参考：
**https://zhuanlan.zhihu.com/p/421962223**
**https://blog.csdn.net/qq_43758883/article/details/116503614**


通过视场角来从$M_p$得到$M_{cam}$:


![](https://i-blog.csdnimg.cn/direct/ef507936ae76496f8d12bb3e68d4fbd8.png#pic_center)




上图，忽略y轴，上半部分是相机坐标系，下班部分是像素坐标系，可见相机坐标下的视场角和像素坐标下的视场角是一致的，
1、
在相机坐标系下：
$$tan(fovx/2) = \frac{r-l}{2n}$$

在像素坐标系下：
$$tan(fovx/2 )= \frac{fx}{w/2} $$
将上式公式同时提取缩放因子：
$$tan(fovx/2) = \frac{ \alpha f}{\alpha w'/2}  = \frac{ f}{ w'/2} =  \frac{r-l}{2n}$$

2、
$$ 
\begin{align}
\alpha (r-l) &= w \\
\alpha (r+l) &= (w - 2cx)
\end{align}
$$
所以，
$$ \frac {r+l}{r-l} = \frac {(w - 2cx)}{w}$$

同理，透视投影矩阵y轴也是如此推导。

#### splatam非标准透视投影
splatam使用的非标准透视投影，与一般不同，splatam是把深度缩放到 [0,1]：
此处深度全取正数，因此第三、四列需要加个负号。
$$
M_{splatam}= \begin{bmatrix}
\frac{2 \cdot fx}{w} & 0.0 & \frac{-(w - 2 \cdot cx)}{w} & 0.0 \\
0.0 & \frac{2 \cdot fy}{h} & \frac{-(h - 2 \cdot cy)}{h} & 0.0 \\
0.0 & 0.0 & \frac{far}{far - near} & \frac{-(far \cdot near)}{far - near} \\
0.0 & 0.0 & 1.0 & 0.0
\end{bmatrix}$$
可以看到矩阵的第三行与之前的不同，它的作用是把深度最后归一化到 [0,1]。
设相机坐标下的点位置为$[x_{cam} , y_{cam} ,z_{cam},1]$，对应裁剪空间坐标为$[x_{c} , y_{c} ,z_{c},w_c]$，对应NDC坐标$[x_{n} , y_{n} ,z_{n},1]$。
故：
$$
\begin{bmatrix}x_{c} \\ y_{c} \\ z_{c} \\w_c \end{bmatrix}
=M_{splatam} \begin{bmatrix}x_{cam} \\ y_{cam} \\ z_{cam} \\1 \end{bmatrix}
$$
考虑深度，即z轴的变换有：
$$
\begin{align}
z_c &= \frac{far}{far - near}   z_{cam}  - \frac{(far \cdot near)}{far - near} \\
w_c &= z_{cam}
\end{align}
$$

归一化处理，得到NDC坐标
$$
\begin{align}
z_n &= \frac {z_c}{w_c}  = \frac{\frac{far}{far - near}   z_{cam}  - \frac{(far \cdot near)}{far - near}}{z_{cam}} \\
&= \frac{far - \frac{(far \cdot near)}{z_{cam}}}{far - near}
\end{align}
$$
故，当$z_{cam} = near$时，$z_n = 0$， 当$z_{cam} = far$时，$z_n = 1$，所以深度最后是限制在 [0,1]。为什么要怎样设计呢，因为使用3DGS渲染时，只考虑深度大于0的。




5、透视投影矩阵线性化：
透视变换是非线性的，高斯分布经过透视变换后会不再符合高斯分布，所以需要线性化。
线性化规则：使用当前点位置来作为泰勒公式的基点来表达领域附近的关系，使用投影矩阵线性化比较麻烦，因此这里直接使用针孔模型的公式线性化。它与使用投影矩阵将3D相机坐标先转换到标准化设备坐标（Normalized Device Coordinates, NDC），然后再转换到像素坐标，这两种方法在数学上是等价的。只不过使用投影矩阵有利于并行处理。
$$
J = 
\begin{bmatrix}
\frac{fx}{t_z} & 0 & -\frac{fx \cdot t_x}{t_z^2} \\
0 & \frac{fy}{t_z} & -\frac{fy \cdot t_y}{t_z^2} \\
0 & 0 & 0
\end{bmatrix}
$$







在针孔相机模型中，三维空间中的点 \( P = (X, Y, Z) \) 投影到二维图像平面上的坐标 \( p = (x, y) \) 可以通过以下公式计算：

\[ x = f_x \frac{X}{Z} + c_x \]
\[ y = f_y \frac{Y}{Z} + c_y \]

这里，\(f_x\) 和 \(f_y\) 分别是图像平面在x轴和y轴方向上的焦距，而 \(c_x\) 和 \(c_y\) 是主点偏移量（通常位于图像中心）。如果我们忽略主点偏移量（即假设 \(c_x = c_y = 0\)），则上述方程简化为：

\[ x = f_x \frac{X}{Z} \]
\[ y = f_y \frac{Y}{Z} \]

如果构建一个描述这些投影方程相对于点 \(P\) 的变化率（即其梯度）的雅可比矩阵：

\[ J = \begin{bmatrix}
\frac{\partial x}{\partial X} & \frac{\partial x}{\partial Y} & \frac{\partial x}{\partial Z} \\
\frac{\partial y}{\partial X} & \frac{\partial y}{\partial Y} & \frac{\partial y}{\partial Z} \\
0 & 0 & 0
\end{bmatrix} \]

根据上面简化的投影方程，我们可以计算出各个偏导数：

- 对于 \(x\) 方向：\[ \frac{\partial x}{\partial X} = \frac{f_x}{Z}, \quad \frac{\partial x}{\partial Y} = 0, \quad \frac{\partial x}{\partial Z} = -\frac{f_x X}{Z^2} \]
- 对于 \(y\) 方向：\[ \frac{\partial y}{\partial X} = 0, \quad \frac{\partial y}{\partial Y} = \frac{f_y}{Z}, \quad \frac{\partial y}{\partial Z} = -\frac{f_y Y}{Z^2} \]



这段代码计算的是损失函数相对于2D协方差矩阵各元素的梯度，给定损失函数相对于该协方差矩阵逆矩阵（即共轭矩阵）的梯度。具体来说，它使用链式法则来从损失关于逆协方差矩阵的梯度推导出损失关于原始协方差矩阵元素的梯度。

## 前向传播：

\[ f(\mathbf{x}|\boldsymbol{\mu},\boldsymbol{\Sigma}) = \frac{1}{(2\pi)^{k/2}|\boldsymbol{\Sigma}|^{1/2}} \exp\left(-\frac{1}{2}(\mathbf{x}-\boldsymbol{\mu})^\top\boldsymbol{\Sigma}^{-1}(\mathbf{x}-\boldsymbol{\mu})\right) \]

$$ power = \left(-\frac{1}{2}(\mathbf{x}-\boldsymbol{\mu})^\top\boldsymbol{\Sigma}^{-1}(\mathbf{x}-\boldsymbol{\mu})\right)$$

$$\alpha = \min(0.99, opacity * \exp(power)); $$

$$ T_{test} = T * (1 - \alpha) $$
这里T是当前层可用的透明度，$T_{test}$是经过当前层后留给的下一层的透明度。
光线穿过多个半透明层（例如玻璃、雾气等）。每一层都会吸收一部分光线，并且让剩余的光线通过。如果一个层的不透明度为 $\alpha$，那么它会阻挡 $\alpha$ 比例的光线，并允许 $(1 - \alpha) $比例的光线通过。
当光线进入第一层时，它保留了 $(1 - \alpha_1)$ 的比例。
然后当这部分光线进入第二层时，它再次被削减，只保留 $(1 - \alpha_2)$ 的比例。
因此，经过两层后的光线量是原始光线量乘以 $(1 - \alpha_1) * (1 - \alpha_2)$。

$$ color = color + color *\alpha *  T $$


## 反向传播：
### 已知：
使用pytorch自动求导得到了损失值对颜色的梯度：$\frac {dL}{dC}$


#### 求取当前层透明度
$$ T = T / (1 - \alpha) $$ 


### Alpha Blending 

Alpha Blending 是一种用于图像合成和透明度处理的常用技术，广泛应用于计算机图形学、图像处理和视频处理等领域。其基本公式用于将两个图像（或图层）按照一定的透明度进行混合。

假设有两个图像 \( C_1 \) 和 \( C_2 \)，分别表示前景和背景图像的颜色值（通常为 RGB 颜色值）。\( \alpha \) 表示前景图像 \( C_1 \) 的透明度（alpha 值），取值范围为 \([0, 1]\)：

- \( \alpha = 0 \) 表示完全透明（前景不可见）。
- \( \alpha = 1 \) 表示完全不透明（前景完全覆盖背景）。

Alpha Blending 的混合公式如下：

\[
C_{\text{out}} = \alpha \cdot C_1 + (1 - \alpha) \cdot C_2
\]

其中：
- \( C_{\text{out}} \) 是混合后的颜色值。
- \( C_1 \) 是前景图像的颜色值。
- \( C_2 \) 是背景图像的颜色值。
- \( \alpha \) 是前景图像的透明度。


1. **前景贡献**：\( \alpha \cdot C_1 \) 表示前景图像对最终颜色的贡献。透明度 \( \alpha \) 越大，前景图像的颜色对最终结果的影响越大。

2. **背景贡献**：\( (1 - \alpha) \cdot C_2 \) 表示背景图像对最终颜色的贡献。透明度 \( \alpha \) 越小，背景图像的颜色对最终结果的影响越大。

3. **混合结果**：将前景和背景的贡献相加，得到最终的混合颜色 \( C_{\text{out}} \)。


#### 公式推导：
1、$\frac {dL}{d\alpha}、\frac{\partial L}{\partial \boldsymbol{g}}$：

使用Alpha Blending公式，设\(\boldsymbol{b}_i\)是第\(i\)个及其后的Gaussians渲染出来的颜色（三个通道），\(\boldsymbol{g}_i\)是第\(i\)个Gaussian的颜色，则有递推公式：

\[
\boldsymbol{b}_i = \alpha_i \boldsymbol{g}_i + (1 - \alpha_i) \boldsymbol{b}_{i+1}
\]

令\(T_i = (1 - \alpha_1)(1 - \alpha_2)\cdots(1 - \alpha_{i-1})\)为第\(i\)个Gaussian对像素点的透光率，\(\boldsymbol{C}\)是像素点的颜色，则有

\[
\frac{\partial L}{\partial \boldsymbol{b}_i} = \frac{\partial L}{\partial \boldsymbol{C}} \cdot \frac{\partial \boldsymbol{b}_1}{\partial \boldsymbol{b}_2} \cdot \frac{\partial \boldsymbol{b}_2}{\partial \boldsymbol{b}_3} \cdots \frac{\partial \boldsymbol{b}_{i-1}}{\partial \boldsymbol{b}_i}
\]

\[
= \frac{\partial L}{\partial \boldsymbol{C}} \cdot (1 - \alpha_1)I \cdot (1 - \alpha_2)I \cdots (1 - \alpha_{i-1})I
\]

\[
= T_i \frac{\partial L}{\partial \boldsymbol{C}}
\]

故

\[
\frac{\partial L}{\partial \alpha_i} = T_i \frac{\partial L}{\partial \boldsymbol{C}} (\boldsymbol{g}_i - \boldsymbol{b}_{i+1})
\]

\[
\frac{\partial L}{\partial \boldsymbol{g}_i} = T_i \alpha_i \frac{\partial L}{\partial \boldsymbol{C}}
\]

2、$\frac{\partial L}{\partial \sigma} 、\frac{\partial L}{\partial G}$


令 \(\alpha = \min(0.99, \sigma G)\)，其中 \(\sigma\) 是“opacity”，\(G\) 是正态分布给出的“exponential falloff”，则

\[
\frac{\partial L}{\partial \sigma} = G \frac{\partial L}{\partial \alpha}
\]

\[
\frac{\partial L}{\partial G} = \sigma \frac{\partial L}{\partial \alpha}
\]



计算损失函数 \(L\) 对于不透明度 \(\sigma\) 和指数衰减因子 \(G\) 的偏导数。

3、计算损失函数对2D位置和2D协方差矩阵的逆的梯度

\(G = \exp\left\{-\frac{1}{2} \boldsymbol{d}^T A \boldsymbol{d}\right\}\)，
\(\boldsymbol{d} = \boldsymbol{\mu} - \boldsymbol{p}\)，
其中 \(\boldsymbol{p}\) 为像素坐标，\(\boldsymbol{\mu}\) 为Gaussian中心的2D坐标，\(\boldsymbol{d}\) 为它们之间的位移向量，\(A\) 为椭圆二次型的矩阵，因此

\[
\frac{\partial L}{\partial \boldsymbol{d}} = -\frac{1}{2} G \frac{\partial L}{\partial G} [ (A^T + A)\boldsymbol{d}]
\]

\[
\frac{\partial L}{\partial \boldsymbol{\mu}} = \frac{\partial L}{\partial \boldsymbol{d}}
\]

\[
\frac{\partial L}{\partial A_{00}} = -\frac{1}{2} G \frac{\partial L}{\partial G} d_x^2
\]

\[
\frac{\partial L}{\partial A_{11}} = -\frac{1}{2} G \frac{\partial L}{\partial G} d_y^2
\]

\[
\frac{\partial L}{\partial A_{01}} = -\frac{1}{2} G \frac{\partial L}{\partial G} d_x d_y
\]

注意计算 \(\frac{\partial L}{\partial \boldsymbol{\mu}}\) 时要乘以像素坐标对像平面坐标的导。



4、损失函数对2D协方差矩阵的梯度


要理解 `dL_da` 的推导过程，我们需要从损失函数关于逆协方差矩阵（共轭矩阵）的梯度出发，通过链式法则推导出损失函数关于原始协方差矩阵元素的梯度。具体来说，我们要计算的是损失 \(L\) 关于协方差矩阵 \(\Sigma\) 中元素 \(a\) 的梯度，即 \(\frac{\partial L}{\partial a}\)。

### 背景

假设我们有一个2x2的协方差矩阵 \(\Sigma\) 和其逆矩阵（共轭矩阵）\(\Sigma^{-1}\)：

\[
\Sigma = \begin{pmatrix} a & b \\ b & c \end{pmatrix}, \quad \Sigma^{-1} = \frac{1}{ac - b^2} \begin{pmatrix} c & -b \\ -b & a \end{pmatrix}
\]

其中，\(ac - b^2\) 是行列式，记作 \(\text{denom}\)。

### 目标

我们的目标是根据损失函数关于 \(\Sigma^{-1}\) 的梯度来推导出损失函数关于 \(\Sigma\) 元素的梯度。

### 推导过程

#### 1. 计算 \(\frac{\partial \Sigma^{-1}}{\partial a}\)

首先，我们需要计算 \(\Sigma^{-1}\) 对 \(a\) 的偏导数。由于 \(\Sigma^{-1}\) 可以写成：

\[
\Sigma^{-1} = \frac{1}{\text{denom}} \begin{pmatrix} c & -b \\ -b & a \end{pmatrix}
\]

我们可以写出 \(\Sigma^{-1}\) 各元素对 \(a\) 的偏导数：

- \(\frac{\partial (\Sigma^{-1})_{11}}{\partial a} = \frac{\partial}{\partial a} \left( \frac{c}{\text{denom}} \right) = \frac{-c \cdot (-2b \frac{\partial b}{\partial a} + c)}{\text{denom}^2} = \frac{-c^2}{\text{denom}^2}\)
- \(\frac{\partial (\Sigma^{-1})_{12}}{\partial a} = \frac{\partial}{\partial a} \left( \frac{-b}{\text{denom}} \right) = \frac{b \cdot (2b \frac{\partial b}{\partial a} - c)}{\text{denom}^2} = \frac{2bc}{\text{denom}^2}\)
- \(\frac{\partial (\Sigma^{-1})_{22}}{\partial a} = \frac{\partial}{\partial a} \left( \frac{a}{\text{denom}} \right) = \frac{\text{denom} - a \cdot (-2b \frac{\partial b}{\partial a} + c)}{\text{denom}^2} = \frac{\text{denom} - ac}{\text{denom}^2}\)

#### 2. 应用链式法则

接下来，应用链式法则计算 \(\frac{\partial L}{\partial a}\)：

\[
\frac{\partial L}{\partial a} = \sum_{i,j} \frac{\partial L}{\partial (\Sigma^{-1})_{ij}} \cdot \frac{\partial (\Sigma^{-1})_{ij}}{\partial a}
\]

将上述偏导数代入公式中：

\[
\frac{\partial L}{\partial a} = \frac{\partial L}{\partial (\Sigma^{-1})_{11}} \cdot \frac{-c^2}{\text{denom}^2} + \frac{\partial L}{\partial (\Sigma^{-1})_{12}} \cdot \frac{2bc}{\text{denom}^2} + \frac{\partial L}{\partial (\Sigma^{-1})_{22}} \cdot \frac{\text{denom} - ac}{\text{denom}^2}
\]

简化得到：

\[
\frac{\partial L}{\partial a} = \frac{1}{\text{denom}^2} \left(-c^2 \cdot \frac{\partial L}{\partial (\Sigma^{-1})_{11}} + 2bc \cdot \frac{\partial L}{\partial (\Sigma^{-1})_{12}} + (\text{denom} - ac) \cdot \frac{\partial L}{\partial (\Sigma^{-1})_{22}}\right)
\]

这正是代码中的表达式：

```cpp
dL_da = denom2inv * (-c * c * dL_dconic.x + 2 * b * c * dL_dconic.y + (denom - a * c) * dL_dconic.z);
```

其中，`denom2inv` 就是 \(\frac{1}{\text{denom}^2}\)。


5、损失函数对3D协方差矩阵的梯度
设T为定值，cov_{3D}为变量

在3D到2D投影的过程中，3D协方差矩阵 \(V_{rk}\) 通过一个变换矩阵 \(T\) 投影到2D空间中形成2D协方差矩阵 \(cov_{2D}\)：

\[
cov_{2D} = T^T \cdot cov_{3D} \cdot T
\]

其中，\(T\) 是从3D空间到2D屏幕空间的变换矩阵。
$$
\frac{\partial L}{\partial cov_{3D} } = 
\frac{\partial L}{\partial cov_{2D} } \frac{\partial cov_{2D}}{\partial cov_{3D} } = 
\frac{\partial L}{\partial cov_{2D} } T^TT
$$


6、损失函数对3D空间到2D屏幕空间的变换矩阵的梯度
设T为变值，cov_{3D}为定值
$$
\frac{\partial L}{\partial T } = 
\frac{\partial L}{\partial cov_{2D} } \frac{\partial cov_{2D}}{\partial T } = 
\frac{\partial L}{\partial cov_{2D} } cov_{3D}(T^T+T)
$$

7、损失函数对J的梯度
$$T=WJ$$
$$
\frac{\partial L}{\partial J } = 
\frac{\partial L}{\partial T } \frac{\partial T}{\partial J} = 
\frac{\partial L}{\partial T } W
$$

8、损失函数对3D位置的梯度（协方差矩阵）

由

\[ J = \begin{bmatrix}
\frac{\partial x}{\partial X} & \frac{\partial x}{\partial Y} & \frac{\partial x}{\partial Z} \\
\frac{\partial y}{\partial X} & \frac{\partial y}{\partial Y} & \frac{\partial y}{\partial Z} \\
0 & 0 & 0
\end{bmatrix} \]

根据上面简化的投影方程，我们可以计算出各个偏导数：

- 对于 \(x\) 方向：\[ \frac{\partial x}{\partial X} = \frac{f_x}{Z}, \quad \frac{\partial x}{\partial Y} = 0, \quad \frac{\partial x}{\partial Z} = -\frac{f_x X}{Z^2} \]
- 对于 \(y\) 方向：\[ \frac{\partial y}{\partial X} = 0, \quad \frac{\partial y}{\partial Y} = \frac{f_y}{Z}, \quad \frac{\partial y}{\partial Z} = -\frac{f_y Y}{Z^2} \]

因此，
$$
\frac{\partial L}{\partial X } = 
\frac{\partial L}{\partial J } \frac{\partial J}{\partial X} 
$$



9、损失函数对3D位置的梯度（位置带来的误差）
（1）透视投影变换

给定一个3D点 \(m = (m_x, m_y, m_z)\)，其齐次坐标为 \([m_x, m_y, m_z, 1]\)。应用4x4的透视投影矩阵 `proj` 后，得到新的齐次坐标：

\[ 
\text{m\_hom} = \text{proj} \times [m_x, m_y, m_z, 1]^T 
\]

其中，`proj` 是一个4x4的投影矩阵。为了获得2D屏幕坐标，我们需要进行透视除法（即除以第四个分量 \(w\)）：

\[ 
x' = \frac{\text{m\_hom}_x}{\text{m\_hom}_w}, \quad y' = \frac{\text{m\_hom}_y}{\text{m\_hom}_w}
\]

假设 `proj` 矩阵的形式如下：
```
proj = | p00 p01 p02 p03 |
       | p10 p11 p12 p13 |
       | p20 p21 p22 p23 |
       | p30 p31 p32 p33 |
```

则有：
\[ 
\text{m\_hom}_x = p00 \cdot m_x + p01 \cdot m_y + p02 \cdot m_z + p03
\]
\[ 
\text{m\_hom}_y = p10 \cdot m_x + p11 \cdot m_y + p12 \cdot m_z + p13
\]
\[ 
\text{m\_hom}_z = p20 \cdot m_x + p21 \cdot m_y + p22 \cdot m_z + p23
\]
\[ 
\text{m\_hom}_w = p30 \cdot m_x + p31 \cdot m_y + p32 \cdot m_z + p33
\]

然后，我们进行透视除法：
\[ 
x' = \frac{\text{m\_hom}_x}{\text{m\_hom}_w}, \quad y' = \frac{\text{m\_hom}_y}{\text{m\_hom}_w}
\]

### 2. 计算偏导数

为了计算损失函数 \(L\) 关于3D均值 \(m\) 的梯度，我们需要首先计算 \(x'\) 和 \(y'\) 对 \(m_x, m_y, m_z\) 的偏导数。

#### 对 \(x'\) 的偏导数

\[ 
\frac{\partial x'}{\partial m_i} = \frac{\partial}{\partial m_i} \left( \frac{\text{m\_hom}_x}{\text{m\_hom}_w} \right)
\]

使用商规则：
\[ 
\frac{\partial x'}{\partial m_i} = \frac{\text{m\_hom}_w \cdot \frac{\partial \text{m\_hom}_x}{\partial m_i} - \text{m\_hom}_x \cdot \frac{\partial \text{m\_hom}_w}{\partial m_i}}{\text{m\_hom}_w^2}
\]

具体到每个维度：

- **对 \(m_x\) 的偏导数**：
  \[
  \frac{\partial x'}{\partial m_x} = \frac{\text{m\_hom}_w \cdot p00 - \text{m\_hom}_x \cdot p30}{\text{m\_hom}_w^2} = \frac{p00}{\text{m\_hom}_w} - \frac{p30 \cdot \text{m\_hom}_x}{\text{m\_hom}_w^2}
  \]

- **对 \(m_y\) 的偏导数**：
  \[
  \frac{\partial x'}{\partial m_y} = \frac{\text{m\_hom}_w \cdot p01 - \text{m\_hom}_x \cdot p31}{\text{m\_hom}_w^2} = \frac{p01}{\text{m\_hom}_w} - \frac{p31 \cdot \text{m\_hom}_x}{\text{m\_hom}_w^2}
  \]

- **对 \(m_z\) 的偏导数**：
  \[
  \frac{\partial x'}{\partial m_z} = \frac{\text{m\_hom}_w \cdot p02 - \text{m\_hom}_x \cdot p32}{\text{m\_hom}_w^2} = \frac{p02}{\text{m\_hom}_w} - \frac{p32 \cdot \text{m\_hom}_x}{\text{m\_hom}_w^2}
  \]

#### 对 \(y'\) 的偏导数

类似地，对于 \(y'\)：

\[ 
\frac{\partial y'}{\partial m_i} = \frac{\partial}{\partial m_i} \left( \frac{\text{m\_hom}_y}{\text{m\_hom}_w} \right)
\]

使用商规则：
\[ 
\frac{\partial y'}{\partial m_i} = \frac{\text{m\_hom}_w \cdot \frac{\partial \text{m\_hom}_y}{\partial m_i} - \text{m\_hom}_y \cdot \frac{\partial \text{m\_hom}_w}{\partial m_i}}{\text{m\_hom}_w^2}
\]

具体到每个维度：

- **对 \(m_x\) 的偏导数**：
  \[
  \frac{\partial y'}{\partial m_x} = \frac{\text{m\_hom}_w \cdot p10 - \text{m\_hom}_y \cdot p30}{\text{m\_hom}_w^2} = \frac{p10}{\text{m\_hom}_w} - \frac{p30 \cdot \text{m\_hom}_y}{\text{m\_hom}_w^2}
  \]

- **对 \(m_y\) 的偏导数**：
  \[
  \frac{\partial y'}{\partial m_y} = \frac{\text{m\_hom}_w \cdot p11 - \text{m\_hom}_y \cdot p31}{\text{m\_hom}_w^2} = \frac{p11}{\text{m\_hom}_w} - \frac{p31 \cdot \text{m\_hom}_y}{\text{m\_hom}_w^2}
  \]

- **对 \(m_z\) 的偏导数**：
  \[
  \frac{\partial y'}{\partial m_z} = \frac{\text{m\_hom}_w \cdot p12 - \text{m\_hom}_y \cdot p32}{\text{m\_hom}_w^2} = \frac{p12}{\text{m\_hom}_w} - \frac{p32 \cdot \text{m\_hom}_y}{\text{m\_hom}_w^2}
  \]

### 3. 使用链式法则计算梯度

现在我们有了所有需要的偏导数，可以使用链式法则来计算损失函数 \(L\) 关于3D均值 \(m\) 的梯度。

假设已知损失函数关于2D投影点的梯度 `dL_dmean2D[idx]`，我们可以将其反向传播回3D空间中的均值 \(m\)。

对于每个维度 \(i \in \{x, y, z\}\)，我们有：

\[
\frac{\partial L}{\partial m_i} = \sum_{j \in \{x', y'\}} \left( \frac{\partial x'}{\partial m_i} \cdot \frac{\partial L}{\partial x'} + \frac{\partial y'}{\partial m_i} \cdot \frac{\partial L}{\partial y'} \right)
\]

10、损失函数对协方差参数S、旋转矩阵R（四元数）的梯度
$$ M = S * R$$
$$ cov_{3D} = M^TM$$
所以
$$ 
\frac{\partial L}{\partial M} = 2M\frac{\partial L}{\partial \sigma}
$$

$$
\frac{\partial L}{\partial S} = R\frac{\partial L}{\partial M^T}
$$

$$
\frac{\partial L}{\partial R} = S\frac{\partial L}{\partial M^T}
$$


为了推导损失函数关于四元数 \(q = (r, x, y, z)\) 的梯度，我们需要从旋转矩阵 \(R\) 对四元数的偏导数出发，并结合链式法则来计算最终的梯度。下面将详细解释如何得到这些公式。

### 1. 四元数与旋转矩阵的关系

给定一个单位四元数 \(q = (r, x, y, z)\)，其对应的旋转矩阵 \(R\) 可以表示为：

\[
R(q) = 
\begin{bmatrix}
1 - 2(y^2 + z^2) & 2(xy - rz) & 2(xz + ry) \\
2(xy + rz) & 1 - 2(x^2 + z^2) & 2(yz - rx) \\
2(xz - ry) & 2(yz + rx) & 1 - 2(x^2 + y^2)
\end{bmatrix}
\]


### 2. 计算旋转矩阵 \(R\) 对四元数 \(q\) 的偏导数

对于每个四元数分量 \(q_i\)（\(i \in \{r, x, y, z\}\)），我们需要计算其对旋转矩阵 \(R\) 中每个元素的偏导数。

#### 偏导数的计算

- **\(\frac{\partial R}{\partial r}\)**：
  \[
  \frac{\partial R}{\partial r} = 
  \begin{bmatrix}
  0 & -2z & 2y \\
  2z & 0 & -2x \\
  -2y & 2x & 0
  \end{bmatrix}
  \]

- **\(\frac{\partial R}{\partial x}\)**：
  \[
  \frac{\partial R}{\partial x} = 
  \begin{bmatrix}
  -4x & 2y & 2z \\
  2y & 0 & -2r \\
  2z & 2r & 0
  \end{bmatrix}
  \]

- **\(\frac{\partial R}{\partial y}\)**：
  \[
  \frac{\partial R}{\partial y} = 
  \begin{bmatrix}
  0 & 2x & -2r \\
  2x & -4y & 2z \\
  2r & 2z & 0
  \end{bmatrix}
  \]

- **\(\frac{\partial R}{\partial z}\)**：
  \[
  \frac{\partial R}{\partial z} = 
  \begin{bmatrix}
  0 & 2r & 2x \\
  -2r & 0 & 2y \\
  2x & 2y & -4z
  \end{bmatrix}
  \]

### 3. 应用链式法则

根据链式法则，损失函数关于四元数 \(q\) 的梯度可以表示为：

\[
\frac{\partial L}{\partial q} = \sum_{i,j} \left( \frac{\partial R_{ij}}{\partial q_k} \cdot \frac{\partial L}{\partial R_{ij}} \right)
\]

其中，\(\frac{\partial L}{\partial R_{ij}}\) 是损失函数关于旋转矩阵 \(R\) 的梯度，使用求和符号把每个元素的梯度累加起来。








## 补充：
在协方差矩阵的梯度转换过程中乘以0.5，主要是因为协方差矩阵是对称矩阵。这意味着它的元素满足 \( \Sigma_{ij} = \Sigma_{ji} \)，因此在计算损失函数关于这些元素的梯度时，需要考虑到这种对称性。

### 协方差矩阵及其对称性

对于一个3D协方差矩阵 \(\Sigma\)，它是一个3x3的对称矩阵：

\[
\Sigma = 
\begin{bmatrix}
\Sigma_{00} & \Sigma_{01} & \Sigma_{02} \\
\Sigma_{10} & \Sigma_{11} & \Sigma_{12} \\
\Sigma_{20} & \Sigma_{21} & \Sigma_{22}
\end{bmatrix}
\]

由于 \(\Sigma\) 是对称的，我们有：

- \(\Sigma_{01} = \Sigma_{10}\)
- \(\Sigma_{02} = \Sigma_{20}\)
- \(\Sigma_{12} = \Sigma_{21}\)

因此，在优化过程中，当我们考虑某个非对角线元素（如 \(\Sigma_{01}\)）的变化对损失函数的影响时，实际上是在同时考虑了两个等效变化：一个是 \(\Sigma_{01}\) 的变化，另一个是 \(\Sigma_{10}\) 的变化。这两个变化对损失函数的影响是相同的，因为它们实际上是同一个值。

### 梯度计算中的对称性处理

假设损失函数 \(L\) 关于协方差矩阵 \(\Sigma\) 的梯度被存储在一个数组 `dL_dcov3D` 中，按照以下顺序存储：

- `dL_dcov3D[0]`: \(\frac{\partial L}{\partial \Sigma_{00}}\)
- `dL_dcov3D[1]`: \(\frac{\partial L}{\partial \Sigma_{01}}\)
- `dL_dcov3D[2]`: \(\frac{\partial L}{\partial \Sigma_{02}}\)
- `dL_dcov3D[3]`: \(\frac{\partial L}{\partial \Sigma_{11}}\)
- `dL_dcov3D[4]`: \(\frac{\partial L}{\partial \Sigma_{12}}\)
- `dL_dcov3D[5]`: \(\frac{\partial L}{\partial \Sigma_{22}}\)

由于 \(\Sigma\) 的对称性，对于每个非对角线元素（如 \(\Sigma_{01}\)），其梯度应该平均分配给 \(\Sigma_{01}\) 和 \(\Sigma_{10}\)。因此，在将这些梯度重新组织成矩阵形式时，我们需要将非对角线元素的梯度除以2。

### 代码实现

```cpp
glm::mat3 dL_dSigma = glm::mat3(
    dL_dcov3D[0], 0.5f * dL_dcov3D[1], 0.5f * dL_dcov3D[2],
    0.5f * dL_dcov3D[1], dL_dcov3D[3], 0.5f * dL_dcov3D[4],
    0.5f * dL_dcov3D[2], 0.5f * dL_dcov3D[4], dL_dcov3D[5]
);
```

- **对角线元素**（如 `dL_dcov3D[0]`, `dL_dcov3D[3]`, `dL_dcov3D[5]`）直接使用，因为它们只对应一个位置。
- **非对角线元素**（如 `dL_dcov3D[1]`, `dL_dcov3D[2]`, `dL_dcov3D[4]`）乘以0.5，因为在反向传播过程中，它们代表了两个等价位置的梯度（例如，`dL_dcov3D[1]` 同时代表 \(\frac{\partial L}{\partial \Sigma_{01}}\) 和 \(\frac{\partial L}{\partial \Sigma_{10}}\)）。

### 具体解释

假设我们有一个损失函数 \(L\)，其关于协方差矩阵 \(\Sigma\) 的梯度为：

- \(\frac{\partial L}{\partial \Sigma_{01}} = g_{01}\)
- \(\frac{\partial L}{\partial \Sigma_{10}} = g_{10}\)

由于 \(\Sigma_{01} = \Sigma_{10}\)，我们有 \(g_{01} = g_{10}\)。因此，在实际计算中，我们将这两个梯度合并为一个值，并在构建梯度矩阵时将其除以2：

\[ 
\text{Gradient matrix element at } (0, 1) = \frac{g_{01}}{2}
\]
\[ 
\text{Gradient matrix element at } (1, 0) = \frac{g_{10}}{2}
\]

这样做确保了我们在反向传播过程中正确地反映了协方差矩阵的对称性，避免了重复计算或不正确的梯度分配。

### 总结

乘以0.5的原因在于协方差矩阵的对称性。为了正确处理这种对称性并确保梯度的正确分配，非对角线元素的梯度在构建梯度矩阵时需要除以2。这样可以保证在优化过程中，每个参数更新步骤都准确反映了损失函数对各个参数的实际影响。这种方法对于涉及复杂3D-2D映射问题的应用至关重要，如计算机视觉、增强现实或机器学习中的应用。