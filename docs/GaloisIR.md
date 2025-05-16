# 基于仿射表达式的IR

形如$`F(x)=Ax+b`$的变换称为仿射表达式, 其中$`A`$是线性变换矩阵, $b$是一个偏移向量, 它们都是常量.
在一个二维循环中的例子

```c++
size_t rows = 10;
size_t cols = 10;
size_t stride = 11;
float *matrix = new float[2 * rows * stride]; //
for (size_t i = 0;i < rows; ++i){
    for (size_t j = 0; j < cols; ++j){
        // const auto &v = matrix(2 * i, j + 1);
        const auto &v = matrix[2 * i * stride + j + 1];
    }
}
```

把$`(i,j)`$设为坐标向量, 我们可以把v在matrix上的下标表示为

$$
\begin{bmatrix} row \\\ col \end{bmatrix} = \begin{bmatrix} 2 \ 0 \\\ 0 \ 1 \end{bmatrix} * \begin{bmatrix} i \\\ j \end{bmatrix}  + \begin{bmatrix} 0 \\\ 1 \end{bmatrix} = \begin{bmatrix} 2 * i \\\ j + 1 \end{bmatrix}
$$

其中访问v对应的matrix下标就是关于$`(i,j)`$的仿射表达式

$$
F = Ax + b 其中,
A = \begin{bmatrix} 2 \ 0 \\\ 0 \ 1 \end{bmatrix}, b = \begin{bmatrix} 0 \\\ 1 \end{bmatrix}, x =  \begin{bmatrix} i \\\ j \end{bmatrix}
$$

我们一般会用一维指针来表示tensor,可以根据matrix的步长$`S=\begin{bmatrix}stride \ 1\end{bmatrix}`$来计算指针偏移地址

$$
    offset = F * S = (\begin{bmatrix} 2 \ 0 \\\ 0 \ 1 \end{bmatrix} * \begin{bmatrix} i \\\ j \end{bmatrix}  + \begin{bmatrix} 0 \\\ 1 \end{bmatrix}) * \begin{bmatrix}stride \ 1\end{bmatrix} = 2 * i * stride + j + 1
$$

到这里我们展示了用一个仿射表达式来表示一个多维数据的访问, 并且可以方便地计算出地址偏移量. 在galois中我们是这样表示一个访问的, 具体可参阅galois/ir/ir.hpp

```c++
    staitc std::shared_ptr<Accessor> Accessor::Create(std::shared_ptr<Tensor> ir_tensor, // 被访问的tensor
                                                      Eigen::MatrixXi64 transform_matrix, // 仿射表达式的A
                                                      Eigen::VectorXi64 shift_vector) // 仿射表达式的b
```

除此之外, Accessor是在循环体里创建的, $x$的定义域就是循环体的的坐标grid, 也就是(0,0)->(rows, cols)的grid. galois中我们把这样的循环体直接称作Grid. 可以像相面这样创建

```c++
static std::shared_ptr<Grid> Grid::Create(Eigen::VectorXi64 shape); // shape就是循环体的rows, cols
```

循环体grid的起点均为原点(0,0), 我们只需要shape作为参数, 因为我们可以通过$b$来偏移坐标.

现在我们通过一个坐标的Swap变换(交换内外层循环)来了解Galios的IR变换

```c++
inline void Swap(std::shared_ptr<ir::Grid> ir_grid, int64_t dim0, int64_t dim1) {
    std::swap(ir_grid->shape[dim0], ir_grid->shape[dim1]); // 交换循环体的shape

    Each<ir::Accessor>(ir_grid, [=](std::shared_ptr<ir::Accessor> ir_accessor) { // 遍历grid内部的所有accessor
        ir_accessor->transform_matrix.col(dim0).swap(ir_accessor->transform_matrix.col(dim1)); // 交换A相应的的column
    });
}
```

相较于很多平台, Galios IR的变换干净利落, 这是Galios的特色之一.

基于仿射变换的IR可以形式化地描述张量计算. 很多资料把这种描述及其变换称为"多面体优化", 这是极不恰当的, 因为只需要关注简单的场景就好, "多面体"这词会让我们舍本逐末.
<<编译原理>>一书中, 把上述的内容放在第11章"Optimizing for Parallelism and Locality"中, 若想进一步了解, "且勿看其他资料, 直接阅读该书此章节即可".
