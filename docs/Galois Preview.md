# Galios平台预览

galios编译器是一个面向TPU, GPU和CPU的张量计算编译器. 围绕该编译器我们将构建了一个计算平台.

galios为人工智能和科学计算等提供强有力的软件栈, 为其提供统一的编程范式. 将LLM视为第一需求, 兼顾有限元分析, 计算机图形和计算机视觉等领域.
目前我们以LLM的工程落地为核心需求和目标

## galois软件栈的特点

现已有很多以编译器为核心组件的人工智能基础设施, galois会充分借鉴他们理念和优点. galois的初步产品规划中, 展现出来了以下特点.

### 基于仿射表达式的IR为核心, 多层次IR平滑过度

这在后面的内容中会有所体现.

### 基于块而不是线程去编程

矩阵乘法是LLM的核心运算. 矩阵乘法的高效实现在软硬件上是对应的, 都是分块处理. 所以可编程性应该体现在块上, 而没有必要精细到线程.

### 动静结合, JIT执行

我们会在宿主语言(c++或者python)中动态构建计算图, 然后从中抽离出静态计算图将其交给galois优化后生成可执行程序. JIT执行的关键优势是, 宿主中的动态shape在galois里会变成静态(常量)shape, 这是非常利于编译器优化的.

### 无控制流

张量计算是传统计算的一个补充和完善, 没必要排除传统计算而在张量计算的体系下去追求图灵完备. 在LLM等关键计算场景中并不需要控制流, 所以也不会引入控制流.

### 异构计算, 各有其长

我们鼓励异构计算, 比如会将Padding运算其提前, 然后在支持线程控制流的CPU/GPU上完成, 让TPU/NPU专注工整的张量计算. 在硬件上, 也可以将张量计算中的数据流控制交给外部的CPU或者内置的微控制器处理.

### 软硬协同发展

软硬件协同发展, 而不是相互钳制. 软件可以通过Pack等方式为硬件提供工整的数据, 而不是让硬件去处理一些碎片的场景.
举个例子, 不应该让硬件去处理3x3的矩阵乘法, 只应该让它处理所支持尺寸的.

硬件也应当为软件提供良好可编程性的高效硬件. 举个反例, 目前几乎所有cpu多层cache可编程性极差, 这对我们实现矩阵乘法很不利.

### 统一的硬件抽象

将存储的概念, 从缓存, 内存, 硬盘等拓展到集群存储, 将它们视为不同层级的存储.

相应的读写概念, 也从缓存, 内存, 硬盘读写等拓展到网络通讯.

这意味着galois平台设计之初就考虑分布式的计算, 并且分布式的逻辑不会外漏. galois寻求直接将计算表达式自动分发到不同硬件上去.

### 以TPU为未来

虽然目前nvidia gpu仍然在llm中占有较大份额, galios也会兼容nvidia gpu. 但galios认为在未来llm应用会以TPU类的加速器为主导, galios会为此而做出努力.

## 基于仿射表达式的IR

形如$F(x)=Ax+b$的变换称为仿射表达式, 其中$A$是线性变换矩阵, $b$是一个偏移向量, 它们都是常量.
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

把$(i,j)$设为坐标向量, 我们可以把v在matrix上的下标表示为

$$
\begin{bmatrix} row \\ col \end{bmatrix} = \begin{bmatrix} 2 \ 0 \\ 0 \ 1 \end{bmatrix} * \begin{bmatrix} i \\ j \end{bmatrix}  + \begin{bmatrix} 0 \\ 1 \end{bmatrix} = \begin{bmatrix} 2 * i \\ j + 1 \end{bmatrix}
$$

其中访问v对应的matrix下标就是关于(i,j)的仿射表达式

$$
F = Ax + b \\其中,

A = \begin{bmatrix} 2 \ 0 \\ 0 \ 1 \end{bmatrix}, b = \begin{bmatrix} 0 \\ 1 \end{bmatrix}, x =  \begin{bmatrix} i \\ j \end{bmatrix}
$$

我们一般会用一维指针来表示tensor,可以根据matrix的步长$S$=$\begin{bmatrix}stride \ 1\end{bmatrix}$来计算指针偏移地址

$$
    offset = F * S = (\begin{bmatrix} 2 \ 0 \\ 0 \ 1 \end{bmatrix} * \begin{bmatrix} i \\ j \end{bmatrix}  + \begin{bmatrix} 0 \\ 1 \end{bmatrix}) * \begin{bmatrix}stride \ 1\end{bmatrix} = 2 * i * stride + j + 1
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

## 一个矩阵乘法例子

正如前文所提到galois是围绕块也就是tensor去编程的, galois IR的所有Value都是Tensor, 其对应着的数据类型就是TensorType. TensorType是galois的第一数据类型, 我们可以这样使用它

```c++
// f32表示float32的标量类型, "f32[]", 注意它是一个rank为零的Tensor Type, 在galois中, 标量也属于张量, 是一种退化形式
auto mat_type_f32_100x200 = f32(100, 200); // 表示"f32[100x200]"
auto mat_packed_type_f32_4x4_8x8 = f32(4,4)(8,8) // "f32[4x4][8x8]", 我们把类似的类型称为packed类型
```

galois提供计算图接口, 下面是实现一个gemm的测试

```c++
TEST(galoisTests, TestGemm) {
    //  一种type的快捷写法, 需要用TensorTypePointer包装后才支持这种写法
    auto ir_ts_type_a = f32(4, 1)(2, 1)(1, 1024)(128, 1); //
    auto ir_ts_type_b = f32(1, 4)(1, 2)(1024, 1)(1, 128);

    auto shape_a = ir_ts_type_a->NormalizeShape();
    auto shape_b = ir_ts_type_b->NormalizeShape();

    auto ir_input_a = graph::Input::Create(f32(shape_a));
    auto ir_input_b = graph::Input::Create(f32(shape_b));
    auto ir_pack_op_a = op::PackCreator::Create(ir_ts_type_a);
    // 创建pack节点
    auto ir_pack_a = graph::ComputeNode::Create(ir_pack_op_a, {ir_input_a});
    auto ir_pack_op_b = op::PackCreator::Create(ir_ts_type_b);
    auto ir_pack_b = graph::ComputeNode::Create(ir_pack_op_b, {ir_input_b});
    auto ir_matrix_multiply_op = std::make_shared<op::MatrixMultiplyCreator>();
    // 创建矩阵乘法节点, 只有packed的数据才能得到极致性能, packed数据的层数会对应着Grid(循环体)的层数
    auto ir_mat_mul = graph::ComputeNode::Create(ir_matrix_multiply_op, {ir_pack_a, ir_pack_b});
    auto ir_unpack_op_c = std::make_shared<op::UnpackCreator>();
    // unpack
    auto ir_unpack_c = graph::ComputeNode::Create(ir_unpack_op_c, {ir_mat_mul});
    // 获取"静态"计算图, 这里的"静态"是指在galois编译器中
    auto ir_module = graph::ComputeGraph::BuildComputeGraph(ir_unpack_c, "tmp_module");

    auto ir_affine_convertor = graph::AffineConvertor::Create();
    // 将计算图->galois IR
    auto ir_operator = ir_affine_convertor->EmitModule(ir_module);
    auto prajna_compiler = prajna::Compiler::Create();
    auto llvm_codegen = std::make_shared<codegen::cpu::LlvmCodegen>(prajna_compiler->_symbol_table);
    // 将galois IR -> Prajna IR
    llvm_codegen->EmitOperatorInstance(ir_operator);
    // Prajna IR -> LLVM IR
    prajna_compiler->GenLlvm(llvm_codegen->prajna_ir_builder->module);
    // LLVM IR -> exe,  获取可执行程序(这里是一个函数指针)
    auto tmp_fun = reinterpret_cast<void (*)(float *, float *, float *)>(
        prajna_compiler->GetSymbolValue("::tmp_module"));

    Eigen::MatrixRXf eigen_matrix_f32_a = Eigen::MatrixRXf::Ones(shape_a[0], shape_a[1]);
    Eigen::MatrixRXf eigen_matrix_f32_b = Eigen::MatrixRXf::Ones(shape_b[0], shape_b[1]);
    auto shape_c = Cast<TensorType>(ir_module->type)->shape;
    // 执行
    tmp_fun(eigen_matrix_f32_a.data(), eigen_matrix_f32_b.data(), eigen_matrix_f32_c.data());
}
```

在M1平台上, 性能已接近Eigen的C++和汇编混合版本

```terminal
[==========] Running 3 tests from 1 test suite.
[----------] 3 tests from galoisTests
[ RUN      ] galoisTests.TestPeakGflops
peak performance: 100.50410400931655g
[       OK ] galoisTests.TestPeakGflops (1273 ms)
[ RUN      ] galoisTests.TestMatrixMultiply
cost time: 21840625ns, galois gemm flops: 98.3251920675347gflops
[       OK ] galoisTests.TestMatrixMultiply (95 ms)
[ RUN      ] galoisTests.TestGemm
cost time: 23845125ns, eigen gemm flops: 90.05965152206164gflops
cost time: 23944500ns, galois gemm flops: 89.68588393994446gflops
[       OK ] galoisTests.TestGemm (325 ms)
[----------] 3 tests from galoisTests (1695 ms total)
```

事实上, 几乎所有的计算场景仅需要使用Packed版本的矩阵乘法,因为矩阵乘法需要Pack, 而Packed的数据不影响进行elementwise运算.
这意味着只需要在模型的开头Pack和结尾Unpack就好了.

若不统计pack和unpack的时间, 那么galois的性能已经超过了Eigen版本的. 达到98g/flops, 已经接近M1芯片的单线程峰值.
实现这个运算, 在galois平台下只需要100行左右的代码, 完整可参阅代码galois/op/matrix_multiply.hpp. 下面一起看一下关键函数AffineExpress

```c++
class MatrixMultiplyCreator : public OperatorCreator {
   public:
    std::shared_ptr<TensorType> InferType(
        std::vector<std::shared_ptr<TensorType>> ir_input_types) override {
        if (ir_input_types[0]->IsScalar() && ir_input_types[1]->IsScalar()) {
            galois_ASSERT(ir_input_types[0] == ir_input_types[1]);
            return ir_input_types[0];
        }

        auto ir_value_type =
            this->InferType({ir_input_types[0]->value_type, ir_input_types[1]->value_type});
        return TensorType::CreateMatrixType(ir_value_type, ir_input_types[0]->shape[0],
                                            ir_input_types[1]->shape[1]);
    }

    void AffineExpress(std::vector<std::shared_ptr<ir::Tensor>> ir_inputs,
                       std::vector<std::shared_ptr<ir::Tensor>> ir_outputs,
                       std::shared_ptr<Builder> ir_builder) override {
        // 某种意义上可以从下往上把这个kernel的实现优化出来的, galois也实验了这样的优化, 但最终并没有这样做,
        // 从上往下的优化更为直接有效, 它可以充分利用已知的规则, 所以galois会预先实现一些比较关键的矩阵乘法kernel
        for (auto ir_kernel : ir_builder->kernel_queue) {
            if (ir_kernel->Match(ir_inputs, ir_outputs, ir_builder)) {
                ir_kernel->Build(ir_inputs, ir_outputs, ir_builder);
                return;
            }
        }

        auto ir_mat_a = ir_inputs[0];
        auto ir_mat_b = ir_inputs[1];
        auto ir_mat_c = ir_outputs[0];
        galois_ASSERT(ir_mat_a->type->shape[1] == ir_mat_b->type->shape[0]);
        galois_ASSERT(ir_mat_c->type->shape[0] == ir_mat_a->type->shape[0]);
        galois_ASSERT(ir_mat_c->type->shape[1] == ir_mat_b->type->shape[1]);

        // 当矩阵退化成标量的时候, 矩阵乘法就变成了c+=a*b普通数值计算
        if (ir_mat_a->type->IsScalar()) {
            auto ir_re =
                ir_builder->Create<Add>(ir_builder->Create<Mul>(ir_mat_a, ir_mat_b), ir_mat_c);
            ir_builder->Create<Write>(ir_re, ir_mat_c);
        }

        // 一个三维的Grid, 在cpu中可以等效为i,k,j的三个循环
        auto [ir_grid, scope_guard] = ir_builder->CreateGrid(Eigen::Vector3i64(
            ir_mat_a->type->shape[0], ir_mat_a->type->shape[1], ir_mat_b->type->shape[1]));

        // 可以把ir_accessor_a理解为ir_mat_a[i,k]. 这是一个"仿射表达式"
        auto ir_accessor_a = ir_builder->CreateAccessor(ir_mat_a);
        ir_accessor_a->transform_matrix(0, 0) = 1;
        ir_accessor_a->transform_matrix(1, 1) = 1;
        // ir_mat_b[k,j]
        auto ir_accessor_b = ir_builder->CreateAccessor(ir_mat_b);
        ir_accessor_b->transform_matrix(0, 1) = 1;
        ir_accessor_b->transform_matrix(1, 2) = 1;
        // ir_mat_c[i,j]
        auto ir_accessor_c = ir_builder->CreateAccessor(ir_mat_c);
        ir_accessor_c->transform_matrix(0, 0) = 1;
        ir_accessor_c->transform_matrix(1, 2) = 1;

        // 矩阵分块乘法的原理 ir_mat_c[i,j] += ir_mat_a[i,k] * ir_mat_b[k,j]. 而这里的"*"其实就是矩阵乘法本身(而非标量乘法)
        this->AffineExpress({ir_accessor_a, ir_accessor_b}, {ir_accessor_c}, ir_builder);
        // 只有在标量形式下, 矩阵乘法里的"*"所表示的运算才对应数值运算里的乘, 在element type是matrix时, "*"就是矩阵乘法它自己(这种递归也体现在了在AffineExpress的递归调用里)
        // 当我们用简单的方式, 描述一个完备的矩阵运算的时, 获得了最佳的性能, 这很好地体现了galois平台的思想
    }
};
```

## 期待更多开发者加入社区

galois项目处于起步阶段, 欢迎对AI基础设施,编译器优化和LLM相关技术感兴趣的朋友加入到项目中来. 并不需要志愿者有什么相关基础, galois期待和大家一块学习成长.

感兴趣的朋友可以先star在github上的项目<https://github.com/galois-stack/galois>, 即将开源

关注"玄青矩阵"微信公众号获取更多资讯, 后续会发布更多相关分享
