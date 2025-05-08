# 从希尔伯特曲线到张量张量

## 什么是张量(Tensor)

张量(Tensor)可以理解为标量和向量概念的延伸，它是数学中用来描述多维关系的一种对象。在机器学习里，通常把多维的数据叫作张量，并用秩来表示它有多少个方向或维度。图-1简单展示了不同维度的张量：

图-1
![alt text](images/image-10.png)

## Galois中的张量表示

### Galois的TensorType类图

```plantuml

RealNumberType --|> TensorType
FloatType --|> RealNumberType
IntType --|> RealNumberType

class RealNumberType{
    bits: int64_t
    bytes: int64_t
}

class IntType {
    signed: bool
}

class TensorType{
    shape: Vector
    layout: LayoutType
    stride: Vector
    value_type: TensorType
}

TensorType::value_type *-- TensorType
```

这是Galois的TensorType类图, 我们可以看到TensorType是Galois的第一类型, 即使FloatType和IntType这类标量类型也是从
TensorType派生出来的.

### 表示标量

在Galois中我们预定义了标量,

```c++
std::shared_ptr<TensorType> f16(FloatType::Create(16));
std::shared_ptr<TensorType> f32(FloatType::Create(32));
std::shared_ptr<TensorType> f64(FloatType::Create(64));
std::shared_ptr<TensorType> i8(IntType::Create(8, true));
std::shared_ptr<TensorType> i16(IntType::Create(16, true));
std::shared_ptr<TensorType> i32(IntType::Create(32, true));
std::shared_ptr<TensorType> i64(IntType::Create(64, true));
std::shared_ptr<TensorType> u8(IntType::Create(8, false));
std::shared_ptr<TensorType> u16(IntType::Create(16, false));
std::shared_ptr<TensorType> u32(IntType::Create(32, false));
std::shared_ptr<TensorType> u64(IntType::Create(64, false));
```

我们可以通过galois::ir::f32类似的形式来使用f32标量类型.

### 创建张量

我们可以通过下面的代码来创建张量,

```c++
auto ir_value_type = ir::f32;
Eigen::VectorXi64 shape(2);
shape[0] = 16;
shape[1] = 16;
auto ir_mat_type = ir::TensorType::Create(ir::f32, shape);
```

实际上我们还有一个stride参数

```c++
Eigen::VectorXi64 stride(2);
stride[1] =  1;
stride[0] = shape[0];
auto ir_mat_type = ir::TensorType::Create(ir::f32, shape, stride);
```

上面的代码是等效的, Galois里默认使用“行优先”的数据布局.

## 行优先和列有限

如下图所示, 分别是行优先和列有限的数据布局,

![alt text](images/image-11.png)

我们可以看出, 当张量的尺寸很大时, 只有一个纬度的相邻两个元素是内存联系的, 其他纬度会存在非常大的跨度.
这是非常不利于我们进行矩阵乘法, 卷积等存在多个纬度领域关系的计算的. 我们必须寻求更为合适的张量布局, 而希尔伯特曲线给予
了我们很大的启发.

## 希尔伯特曲线

希尔伯特曲线是一种类似分形的自相似​​空间填充曲线，由数学家大卫·希尔伯特于 1891 年首次描述。除其他有趣特性外，它还允许通过保持局部性在一维和二维空间之间创建映射：这意味着一维空间中彼此靠近的两个点在二维空间折叠后也会彼此靠近。反之亦然，因为从二维到一维时不可避免地会出现这种情况。然而，即使在这种情况下，曲线也表现出尽可能保持局部性的趋势，这使其成为计算机科学和生物信息学中多种应用的宝贵工具 [1]。

![alt text](images/Hilbert_curve.png)

我们从上面的图片可以看到希尔伯特曲线的两个关键特性“局部连续”和“自相似”.

### 局部连续

希尔伯特曲线的核心特性是局部连续性，在多个纬度上相邻的点在曲线上也接近，这种内存排是远优于单纯的行优先或列优先布局的.

### 自相似

希尔伯特曲线具有分形结构，表现出自相似性：局部形状与整体形状相似, 这种性质通过递归生成实现.
**这意味着我们针对该数据结构的算法(规则)也可以递归实现**. (这个结论的证明应该是需要极高超的数学技巧的, 我这里是凭经验得出)

事实上, 个人认为“自相似”并非现象, 而是奠定宇宙万物的宇宙级公理.

## 张量张量

事实上我们的确可以实现一个和希尔伯特曲线一样的内存布局, 但我们并不会那么做, 因为那样张量的索引会变得复杂, 也会需要更多的规则,
我们借鉴它来解释张量张量即可.
我们只需要使用“张量张量”的数据形式就能获取和希尔伯特曲线一样的效果. “张量张量”本身也是一种更为简单的
“分形”.

```c++
auto ir_mat_2x2_type = ir::f32->Tile(2,2);
```

这是创建一个类型为f32的2x2的矩阵的简化写法. 我们可以如下创建张量张量,

```c++
auto ir_packed_mat_type = ir::f32->Tile(2,2)->Tile(3, 2);
```

该张量张量的布局如下图所示

![alt text](images/image-12.png)

张量张量的每一级都会有比下一级更好的连通性, 这和我们的多层存储层级是对应的.

```c++
ir::f32->Tile(32, 32)->Tile(32, 32)->Tile(32, 32)->Tile(32, 32)
//            l1 cache      l2 cache      ddr           distribute
```

如果我们设计算法之初就以张量张量为基本类型, 那我们的系统必然更加高效, 也许算法的效果也会更好, 毕竟遵循了宇宙级公理.
