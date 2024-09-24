# Galois IR

在传统编译领域, 已经存在了比较知名的LLVM IR. 针对AI编译器, 也有很多开源的MLIR, TVM IR等. Galois IR在设计之初借鉴了MLIR的多层次IR的思想, 也参考了一些其他的设计. 之所以没有使用MLIR, 还是觉得它过于庞大, "方言"(dialet)过多, 会导致生态的碎片化. 故玄青矩阵重新设计的Galois IR, 以简单, 直接, 高效为设计理念来设计, 下面是Galois IR的一些关键特性

## 基于块的编程

Galois IR是一块(tile)为基本单位的, 每个tile都是一个TensorType, TensorType就是用户所能接触到的基本类型

## 基于仿射变换的访问

我们通过仿射变换来描述对张量的数据访问, 这种形式化的描述会极大得简化我们后续IR的转换和优化. 不同于MLIR, Galois IR以此为核心, 不再支持其他同级别的方言

## 无控制流

Galois IR是无控制流的, 虽然里面很多IR最终会向下转换为低层次的控制流IR, 但在用户端, Galois IR会体现出无控制流的特性. 这会让Galois IR兼容更多的芯片, 因为它不需要芯片具备控制流逻辑

## 重要IR

目前Galois IR还没定义严格的字符格式, 不过不影响我们用一些常见的规则来表示它们.

### TensorType

TensorType是最基本的数据类型, 在Galois中, 标量类型也是一种特殊的张量类型.

* f32[] 表示f32标量
* f32[4] 向量
* f32[124x100] 矩阵

除此之外, TensorType还是可以嵌套的

* f32[4x4][128x1] Packed类型

### Tensor

### Grid

### GRidIndexVector

### Accessor

### Write

### Slice

### ArithmeticInstruction

### OperatorFunction

### Call

### Alloca

### Free
