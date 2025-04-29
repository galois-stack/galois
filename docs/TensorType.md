# 从希尔伯特曲线到张量张量

## TensorType类图

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
' CompositeTensorType --|> TensorType
' SparseTensorType --|> TensorType
```

## 什么是张量(Tensor)

### 如何表示矩阵

### 如何表示标量

标量也是张量的一种, 属于其退化形式, 我们可以这样表示标量

### 内存布局

### 我们为什么推荐使用Matrix来表示Vector

### 行向量和列向量

## 希尔伯特曲线

网上找点资料,插图, 把这个章节写一下

### 局部连续

### 自相似

## 张量张量

### 张量张量的Layout

### LLM的关键运算矩阵乘法·
