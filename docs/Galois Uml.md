# Galois UML

## 类图

### Galios IR

```plantuml

class Vector
class Matrix
class LayoutType

class Type

RealNumberType --> Type
FloatType --> RealNumberType
IntType --> RealNumberType



class RealNumberType{
    bits: int64_t
    bytes: int64_t
}

class IntType {
    signed: bool
}


class TensorType{
    value_type: TensorType
    shape: Vector
    layout: LayoutType
    -data_type: Type
    - stride: Vector
}

TensorType::data_type *-left- Type
TensorType::value_type *-- TensorType

class Tensor{
    type: TensorType
}

Tensor::type *--> TensorType

Tensor <|-- Constant

class Block{
    tensors: list<Tensor>
}

Tensor <|-- Block
Block::tensors *-- Tensor

class Instruction

Tensor <|-- Instruction

Instruction <|-- ArithmeticInstruction

ArithmeticInstruction <|-- Add
ArithmeticInstruction <|-- Sub
ArithmeticInstruction <|-- Mul
ArithmeticInstruction <|-- Div

Instruction <|-- Alloca
Instruction <|-- Free

class Grid {

}

Block <|-- Grid

class Accessor

Instruction <|-- Accessor
Grid --o Accessor
GridIndexVector --o Accessor
Grid o-- GridIndexVector

Instruction <|-- Slice

Instruction <|-- View

class View{
    static Stride(strides): Tensor
    static Shift(offsets): Tensor
}

class Write{
    value: Tensor
    variable: Tensor
}

Instruction <|-- Write

class OperatorFunction{
    inputs: vector<Tensor>
    outputs: vector<Tensor>
}

Instruction <|-- OperatorFunction
Instruction <|-- Call

class Call{
    Callee: OperatorFunction
    Paramters: vector<Tensor>
}

```

Galios 执行图

```mermaid
stateDiagram
    direction LR
    c++api: C++ API
    galoisir: Galois IR
    prajnair: Prajna IR
    llvmir: LLVM IR
    jitengine: JIT Engine
    c++api --> galoisir
    galoisir --> prajnair
    prajnair --> llvmir
    llvmir --> jitengine
```

Galois到Prajna的IR转换的一个典型时序图, 图中展示了多次Grid场景下, 其Grid坐标的有效区间

```plantuml


Galois -> Prajna:EmitOperatorFunction
Galois -> Prajna:EmitGrid0
Galois -> Prajna:EmitGridIndexVector0
note right:最外层Grid坐标有效
activate Prajna #Red
Galois -> Prajna: EmitAccess0
Galois -> Prajna:EmitGrid1
Galois -> Prajna:EmitGridIndexVector1
activate Prajna #Blue
note right: 里层Grid坐标有效
Galois -> Prajna:EmitAccess1
Galois <-- Prajna:EmitGrid0
deactivate Prajna
note right: "里层Grid坐标无效, 最外层坐标重新生效"
Galois -> Prajna:EmitAccess2
Galois <-- Prajna:EmitGrid1
deactivate Prajna
Galois <-- Prajna:EmitOperatorFunction
```
