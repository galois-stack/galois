# 使用Z3求解矩阵乘法核大小

在Gemm优化中, 我们会使用多层分块的策略来减少内存的访问, 不同的层级的Tile, 其内存会放在不同的cache上. 而最小一层的Tile我们这里把它称为“Kernel Tile”. Kernel Tile是和向量化指令集紧密联系的, 本文就关注如何使用Z3求解出Kernel Tile的尺寸

## 向量化指令如何实现矩阵乘法

### 内积实现(Dot Product)
![alt text](image-6.png)
内积实现通过将一个向量与另一个向量对应元素相乘并累计，最终得到一个标量结果。例如，在ARM NEON([ARM Intrinsics Guide](https://developer.arm.com/architectures/instruction-sets/intrinsics/))中，可以通过vmulq_f32 和 vaddvq_f32 高效计算浮点向量的点积: 





```
//伪代码
float32x4_t A[i][:], B[:][j];
float32x4_t prod = vmulq_f32(A[i][:], B[:][j]);// 对应元素相乘
float32_t result;
result = vaddvq_f32(prod);// 累加所有元素到标量

```
在这种方式下，C[i][j] 的每个值都是通过对A[i][:]行和B[:][j]列的dot product实现。

 

### 外积实现(Out Product)


![alt text](image-7.png)
外积实现是从RAM加载A的一列和B的一行到寄存器中，计算两个向量之间的外积，并将外积的结果添加到矩阵C中。
经过K次迭代后，矩阵C的计算完成，可以存储到RAM中。这里通常把C称为累加器，因为它沿着维度K累加外积。


```
伪代码：

// 矩阵乘法 C = A × B
// A: M × K, B: K × N, C: M × N
// 初始化 C 的 4x4 小块
float32x4x4_t c_tile;
c_tile.val[0] = vld1q_f32(&C[i * N + j]);
c_tile.val[1] = vld1q_f32(&C[(i + 1) * N + j]);
c_tile.val[2] = vld1q_f32(&C[(i + 2) * N + j]);
c_tile.val[3] = vld1q_f32(&C[(i + 3) * N + j]);

// 沿 K 维度迭代，累加外积
for (int k = 0; k < K; k++) {
    // 加载 A 的一列 (A[i:i+4,k])，从 A_transpose 加载
    float32x4_t a = vld1q_f32(&A_transpose[k * M + i]);

    // 加载 B 的一行 (B[k][j:j+4])
    float32x4_t b = vld1q_f32(&B[k * N + j]);

    // 计算外积并累加到 c_tile
    for (int l = 0; l < 4; ++l) {
        c_tile.val[l] = vfmaq_laneq_f32(c_tile.val[l], b, a, l);
    }
}

// 将 c_tile 写回 C
vst1q_f32(&C[i * N + j], c_tile.val[0]);
vst1q_f32(&C[(i + 1) * N + j], c_tile.val[1]);
vst1q_f32(&C[(i + 2) * N + j], c_tile.val[2]);
vst1q_f32(&C[(i + 3) * N + j], c_tile.val[3]);
  

```



目前主流平台采用的还是外积实现, 其更为简单直接, 并不需要增加特殊的指令
为了便于和下文区分, 我们把基于的外积的上术矩阵乘法实现成为simd kernel. “simd kernel”是我们实际意义上的不可拆分单元,
再拆就无法有效使用向量化指令了. 实际中neon和avx的“simd kernel”是有区别的, 本文会以Neon的实现为例.

## 存在什么问题

### 指令延迟
**指令延迟：** 指令固有的执行时间，一条指令的数据可供另一条指令使用所需的处理器时钟数。


Intel的AVX-512指令([Intel® Intrinsics Guide](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#ig_expand=4407,3067,3107))，用于执行融合乘加操作（Fused Multiply-Add），如图所示：

![alt text](image-4.png)

该指令的延迟是 4 个时钟周期。也就是说，从指令开始执行到其结果准备好供后续指令使用，需要经过 4 个时钟周期。如果后续指令依赖于 _mm256_fmadd_ps 的结果，那么至少需要等待 4 个时钟周期才能继续执行。因此，延迟决定了流水线中的指令之间的时间间隔。

流水线技术可以提高整体的指令吞吐率（单位时间内完成更多指令），但并不能缩短单条指令自身的执行延迟。即使流水线再深，每条指令完成所需的时间（延迟）仍然由指令的复杂性和硬件实现决定。

### 指令流水线


**指令依赖:** 某些指令需等待前序指令完成才能执行。

**流水线:** 一种能使多条指令重叠执行的实现技术。

简化的流水线的指令执行，如图：
![alt text](image-5.png)
流水线指令执行通常包含5个步骤：
1. 指令提取(IF)
2. 指令译码(ID)
3. 执行/有效地址(EX)
4. 存储器访问(MEM)
5. 写回(WB)

在CPU流水线中，指令依赖会导致数据冒险，即一条指令依赖于前面一条尚在流水线中的指令。
 例如，假设有一条加法指令，它后面紧跟着一条使用加法的和的减法指令(x10)：
 ```
 ADD X10, X1, X2      // 第1条指令：X10 = X1 + X2
SUB X12, X10, X3     // 第2条指令：X12 = X10 - X3 （依赖X10）
```



不存依赖的指令, 可以通过指令流水线来掩盖延迟

我们可以看到simd kernel里的计算指令是没有依赖的. 但simd的指令延迟比较大,
如果我们直接把simd kernel应用到分块矩阵乘法中, 就会存在一个问题, 指令数目不足以掩盖指令延迟, 那样性能就无法发挥到极致

## Kernel Tile建模

现在我们把Kernel Tile的问题转化成了一个最优化问题

* 目标: 最大化无依赖的乘加指令
* 约束: 所使用的向量寄存器数量不超过cpu支持的

## 通过Z3求解

### Z3介绍


Z3 是由微软开发的一个高性能 SMT（Satisfiability Modulo Theories）求解器，广泛用于程序验证、自动化推理和约束求解等场景。Z3 支持整数、布尔、实数等类型约束建模。Z3地址：https://github.com/Z3Prover/z3

### 通过z3求解

``` c++

std::tuple<int64_t, int64_t> GetKernelTileShape(std::shared_ptr<ir::TensorType> ir_data_type,
                                                    std::shared_ptr<NativeCpuInfo> cpu_info) {
        int32_t simd_register_count = cpu_info->SimdRegisterCount();
        int32_t simd_lanes = (cpu_info->SimdBits() / 8) / ir_data_type->bytes;
        /// 通过Z3来求解寄存器分块， 该问题不是一个线性规划问题， 所以采用Z3来处理
        z3::context z3_context;
        z3::params z3_params(z3_context);
        z3_params.set("priority", z3_context.str_symbol("register tile"));
        z3::optimize z3_optimize(z3_context);
        z3_optimize.set(z3_params);
        z3::expr z3_register_tile_rows = z3_context.int_const("z3_register_tile_rows");
        z3::expr z3_register_tile_cols = z3_context.int_const("z3_register_tile_cols");
        z3_optimize.add(z3_register_tile_rows > 0);
        z3_optimize.add(z3_register_tile_cols > 0);
        z3_optimize.add(z3_register_tile_rows >= z3_register_tile_cols);
        // z3_register_tile_rows + z3_register_tile_cols : 行和列的寄存器都需要保留，
        // 这样才能复用数据 z3_register_tile_rows * z3_register_tile_cols * int32_t(simd_lanes)：
        // 用于存储外积的结果
        z3_optimize.add(z3_register_tile_rows + z3_register_tile_cols +
                            z3_register_tile_rows * z3_register_tile_cols * simd_lanes <
                        simd_register_count);
        // 最大化无依赖的计算指令数目, 同时也最大化了计算强度
        z3::optimize::handle z3_handle_x =
            z3_optimize.maximize(z3_register_tile_rows * z3_register_tile_cols);
        GALOIS_ASSERT(z3_optimize.check() == z3::sat);
        z3::model z3_model = z3_optimize.get_model();
        auto register_tile_rows = z3_model.eval(z3_register_tile_rows).get_numeral_int64();
        auto register_tile_cols = z3_model.eval(z3_register_tile_cols).get_numeral_int64();
        return {register_tile_rows * simd_lanes, register_tile_cols * simd_lanes};
    }
```

## 总结

总结一下我们方案的优点, 介绍一下其在我们galois项目中的应用. 附上我们galois项目的地址

## 作者介绍

### 孙腾

介绍一下你自己
