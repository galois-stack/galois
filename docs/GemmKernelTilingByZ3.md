# 使用Z3求解矩阵乘法Kernel Tile--基于NEON

在Gemm优化中, 我们会使用多层分块的策略来减少内存的访问, 不同的层级的Tile, 其内存会放在不同的cache上. 而最小一层的Tile我们这里把它称为“Kernel Tile”. Kernel Tile是和向量化指令集紧密联系的, 本文就关注如何使用Z3求解出Kernel Tile的尺寸.

## NEON 指令集概述

NEON 是 ARM 架构中的高级 SIMD（Single Instruction, Multiple Data）扩展指令集，一次可进行多个元素(向量)的运算，常用于科学计算, 图像处理和人工智能等计算密集领域.

- **向量宽度**：支持 128 位向量寄存器（Q 寄存器）, 可存储多种数据类型，例如 4 个 32 位浮点数（float32x4_t）、8 个 16 位整数（int16x8_t）等.
- **寄存器**：在 ARMv8-A 架构中, NEON 有 32 个 128 位向量寄存器（Q0-Q31）.
- **指令类型**：包括加载/存储（如 vld1q_f32, vst1q_f32）; 算术运算（如加法 vaddq_f32、乘法 vmulq_f32、融合乘加 vfmaq_f32）; 逻辑运算（如与 vandq_s32、或 vorrq_s32）; 比较和选择（如比较 vceqq_f32、选择 vbslq_f32）; 数据重排（如转置 vtrnq_f32、交错 vzipq_f32）; 以及归约操作（如 vaddvq_f32）.

### NEON中的FMA指令

NEON中的融合"乘加"（FMA, Fused Multiply-Add）指令是NEON指令集中非常重要的一部分, 因为它可以将乘法和加法操作融合为一个指令, 减少指令依赖、降低延迟, 并提升计算吞吐量. FMA执行`c = a * b + c`的操作, 其中`a`和`b`相乘，结果与`c`相加并存回c.

## 向量化指令如何实现矩阵乘法

### 内积实现(Dot Product)

内积是通过将一个向量与另一个向量对应元素相乘并累加, 最终得到一个标量结果.

图1![alt text](images/image-5.png)

例如, 在ARM NEON中, 可以通过vmulq_f32和vaddvq_f32计算浮点向量的点积:

  ```c++
  float32x4_t a;
  float32x4_t b;
  float32x4_t prod = vmulq_f32(a, b); // prod[i] = a[i] * b[i]
  float c = vaddvq_f32(prod);  // c = prod[0] + prod[1] + prod[2] + prod[3]
  ```

### 外积实现(Out Product)

外积实现是从RAM加载A的一列和B的一行到向量寄存器中, 计算两个向量之间的外积, 并将外积的结果添加到矩阵C中.

图2 ![alt text](images/image-4.png)

例如，在ARM NEON中, 可以通过带广播的vfmaq_laneq_f32计算浮点向量的外积:

  ```c++
  float32x4_t A;
  float32x4_t B;
  float32x4x4_t C;

  C.val[0] = vfmaq_laneq_f32(C.val[0], B, A, 0);
  C.val[1] = vfmaq_laneq_f32(C.val[1], B, A, 1);
  C.val[2] = vfmaq_laneq_f32(C.val[2], B, A, 2);
  C.val[3] = vfmaq_laneq_f32(C.val[3], B, A, 3);
  ```

目前主流平台采用的还是外积实现, 其更为简单直接, 并不需要增加特殊的指令
为了便于和下文区分, 我们把基于外积的上述矩阵乘法实现称为simd kernel. “simd kernel”是我们实际意义上的不可拆分单元,
再拆就无法有效使用向量化指令了. 实际中neon和avx的“simd kernel”是有区别的, 本文会以Neon的实现为例.

## 存在什么问题

### 指令流水线

**指令延迟:** FMA等向量化指令是存在较大延迟的, 需要多个时钟周期才能执行完毕.

**指令依赖:** 如果我们的指令存在依赖关系, 则需要所依赖的指令执行完毕才能执行.

为此很多cpu支持**指令流水线**技术, 一种能使多条无依赖指令重叠执行的实现技术.

我们以FMA为例, 如果多条FMA指令存在依赖，那它们的执行如下图所示:

图3![alt text](images/image.png)

FMA的输入值（a、b、c）**依赖前面的指令结果, 那么就必须等前面的指令执行完. 尤其是累加型的循环, 比如：

  ```c
  for (int i = 0; i < N; ++i)
      sum = sum + a[i] * b[i]; // fma(a[i], b[i], sum)
  ```

  每一次FMA必须等上一次的sum算完, 所以存在指令依赖, 不能流水线执行, 必须等待上一条执行完毕.

  只要是FMA的输入参数来自“前一条FMA输出”, 就有指令依赖. 这里的sum即是上一条fma指令的输出, 也是下一条指令的输入.
  所以是存在依赖关系的

**无指令依赖：**

图4![alt text](images/image-1.png)

如果我们的fma指令不存在依赖关系, 那它们就可以如上图所示流水线执行, 它们的指令延迟会得到很好的掩盖.

举例：

```c
// 并行计算不同位置
for (int i = 0; i < 4; ++i)
  for (int j = 0; j < 4; ++j)
    C[i][j] = A[i][k] * B[k][j] + C[i][j];  // fma(A[i][j], B[k][j], C[i][j])
```

这个外积实现中, 不同的 \( C[i][j] \) 之间是独立的, 每个位置累加的是自己的, 不依赖别人的结果. 所以fma指令是不存在
依赖关系的.

回到开头我们的向量化外积实现, 可以看到simd kernel里的计算指令是没有依赖的. 但simd的指令延迟比较大,
如果我们直接把simd kernel应用到分块矩阵乘法中, 就会存在一个问题, 指令数目不足以掩盖指令延迟, 那样性能就无法发挥到极致.
所以simd kernel是不可拆分的”, 但并不是最佳的kernel tile. 要获得最佳性能, 需要将simd kernel以外积的形式进一步展开.

## Kernel Tile建模

如下图所示我们可以将simd kernel进一步展开, 很多资料里把它称作register tile.

图5 ![alt text](images/image-2.png)

这些参数存在这样的关系:

- `simd_lanes = simd_bits / data_type->bits`
- `simd_kernel_tile_rows / cols = simd_lanes`
- `kernel_tile_rows / cols = register_tile_rows / cols * simd_lanes`

我们看到, 上述形式的所有fma向量指令都是无依赖的. 我们现在所要获取的就是求得最佳kernel rows和kernel cols, 这样我们
就可以合理的生成MatrixMultiplyKernel了.

通过建模, 我们把Kernel Tile的问题转化成了一个最优化问题:

- 目标: 最大化无依赖的fma向量指令数目
- 约束: 所使用的向量寄存器数量不超过cpu支持的

上图所示的fma向量指令数目可以表示为`register_tile_rows * register_tile_cols * simd_lanes`.

所需要的寄存器数目:

- `tile A: register_tile_rows`
- `tile B: register_tile_cols`
- `tile C: register_tile_rows * register_tile_cols * simd_lanes`

合计就是`z3_register_tile_rows + z3_register_tile_cols + z3_register_tile_rows * z3_register_tile_cols * simd_lanes`个

 这应该是一个非线形优化问题, 下面我们就可以通过常用的最优化工具Z3来求解该问题了.

## 通过Z3求解

### Z3介绍

Z3 是由微软开发的一个高性能 SMT（Satisfiability Modulo Theories）求解器, 广泛用于程序验证、自动化推理和约束求解等场景. Z3 支持整数、布尔、实数等类型约束建模. Z3地址：<https://github.com/Z3Prover/z3>

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
    z3_optimize.add(z3_register_tile_rows <= z3_register_tile_cols);  // 我们不需要镜像的解
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

上述就是我们通过Z3求解kernel tile的代码, 并不复杂. 在Neon指令集下, 当数据类型为f32时, 我们求得kernel tile的尺寸为
`8x12`. 上文的图5显示的就是该结果, 一共用了`2 * 3 * 4 + 2 + 3`共29个寄存器.

下面是我们基于Galois的IR实现的NeonMatrixMultiplyKernel,

```c++
class NeonMatrixMultiplyKernel : public MatrixMultiplyMicroKernel {
   public:
    static std::shared_ptr<NeonMatrixMultiplyKernel> Create(int64_t bits, int64_t rows,
                                                            int64_t cols) {
        std::shared_ptr<NeonMatrixMultiplyKernel> self(new NeonMatrixMultiplyKernel);
        self->bits = bits;
        self->bytes = self->bits / 8;
        self->rows = rows;
        self->cols = cols;
        return self;
    }

    bool Match(std::shared_ptr<ir::TensorType> ir_mat_type_a,
               std::shared_ptr<ir::TensorType> ir_mat_type_b) override {
        GALOIS_ASSERT(ir_mat_type_a->shape.size() == 2);
        GALOIS_ASSERT(ir_mat_type_b->shape.size() == 2);
        auto simd_lanes = this->bytes / ir_mat_type_a->value_type->bytes;
        if (ir_mat_type_a->value_type == ir_mat_type_b->value_type) {
            if (ir_mat_type_a->shape[1] == 1 && ir_mat_type_a->shape[0] == this->rows) {
                if (ir_mat_type_b->shape[0] == 1 && ir_mat_type_b->shape[1] == this->cols) {
                    return true;
                }
            }
        }

        return false;
    }

    void Express(std::shared_ptr<ir::Tensor> ir_mat_a, std::shared_ptr<ir::Tensor> ir_mat_b,
                 std::shared_ptr<ir::Tensor> ir_mat_c,
                 std::shared_ptr<ir::Builder> ir_builder) override {
        int64_t lanes_a = ir_mat_a->type->shape[0];
        int64_t lanes_b = ir_mat_b->type->shape[1];
        auto ir_data_type = ir_mat_a->type->DataType();
        auto simd_lanes = this->bytes / ir_data_type->bytes;
        GALOIS_ASSERT(lanes_a % simd_lanes == 0);
        GALOIS_ASSERT(lanes_b % simd_lanes == 0);
        auto ir_simd_type_a = ir_data_type->Tile(simd_lanes)->Tile(lanes_a / simd_lanes);
        auto ir_simd_type_b = ir_data_type->Tile(simd_lanes)->Tile(lanes_b / simd_lanes);

        auto ir_vec_bit_cast_a = ir_builder->Create<ir::BitCast>(ir_mat_a, ir_simd_type_a);
        auto ir_vec_bit_cast_b = ir_builder->Create<ir::BitCast>(ir_mat_b, ir_simd_type_b);
        auto ir_mat_bit_cast_c =
            ir_builder->Create<ir::BitCast>(ir_mat_c, ir_simd_type_b->Tile(lanes_a));

        for (int64_t r = 0; r < ir_simd_type_a->shape[0]; ++r) {
            auto ir_accessor_a = ir_builder->CreateAccessor(ir_vec_bit_cast_a);
            ir_accessor_a->transform_matrix.resize(0, 0);
            ir_accessor_a->shift_vector[0] = r;
            for (int64_t c = 0; c < ir_simd_type_b->shape[0]; ++c) {
                auto ir_accessor_b = ir_builder->CreateAccessor(ir_vec_bit_cast_b);
                ir_accessor_b->transform_matrix.resize(0, 0);
                ir_accessor_b->shift_vector[0] = c;

                for (int64_t i = 0; i < simd_lanes; ++i) {
                    auto ir_vector_broadcast_a = ir_builder->Create<ir::VectorBroadcast>(
                        ir_accessor_a, ir_accessor_b->type, i);
                    auto ir_mul = ir_builder->Mul(ir_vector_broadcast_a, ir_accessor_b);
                    auto ir_accessor_c_row = ir_builder->CreateAccessor(ir_mat_bit_cast_c);
                    ir_accessor_c_row->transform_matrix.resize(0, 0);
                    ir_accessor_c_row->shift_vector[0] = r * simd_lanes + i;
                    auto ir_accessor_c = ir_builder->CreateAccessor(ir_accessor_c_row);
                    ir_accessor_c->transform_matrix.resize(0, 0);
                    ir_accessor_c->shift_vector[0] = c;
                    auto ir_sum = ir_builder->Add(ir_mul, ir_accessor_c);
                    auto ir_write = ir_builder->Create<ir::Write>(ir_sum, ir_accessor_c);
                }
            }
        }
    }

   private:
    int64_t bits = 128;
    int64_t bytes = 16;
    int64_t rows;
    int64_t cols;
};
```

将kernel tile的shape代入之后, 我们可以得到下面的汇编代码：

```asm
      58: 3cdf01c8      ldur    q8, [x14, #-0x10]
      5c: ad7f29a9      ldp     q9, q10, [x13, #-0x20]
      60: 4f88113d      fmla.4s v29, v9, v8[0]
      64: 4fa8113c      fmla.4s v28, v9, v8[1]
      68: 4f88193b      fmla.4s v27, v9, v8[2]
      6c: 4fa8193a      fmla.4s v26, v9, v8[3]
      70: 4f881159      fmla.4s v25, v10, v8[0]
      74: 4fa81158      fmla.4s v24, v10, v8[1]
      78: 4f881957      fmla.4s v23, v10, v8[2]
      7c: 4fa81956      fmla.4s v22, v10, v8[3]
      80: 3cc305ab      ldr     q11, [x13], #0x30
      84: 4f881175      fmla.4s v21, v11, v8[0]
      88: 4fa81174      fmla.4s v20, v11, v8[1]
      8c: 4f881973      fmla.4s v19, v11, v8[2]
      90: 4fa81972      fmla.4s v18, v11, v8[3]
      94: 3cc205c8      ldr     q8, [x14], #0x20
      98: 4f881131      fmla.4s v17, v9, v8[0]
      9c: 4fa81130      fmla.4s v16, v9, v8[1]
      a0: 4f881927      fmla.4s v7, v9, v8[2]
      a4: 4fa81926      fmla.4s v6, v9, v8[3]
      a8: 4f881145      fmla.4s v5, v10, v8[0]
      ac: 4fa81144      fmla.4s v4, v10, v8[1]
      b0: 4f881943      fmla.4s v3, v10, v8[2]
      b4: 4fa8195f      fmla.4s v31, v10, v8[3]
      b8: 4f881162      fmla.4s v2, v11, v8[0]
      bc: 9100058c      add     x12, x12, #0x1
      c0: 4fa81161      fmla.4s v1, v11, v8[1]
      c4: 4f881960      fmla.4s v0, v11, v8[2]
      c8: 4fa8197e      fmla.4s v30, v11, v8[3]
      cc: f1007d9f      cmp     x12, #0x1f
      d0: 54fffc43      b.lo    0x58 <ltmp0+0x58>
```

这段代码是符合我们的期望的, 一共24条fma向量指令被展开, 也获得了预期的性能, 达到了130gflops, 在M4 pro平台, f32的单核峰值是130gflops多一点, 基本是接近的.

## Galois项目

Galois项目通过上述方案, 在Gemm最为关键的Kernel实现上获得了非常理想性能, 该方案具备一下优点:

- 核心代码少, 且具备良好可读性
- 具备良好的兼容能力和拓展能力
- 可维护性高

这是<https://github.com/galois-stack/galois/>的项目地址, 欢迎大家star和参与.
Galois项目的最终目标是构建一个基于编译器的AI基础设施, 以"端侧(本地)部署LLM"为主要目标.

后续我们还会更新更多的技术文档, 大家感兴趣可以关注公众号.

## 参考资料

- [Blis](https://github.com/flame/blis)
- [Eigen](https://eigen.tuxfamily.org/index.php?title=Main_Page)
- [Anatomy of High-Performance Matrix Multiplication](https://www.cs.utexas.edu/~flame/pubs/GotoTOMS_revision.pdf)
- [cpufp](https://github.com/pigirons/cpufp.git)
