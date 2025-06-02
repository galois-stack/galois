#include "galois_test.hpp"

#include <sys/syscall.h>

TEST(GaloisTest, TestAmx) {
    auto ir_builder = ir::Builder::Create();
    auto [ir_operator, scope_operator] =
        ir_builder->CreateOperator(ir::OperatorType::Create({ir::i8->Tile(16, 64), ir::i8->Tile(16, 64), ir::i32->Tile(16, 16)}, ir::void_), "amx_test");
    ir_builder->Create<ir::amx::LoadTileConfig>();
    ir_builder->Create<ir::amx::TileLoad>(ir_builder->GetConstant(ir::i8, 0), ir_operator->inputs[0], ir_builder->GetConstant(ir::i64, 64));
    ir_builder->Create<ir::amx::TileLoad>(ir_builder->GetConstant(ir::i8, 1), ir_operator->inputs[1], ir_builder->GetConstant(ir::i64, 64));
    ir_builder->Create<ir::amx::TileProduct>(ir_builder->GetConstant(ir::i8, 0), ir_builder->GetConstant(ir::i8, 1), ir_builder->GetConstant(ir::i8, 2));
    ir_builder->Create<ir::amx::TileStore>(ir_builder->GetConstant(ir::i8, 2), ir_operator->inputs[2], ir_builder->GetConstant(ir::i64, 16 * 4));
    ir_builder->Create<ir::amx::TileRelease>();

    auto jit_engine = jit::Engine::Create();
    auto amx_test_fun = jit_engine->EmitOperatorSymbol<void (*)(int8_t *, int8_t *, int32_t *)>(ir_operator);

    alignas(64) int8_t tile_a[16*64] = {1};
    alignas(64) int8_t tile_b[16*64] = {1};
    alignas(64) int32_t tile_c[16*16] = {0};

    for (int64_t i = 0; i < 16 * 64; ++i) {
        tile_a[i] = 1;
        tile_b[i] = 1;
    }

    int ARCH_REQ_XCOMP_PERM = 0x1023;
    int XFEATURE_XTILEDATA = 18;

    auto res = syscall(SYS_arch_prctl, ARCH_REQ_XCOMP_PERM, XFEATURE_XTILEDATA);
    ASSERT_EQ(res, 0); // fail:Invoke syscall to set ARCH_SET_STATE_USE

    amx_test_fun(tile_a, tile_b, tile_c);
    for (int i = 0; i < 16; ++i) {
        for (int j = 0; j < 16; ++j) {
            ASSERT_EQ(tile_c[i * 16 + j], 64);
        }
    }
}
