#pragma once

#include "galois/ir/tensor.hpp"

namespace galois::ir::amx {

class LoadTileConfig : public Instruction {
   public:
    static std::shared_ptr<LoadTileConfig> Create() {
        std::shared_ptr<LoadTileConfig> self(new LoadTileConfig);
        self->tag = "LoadTileConfig";
        return self;
    }

    void ApplyVisitor(std::shared_ptr<Visitor> interpreter) override {
        interpreter->Visit(Cast<LoadTileConfig>(this->shared_from_this()));
    }
};

class TileRelease : public Instruction {
   public:
    static std::shared_ptr<TileRelease> Create() {
        std::shared_ptr<TileRelease> self(new TileRelease);
        self->OperandResize(0);
        self->tag = "TileRelease";
        return self;
    }

    void ApplyVisitor(std::shared_ptr<Visitor> interpreter) override {
        interpreter->Visit(Cast<TileRelease>(this->shared_from_this()));
    }
};

class TileLoad : public Instruction {
   public:
    static std::shared_ptr<TileLoad> Create(std::shared_ptr<Tensor> ir_mat_id,
                                            std::shared_ptr<Tensor> ir_mat,
                                            std::shared_ptr<Tensor> ir_stride) {
        std::shared_ptr<TileLoad> self(new TileLoad);
        self->OperandResize(3);
        self->TileRegisterId(ir_mat_id);
        self->Matrix(ir_mat);
        self->Stride(ir_stride);
        self->tag = "TileLoad";
        return self;
    }

    std::shared_ptr<Tensor> TileRegisterId() { return this->GetOperand(0); }
    void TileRegisterId(std::shared_ptr<Tensor> ir_mat_id) { this->SetOperand(0, ir_mat_id); }

    std::shared_ptr<Tensor> Matrix() { return this->GetOperand(1); }
    void Matrix(std::shared_ptr<Tensor> ir_mat) { this->SetOperand(1, ir_mat); }

    std::shared_ptr<Tensor> Stride() { return this->GetOperand(2); }
    void Stride(std::shared_ptr<Tensor> ir_mat) { this->SetOperand(2, ir_mat); }

    void ApplyVisitor(std::shared_ptr<Visitor> interpreter) override {
        interpreter->Visit(Cast<TileLoad>(this->shared_from_this()));
    }
};

class TileStore : public Instruction {
   public:
    static std::shared_ptr<TileStore> Create(std::shared_ptr<Tensor> ir_mat_id,
                                             std::shared_ptr<Tensor> ir_mat,
                                             std::shared_ptr<Tensor> ir_stride) {
        std::shared_ptr<TileStore> self(new TileStore);
        self->OperandResize(3);
        self->TileRegisterId(ir_mat_id);
        self->Matrix(ir_mat);
        self->Stride(ir_stride);

        self->tag = "TileStore";
        return self;
    }

    std::shared_ptr<Tensor> TileRegisterId() { return this->GetOperand(0); }
    void TileRegisterId(std::shared_ptr<Tensor> ir_mat_id) { this->SetOperand(0, ir_mat_id); }

    std::shared_ptr<Tensor> Matrix() { return this->GetOperand(1); }
    void Matrix(std::shared_ptr<Tensor> ir_mat) { this->SetOperand(1, ir_mat); }

    std::shared_ptr<Tensor> Stride() { return this->GetOperand(2); }
    void Stride(std::shared_ptr<Tensor> ir_mat) { this->SetOperand(2, ir_mat); }

    void ApplyVisitor(std::shared_ptr<Visitor> interpreter) override {
        interpreter->Visit(Cast<TileStore>(this->shared_from_this()));
    }
};

class TileProduct : public Instruction {
   public:
    static std::shared_ptr<TileProduct> Create(std::shared_ptr<Tensor> ir_mat_a,
                                               std::shared_ptr<Tensor> ir_mat_b,
                                               std::shared_ptr<Tensor> ir_mat_c) {
        std::shared_ptr<TileProduct> self(new TileProduct);
        self->OperandResize(3);
        self->MatrixA(ir_mat_a);
        self->MatrixB(ir_mat_b);
        self->MatrixC(ir_mat_c);
        self->tag = "TileProduct";
        return self;
    }

    void ApplyVisitor(std::shared_ptr<Visitor> interpreter) override {
        interpreter->Visit(Cast<TileProduct>(this->shared_from_this()));
    }

    std::shared_ptr<Tensor> MatrixA() { return this->GetOperand(0); }
    void MatrixA(std::shared_ptr<Tensor> ir_mat_a) { this->SetOperand(0, ir_mat_a); }

    std::shared_ptr<Tensor> MatrixB() { return this->GetOperand(1); }
    void MatrixB(std::shared_ptr<Tensor> ir_mat_b) { this->SetOperand(1, ir_mat_b); }

    std::shared_ptr<Tensor> MatrixC() { return this->GetOperand(2); }
    void MatrixC(std::shared_ptr<Tensor> ir_mat_c) { this->SetOperand(2, ir_mat_c); }
};

}  // namespace galois::ir::amx
