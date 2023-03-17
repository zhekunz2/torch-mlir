//===- RewriteCustomOp.cpp ---------------------------------------*- C++-*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Also available under a BSD-style license. See LICENSE.
//
//===----------------------------------------------------------------------===//

#include "PassDetail.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinDialect.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "torch-mlir/Dialect/Torch/IR/TorchDialect.h"
#include "torch-mlir/Dialect/Torch/IR/TorchOps.h"
#include "torch-mlir/Dialect/Torch/Transforms/Passes.h"
#include "torch-mlir/Dialect/Torch/Utils/CustomOpUtils.h"
#include "llvm/ADT/SmallVector.h"
#include <cstddef>

using namespace mlir;
using namespace mlir::torch;
using namespace mlir::torch::Torch;

namespace {

struct RewriteDynamicPartitionPattern
    : public OpConversionPattern<CustomDynamicPartitionOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(CustomDynamicPartitionOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    int64_t numPartitions;
    if (!matchPattern(op.getNumPartitions(),
                      m_TorchConstantInt(&numPartitions))) {
      return rewriter.notifyMatchFailure(op,
                                         "non-int num_partitions unsupported");
    }
    SmallVector<Type, 4> resultTypes;
    resultTypes.reserve(numPartitions);
    auto listType = op->getResult(0).getType().cast<Torch::ListType>();
    for (int64_t i = 0; i < numPartitions; i++)
      resultTypes.push_back(listType.getContainedType());

    std::vector<NamedAttribute> customOpAttrs;
    customOpAttrs.emplace_back(rewriter.getStringAttr("num_partitions"),
                               rewriter.getI64IntegerAttr(numPartitions));
    llvm::SmallVector<NamedAttribute> attrs;
    attrs.emplace_back(rewriter.getStringAttr(getCustomOpName()),
                       rewriter.getStringAttr(getDynamicPartitionCustomName()));
    attrs.emplace_back(rewriter.getStringAttr(getCustomOpAttrName()),
                       rewriter.getDictionaryAttr(customOpAttrs));

    auto customOp =
        rewriter.create<CustomOp>(op.getLoc(), TypeRange(resultTypes),
                                  ValueRange{op.getSelf(), op.getPartition()},
                                  ArrayRef<NamedAttribute>(attrs));

    Value result = rewriter.create<PrimListConstructOp>(
        op->getLoc(), Torch::ListType::get(listType.getContainedType()),
        customOp->getResults());
    rewriter.replaceOp(op, result);
    return success();
  }
};

struct RewriteDynamicStitchPattern
    : public OpConversionPattern<CustomDynamicStitchOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(CustomDynamicStitchOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    std::vector<NamedAttribute> customOpAttrs;
    SmallVector<int64_t> outputShape;
    if (!matchPattern(op.getOutputShape(),
                      m_TorchListOfConstantInts(outputShape)))
      return rewriter.notifyMatchFailure(
          op, "only support constant int output shape");
    customOpAttrs.emplace_back(rewriter.getStringAttr("output_shape"),
                               rewriter.getI64VectorAttr(outputShape));
    llvm::SmallVector<NamedAttribute> attrs;
    attrs.emplace_back(rewriter.getStringAttr(getCustomOpName()),
                       rewriter.getStringAttr(getDynamicStitchCustomName()));
    attrs.emplace_back(rewriter.getStringAttr(getCustomOpAttrName()),
                       rewriter.getDictionaryAttr(customOpAttrs));

    SmallVector<Value> operands;
    if (auto listConstruct =
            dyn_cast<PrimListConstructOp>(op.getIndices().getDefiningOp())) {
      for (size_t i = 0; i < listConstruct->getNumOperands(); i++)
        operands.push_back(listConstruct->getOperand(i));
    } else {
      operands.push_back(op.getIndices());
    }
    if (auto listConstruct =
            dyn_cast<PrimListConstructOp>(op.getData().getDefiningOp())) {
      for (size_t i = 0; i < listConstruct->getNumOperands(); i++)
        operands.push_back(listConstruct->getOperand(i));
    } else {
      operands.push_back(op.getData());
    }
    auto customOp = rewriter.create<CustomOp>(
        op.getLoc(), TypeRange(op->getResult(0).getType()), operands,
        ArrayRef<NamedAttribute>(attrs));

    rewriter.replaceOp(op, customOp.getResult(0));
    return success();
  }
};

void populateRewriteCustomOpPatterns(RewritePatternSet &patterns) {
  patterns.add<RewriteDynamicPartitionPattern>(patterns.getContext());
  patterns.add<RewriteDynamicStitchPattern>(patterns.getContext());
}

struct RewriteCustomOpPass : public RewriteCustomOpBase<RewriteCustomOpPass> {
  void runOnOperation() override {
    ConversionTarget target(getContext());
    target.addIllegalOp<CustomDynamicPartitionOp>();
    target.addIllegalOp<CustomDynamicStitchOp>();
    target.addLegalDialect<func::FuncDialect, TorchDialect>();

    RewritePatternSet patterns(&getContext());
    populateRewriteCustomOpPatterns(patterns);

    if (failed(applyPartialConversion(getOperation(), target,
                                      std::move(patterns)))) {
      return signalPassFailure();
    }
  }
};
} // namespace

std::unique_ptr<OperationPass<func::FuncOp>>
mlir::torch::Torch::createRewriteCustomOpPass() {
  return std::make_unique<RewriteCustomOpPass>();
}
