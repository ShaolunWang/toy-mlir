#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "toy/Dialect.h"
#include <numeric>
using namespace mlir;
using namespace toy;

namespace {
/// Include the patterns defined in the Declarative Rewrite framework.
// #include "ToyCombine.inc"
} // namespace

/// This is an example of a c++ rewrite pattern for the TransposeOp. It
/// optimizes the following scenario: transpose(transpose(x)) -> x
struct SimplifyRedundantTranspose : public mlir::OpRewritePattern<TransposeOp> {
  /// We register this pattern to match every toy.transpose in the IR.
  /// The "benefit" is used by the framework to order the patterns and process
  /// them in order of profitability.
  SimplifyRedundantTranspose(mlir::MLIRContext *context)
      : OpRewritePattern<TransposeOp>(context, /*benefit=*/1) {}

  /// This method attempts to match a pattern and rewrite it. The rewriter
  /// argument is the orchestrator of the sequence of rewrites. The pattern is
  /// expected to interact with it to perform any changes to the IR from here.
  mlir::LogicalResult
  matchAndRewrite(TransposeOp op,
                  mlir::PatternRewriter &rewriter) const override {
    // Look through the input of the current transpose.
    mlir::Value transposeInput = op.getOperand();
    TransposeOp transposeInputOp = transposeInput.getDefiningOp<TransposeOp>();

    // Input defined by another transpose? If not, no match.
    if (!transposeInputOp)
      return failure();

    // Otherwise, we have a redundant transpose. Use the rewriter.
    rewriter.replaceOp(op, {transposeInputOp.getOperand()});
    return success();
  }
};

struct ReshapeReshapeOptPattern : public mlir::OpRewritePattern<ReshapeOp> {
  ReshapeReshapeOptPattern(mlir::MLIRContext *context)
      : OpRewritePattern<ReshapeOp>(context, 1) {}
  mlir::LogicalResult
  matchAndRewrite(ReshapeOp op,
                  mlir::PatternRewriter &rewriter) const override {
    mlir::Value reshapeInput = op.getOperand();
    ReshapeOp reshapeInputOp = reshapeInput.getDefiningOp<ReshapeOp>();
    if (!reshapeInputOp)
      return failure();
    rewriter.replaceOp(op, {reshapeInputOp});
    return success();
  }
};
struct RedundantReshapeOptPattern : public mlir::OpRewritePattern<ReshapeOp> {
  RedundantReshapeOptPattern(mlir::MLIRContext *context)
      : mlir::OpRewritePattern<ReshapeOp>(context, 1) {}
  mlir::LogicalResult
  matchAndRewrite(ReshapeOp op,
                  mlir::PatternRewriter &rewriter) const override {
    mlir::Value reshapeInput = op.getOperand();
    mlir::Value reshapeOutput = op.getResult();
    auto reshapeInputOp = reshapeInput.getDefiningOp<ReshapeOp>();
    if (reshapeInput.getType() != reshapeOutput.getType())
      return failure();
    rewriter.replaceOp(op, {reshapeInputOp});
    return success();
  }
};

struct FoldConstantReshapeOptPattern
    : public mlir::OpRewritePattern<ReshapeOp> {
  FoldConstantReshapeOptPattern(mlir::MLIRContext *context)
      : mlir::OpRewritePattern<ReshapeOp>(context, 1) {}
  mlir::LogicalResult
  matchAndRewrite(ReshapeOp op,
                  mlir::PatternRewriter &rewriter) const override {
    mlir::Value reshapeInput = op.getOperand();
    ConstantOp reshapeconstInputOp = reshapeInput.getDefiningOp<ConstantOp>();
    if (!reshapeconstInputOp)
      return failure();
    rewriter.replaceOp(op, {reshapeconstInputOp});
    return success();
  }
};
/// Register our patterns as "canonicalization" patterns on the TransposeOp so
/// that they can be picked up by the Canonicalization framework.
void TransposeOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                              MLIRContext *context) {
  results.add<SimplifyRedundantTranspose>(context);
}

/// Register our patterns as "canonicalization" patterns on the ReshapeOp so
/// that they can be picked up by the Canonicalization framework.
void ReshapeOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                            MLIRContext *context) {
  results.add<ReshapeReshapeOptPattern, RedundantReshapeOptPattern,
              FoldConstantReshapeOptPattern>(context);
}
