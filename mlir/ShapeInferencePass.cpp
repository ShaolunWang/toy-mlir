#include "mlir/Pass/Pass.h"
#include "toy/Dialect.h"
#include "toy/Passes.h"
#include "toy/ShapeInferenceInterface.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"

#define DEBUG_TYPE "shape-inference"

using namespace mlir;
using namespace toy;

/// Include the auto-generated definitions for the shape inference interfaces.
#include "toy/ShapeInferenceOpInterfaces.cpp.inc"

namespace {
/// The ShapeInferencePass is a pass that performs intra-procedural
/// shape inference.
///
///    Algorithm:
///
///   1) Build a worklist containing all the operations that return a
///      dynamically shaped tensor: these are the operations that need shape
///      inference.
///   2) Iterate on the worklist:
///     a) find an operation to process: the next ready operation in the
///        worklist has all of its arguments non-generic,
///     b) if no operation is found, break out of the loop,
///     c) remove the operation from the worklist,
///     d) infer the shape of its output from the argument types.
///   3) If the worklist is empty, the algorithm succeeded.
///
struct ShapeInferencePass
    : public mlir::PassWrapper<ShapeInferencePass, OperationPass<toy::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ShapeInferencePass)

  static bool returnsDynamicShape(Operation *op) {
    return llvm::any_of(op-> getResultTypes(), [](Type operandType){
      return !operandType.isa<RankedTensorType>();
    });
  }
  static bool allOperandsInferred(Operation *op){
    return llvm::all_of(op->getOperandTypes(), [](Type operandType){
      return operandType.isa<RankedTensorType>();
    });
  }

  // Populate the worklist with the operations that need shape inference:
  // these are operations that return a dynamic shape.
  void runOnOperation() override {
    auto f = getOperation();
    llvm::SmallPtrSet<mlir::Operation *, 16> opWorklist;
    f.walk([&](mlir::Operation *op) {
      if (returnsDynamicShape(op)) {
        opWorklist.insert(op);
      }
    });

    // Find the next operation ready for inference, that is an operation
    // with all operands already resolved (non-generic).
    while (!opWorklist.empty()){
      auto nextop = llvm::find_if(opWorklist, allOperandsInferred);
      if (nextop == opWorklist.end())
        break;
      Operation *op = *nextop;
      LLVM_DEBUG(llvm::dbgs() << "Inferring shape for: " << *op << "\n");
      if (auto shapeop = dyn_cast<ShapeInference>(op)){
        shapeop.inferShapes();
      }
      else{
        op->emitError("unable to infer the shape");
        signalPassFailure();
      }
    }
  }
};
} // namespace

std::unique_ptr<mlir::Pass> mlir::toy::createShapeInferencePass() {
  return std::make_unique<ShapeInferencePass>();
}
