#ifndef TOY_PASSES_H
#define TOY_PASSES_H
#include <memory>
class Pass;

namespace mlir {
class Pass;
namespace toy {
std::unique_ptr<Pass> createShapeInferencePass();

std::unique_ptr<mlir::Pass> createLowerToAffinePass();
} // namespace toy
} // namespace mlir

#endif
