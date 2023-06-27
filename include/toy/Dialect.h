#ifndef MLIR_TUTORIAL_TOY_DIALECT_H_
#define MLIR_TUTORIAL_TOY_DIALECT_H_

#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/FunctionInterfaces.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Interfaces/CallInterfaces.h"
#include "mlir/Interfaces/CastInterfaces.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "toy/ShapeInferenceInterface.h"

#include "toy/Dialect.h.inc"

#define GET_OP_CLASSES
#include "toy/Ops.h.inc"

#endif
