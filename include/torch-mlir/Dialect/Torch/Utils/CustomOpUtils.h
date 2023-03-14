//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Also available under a BSD-style license. See LICENSE.
//
//===----------------------------------------------------------------------===//

#include "mlir/Pass/Pass.h"
#include "llvm/ADT/StringMap.h"
#include <memory>
#include <string>

namespace mlir {

constexpr StringRef getCustomOpAttrName() { return "custom_op_attrs"; }

constexpr StringRef getCustomOpName() { return "custom_op_name"; }

constexpr StringRef getDynamicPartitionCustomName() { return "dynamic_partition"; }

constexpr StringRef getDynamicStitchCustomName() { return "dynamic_stitch"; }
} // namespace mlir