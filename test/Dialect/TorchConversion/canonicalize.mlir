// RUN: torch-mlir-opt %s -canonicalize | FileCheck %s

// CHECK-LABEL:   func.func @torch_c.from_i64() -> !torch.int {
// CHECK:     %[[INT5:.*]] = torch.constant.int 5
// CHECK:     return %[[INT5]] : !torch.int
func.func @torch_c.from_i64() -> !torch.int {
  %c5_i64 = arith.constant 5 : i64
  %0 = torch_c.from_i64 %c5_i64
  return %0 : !torch.int
}

// CHECK-LABEL:   func.func @torch_c.to_i64$i64() -> i64 {
// CHECK:     %[[C5_I64:.*]] = arith.constant 5 : i64
// CHECK:     return %[[C5_I64]] : i64
func.func @torch_c.to_i64() -> i64 {
  %int5 = "torch.constant.int"() {value = 5 : i64} : () -> !torch.int
  %0 = torch_c.to_i64 %int5
  return %0 : i64
}

// CHECK-LABEL:   func.func @torch_c.to_i64$si64() -> i64 {
// CHECK:     %[[C1_I64:.*]] = arith.constant 1 : i64
// CHECK:     return %[[C1_I64]] : i64
func.func @torch_c.to_i64$si64() -> i64 {
  %int1 = "torch.constant.int"() {value = 1 : si64} : () -> !torch.int
  %0 = torch_c.to_i64 %int1
  return %0 : i64
}

// CHECK-LABEL:   func.func @torch_c.from_i64$to_i64() -> i64 {
// CHECK:     %[[C5_I64:.*]] = arith.constant 5 : i64
// CHECK:     return %[[C5_I64]] : i64
func.func @torch_c.from_i64$to_i64() -> i64 {
  %c5_i64 = arith.constant 5 : i64
  %0 = torch_c.from_i64 %c5_i64
  %1 = torch_c.to_i64 %0
  return %1 : i64
}

// CHECK-LABEL:   func.func @torch_c.to_i64$from_i64() -> !torch.int {
// CHECK:     %[[INT5:.*]] = torch.constant.int 5
// CHECK:     return %[[INT5]] : !torch.int
func.func @torch_c.to_i64$from_i64() -> !torch.int {
  %int5 = torch.constant.int 5
  %0 = torch_c.to_i64 %int5
  %1 = torch_c.from_i64 %0
  return %1 : !torch.int
}

// CHECK-LABEL:   func.func @torch_c.from_f64() -> !torch.float {
// CHECK:     %[[FLOAT5:.*]] = torch.constant.float 5.000000e+00
// CHECK:     return %[[FLOAT5]] : !torch.float
func.func @torch_c.from_f64() -> !torch.float {
  %c5_f64 = arith.constant 5.000000e+00 : f64
  %0 = torch_c.from_f64 %c5_f64
  return %0 : !torch.float
}

// CHECK-LABEL:   func.func @torch_c.to_f64() -> f64 {
// CHECK:     %[[C5_f64:.*]] = arith.constant 5.000000e+00 : f64
// CHECK:     return %[[C5_f64]] : f64
func.func @torch_c.to_f64() -> f64 {
  %float5 = torch.constant.float 5.000000e+00
  %0 = torch_c.to_f64 %float5
  return %0 : f64
}

// CHECK-LABEL:   func.func @torch_c.from_f64$to_f64() -> f64 {
// CHECK:     %[[C5_f64:.*]] = arith.constant 5.000000e+00 : f64
// CHECK:     return %[[C5_f64]] : f64
func.func @torch_c.from_f64$to_f64() -> f64 {
  %c5_f64 = arith.constant 5.000000e+00 : f64
  %0 = torch_c.from_f64 %c5_f64
  %1 = torch_c.to_f64 %0
  return %1 : f64
}

// CHECK-LABEL:   func.func @torch_c.to_f64$from_f64() -> !torch.float {
// CHECK:     %[[FLOAT5:.*]] = torch.constant.float 5.000000e+00
// CHECK:     return %[[FLOAT5]] : !torch.float
func.func @torch_c.to_f64$from_f64() -> !torch.float {
  %float5 = torch.constant.float 5.000000e+00
  %0 = torch_c.to_f64 %float5
  %1 = torch_c.from_f64 %0
  return %1 : !torch.float
}
