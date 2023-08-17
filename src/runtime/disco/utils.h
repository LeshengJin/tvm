/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */
#ifndef TVM_RUNTIME_DISCO_UTILS_H
#define TVM_RUNTIME_DISCO_UTILS_H

#include <dlpack/dlpack.h>

#include "./session.h"
#include "./worker.h"

namespace tvm {
namespace runtime {

inline Device UseDefaultDeviceIfNone(Device device) {
  if (device.device_type == 0 && device.device_id == 0) {
    return DiscoWorker::ThreadLocal()->default_device;
  }
  return device;
}

inline std::string DLDeviceType2Str(DLDeviceType ty) {
  switch (ty) {
    case DLDeviceType::kDLCPU:
      return "cpu";
    case DLDeviceType::kDLCUDA:
      return "cuda";
    case DLDeviceType::kDLCUDAHost:
      return "cuda-host";
    case DLDeviceType::kDLOpenCL:
      return "opencl";
    case DLDeviceType::kDLVulkan:
      return "vulkan";
    case DLDeviceType::kDLMetal:
      return "metal";
    case DLDeviceType::kDLVPI:
      return "vpi";
    case DLDeviceType::kDLROCM:
      return "rocm";
    case DLDeviceType::kDLROCMHost:
      return "rocm-host";
    case DLDeviceType::kDLCUDAManaged:
      return "cuda-managed";
    case DLDeviceType::kDLOneAPI:
      return "oneapi";
    case DLDeviceType::kDLWebGPU:
      return "webgpu";
    case DLDeviceType::kDLHexagon:
      return "hexagon";
    default:
      return "Device(" + std::to_string(ty) + ")";
  }
  throw;
}

inline std::string Device2String(Device device) {
  return DLDeviceType2Str(device.device_type) + ":" + std::to_string(device.device_id);
}

}  // namespace runtime
}  // namespace tvm
#endif  // TVM_RUNTIME_DISCO_UTILS_H
