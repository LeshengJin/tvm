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
#include "./session.h"

#include <tvm/runtime/packed_func.h>
#include <tvm/runtime/registry.h>

#include "./worker.h"

namespace tvm {
namespace runtime {

TVM_REGISTER_OBJECT_TYPE(DRefObj);
TVM_REGISTER_OBJECT_TYPE(CommandQueueObj);
TVM_REGISTER_OBJECT_TYPE(SessionObj);

DRefObj::~DRefObj() { Downcast<Session>(this->session)->DeallocReg(reg_id); }

DRef::DRef(int reg_id, Session session) {
  ObjectPtr<DRefObj> p = make_object<DRefObj>();
  p->reg_id = reg_id;
  p->session = session;
  data_ = std::move(p);
}

TVMRetValue DRefObj::DebugGetFromRemote(int worker_id) {
  return Downcast<Session>(this->session)->DebugGetFromRemote(this->reg_id, worker_id);
}

void CommandQueueObj::SyncWorker(int worker_id) {
  this->BroadcastUnpacked(DiscoAction::kSyncWorker, worker_id);
  TVMArgs args = this->RecvReplyPacked(worker_id);
  ICHECK_EQ(args.size(), 2);
  DiscoAction action = static_cast<DiscoAction>(args[0].operator int());
  int ret_worker_id = args[1];
  ICHECK(action == DiscoAction::kSyncWorker);
  ICHECK_EQ(ret_worker_id, worker_id);
}

int SessionObj::AllocateReg() {
  if (this->free_regs_.empty()) {
    return this->reg_count_++;
  }
  int reg_id = this->free_regs_.back();
  this->free_regs_.pop_back();
  return reg_id;
}

void SessionObj::DeallocReg(int reg_id) {
  cmd_->BroadcastUnpacked(DiscoAction::kKillReg, reg_id);
  this->free_regs_.push_back(reg_id);
}

DRef SessionObj::CallPackedWithPacked(const TVMArgs& args) {
  constexpr uint64_t one = 1;
  constexpr int offset = 4;
  TVMValue* values = const_cast<TVMValue*>(args.values);
  int* type_codes = const_cast<int*>(args.type_codes);
  int num_args = args.num_args;
  int reg_id = AllocateReg();
  uint64_t is_dref = 0;
  CHECK_LE(num_args - offset, 64) << "ValueError: Too many arguments, maximum is 64, but got "
                                  << (num_args - offset);
  TVMArgsSetter setter(values, type_codes);
  for (int i = offset; i < num_args; ++i) {
    TVMValue t_value = values[i];
    int type_code = type_codes[i];
    CHECK(type_code != kTVMDLTensorHandle)
        << "ValueError: Cannot pass DLTensor to Session.CallPacked";
    CHECK(type_code != kTVMModuleHandle) << "ValueError: Cannot pass Module to Session.CallPacked";
    CHECK(type_code != kTVMPackedFuncHandle)
        << "ValueError: Cannot pass PackedFunc to Session.CallPacked";
    CHECK(type_code != kTVMNDArrayHandle)
        << "ValueError: Cannot pass NDArray to Session.CallPacked";
    CHECK(type_code != kTVMObjectRValueRefArg)
        << "ValueError: Cannot pass RValue to Session.CallPacked";
    if (type_code == kTVMObjectHandle) {
      TVMArgValue arg(t_value, type_code);
      if (arg.IsObjectRef<DRef>()) {
        setter(i, arg.AsObjectRef<DRef>()->reg_id);
        is_dref |= one << (i - offset);
      } else if (arg.IsObjectRef<String>()) {
        // NOTE: String is always on heap during `cmd_->BroadcastPacked` because there is no chance
        // its reference counter decreases.
        setter(i, arg.AsObjectRef<String>()->data);
      } else {
        LOG(FATAL) << "ValueError: Cannot pass Object to Session.CallPacked";
      }
    }
  }
  DRef func = args[2];
  setter(0, static_cast<int>(DiscoAction::kCallPacked));
  setter(1, reg_id);
  setter(2, func->reg_id);
  setter(3, is_dref);
  cmd_->BroadcastPacked(TVMArgs(values, type_codes, num_args));
  return DRef(reg_id, GetRef<Session>(this));
}

DRef SessionObj::GetGlobalFunc(const std::string& name) {
  int reg_id = AllocateReg();
  cmd_->BroadcastUnpacked(DiscoAction::kGetGlobalFunc, reg_id, name);
  return DRef(reg_id, GetRef<Session>(this));
}

void SessionObj::CopyFromWorker0(const NDArray& host_array, const DRef& remote_array) {
  cmd_->AppendHostNDArray(host_array);
  cmd_->BroadcastUnpacked(DiscoAction::kCopyFromWorker0, remote_array->reg_id);
}

void SessionObj::CopyToWorker0(const NDArray& host_array, const DRef& remote_array) {
  cmd_->AppendHostNDArray(host_array);
  cmd_->BroadcastUnpacked(DiscoAction::kCopyToWorker0, remote_array->reg_id);
}

void SessionObj::SyncWorker(int worker_id) { this->cmd_->SyncWorker(worker_id); }

TVMRetValue SessionObj::DebugGetFromRemote(int64_t reg_id, int worker_id) {
  return this->cmd_->DebugGetFromRemote(reg_id, worker_id);
}

struct Internal {
  static DRef CallPackedWithPacked(Session sess, const TVMArgs& args) {
    return sess->CallPackedWithPacked(args);
  }
};

TVM_REGISTER_GLOBAL("runtime.disco.SessionThreaded").set_body_typed(Session::ThreadedSession);
TVM_REGISTER_GLOBAL("runtime.disco.DRefSession").set_body_typed([](DRef obj) {
  return obj->session;
});
TVM_REGISTER_GLOBAL("runtime.disco.DRefDebugGetFromRemote")
    .set_body_method<DRef>(&DRefObj::DebugGetFromRemote);
TVM_REGISTER_GLOBAL("runtime.disco.SessionGetGlobalFunc")
    .set_body_method<Session>(&SessionObj::GetGlobalFunc);
TVM_REGISTER_GLOBAL("runtime.disco.SessionCopyFromWorker0")
    .set_body_method<Session>(&SessionObj::CopyFromWorker0);
TVM_REGISTER_GLOBAL("runtime.disco.SessionCopyToWorker0")
    .set_body_method<Session>(&SessionObj::CopyToWorker0);
TVM_REGISTER_GLOBAL("runtime.disco.SessionSyncWorker")
    .set_body_method<Session>(&SessionObj::SyncWorker);
TVM_REGISTER_GLOBAL("runtime.disco.SessionCallPacked").set_body([](TVMArgs args, TVMRetValue* rv) {
  Session self = args[0];
  *rv = Internal::CallPackedWithPacked(
      self, TVMArgs(args.values + 1, args.type_codes + 1, args.num_args - 1));
});

}  // namespace runtime
}  // namespace tvm
