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
#include <dmlc/io.h>

#include <condition_variable>
#include <memory>
#include <mutex>
#include <thread>
#include <utility>

#include "../../support/arena.h"
#include "../../support/ring_buffer.h"
#include "../minrpc/rpc_reference.h"
#include "./session.h"
#include "./worker.h"

namespace tvm {
namespace runtime {

class ThreadedMessageQueue : public dmlc::Stream {
 public:
  void Send(const TVMArgs& args) {
    RPCReference::ReturnPackedSeq(args.values, args.type_codes, args.num_args, this);
    NotifyEnqueue();
  }

  TVMArgs Recv() {
    WaitDequeue();
    uint64_t packet_nbytes = 0;
    RPCCode code = RPCCode::kReturn;
    this->Read(&packet_nbytes);
    this->Read(&code);
    TVMValue* values = nullptr;
    int* type_codes = nullptr;
    int num_args = 0;
    RPCReference::RecvPackedSeq(&values, &type_codes, &num_args, this);
    return TVMArgs(values, type_codes, num_args);
  }

 protected:
  void NotifyEnqueue() {
    {
      std::lock_guard<std::mutex> lock{mutex_};
      ++msg_cnt_;
    }
    condition_.notify_one();
  }

  void WaitDequeue() {
    std::unique_lock<std::mutex> lock(mutex_);
    condition_.wait(lock, [this] { return msg_cnt_.load() > 0; });
    --msg_cnt_;
  }

  void MessageStart(uint64_t packet_nbytes) {
    std::lock_guard<std::mutex> lock(mutex_);
    size_t n = ring_buffer_.bytes_available();
    n += packet_nbytes + sizeof(uint64_t);
    this->ring_buffer_.Reserve(n);
  }

  void MessageDone() {}

  void ThrowError(RPCServerStatus status) {
    LOG(FATAL) << "InternalError: Unexpected error in RPC: " << RPCServerStatusToString(status);
  }

  template <typename T>
  T* ArenaAlloc(int count) {
    static_assert(std::is_pod<T>::value, "need to be trival");
    return arena_.template allocate_<T>(count);
  }

  size_t Read(void* data, size_t size) final {
    std::lock_guard<std::mutex> lock(mutex_);
    ring_buffer_.Read(data, size);
    return size;
  }

  void Write(const void* data, size_t size) final {
    std::lock_guard<std::mutex> lock(mutex_);
    ring_buffer_.Write(data, size);
  }

  using dmlc::Stream::Read;
  using dmlc::Stream::ReadArray;
  using dmlc::Stream::Write;
  using dmlc::Stream::WriteArray;
  friend struct RPCReference;

  std::mutex mutex_;
  std::atomic<int> msg_cnt_{0};
  std::condition_variable condition_;

  support::RingBuffer ring_buffer_;
  support::Arena arena_;
};

class DiscoThreadChannel final : public DiscoChannel {
 public:
  void Send(const TVMArgs& args) { controler_to_worker_.Send(args); }
  TVMArgs Recv() { return controler_to_worker_.Recv(); }
  void Reply(const TVMArgs& args) { worker_to_controler_.Send(args); }
  TVMArgs RecvReply() { return worker_to_controler_.Recv(); }

  ThreadedMessageQueue controler_to_worker_;
  ThreadedMessageQueue worker_to_controler_;
};

class ThreadedCommandQueueObj final : public CommandQueueObj {
 public:
  explicit ThreadedCommandQueueObj(int n_workers) {
    for (int i = 0; i < n_workers; ++i) {
      std::shared_ptr<DiscoThreadChannel> channel = std::make_shared<DiscoThreadChannel>();
      WorkerZeroData* data = (i == 0) ? &worker_zero_data_ : nullptr;
      workers_.emplace_back(std::make_unique<DiscoWorker>(i, n_workers, data, channel));
      channels_.emplace_back(std::move(channel));
      worker_threads_.emplace_back([worker = workers_.back().get()] { worker->MainLoop(); });
    }
  }

  ~ThreadedCommandQueueObj() {
    this->BroadcastUnpacked(DiscoAction::kShutDown, 0);
    for (std::thread& worker : this->worker_threads_) {
      worker.join();
    }
  }

  void BroadcastPacked(const TVMArgs& args) final {
    for (std::shared_ptr<DiscoThreadChannel>& channel : this->channels_) {
      channel->Send(args);
    }
  }

  void AppendHostNDArray(const NDArray& host_array) final {
    std::lock_guard<std::mutex> lock(worker_zero_data_.queue_mutex_);
    worker_zero_data_.host_arrays.push(host_array);
  }

  TVMArgs RecvReplyPacked(int worker_id) final { return channels_[worker_id]->RecvReply(); }

  TVMRetValue DebugGetFromRemote(int64_t reg_id, int worker_id) {
    this->SyncWorker(worker_id);
    return this->workers_.at(worker_id)->register_file.at(reg_id);
  }

  static constexpr const char* _type_key = "runtime.disco.ThreadedCommandQueue";
  TVM_DECLARE_FINAL_OBJECT_INFO(ThreadedCommandQueueObj, CommandQueueObj);

 private:
  std::vector<std::shared_ptr<DiscoThreadChannel>> channels_;
  std::vector<std::unique_ptr<DiscoWorker>> workers_;
  std::vector<std::thread> worker_threads_;
  WorkerZeroData worker_zero_data_;
};

TVM_REGISTER_OBJECT_TYPE(ThreadedCommandQueueObj);

Session Session::ThreadedSession(int num_workers) {
  ObjectPtr<SessionObj> n = make_object<SessionObj>();
  n->cmd_ = CommandQueue(make_object<ThreadedCommandQueueObj>(num_workers));
  n->reg_count_ = 1;
  n->free_regs_.clear();
  return Session(std::move(n));
}

}  // namespace runtime
}  // namespace tvm
