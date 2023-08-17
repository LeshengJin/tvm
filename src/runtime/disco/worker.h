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
/*!
 * \file worker.h
 * \brief This file defines a worker in Disco. A worker can be launched in a separate thread,
 * process or node as long as the channel supports bi-directional communication in-between the
 * worker and the controler.
 */
#ifndef TVM_RUNTIME_DISCO_WORKER_H_
#define TVM_RUNTIME_DISCO_WORKER_H_

#include <tvm/runtime/packed_func.h>

#include <atomic>
#include <condition_variable>
#include <memory>
#include <mutex>
#include <queue>
#include <vector>

namespace tvm {
namespace runtime {

class DiscoWorker;

/*!
 * \brief Controler-to-worker commands.
 * Each command is broadcasted to all workers.
 *
 * Worker-0 is a special worker that is always co-located with the controller process,
 * and thus could conveniently exchange data with the controler in memory.
 */
enum class DiscoAction : int32_t {
  kShutDown = 0,
  kKillReg = 1,
  kGetGlobalFunc = 2,
  kCallPacked = 3,
  kSyncWorker = 4,
  kCopyFromWorker0 = 5,
  kCopyToWorker0 = 6,
};

inline std::string DiscoAction2String(DiscoAction action) {
  switch (action) {
    case DiscoAction::kShutDown:
      return "kShutDown";
    case DiscoAction::kKillReg:
      return "kKillReg";
    case DiscoAction::kGetGlobalFunc:
      return "kGetGlobalFunc";
    case DiscoAction::kCallPacked:
      return "kCallPacked";
    case DiscoAction::kSyncWorker:
      return "kSyncWorker";
    case DiscoAction::kCopyFromWorker0:
      return "kCopyFromWorker0";
    case DiscoAction::kCopyToWorker0:
      return "kCopyToWorker0";
  }
  LOG(FATAL) << "ValueError: Unknown DiscoAction: " << static_cast<int>(action);
}

/*! \brief Possible kinds of allreduce or reduce operations. */
enum class ReduceKind : int32_t {
  kSum = 0,
  kProd = 1,
  kMin = 2,
  kMax = 3,
  kAvg = 4,
};

inline std::string ReduceKind2String(ReduceKind kind) {
  switch (kind) {
    case ReduceKind::kSum:
      return "kSum";
    case ReduceKind::kProd:
      return "kProd";
    case ReduceKind::kMin:
      return "kMin";
    case ReduceKind::kMax:
      return "kMax";
    case ReduceKind::kAvg:
      return "kAvg";
  }
  LOG(FATAL) << "ValueError: Unknown ReduceKind: " << static_cast<int>(kind);
}

/*! \brief A bi-directional channel for controler-worker communication. */
class DiscoChannel {
 public:
  /*! \brief Send a packed sequence to the receiver */
  virtual void Send(const TVMArgs& args) = 0;
  /*! \brief Receive a packed sequence from worker */
  virtual TVMArgs Recv() = 0;
  /*! \brief Reply a packed sequence to the sender */
  virtual void Reply(const TVMArgs& args) = 0;
  /*! \brief Receive a reply from the worker */
  virtual TVMArgs RecvReply() = 0;
};

/*!
 * \brief A special communication channel between controler and worker-0,
 * assuming they are always collocated in the same process.
 */
class WorkerZeroData {
 public:
  /*!
   * \brief The host-side arrays to passed to worker-0 for special uses, for example,
   * copy-to-worker0 and copy-from-worker0
   */
  std::queue<NDArray> host_arrays;
  /*! \brief The mutex that guards `host_arrays` */
  std::mutex queue_mutex_;
};

/*!
 * \brief A worker in Disco. It takes a channel to communication with the controler.
 * The worker can be run in a separate thread, process or node as long as the channel
 * supports bi-directional communication in-between.
 */
class DiscoWorker {
 public:
  /*!
   * \brief Construct a worker.
   * \param worker_id The id of the worker.
   * \param worker_zero_data The data shared between worker-0 and the controler. It's a nullptr if
   * the worker is not worker-0.
   * \param channel The communication channel between the worker and the controler.
   */
  explicit DiscoWorker(int worker_id, int num_workers, WorkerZeroData* worker_zero_data,
                       std::shared_ptr<DiscoChannel> channel)
      : worker_id(worker_id),
        num_workers(num_workers),
        default_device(Device{DLDeviceType::kDLCPU, 0}),
        worker_zero_data(worker_zero_data),
        channel(std::move(channel)),
        register_file{} {}

  /*! \brief Main loop of the worker */
  void MainLoop();
  /*! \brief Get the worker instance on the current thread */
  static DiscoWorker* ThreadLocal();

  /*! \brief The id of the worker.*/
  int worker_id;
  /*! \brief Total number of workers */
  int num_workers;
  /*! \brief The default device to allocate data if not specified */
  Device default_device;
  /*! \brief The name of the underlying collective communication library. */
  String ccl;
  /*!
   * \brief The data shared between worker-0 and the controler. It's a nullptr if
   * the worker is not worker-0.
   */
  WorkerZeroData* worker_zero_data;
  /*! \brief The communication channel between the worker and the controler. */
  std::shared_ptr<DiscoChannel> channel;
  /*! \brief The registers in the worker */
  std::vector<TVMRetValue> register_file;

  struct Impl;
  friend struct DiscoWorker::Impl;
};

}  // namespace runtime
}  // namespace tvm
#endif  // TVM_RUNTIME_DISCO_WORKER_H_
