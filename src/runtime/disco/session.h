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
 * \file session.h
 * \brief This file defines a Session in Disco. Session is the primary interface that users interact
 * with the distributed runtime.
 *
 * This file contains the definition of the following objects:
 * 1) DRef: A controler-side reference to an object that exists on all workers.
 * 2) CommandQueue: An abstract class that defines the interface of a command queue that can be used
 * to broadcast commands to all workers. The command queue could be implemented differently for each
 * runtime, for example, a thread pool of workers, shared-memory queue or a TCP socket.
 * The unlderying implementation of a command queue, although hidden, is required to create a
 * channel between the controler and each of the worker.
 * 3) Session: The primary interface to interact with the distributed runtime.
 */
#ifndef TVM_RUNTIME_DISCO_SESSION_H_
#define TVM_RUNTIME_DISCO_SESSION_H_

#include <tvm/runtime/packed_func.h>

#include "./worker.h"

namespace tvm {
namespace runtime {

class Session;

/*!
 * \brief An object that exists on all workers.
 *
 * The controler process will assign a unique "register id" to each object,
 * and the worker process will use this id to refer to the object residing on itself.
 */
class DRefObj : public Object {
 public:
  /*!\ brief Virtual destructor that send dellocation command for `reg_id` */
  ~DRefObj();

  /*! \brief The id of the register */
  int64_t reg_id;
  /*! \brief Back-pointer to the host controler session */
  ObjectRef session{nullptr};
  /*!
   * \brief Get the value of a DRef from a remote worker.
   * \param worker_id The id of the worker to be fetched from.
   * \return The value of the register.
   */
  TVMRetValue DebugGetFromRemote(int worker_id);

  static constexpr const char* _type_key = "runtime.disco.DRef";
  TVM_DECLARE_FINAL_OBJECT_INFO(DRefObj, Object);
};

/*!
 * \brief Managed reference to DRefObj
 * \sa DRefObj
 */
class DRef : public ObjectRef {
 public:
  TVM_DEFINE_MUTABLE_NOTNULLABLE_OBJECT_REF_METHODS(DRef, ObjectRef, DRefObj);

 protected:
  /*! \brief Non-public constructor: DRef is only allowed to be created by Session */
  explicit DRef(int reg_id, Session session);
  friend class SessionObj;
};

/*!
 * \brief A command queue to broadcast commands to all workers.
 * The underlying implementation could be different depending on the runtime,
 * for example, a thread pool of workers, shared-memory queue or a TCP socket.
 *
 * A CommandQueue needs to implement the pure virtual methods as defined in this class,
 * and Disco Session will be able to assemble all the commands into a command queue
 */
class CommandQueueObj : public Object {
 public:
  /*! \brief Virtual destructor. */
  virtual ~CommandQueueObj() {}
  /*!
   * \brief Broadcast a command to all workers.
   * \param action The action to be broadcasted.
   * \param reg_id The register id attached to the action.
   * \param args The variadic arguments
   * \tparam Args In the variadic arguments, the supported types include:
   * - integers and floating point numbers;
   * - DLDataType;
   * - DLDevice;
   * - std::string;
   * - DRef.
   * Examples of unsupported types:
   * - NDArray, DLTensor;
   * - TVM Objects, including PackedFunc and Module;
   */
  template <typename... Args>
  void TVM_ALWAYS_INLINE BroadcastUnpacked(DiscoAction action, int64_t reg_id, Args&&... args);
  /*!
   * \brief Broadcast a command to all workers via TVM's PackedFunc calling convention.
   * As part of the calling convention, The first argument in the packed sequence must be
   * the action, and the second argument must be the register id.
   * \param TVMArgs The input arguments in TVM's PackedFunc calling convention
   * \sa CommandQueueObj::BroadcastUnpacked
   */
  virtual void BroadcastPacked(const TVMArgs& args) = 0;
  /*!
   * \brief Receive a packed sequence from a worker. This function is usually called by the
   * controler to communicate with worker-0, because the worker-0 is assumed to be always collocated
   * with the controler. Receiving from other workers may not be supported.
   * \return The packed sequence received.
   */
  virtual TVMArgs RecvReplyPacked(int worker_id) = 0;
  /*!
   * \brief Get the value of a register from a remote worker.
   * \param reg_id The id of the register to be fetched.
   * \param worker_id The id of the worker to be fetched from.
   * \return The value of the register.
   */
  virtual TVMRetValue DebugGetFromRemote(int64_t reg_id, int worker_id) = 0;
  /*!
   * \brief Append an controler-side NDArray to a special queue used to communicate with worker-0.
   * \param host_array The array to be appended to worker-0
   */
  virtual void AppendHostNDArray(const NDArray& host_array) = 0;
  /*!
   * \brief Synchrnoize the controler with a worker, and it will wait until worker finishes
   * executing this instruction.
   * \param worker_id The id of the worker to be synced with.
   * \note This function is usually used for worker-0, because it is the only worker that is
   * assumed to collocate with the controler. Syncing with other workers may not be supported.
   */
  virtual void SyncWorker(int worker_id);

  static constexpr const char* _type_key = "runtime.disco.CommandQueue";
  TVM_DECLARE_BASE_OBJECT_INFO(CommandQueueObj, Object);

 private:
  /*!
   * \brief An intermediate step used by `BroadcastPacked` that packs `args` into TVM's PackedFunc
   * calling convention, and feed to `BoradcastPacked`.
   */
  template <typename... Args>
  void TVM_ALWAYS_INLINE BroadcastUnpackedImpl(Args&&... args);
};

/*!
 * \brief Managed reference to CommandQueueObj
 * \sa CommandQueueObj
 */
class CommandQueue : public ObjectRef {
 public:
  TVM_DEFINE_MUTABLE_NOTNULLABLE_OBJECT_REF_METHODS(CommandQueue, ObjectRef, CommandQueueObj);
};

/*!
 * \brief A Disco interactive session. It allows users to interact with the Disco command queue with
 * various PackedFunc calling convention.
 */
class SessionObj : public Object {
 public:
  /*! \brief Get a global functions on workers. */
  DRef GetGlobalFunc(const std::string& name);
  /*!
   * \brief Call a PackedFunc on workers providing variadic arguments.
   * \tparam Args In the variadic arguments, the supported types include:
   * - integers and floating point numbers;
   * - DLDataType;
   * - DLDevice;
   * - std::string;
   * - DRef.
   * Examples of unsupported types:
   * - NDArray, DLTensor;
   * - TVM Objects, including PackedFunc and Module;
   * \param func The function to be called.
   * \param args The variadic arguments.
   * \return The return value of function call
   */
  template <typename... Args>
  DRef TVM_ALWAYS_INLINE CallPacked(const DRef& func, Args&&... args);
  /*!
   * \brief Copy the controler-side NDArray to worker-0
   * \param host_array The array to be copied to worker-0
   * \param remote_array The NDArray on worker-0
   */
  void CopyFromWorker0(const NDArray& host_array, const DRef& remote_array);
  /*!
   * \brief Copy an NDArray from worker-0 to the controler-side NDArray
   * \param host_array The array to be copied to worker-0
   * \param remote_array The NDArray on worker-0
   */
  void CopyToWorker0(const NDArray& host_array, const DRef& remote_array);
  /*!
   * \brief Synchrnoize the controler with a worker, and it will wait until worker finishes
   * executing this instruction.
   * \param worker_id The id of the worker to be synced with.
   * \note This function is usually used for worker-0, because it is the only worker that is
   * assumed to collocate with the controler. Syncing with other workers may not be supported.
   */
  void SyncWorker(int worker_id);
  /*!
   * \brief Get the value of a register from a remote worker.
   * \param reg_id The id of the register to be fetched.
   * \param worker_id The id of the worker to be fetched from.
   * \return The value of the register.
   */
  TVMRetValue DebugGetFromRemote(int64_t reg_id, int worker_id);

  /*! \brief The command queue */
  CommandQueue cmd_{nullptr};
  /*! \brief Number of registers used, including those in `free_regs_` */
  int reg_count_;
  /*! \brief The regsiter ids that have been deallocated */
  std::vector<int64_t> free_regs_;

  static constexpr const char* _type_key = "runtime.disco.Session";
  TVM_DECLARE_FINAL_OBJECT_INFO(SessionObj, Object);

 protected:
  /*! \brief Call packed function on each worker using a packed sequence */
  DRef CallPackedWithPacked(const TVMArgs& args);
  /*! \brief intermediate method used by `CallPacked` to convert unpacked inputs to packed ones */
  template <typename... Args>
  DRef TVM_ALWAYS_INLINE CallPackedWithUnpacked(Args&&... args);
  /*! \brief Allocate a register id, either from `free_regs_` or by incrementing `reg_count_` */
  int AllocateReg();
  /*! \brief Deallocate a register id, kill it on all workers, and append it to `free_regs_`. */
  void DeallocReg(int reg_id);
  friend class DRefObj;
  friend struct Internal;
};

/*!
 * \brief Managed reference to SessionObj
 * \sa SessionObj
 */
class Session : public ObjectRef {
 public:
  /*! \brief Create a session backed by a thread pool of workers */
  static Session ThreadedSession(int num_workers);
  TVM_DEFINE_MUTABLE_NOTNULLABLE_OBJECT_REF_METHODS(Session, ObjectRef, SessionObj);
};

// Details

template <typename... Args>
void CommandQueueObj::BroadcastUnpacked(DiscoAction action, int64_t reg_id, Args&&... args) {
  this->BroadcastUnpackedImpl(static_cast<int>(action), reg_id, std::forward<Args>(args)...);
}

template <typename... Args>
void CommandQueueObj::BroadcastUnpackedImpl(Args&&... args) {
  constexpr int kNumArgs = (sizeof...(Args) > 0) ? (sizeof...(Args)) : 1;
  TVMValue values[kNumArgs];
  int type_codes[kNumArgs];
  PackArgs(values, type_codes, std::forward<Args>(args)...);
  this->BroadcastPacked(TVMArgs(values, type_codes, sizeof...(Args)));
}

template <typename... Args>
DRef SessionObj::CallPacked(const DRef& func, Args&&... args) {
  return this->CallPackedWithUnpacked(static_cast<int>(DiscoAction::kCallPacked), /*reg_id=*/0,
                                      /*func=*/func, /*is_dref=*/0, std::forward<Args>(args)...);
}

template <typename... Args>
DRef SessionObj::CallPackedWithUnpacked(Args&&... args) {
  constexpr int kNumArgs = (sizeof...(Args) > 0) ? (sizeof...(Args)) : 1;
  TVMValue values[kNumArgs];
  int type_codes[kNumArgs];
  PackArgs(values, type_codes, std::forward<Args>(args)...);
  return this->CallPackedWithPacked(values, type_codes, sizeof...(Args));
}

}  // namespace runtime
}  // namespace tvm
#endif  // TVM_RUNTIME_DISCO_SESSION_H_
