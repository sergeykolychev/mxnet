//
// Created by coolivie on 11/25/17.
//

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
 *  Copyright (c) 2017 by Contributors
 * \file c_api_profile.cc
 * \brief C API of mxnet profiler and support functions
 */
#include <dmlc/base.h>
#include <dmlc/logging.h>
#include <dmlc/thread_group.h>
#include <stack>
#include "./c_api_common.h"
#include "profiler/profiler.h"

namespace mxnet {

#if MXNET_USE_PROFILER
static profiler::ProfileDomain api_domain("MXNET_C_API");
static profiler::ProfileCounter api_call_counter("MXNet C API Call Count", &api_domain);
static profiler::ProfileCounter api_concurrency_counter("MXNet C API Concurrency Count",
                                                        &api_domain);

/*! \brief Per-API-call timing data */
struct APICallTimingData {
  const char *name_;
  profiler::ProfileTask *task_;
  profiler::ProfileEvent *event_;
};

template<typename T, typename... Args>
inline std::unique_ptr<T> make_unique(Args&&... args) {
  return std::unique_ptr<T>(new T(std::forward<Args>(args)...));
}

/*!
 * \brief Per-thread profiling data
 */
class ProfilingThreadData {
 public:
  /*!
   * \brief Constructor, nothrow
   */
  inline ProfilingThreadData() noexcept {}

  /*!
   * \brief Retreive ProfileTask object of the given name, or create if it doesn't exist
   * \param name Name of the task
   * \param domain Domain of the task
   * \return Pointer to the stored or created ProfileTask object
   */
  profiler::ProfileTask *profile_task(const char *name, profiler::ProfileDomain *domain) {
    // Per-thread so no lock necessary
    auto iter = tasks_.find(name);
    if (iter == tasks_.end()) {
      iter = tasks_.emplace(std::make_pair(
        name, make_unique<profiler::ProfileTask>(name, domain))).first;
    }
    return iter->second.get();
  }

  /*!
   * \brief Retreive ProfileEvent object of the given name, or create if it doesn't exist
   * \param name Name of the event
   * \return Pointer to the stored or created ProfileEvent object
   */
  profiler::ProfileEvent *profile_event(const char *name) {
    // Per-thread so no lock necessary
    auto iter = events_.find(name);
    if (iter == events_.end()) {
      iter = events_.emplace(std::make_pair(name, make_unique<profiler::ProfileEvent>(name))).first;
    }
    return iter->second.get();
  }

  /*! \brief nestable call stack */
  std::stack<APICallTimingData> calls_;
  /*! \brief Whether profiling actions should be ignored/excluded */
  volatile bool ignore_call_ = false;  // same-thread only, so not atomic

 private:
  /*! \brief tasks */
  std::unordered_map<std::string, std::unique_ptr<profiler::ProfileTask>> tasks_;
  /*! \brief events */
  std::unordered_map<std::string, std::unique_ptr<profiler::ProfileEvent>> events_;
};

static thread_local ProfilingThreadData thread_profiling_data;
#endif  // MXNET_USE_PROFILER

extern void on_enter_api(const char *function) {
#if MXNET_USE_PROFILER
  profiler::Profiler *prof = profiler::Profiler::Get();
  if (prof->GetState() == profiler::Profiler::kRunning
      && (prof->GetMode() & profiler::Profiler::kAPI) != 0) {
    if (!thread_profiling_data.ignore_call_) {
      ++api_call_counter;
      ++api_concurrency_counter;
      APICallTimingData data = {
        function,
        thread_profiling_data.profile_task(function, &api_domain),
        thread_profiling_data.profile_event(function)
      };
      thread_profiling_data.calls_.push(data);
      data.task_->start();
      data.event_->start();
    }
  }
#endif  // MXNET_USE_PROFILER
}

extern void on_exit_api() {
#if MXNET_USE_PROFILER
  profiler::Profiler *prof = profiler::Profiler::Get();
  if (prof->GetState() == profiler::Profiler::kRunning
      && (prof->GetMode() & profiler::Profiler::kAPI) != 0) {
    if (!thread_profiling_data.ignore_call_) {
      CHECK(!thread_profiling_data.calls_.empty());
      APICallTimingData data = thread_profiling_data.calls_.top();
      data.event_->stop();
      data.task_->stop();
      thread_profiling_data.calls_.pop();
      --api_concurrency_counter;
    }
  }
#endif  // MXNET_USE_PROFILER
}

/*!
 * \brief Don't profile calls in this scope using RAII
 */
struct IgnoreProfileCallScope {
  IgnoreProfileCallScope()  {
#if MXNET_USE_PROFILER
    DCHECK_EQ(thread_profiling_data.ignore_call_, false);
    thread_profiling_data.ignore_call_ = true;
#endif  // MXNET_USE_PROFILER
  }
  ~IgnoreProfileCallScope() {
#if MXNET_USE_PROFILER
    DCHECK_EQ(thread_profiling_data.ignore_call_, true);
    thread_profiling_data.ignore_call_ = false;
#endif  // MXNET_USE_PROFILER
  }
};

}  // namespace mxnet

/*!
 * \brief Simple global profile objects created from Python
 * \note These mutexes will almost never have a collision, so internal futexes will be able
 *       to lock in user mode (good performance)
 *       I would use dmlc::SpinLock, except that I am concerned that if conditions change and
 *       there are frequent collisions (ie multithreaded inference), then the spin locks may
 *       start burning CPU unnoticed
 */
struct PythonProfileObjects {
  std::mutex cs_domains_;
  std::mutex cs_counters_;
  std::mutex cs_tasks_;
  std::mutex cs_frames_;
  std::mutex cs_events_;
  std::list<std::shared_ptr<profiler::ProfileDomain>> domains_;
  std::unordered_map<profiler::ProfileCounter *, std::shared_ptr<profiler::ProfileCounter>> counters_;
  std::unordered_map<profiler::ProfileDuration *, std::shared_ptr<profiler::ProfileDuration>> tasks_;
  std::unordered_map<profiler::ProfileDuration *, std::shared_ptr<profiler::ProfileDuration>> frames_;
  std::unordered_map<profiler::ProfileDuration *, std::shared_ptr<profiler::ProfileDuration>> events_;
};
static PythonProfileObjects python_profile_objects;

#if !defined(MXNET_USE_PROFILER) || !MXNET_USE_PROFILER
static void warn_not_built_with_profiler_enabled() {
  static volatile bool warned_not_built_with_profiler_enabled = false;
  if (!warned_not_built_with_profiler_enabled) {
    warned_not_built_with_profiler_enabled = true;
    LOG(WARNING) << "Need to compile with USE_PROFILER=1 for MXNet Profiling";
  }
}
#endif  // MXNET_USE_PROFILER

struct ProfileModeParam : public dmlc::Parameter<ProfileModeParam> {
  int mode;
  DMLC_DECLARE_PARAMETER(ProfileModeParam) {
    DMLC_DECLARE_FIELD(mode).set_default(profiler::Profiler::kSymbolic)
      .add_enum("symbolic", profiler::Profiler::kSymbolic)
      .add_enum("imperative", profiler::Profiler::kImperative)
      .add_enum("api", profiler::Profiler::kAPI)
      .add_enum("memory", profiler::Profiler::kMemory)
      .add_enum("all_ops", profiler::Profiler::kSymbolic|profiler::Profiler::kImperative)
      .add_enum("all", profiler::Profiler::kSymbolic|profiler::Profiler::kImperative
                       |profiler::Profiler::kAPI|profiler::Profiler::kMemory)
      .describe("Profile mode.");
  }
};

DMLC_REGISTER_PARAMETER(ProfileModeParam);

struct ProfileInstantScopeParam : public dmlc::Parameter<ProfileInstantScopeParam> {
  int scope;
  DMLC_DECLARE_PARAMETER(ProfileInstantScopeParam) {
    DMLC_DECLARE_FIELD(scope).set_default(profiler::ProfileMarker::kProcess)
      .add_enum("global", profiler::ProfileMarker::kGlobal)
      .add_enum("process", profiler::ProfileMarker::kProcess)
      .add_enum("thread", profiler::ProfileMarker::kThread)
      .add_enum("task", profiler::ProfileMarker::kTask)
      .add_enum("marker", profiler::ProfileMarker::kMarker)
      .describe("Profile Instant-Marker scope.");
  }
};

DMLC_REGISTER_PARAMETER(ProfileInstantScopeParam);

int MXSetProfilerConfig(const char *mode, const char* filename, const int append_mode) {
  mxnet::IgnoreProfileCallScope ignore;
  API_BEGIN();
#if MXNET_USE_PROFILER
    ProfileModeParam param;
    std::vector<std::pair<std::string, std::string>> kwargs = {{ "mode", mode }};
    param.Init(kwargs);
    profiler::Profiler::Get()->SetConfig(profiler::Profiler::ProfilerMode(param.mode),
                                        std::string(filename), append_mode != 0);
#else
    warn_not_built_with_profiler_enabled();
#endif
  API_END();
}

int MXDumpProfile() {
  mxnet::IgnoreProfileCallScope ignore;
  API_BEGIN();
#if MXNET_USE_PROFILER
    profiler::Profiler *profiler = profiler::Profiler::Get();
    CHECK(profiler->IsEnableOutput())
      << "Profiler hasn't been run. Config and start profiler first";
    profiler->DumpProfile(true);
#else
    warn_not_built_with_profiler_enabled();
#endif
  API_END()
}

int MXSetProfilerState(int state) {
  mxnet::IgnoreProfileCallScope ignore;
  // state, kNotRunning: 0, kRunning: 1
  API_BEGIN();
#if MXNET_USE_PROFILER
    switch (state) {
      case profiler::Profiler::kNotRunning:
        profiler::vtune::vtune_pause();
        break;
      case profiler::Profiler::kRunning:
        profiler::vtune::vtune_resume();
        break;
    }
    profiler::Profiler::Get()->SetState(profiler::Profiler::ProfilerState(state));
#else
    warn_not_built_with_profiler_enabled();
#endif
  API_END();
}

int MXSetContinuousProfileDump(int continuous_dump, float delay_in_seconds) {
  API_BEGIN();
#if MXNET_USE_PROFILER
    profiler::Profiler::Get()->SetContinuousProfileDump(continuous_dump != 0, delay_in_seconds);
#else
    warn_not_built_with_profiler_enabled();
#endif
  API_END();
}

int MXProfileCreateDomain(const char *domain, ProfileHandle *out) {
  mxnet::IgnoreProfileCallScope ignore;
  API_BEGIN();
#if MXNET_USE_PROFILER
    auto dom = std::make_shared<profiler::ProfileDomain>(domain);
    {
      std::unique_lock<std::mutex> lock(python_profile_objects.cs_domains_);
      python_profile_objects.domains_.push_back(dom);
    }
    *out = dom.get();
#else
    warn_not_built_with_profiler_enabled();
    *out = nullptr;
#endif
  API_END();
}

int MXProfileCreateTask(ProfileHandle domain,
                        const char *task_name,
                        ProfileHandle *out) {
  mxnet::IgnoreProfileCallScope ignore;
  API_BEGIN();
#if MXNET_USE_PROFILER
    auto ctr =
      std::make_shared<profiler::ProfileTask>(task_name,
                                                static_cast<profiler::ProfileDomain *>(domain));
    {
      std::unique_lock<std::mutex> lock(python_profile_objects.cs_tasks_);
      python_profile_objects.tasks_.emplace(std::make_pair(ctr.get(), ctr));
    }
    *out = ctr.get();
#else
    warn_not_built_with_profiler_enabled();
    *out = nullptr;
#endif
  API_END();
}

int MXProfileCreateFrame(ProfileHandle domain,
                         const char *frame_name,
                         ProfileHandle *out) {
  mxnet::IgnoreProfileCallScope ignore;
  API_BEGIN();
#if MXNET_USE_PROFILER
    auto ctr =
      std::make_shared<profiler::ProfileFrame>(frame_name,
                                              static_cast<profiler::ProfileDomain *>(domain));
    {
      std::unique_lock<std::mutex> lock(python_profile_objects.cs_frames_);
      python_profile_objects.frames_.emplace(std::make_pair(ctr.get(), ctr));
    }
    *out = ctr.get();
#else
    warn_not_built_with_profiler_enabled();
    *out = nullptr;
#endif
  API_END();
}

int MXProfileCreateEvent(const char *event_name, ProfileHandle *out) {
  mxnet::IgnoreProfileCallScope ignore;
  API_BEGIN();
#if MXNET_USE_PROFILER
    auto ctr =
      std::make_shared<profiler::ProfileEvent>(event_name);
    {
      std::unique_lock<std::mutex> lock(python_profile_objects.cs_events_);
      python_profile_objects.events_.emplace(std::make_pair(ctr.get(), ctr));
    }
    *out = ctr.get();
#else
    warn_not_built_with_profiler_enabled();
    *out = nullptr;
#endif
  API_END();
}

int MXProfileDestroyHandle(ProfileHandle object_handle) {
  mxnet::IgnoreProfileCallScope ignore;
  API_BEGIN();
#if MXNET_USE_PROFILER
    CHECK_NE(object_handle, static_cast<ProfileHandle>(nullptr))
      << "Invalid NULL handle passed to MXProfileDestroyHandle";
    std::shared_ptr<profiler::ProfileObject> shared_object_ptr(nullptr);
    {
      auto object = static_cast<profiler::ProfileObject *>(object_handle);
      switch (object->type()) {
        case profiler::kTask: {
          auto p = static_cast<profiler::ProfileDuration *>(object_handle);
          std::unique_lock<std::mutex> lock(python_profile_objects.cs_tasks_);
          auto iter = python_profile_objects.tasks_.find(p);
          if (iter != python_profile_objects.tasks_.end()) {
            shared_object_ptr = iter->second;
            python_profile_objects.tasks_.erase(iter);
          }
          break;
        }
        case profiler::kEvent: {
          auto p = static_cast<profiler::ProfileDuration *>(object_handle);
          std::unique_lock<std::mutex> lock(python_profile_objects.cs_events_);
          auto iter = python_profile_objects.events_.find(p);
          if (iter != python_profile_objects.events_.end()) {
            shared_object_ptr = iter->second;
            python_profile_objects.events_.erase(iter);
          }
          break;
        }
        case profiler::kFrame: {
          auto p = static_cast<profiler::ProfileDuration *>(object_handle);
          std::unique_lock<std::mutex> lock(python_profile_objects.cs_frames_);
          auto iter = python_profile_objects.frames_.find(p);
          if (iter != python_profile_objects.frames_.end()) {
            shared_object_ptr = iter->second;
            python_profile_objects.frames_.erase(iter);
          }
          break;
        }
        case profiler::kCounter: {
          auto p = static_cast<profiler::ProfileCounter *>(object_handle);
          std::unique_lock<std::mutex> lock(python_profile_objects.cs_counters_);
          auto iter = python_profile_objects.counters_.find(p);
          if (iter != python_profile_objects.counters_.end()) {
            shared_object_ptr = iter->second;
            python_profile_objects.counters_.erase(iter);
          }
          break;
        }
        case profiler::kDomain:
          // Not destroyed
          break;
      }
    }
    shared_object_ptr.reset();  // Destroy out of lock scope
#else
    warn_not_built_with_profiler_enabled();
#endif
  API_END();
}

int MXProfileDurationStart(ProfileHandle duration_handle) {
  mxnet::IgnoreProfileCallScope ignore;
  API_BEGIN();
#if MXNET_USE_PROFILER
    CHECK_NOTNULL(duration_handle);
    static_cast<profiler::ProfileDuration *>(duration_handle)->start();
#else
    warn_not_built_with_profiler_enabled();
#endif
  API_END();
}

int MXProfileDurationStop(ProfileHandle duration_handle) {
  mxnet::IgnoreProfileCallScope ignore;
  API_BEGIN();
#if MXNET_USE_PROFILER
    CHECK_NOTNULL(duration_handle);
    static_cast<profiler::ProfileDuration *>(duration_handle)->stop();
#else
    warn_not_built_with_profiler_enabled();
#endif
  API_END();
}

int MXProfilePause(int paused) {
  mxnet::IgnoreProfileCallScope ignore;
  API_BEGIN();
#if MXNET_USE_PROFILER
    if (paused) {
      profiler::vtune::vtune_pause();
      profiler::Profiler::Get()->set_paused(true);
    } else {
      profiler::Profiler::Get()->set_paused(false);
      profiler::vtune::vtune_resume();
    }
#else
    warn_not_built_with_profiler_enabled();
#endif
  API_END();
}

int MXProfileCreateCounter(ProfileHandle domain,
                           const char *counter_name,
                           ProfileHandle *out) {
  mxnet::IgnoreProfileCallScope ignore;
  API_BEGIN();
#if MXNET_USE_PROFILER
    auto ctr =
      std::make_shared<profiler::ProfileCounter>(counter_name,
                                                static_cast<profiler::ProfileDomain *>(domain));
    {
      std::unique_lock<std::mutex> lock(python_profile_objects.cs_counters_);
      python_profile_objects.counters_.emplace(std::make_pair(ctr.get(), ctr));
    }
    *out = ctr.get();
#else
    warn_not_built_with_profiler_enabled();
    *out = nullptr;
#endif
  API_END();
}

int MXProfileSetCounter(ProfileHandle counter_handle, uint64_t value) {
  mxnet::IgnoreProfileCallScope ignore;
  API_BEGIN();
#if MXNET_USE_PROFILER
    static_cast<profiler::ProfileCounter *>(counter_handle)->operator=(value);
#else
    warn_not_built_with_profiler_enabled();
#endif
  API_END();
}

int MXProfileAdjustCounter(ProfileHandle counter_handle, int64_t by_value) {
  mxnet::IgnoreProfileCallScope ignore;
  API_BEGIN();
#if MXNET_USE_PROFILER
    static_cast<profiler::ProfileCounter *>(counter_handle)->operator+=(by_value);
#else
    warn_not_built_with_profiler_enabled();
#endif
  API_END();
}

int MXProfileSetInstantMarker(ProfileHandle domain,
                              const char *instant_marker_name,
                              const char *scope) {
  mxnet::IgnoreProfileCallScope ignore;
  API_BEGIN();
#if MXNET_USE_PROFILER
    ProfileInstantScopeParam param;
    std::vector<std::pair<std::string, std::string>> kwargs = {{ "scope", scope }};
    param.Init(kwargs);
    profiler::ProfileMarker marker(instant_marker_name,
                                         static_cast<profiler::ProfileDomain *>(domain),
                                         static_cast<profiler::ProfileMarker::MarkerScope>(
                                           param.scope));
    marker.mark();
#else
    warn_not_built_with_profiler_enabled();
#endif
  API_END();
}
