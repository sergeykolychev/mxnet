# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

# coding: utf-8
# pylint: disable=fixme, invalid-name, too-many-arguments, too-many-locals, too-many-lines
# pylint: disable=too-many-branches, too-many-statements
"""Profiler setting methods."""
from __future__ import absolute_import
import ctypes
from .base import _LIB, check_call, c_str, ProfileDomainHandle, \
    ProfileCounterHandle, ProfileTaskHandle, ProfileFrameHandle, ProfileEventHandle

def profiler_set_config(mode='symbolic', filename='profile.json'):
    """Set up the configure of profiler.

    Parameters
    ----------
    mode : string, optional
        Indicates whether to enable the profiler, can
        be 'symbolic', or 'all'. Defaults to `symbolic`.
    filename : string, optional
        The name of output trace file. Defaults to 'profile.json'.
    """
    mode2int = {'symbolic': 1,
                'imperative': 2,
                'api': 4,
                'memory': 8,
                # Combined alias items
                'all_ops': 3,
                'all': 15
                }
    mode_flag = 0
    if isinstance(mode, list):
      for i in mode:
        mode_flag |= mode2int[i]
    elif isinstance(mode, str):
        mode_flag |= mode2int[mode]
    elif isinstance(mode, (int, long)):
      mode_flag = mode
    check_call(_LIB.MXSetProfilerConfig(ctypes.c_int(mode_flag), c_str(filename)))

def profiler_set_state(state='stop'):
    """Set up the profiler state to record operator.

    Parameters
    ----------
    state : string, optional
        Indicates whether to run the profiler, can
        be 'stop' or 'run'. Default is `stop`.
    """
    state2int = {'stop': 0, 'run': 1}
    check_call(_LIB.MXSetProfilerState(ctypes.c_int(state2int[state])))

def dump_profile():
    """Dump profile and stop profiler. Use this to save profile
    in advance in case your program cannot exit normally."""
    check_call(_LIB.MXDumpProfile())

def create_domain(name):
  domain_handle = ProfileDomainHandle()
  check_call(_LIB.MXProfileCreateDomain(c_str(name), ctypes.byref(domain_handle)))
  return domain_handle

def create_task(domain_handle, name):
    task_handle = ProfileTaskHandle()
    check_call(_LIB.MXProfileCreateTask(domain_handle,
                                        c_str(name),
                                        ctypes.byref(task_handle)))
    return task_handle

def destroy_task(task_handle):
    check_call(_LIB.MXProfileDestroyTask(task_handle))

def task_start(task_handle):
    check_call(_LIB.MXProfileTaskStart(task_handle))

def task_stop(task_handle):
    check_call(_LIB.MXProfileTaskStop(task_handle))

def create_frame(domain_handle, name):
    frame_handle = ProfileFrameHandle()
    check_call(_LIB.MXProfileCreateFrame(domain_handle,
                                        c_str(name),
                                        ctypes.byref(frame_handle)))
    return frame_handle

def destroy_frame(frame_handle):
    check_call(_LIB.MXProfileDestroyFrame(frame_handle))

def frame_start(frame_handle):
    check_call(_LIB.MXProfileFrameStart(frame_handle))

def frame_stop(frame_handle):
    check_call(_LIB.MXProfileFrameStop(frame_handle))

def create_event(name):
    event_handle = ProfileEventHandle()
    check_call(_LIB.MXProfileCreateEvent(c_str(name), ctypes.byref(event_handle)))
    return event_handle

def destroy_event(event_handle):
    check_call(_LIB.MXProfileDestroyEvent(event_handle))

def event_start(event_handle):
    check_call(_LIB.MXProfileEventStart(event_handle))

def event_stop(event_handle):
    check_call(_LIB.MXProfileEventStop(event_handle))

def tune_pause():
    check_call(_LIB.MXProfileTunePause())

def tune_resume():
    check_call(_LIB.MXProfileTuneResume())

def create_counter(domain_handle, name, value=None):
    counter_handle = ProfileCounterHandle()
    check_call(_LIB.MXProfileCreateCounter(domain_handle,
                                           c_str(name),
                                           ctypes.byref(counter_handle)))
    if value is not None:
        set_counter(counter_handle, value)
    return counter_handle

def destroy_counter(counter_handle):
    check_call(_LIB.MXProfileDestroyCounter(counter_handle))

def set_counter(counter_handle, value):
    check_call(_LIB.MXProfileSetCounter(counter_handle, int(value)))

def increment_counter(counter_handle, by_value):
    check_call(_LIB.MXProfileAdjustCounter(counter_handle, int(by_value)))

def decrement_counter(counter_handle, by_value):
    check_call(_LIB.MXProfileAdjustCounter(counter_handle, -int(by_value)))

def set_append_mode(mode):
  if mode is False:
    mode = 0
  else:
    mode = 1
  check_call(_LIB.MXSetDumpProfileAppendMode(int(mode)))

def set_continuous_dump(continuous_dump=True, delay_in_seconds=1.0):
  if continuous_dump is False:
    cd = 0
  else:
    cd = 1
  ds = float(delay_in_seconds)
  check_call(_LIB.MXSetContinuousProfileDump(ctypes.c_int(cd), ctypes.c_float(ds)))

def set_instant_marker(domain_handle, name, scope='process'):
    marker_scope2int = { 'global': 1, 'process': 2, 'thread': 3, 'task': 4, 'marker': 5 }
    scope_int = marker_scope2int[scope]
    check_call(_LIB.MXProfileSetInstantMarker(domain_handle, c_str(name), scope_int))


class Domain:
    """Profiling domain, used to group sub-objects like tasks, counters, etc into categories
    Serves as part of 'categories' for chrome://tracing
    Note: Domain handles are never destroyed
    """
    def __init__(self, name):
        self.name = name
        self.handle = create_domain(name)

    def __str__(self):
        return self.name


class Task:
    """Profiling Task class
    A task is a logical unit of work performed by a particular thread.
    Tasks can nest; thus, tasks typically correspond to functions, scopes, or a case block
    in a switch statement.
    You can use the Task API to assign tasks to threads
    """
    def __init__(self, domain, name):
        self.domain = domain
        self.name = name
        self.handle = create_task(domain.handle, name)

    def start(self):
        task_start(self.handle)

    def stop(self):
        task_stop(self.handle)

    def __str__(self):
        return self.name

    def __del__(self):
        if self.handle is not None:
            destroy_task(self.handle)


class Frame:
    """Profiling Frame class
    Use the frame API to insert calls to the desired places in your code and analyze
    performance per frame, where frame is the time period between frame begin and end points.
    When frames are displayed in Intel VTune Amplifier, they are displayed in a
    separate track, so they provide a way to visually separate this data from normal task data.
    """
    def __init__(self, domain, name):
        self.domain = domain
        self.name = name
        self.handle = create_frame(domain.handle, name)

    def start(self):
        frame_start(self.handle)

    def stop(self):
        frame_stop(self.handle)

    def __str__(self):
        return self.name

    def __del__(self):
        if self.handle is not None:
            destroy_frame(self.handle)


class Event:
    """Profiling Event class
    The event API is used to observe when demarcated events occur in your application, or to
    identify how long it takes to execute demarcated regions of code. Set annotations in the
    application to demarcate areas where events of interest occur.
    After running analysis, you can see the events marked in the Timeline pane.
    Event API is a per-thread function that works in resumed state.
    This function does not work in paused state.
    """
    def __init__(self, name):
        self.name = name
        self.handle = create_event(name)

    def start(self):
        event_start(self.handle)

    def stop(self):
        event_stop(self.handle)

    def __str__(self):
        return self.name

    def __del__(self):
        if self.handle is not None:
            destroy_event(self.handle)


class Counter:
    """Profiling Counter class
    The counter event can track a value as it changes over time.
    """
    def __init__(self, domain, name, value=0):
        self.name = name
        self.handle = create_counter(domain.handle, name, value)

    def set_value(self, value):
        set_counter(self.handle, value)

    def increment(self, value_change):
        increment_counter(self.handle, value_change)

    def decrement(self, value_change):
        decrement_counter(self.handle, value_change)

    def __iadd__(self, value_change):
        self.increment(value_change)
        return self

    def __isub__(self, value_change):
        self.decrement(value_change)
        return self

    def __str__(self):
        return self.name

    def __del__(self):
        if self.handle is not None:
            destroy_counter(self.handle)


class InstantMarker:
    """Set marker for an instant in time"""
    def __init__(self, domain, name):
        self.name = name
        self.domain = domain

    def signal(self, scope='process'):
        set_instant_marker(self.domain.handle, self.name, scope)

