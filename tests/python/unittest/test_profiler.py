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

from __future__ import print_function
import logging
import mxnet as mx
from mxnet import profiler
import time
import os

def enable_profiler(run=True, append_mode=False):
    profile_filename = "test_profile.json"
    profiler.profiler_set_config(mode='all', filename=profile_filename, append_mode=append_mode)
    print('profile file save to {0}'.format(profile_filename))
    if run is True:
      profiler.profiler_set_state('run')


def test_profiler():
    iter_num = 5
    begin_profiling_iter = 2
    end_profiling_iter = 4

    enable_profiler(False, False)

    A = mx.sym.Variable('A')
    B = mx.sym.Variable('B')
    C = mx.symbol.dot(A, B)

    executor = C.simple_bind(mx.cpu(1), 'write', A=(4096, 4096), B=(4096, 4096))

    a = mx.random.uniform(-1.0, 1.0, shape=(4096, 4096))
    b = mx.random.uniform(-1.0, 1.0, shape=(4096, 4096))

    a.copyto(executor.arg_dict['A'])
    b.copyto(executor.arg_dict['B'])

    print("execution begin")
    for i in range(iter_num):
        print("Iteration {}/{}".format(i + 1, iter_num))
        if i == begin_profiling_iter:
            t0 = time.clock()
            profiler.profiler_set_state('run')
        if i == end_profiling_iter:
            t1 = time.clock()
            profiler.profiler_set_state('stop')
        executor.forward()
        c = executor.outputs[0]
        c.wait_to_read()
    print("execution end")
    duration = t1 - t0
    print('duration: {0}s'.format(duration))
    print('          {0}ms/operator'.format(duration*1000/iter_num))
    profiler.dump_profile()


def test_profile_create_domain():
    enable_profiler()
    domain = profiler.Domain(name='PythonDomain')
    print("Domain created: {}".format(str(domain)))


def test_profile_task():
    def makeParams():
        objects = tuple('foo' for _ in range(50))
        template = ''.join('{%d}' % i for i in range(len(objects)))
        return template, objects

    def doLog():
        template, objects = makeParams()
        for _ in range(100000):
            logging.info(template.format(*objects))

    logging.basicConfig()
    enable_profiler()
    python_domain = profiler.Domain('PythonDomain::test_profile_task')
    task = profiler.Task(python_domain, "test_profile_task")
    task.start()
    start = time.time()
    var = mx.nd.ones((1000, 500))
    doLog()
    var.asnumpy()
    stop = time.time()
    task.stop()
    print('run took: %.3f' % (stop - start))


def test_profile_frame():
    def makeParams():
        objects = tuple('foo' for _ in range(50))
        template = ''.join('{%d}' % i for i in range(len(objects)))
        return template, objects

    def doLog():
        template, objects = makeParams()
        for _ in range(100000):
            logging.info(template.format(*objects))

    logging.basicConfig()
    enable_profiler()
    python_domain = profiler.Domain('PythonDomain::test_profile_frame')
    frame = profiler.Frame(python_domain, "test_profile_frame")
    frame.start()
    start = time.time()
    var = mx.nd.ones((1000, 500))
    doLog()
    var.asnumpy()
    stop = time.time()
    frame.stop()
    print('run took: %.3f' % (stop - start))


def test_profile_event(do_enable_profiler=True):
    def makeParams():
        objects = tuple('foo' for _ in range(50))
        template = ''.join('{%d}' % i for i in range(len(objects)))
        return template, objects

    def doLog():
        template, objects = makeParams()
        for _ in range(100000):
            logging.info(template.format(*objects))

    logging.basicConfig()
    if do_enable_profiler is True:
      enable_profiler()
    event = profiler.Event("test_profile_event")
    event.start()
    start = time.time()
    var = mx.nd.ones((1000, 500))
    doLog()
    var.asnumpy()
    stop = time.time()
    event.stop()
    print('run took: %.3f' % (stop - start))


def test_profile_tune_pause_resume():
    enable_profiler()
    profiler.profiler_pause()
    # "test_profile_task" should *not* show up in tuning analysis
    test_profile_task()
    profiler.profiler_resume()
    # "test_profile_event" should show up in tuning analysis
    test_profile_event()
    profiler.profiler_pause()


def test_profile_counter(do_enable_profiler=True):
    def makeParams():
        objects = tuple('foo' for _ in range(50))
        template = ''.join('{%d}' % i for i in range(len(objects)))
        return template, objects

    def doLog(counter):
        template, objects = makeParams()
        range_size = 100000
        for i in range(range_size):
            if i <= range_size / 2:
                counter += 1
            else:
                counter -= 1
            logging.info(template.format(*objects))

    if do_enable_profiler is True:
      enable_profiler()
    python_domain = profiler.Domain('PythonDomain::test_profile_counter')
    counter = profiler.Counter(python_domain, "PythonCounter::test_profile_counter")
    counter.set_value(5)
    counter += 1
    start = time.time()
    doLog(counter)
    stop = time.time()
    print('run took: %.3f' % (stop - start))


def test_continuous_profile_and_instant_marker():
    enable_profiler(True, True)
    profiler.set_continuous_dump(True, 0.1)
    python_domain = profiler.Domain('PythonDomain::test_continuous_profile')
    last_file_size = 0
    for i in range(10):
        profiler.Marker(python_domain, "StartIteration-" + str(i)).mark('process')
        if i > 1 and i % 10 == 0:
            print("{}...".format(i))
        test_profile_event(False)
        test_profile_counter(False)
        # File size should keep increasing
        new_file_size = os.path.getsize("test_profile.json")
        assert new_file_size >= last_file_size
        last_file_size = new_file_size


if __name__ == '__main__':
    test_profile_create_domain()
    test_profiler()
    test_profile_task()
    test_profile_event()
    test_profile_tune_pause_resume()
    test_profile_frame()
    test_continuous_profile_and_instant_marker()
    test_profile_counter()
