#!/bin/bash
SCRIPTDIR=$(realpath $(dirname $0))
TOPDIR=$(realpath ${SCRIPTDIR}/../../)
export LD_LIBRARY_PATH=${TOPDIR}/cmake-build-relwithdebinfo:${TOPDIR}/../cmake-build-relwithdebinfo/mxnet:$LD_LIBRARY_PATH
export PYTHONPATH=${TOPDIR}/python:$PYTHONPATH
cd ${TOPDIR}
/opt/intel/vtune_amplifier_xe/bin64/amplxe-cl -collect hotspots -run-pass-thru=-timestamp=sys -knob analyze-openmp=true -knob sampling-interval=1 -knob enable-user-tasks=true -- /usr/bin/python2.7 $@
