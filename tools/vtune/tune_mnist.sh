#!/bin/bash
SCRIPTDIR=$(realpath $(dirname $0))
TOPDIR=${SCRIPTDIR}/../../
cd ${TOPDIR}
/bin/bash ${SCRIPTDIR}/tune_python.sh example/image-classification/train_mnist.py --network=mlp --profile=all --num-epochs=1 $@

