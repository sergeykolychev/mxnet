#!/usr/bin/perl
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

# Train mnist, see more explanation at http://mxnet.io/tutorials/python/mnist.html
use strict;
use warnings;
use Getopt::Long ();
#import logging
#logging.basicConfig(level=logging.DEBUG)
#from common import find_mxnet, fit
use common::util ();
use common::fit ();
use AI::MXNet 'mx';
use IO::Zlib ();

sub read_data { my($label, $image) = @_;
    # download and read data into numpy
    my $base_url = 'http://yann.lecun.com/exdb/mnist/';
    for my $flbl (IO::Zlib->new(
            common::util::download_file("$base_url$label", "data/$label"), 'rb')) {
        read $flbl, my($buf), 8;
        my($magic, $num) = unpack 'N2', $buf;
        $label = PDL->new();
        $label->set_datatype($PDL::Types::PDL_B);
        $label->setdims([ $num ]);
        read $flbl, ${$label->get_dataref}, $num;
        $label->upd_data();
    }
    for my $fimg (IO::Zlib->new(
            common::util::download_file("$base_url$image", "data/$image"), 'rb')) {
        read $fimg, my($buf), 16;
        my($magic, $num, $rows, $cols) = unpack 'N4', $buf;
        $image = PDL->new();
        $image->set_datatype($PDL::Types::PDL_B);
        $image->setdims([ $rows, $cols, $num ]);
        read $fimg, ${$image->get_dataref}, $num * $rows * $cols;
        $image->upd_data();
    }
    return($label, $image);
}

sub to4d { my($img) = @_;
    # reshape to 4D arrays
    return $img->reshape(28, 28, 1, ($img->dims)[2])->float / 255;
}

sub get_mnist_iter { my($args, $kv) = @_;
    # create data iterator with NDArrayIter
    my($train_lbl, $train_img) = read_data(
            'train-labels-idx1-ubyte.gz', 'train-images-idx3-ubyte.gz');
    my($val_lbl, $val_img) = read_data(
            't10k-labels-idx1-ubyte.gz', 't10k-images-idx3-ubyte.gz');
    my $train = mx->io->NDArrayIter(
        data => to4d($train_img),
        label => $train_lbl,
        batch_size => $args->{batch_size},
        shuffle => 1,
    );
    my $val = mx->io->NDArrayIter(
        data => to4d($val_img),
        label => $val_lbl,
        batch_size => $args->{batch_size},
    );
    return ($train, $val);
}

if(not defined caller) {
    # parse args
    my $parser = [
        {
            'num-classes' => 10,
            'num-examples' => 60_000,
            'add_stn' => undef,
        },
        'num-classes=i',
        'num-examples=i',
        'add_stn',
    ];
    common::fit::add_fit_args($parser);
    my $args = $parser->[0];
    %$args = (%$args,
        # network
        network        => 'mlp',
        # train
        gpus           => undef,
        batch_size     => 64,
        disp_batches   => 100,
        num_epochs     => 20,
        lr             => .05,
        lr_step_epochs => '10'
    );

    unless(Getopt::Long::GetOptions(@$parser)) {
        my $pre = "usage: $0 ";
        my $msg = "$pre\[-h] [--num-classes NUM_CLASSES] [--num-examples NUM_EXAMPLES] [--add_stn] [--network NETWORK] [--num-layers NUM_LAYERS] [--gpus GPUS] [--kv-store KV_STORE] [--num-epochs NUM_EPOCHS] [--lr LR] [--lr-factor LR_FACTOR] [--lr-step-epochs LR_STEP_EPOCHS] [--optimizer OPTIMIZER] [--mom MOM] [--wd WD] [--batch-size BATCH_SIZE] [--disp-batches DISP_BATCHES] [--model-prefix MODEL_PREFIX] [--monitor MONITOR] [--load-epoch LOAD_EPOCH] [--top-k TOP_K] [--test-io TEST_IO] [--dtype DTYPE]";
        die $msg;
    }

    # load network
    my $net = "symbols::$args->{network}";
    eval "require $net";
    die if $@;
    my $sym = $net->get_symbol(%$args);

    # train
    common::fit::fit($args, $sym, \&get_mnist_iter);
}
