package symbols::lenet;
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

# LeCun, Yann, Leon Bottou, Yoshua Bengio, and Patrick Haffner.
# Gradient-based learning applied to document recognition.
# Proceedings of the IEEE (1998)
use strict;
use warnings;
use AI::MXNet 'mx';

sub get_loc { my($data, $attr) = @_;
    $attr //= { 'lr_mult' => '0.01'};
    # the localisation network in lenet-stn, it will increase acc about more than 1%,
    # when num-epoch >=15
    my $loc = mx->symbol->Convolution(data=>$data, num_filter=>30, kernel=>[5, 5], stride=>[2,2]);
    $loc = mx->symbol->Activation(data => $loc, act_type=>'relu');
    $loc = mx->symbol->Pooling(data=>$loc, kernel=>[2, 2], stride=>[2, 2], pool_type=>'max');
    $loc = mx->symbol->Convolution(data=>$loc, num_filter=>60, kernel=>[3, 3], stride=>[1,1], pad=>[1, 1]);
    $loc = mx->symbol->Activation(data => $loc, act_type=>'relu');
    $loc = mx->symbol->Pooling(data=>$loc, global_pool=>1, kernel=>[2, 2], pool_type=>'avg');
    $loc = mx->symbol->Flatten(data=>$loc);
    $loc = mx->symbol->FullyConnected(data=>$loc, num_hidden=>6, name=>"stn_loc", attr=>$attr);
    return $loc;
}

sub get_symbol { my($class, %kwargs) = @_;
    my $num_classes = $kwargs{'num-classes'} // 10;
    my $add_stn = $kwargs{'add_stn'};
    my $data = mx->symbol->Variable('data');
    $data = mx->sym->SpatialTransformer(data=>$data, loc=>get_loc($data), target_shape => [28,28],
                                     transform_type=>"affine", sampler_type=>"bilinear")
        if $add_stn;
    # first conv
    my $conv1 = mx->symbol->Convolution(data=>$data, kernel=>[5,5],
            num_filter=>20);
    my $tanh1 = mx->symbol->Activation(data=>$conv1, act_type=>"tanh");
    my $pool1 = mx->symbol->Pooling(data=>$tanh1, pool_type=>"max",
                              kernel=>[2,2], stride=>[2,2]);
    # second conv
    my $conv2 = mx->symbol->Convolution(data=>$pool1, kernel=>[5,5], num_filter=>50);
    my $tanh2 = mx->symbol->Activation(data=>$conv2, act_type=>"tanh");
    my $pool2 = mx->symbol->Pooling(data=>$tanh2, pool_type=>"max",
                              kernel=>[2,2], stride=>[2,2]);
    # first fullc
    my $flatten = mx->symbol->Flatten(data=>$pool2);
    my $fc1 = mx->symbol->FullyConnected(data=>$flatten, num_hidden=>500);
    my $tanh3 = mx->symbol->Activation(data=>$fc1, act_type=>"tanh");
    # second fullc
    my $fc2 = mx->symbol->FullyConnected(data=>$tanh3, num_hidden=>$num_classes);
    # loss
    my $lenet = mx->symbol->SoftmaxOutput(data=>$fc2, name=>'softmax');
    return $lenet;
}

1;
