package common::fit;
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

use strict;
use warnings;
use AI::MXNet 'mx';
#import logging
#import os
#import time
#
sub _get_lr_scheduler { my($args, $kv) = @_;
    return ($args->{lr}, undef)
        if not exists $args->{'lr_factor'} or $args->{lr_factor} >= 1;
    my $epoch_size = $args->{num_examples} / $args->{batch_size};
    $epoch_size /= $kv->{num_workers}
        if $args->{kv_store} =~ /dist/;
    my $begin_epoch = $args->{load_epoch} ? $args->{load_epoch} : 0;
    my @step_epochs = split /,/, $args->{lr_step_epochs};
    my $lr = $args->{lr};
    for my $s (@step_epochs) {
        $lr *= $args->{lr_factor}
            if $begin_epoch >= $s;
    }
    logging->info(sprintf 'Adjust learning rate to %e for epoch %d', $lr, $begin_epoch)
        if $lr != $args->{lr};

    my @steps = map { $epoch_size * $_ } grep { $_ > 0 } map { $_ - $begin_epoch } @step_epochs;
    return ($lr, mx->lr_scheduler->MultiFactorScheduler(step => \@steps, factor => $args->{lr_factor}));
}

sub _load_model{ my($args, $rank) = @_;
    $rank //= 0;
    return (undef, undef, undef)
        if not $args->{'load_epoch'};
    die "assert" unless defined $args->{model_prefix};
    my $model_prefix = $args->{model_prefix};
    $model_prefix .= sprintf "-%d", $rank
        if $rank > 0 and -e sprintf "%s-%d-symbol.json", $model_prefix, $rank;
    my($sym, $arg_params, $aux_params) = mx->mod->load_checkpoint(
        $model_prefix, $args->{load_epoch});
    logging->info(sprintf 'Loaded model %s_%04d.params', $model_prefix, $args->{load_epoch});
    return ($sym, $arg_params, $aux_params);
}

sub _save_model{ my($args, $rank) = @_;
    $rank //= 0;
    return undef unless defined $args->{model_prefix};
    my($dst_dir) = $args->{model_prefix} =~ m{^(.*)/};
    mkdir $dst_dir
        unless -d $dst_dir;
    return $rank == 0
        ? mx->callback->do_checkpoint($args->{model_prefix})
        : sprintf "%s-%d", $args->{model_prefix}, $rank;
}

sub add_fit_args { my($parser) = @_;
    # parser : argparse.ArgumentParser
    # return a parser added with args required by fit
    my(%default, @train);
    push @train, 'network=s';
    push @train, 'num-layers=i';
    push @train, 'gpus=s';
    push @train, 'kv-store=s';
    $default{'kv-store'} = 'device';
    push @train, 'num-epochs=i';
    $default{'num-epochs'} = 100;
    push @train, 'lr=f';
    $default{'lr'} = 0.1;
    push @train, 'lr-factor=f';
    $default{'lr-factor'} = 0.1;
    push @train, 'lr-step-epochs=s';
    push @train, 'optimizer=s';
    $default{'optimizer'} = 'sgd';
    push @train, 'mom=f';
    $default{'mom'} = 0.9;
    push @train, 'wd=f';
    $default{'wd'} = 0.0001;
    push @train, 'batch-size=i';
    $default{'batch-size'} = 128;
    push @train, 'disp-batches=i';
    $default{'disp-batches'} = 20;
    push @train, 'model-prefix=s';
    push @train, 'monitor=i';
    $default{'monitor'} = 0;
    push @train, 'load-epoch=i';
    push @train, 'top-k=i';
    $default{'top-k'} = 0;
    push @train, 'test-io=i';
    $default{'test-io'} = 0;
    push @train, 'dtype=s';
    $default{'dtype'} = 'float32';
    %{$parser->[0]} = (%{$parser->[0]}, %default);
    push @$parser, @train;
    return(\%default, @train);
}

sub fit { my($args, $network, $data_loader, %kwargs) = @_;
    # train a model
    # args : argparse returns
    # network : the symbol definition of the nerual network
    # data_loader : function that returns the train and val data iterators

    # kvstore
    my $kv = mx->kv->create($args->{'kv-store'});

    # logging
#    head = '%(asctime)-15s Node[' + str(kv.rank) + '] %(message)s'
#    logging.basicConfig(level=logging.DEBUG, format=head)
#    logging.info('start with arguments %s', args)
#
#    # data iterators
    my($train, $val) = $data_loader->($args, $kv);
#    if args.test_io:
#        tic = time.time()
#        for i, batch in enumerate(train):
#            for j in batch.data:
#                j.wait_to_read()
#            if (i+1) % args.disp_batches == 0:
#                logging.info('Batch [%d]\tSpeed: %.2f samples/sec' % (
#                    i, args.disp_batches*args.batch_size/(time.time()-tic)))
#                tic = time.time()
#
#        return
#
#
    # load model
    my($arg_params, $aux_params, $sym);
    if(exists $kwargs{'arg_params'} and exists $kwargs{'aux_params'}) {
        $arg_params = $kwargs{'arg_params'};
        $aux_params = $kwargs{'aux_params'};
    } else {
        ($sym, $arg_params, $aux_params) = _load_model($args, $kv->{rank});
        if(defined $sym) {
            die "assert" unless $sym->tojson() eq $network->tojson();
        }
    }
    # save model
    my $checkpoint = _save_model($args, $kv->{rank});

    # devices for training
    my $devs = $args->{gpus} ? [ map {
        mx->gpu($_)
    } split /,/, $args->{gpus} ] : mx->cpu();

    # learning rate
    my($lr, $lr_scheduler) = _get_lr_scheduler($args, $kv);

    # create model
    my $model = mx->mod->Module(
        context       => $devs,
        symbol        => $network
    );

    $lr_scheduler  = $lr_scheduler;
    my $optimizer_params = {
            'learning_rate' => $lr,
            'wd' => $args->{wd},
            'lr_scheduler' => $lr_scheduler};

    # Add 'multi_precision' parameter only for SGD optimizer
    $optimizer_params->{'multi_precision'} = 1
        if $args->{optimizer} eq'sgd';

    # Only a limited number of optimizers have 'momentum' property
    $optimizer_params->{'momentum'} = $args->{mom}
        if grep { $args->{optimizer} eq $_ } 'sgd', 'dcasgd';

    my $monitor = mx->mon->Monitor($args->{monitor}, pattern=>".*")
        if $args->{monitor} > 0;

    my $initializer;
    if($args->{network} eq 'alexnet') {
        # AlexNet will not converge using Xavier
        $initializer = mx->init->Normal();
    } else {
        $initializer = mx->init->Xavier(
            rnd_type=>'gaussian', factor_type=>"in", magnitude=>2);
    }
    # $initializer   = mx->init->Xavier(factor_type=>"in", magnitude=>2.34),

    # evaluation metrices
    my $eval_metrics = 'accuracy';
    if($args->{'top-k'} > 0) {
        my $comp = AI::MXNet::CompositeEvalMetric->new();
        $comp->add($eval_metrics);
        $comp->add(mx->metric->create('top_k_accuracy',
            top_k=>$args->{'top-k'}));
        $eval_metrics = $comp;
    }

    # callbacks that run after each batch
    my @batch_end_callbacks = mx->callback->Speedometer($args->{batch_size}, $args->{disp_batches});
    if(exists $kwargs{'batch_end_callback'}) {
        my $cbs = $kwargs{'batch_end_callback'};
        push @batch_end_callbacks, ref $cbs eq 'ARRAY' ? @$cbs : $cbs;
    }

    # run
    $model->fit($train,
        begin_epoch        => ($args->{'load-epoch'} // 0),
        num_epoch          => $args->{'num-epochs'},
        eval_data          => $val,
        eval_metric        => $eval_metrics,
        kvstore            => $kv,
        optimizer          => $args->{optimizer},
        optimizer_params   => $optimizer_params,
        initializer        => $initializer,
        arg_params         => $arg_params,
        aux_params         => $aux_params,
        batch_end_callback => \@batch_end_callbacks,
        epoch_end_callback => $checkpoint,
        allow_missing      => 1,
        monitor            => $monitor);
}

1;
