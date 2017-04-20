#!/usr/bin/perl
use strict;
use warnings;
use AI::MXNet ('mx');

sub samples {
    my($batch_size, $func) = @_;
    # get samples
    my $n = 16384;
    my $d = PDL->random(2, $n);
    my $l = $func->($d->slice('0,:'), $d->slice('1,:'));
    # partition into train/eval sets
    my $edge = int($n / 8);
    my $vd = $d->slice(":,0:@{[ $edge - 1 ]}");
    my $vl = $l->slice(":,0:@{[ $edge - 1 ]}");
    my $td = $d->slice(":,$edge:");
    my $tl = $l->slice(":,$edge:");
    # build iters around them
    return(mx->io->NDArrayIter(
        batch_size => $batch_size,
        data => $td,
        label => $tl,
    ), mx->io->NDArrayIter(
        batch_size => $batch_size,
        data => $vd,
        label => $vl,
    ));
}

sub nn_fc {
    my $data = mx->sym->Variable('data');
    my $ln = mx->sym->exp(mx->sym->FullyConnected(
        data => mx->sym->log($data),
        num_hidden => 1,
    ));
    my $wide = mx->sym->Concat($data, $ln);
    my $fc = mx->sym->FullyConnected(data => $wide, num_hidden => 1);
    return mx->sym->MAERegressionOutput(data => $fc, name => 'softmax');
}

sub learn_function {
    my(%args) = @_;
    my $func = $args{func};
    my $batch_size = 128;
    my($train_iter, $eval_iter) = samples($batch_size, $func);
    my $sym = nn_fc(mx->sym->Variable('data'), 'softmax');

    if(0) {
        my @dsz = @{$train_iter->data->[0][1]->shape};
        my @lsz = @{$train_iter->label->[0][1]->shape};
        my $shape = {
            data          => [ $batch_size, splice @dsz,  1 ],
            softmax_label => [ $batch_size, splice @lsz, 1 ],
        };
        print mx->viz->plot_network($sym, shape => $shape)->graph->as_png;
        exit;
    }

    my $model = mx->mod->Module(
        symbol => $sym,
        context => mx->cpu(),
    );
    $model->fit($train_iter,
        eval_data => $eval_iter,
        optimizer => 'adam',
        optimizer_params => {
            learning_rate => 0.01,
        },
        eval_metric => 'mse',
        num_epoch => 4,
    );

    # refit the model for calling on 1 sample at a time
    my $iter = mx->io->NDArrayIter(
        batch_size => 1,
        data => PDL->pdl([[ 0, 0 ]]),
        label => PDL->pdl([[ 0 ]]),
    );
    $model->reshape(
        data_shapes => $iter->provide_data,
        label_shapes => $iter->provide_label,
    );

    # wrap a helper around making predictions
    return sub {
        my($n, $m) = @_;
        return $model->predict(mx->io->NDArrayIter(
            batch_size => 1,
            data => PDL->new([[ $n, $m ]]),
        ))->aspdl->list;
    };
}

my $add = learn_function(func => sub {
    my($n, $m) = @_;
    return $n + $m;
});
my $sub = learn_function(func => sub {
    my($n, $m) = @_;
    return $n - $m;
});
my $mul = learn_function(func => sub {
    my($n, $m) = @_;
    return $n * $m;
});
my $div = learn_function(func => sub {
    my($n, $m) = @_;
    return $n / $m;
});

print "12345 + 54321 ≈ ", $add->(12345, 54321), "\n";
print "188 - 88 ≈ ", $sub->(188, 88), "\n";
print "0.5 * 2 ≈ ", $mul->(0.5, 2), "\n";
print "0.5 / 2 ≈ ", $div->(0.5, 2), "\n";


sub tbl {
    my @l = ('     ');
    
    {
        my @l = ('     ');
        for(my $j = 0; $j < @_; $j++) {
            push @l, sprintf "%5.2f", $_[$j];
        }
        print "@l\n";
    }
    for(my $i = 0; $i < @_; $i++) {
        my @l = (sprintf "%5.2f", $_[$i]);
        for(my $j = 0; $j < @_; $j++) {
            push @l, sprintf "%5.2f", $mul->($_[$i], $_[$j]);
        }
        print "@l\n";
    }
}
tbl(map { ($_ - 5) / 10 } 0 .. 20);

#printf "1.0 * %.2f = %.2f\n", $_, $mul->(2, $_ / 10)
#    for map { $_ / 10 } 0 .. 20
