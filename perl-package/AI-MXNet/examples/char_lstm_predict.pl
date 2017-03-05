#!/usr/bin/perl
use strict;
use warnings;
use PDL;
use AI::MXNet qw(mx);
use AI::MXNet::Function::Parameters;
use Getopt::Long qw(HelpMessage);

GetOptions(
    'num-layers=i'   => \(my $num_layers   = 2       ),
    'num-hidden=i'   => \(my $num_hidden   = 256     ),
    'num-seq=i'      => \(my $seq_size     = 32      ),
    'gpus=s'         => \(my $gpus                   ),
    'kv-store=s'     => \(my $kv_store     = 'device'),
    'num-epoch=i'    => \(my $num_epoch    = 25      ),
    'lr=f'           => \(my $lr           = 0.01    ),
    'optimizer=s'    => \(my $optimizer    = 'sgd'   ),
    'mom=f'          => \(my $mom          = 0       ),
    'wd=f'           => \(my $wd           = 0.00001 ),
    'batch_size'     => \(my $batch_size   = 1       ),
    'disp-batches=i' => \(my $disp_batches = 50      ),
    'chkp-prefix=s'  => \(my $chkp_prefix  = 'lstm_' ),
    'chkp-epoch=i'   => \(my $chkp_epoch   = 0       ),
    'help'           => sub { HelpMessage(0) },
) or HelpMessage(1);

=head1 NAME

    char_lstm.pl - Example of training char LSTM RNN on tiny shakespeare using high level RNN interface

=head1 SYNOPSIS

    --num-layers     number of stacked RNN layers, default=2
    --num-hidden     hidden layer size, default=200
    --num-seq        sequence size, default=60
    --gpus           list of gpus to run, e.g. 0 or 0,2,5. empty means using cpu.
                     Increase batch size when using multiple gpus for best performance.

    --kv-store       key-value store type, default='device'
    --num-epochs     max num of epochs, default=25
    --lr             initial learning rate, default=0.01
    --optimizer      the optimizer type, default='sgd'
    --mom            momentum for sgd, default=0.0
    --wd             weight decay for sgd, default=0.00001
    --batch-size     the batch size type, default=32
    --disp-batches   show progress for every n batches, default=50
    --chkp-prefix    prefix for checkpoint files, default='lstm_'
    --chkp-epoch     save checkpoint after this many epoch, default=0 (saving checkpoints is disabled)

=cut

package AI::MXNet::RNN::IO::ASCIIIterator;
use Mouse;
extends AI::MXNet::DataIter;
has 'data'          => (is => 'ro',  isa => 'PDL',   required => 1);
has 'seq_size'      => (is => 'ro',  isa => 'Int',   required => 1);
has '+batch_size'   => (is => 'ro',  isa => 'Int',   required => 1);
has 'data_name'     => (is => 'ro',  isa => 'Str',   default => 'data');
has 'label_name'    => (is => 'ro',  isa => 'Str',   default => 'softmax_label');
has 'dtype'         => (is => 'ro',  isa => 'Dtype', default => 'float32');
has [qw/nd counter vocab_size
    data_size provide_data provide_label loop/] => (is => 'rw', init_arg => undef);

sub BUILD
{
    my $self = shift;
    $self->data_size($self->data->nelem);
    $self->vocab_size(65);
    my $nd = mx->nd->zeros([$self->batch_size*$self->seq_size], dtype => $self->dtype);
    $nd .= $self->data->slice([0,$self->seq_size-1]);
    $self->counter(0);
    $self->nd($nd);
    my $shape = [$self->batch_size, $self->seq_size];
    $self->provide_data([
        AI::MXNet::DataDesc->new(
            name  => $self->data_name,
            shape => $shape,
            dtype => $self->dtype
        )
    ]);
    $self->provide_label([
        AI::MXNet::DataDesc->new(
            name  => $self->label_name,
            shape => $shape,
            dtype => $self->dtype
        )
    ]);
}

method reset()
{
    $self->counter(0);
}

method next()
{
    if(defined $self->loop)
    {
	$self->nd->slice([0,$self->seq_size-2]) .= $self->nd->slice([1,$self->seq_size-1]);
	$self->nd->at($self->seq_size-1) .= $self->loop;
    }
    return AI::MXNet::DataBatch->new(
        data          => [$self->nd->reshape([$self->batch_size, $self->seq_size])],
        provide_data  => [
            AI::MXNet::DataDesc->new(
                name  => $self->data_name,
                shape => [1,32],
                dtype => $self->dtype
            )
        ],
    );
}

package main;
my $file = "data/input.txt";
open(F, $file) or die "can't open $file: $!";
my $fdata;
{ local($/) = undef; $fdata = <F>; close(F) };
my %vocab; my $i = 0;
$fdata = pdl(map{ exists $vocab{ $_ } ? $vocab{ $_ } : ($vocab{ $_ }=$i++) } split(//, $fdata));
my %reverse_vocab = reverse %vocab;
my $data_iter = AI::MXNet::RNN::IO::ASCIIIterator->new(
    batch_size => $batch_size,
    data       => $fdata,
    seq_size   => $seq_size
);

my $stack = mx->rnn->SequentialRNNCell();
for my $i (0..$num_layers-1)
{
    $stack->add(mx->rnn->LSTMCell(num_hidden => $num_hidden, prefix => "lstm_l${i}_"));
}

my $contexts;
if($gpus)
{
    $contexts = [map { mx->gpu($_) } split(/,/, $gpus)];
}
else
{
    $contexts = mx->cpu(0);
}

my ($sym, $arg_params, $aux_params) = mx->rnn->load_rnn_checkpoint($stack, 'shakespeare1', 2);

my $model = mx->mod->Module(
    symbol  => $sym,
    context => $contexts
);

$model->bind(
    data_shapes  => $data_iter->provide_data,
    label_shapes => $data_iter->provide_label,
    for_training => 0,
    force_rebind => 0
);
$model->set_params(
    $arg_params,
    $aux_params,
    allow_missing => 1,
    force_init    => 0
);
srand(time);
use Math::Random::Discrete;
map { print $reverse_vocab{$_} } @{ $data_iter->nd->aspdl->unpdl };
while(1)
{
    my @out = $model->iter_predict($data_iter, num_batch => 1);
    my $pred = $out[0][0][0];
    my $sentence_size = $pred->shape->[0];
    my $ix = Math::Random::Discrete->new($pred->at($sentence_size-1)->aspdl->unpdl, [0..$data_iter->vocab_size-1])->rand;
    #my $ix = $pred->at($sentence_size-1)->aspdl->maximum_ind;
    print "$ix\n";
    print $reverse_vocab{$ix};
    $data_iter->loop($ix);
}
