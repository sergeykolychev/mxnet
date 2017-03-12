package AI::MXNet::TestUtils;
use strict;
use warnings;
use PDL;
use Scalar::Util qw(blessed);
use AI::MXNet::Function::Parameters;
use Exporter;
use base qw(Exporter);
@AI::MXNet::TestUtils::EXPORT_OK = qw(same reldiff almost_equal GetMNIST_ubyte
                                      GetCifar10 pdl_maximum pdl_minimum mlp2 conv);
use constant default_numerical_threshold => 1e-6;
=head2 same

Test if two numpy arrays are the same

    Parameters
    ----------
    a : pdl
    b : pdl
=cut

func same(PDL $a, PDL $b)
{
    return not which($a - $b)->shape->at(0);
}

=head2 reldiff
    Calculate the relative difference between two input arrays

    Calculated by :math:`\\frac{|a-b|_1}{|a|_1 + |b|_1}`

    Parameters
    ----------
    a : pdl
    b : pdl
=cut

func reldiff(PDL $a, PDL $b)
{
    my $diff = sum(abs($a - $b));
    my $norm = sum(abs($a)) + sum(abs($b));
    if($diff == 0)
    {
        return 0;
    }
    my $ret = $diff / $norm;
    return $ret;
}

=head2 almost_equal

Test if two pdl arrays are almost equal.
=cut

func almost_equal(PDL $a, PDL $b, Maybe[Num] $threshold=)
{
    $threshold //= default_numerical_threshold;
    my $rel = reldiff($a, $b);
    return $rel <= $threshold;
}

func GetMNIST_ubyte()
{
    if(not -d "data")
    {
        mkdir "data";
    }
    if (
        not -f 'data/train-images-idx3-ubyte'
            or
        not -f 'data/train-labels-idx1-ubyte'
            or
        not -f 'data/t10k-images-idx3-ubyte'
            or
        not -f 'data/t10k-labels-idx1-ubyte'
    )
    {
        `wget http://data.mxnet.io/mxnet/data/mnist.zip -P data`;
        chdir 'data';
        `unzip -u mnist.zip`;
        chdir '..';
    }
}

func GetCifar10()
{
    if(not -d "data")
    {
        mkdir "data";
    }
    if (not -f 'data/cifar10.zip')
    {
        `wget http://data.mxnet.io/mxnet/data/cifar10.zip -P data`;
        chdir 'data';
        `unzip -u cifar10.zip`;
        chdir '..';
    }
}

func _pdl_compare(PDL $a, PDL|Num $b, Str $criteria)
{
    if(not blessed $b)
    {
        my $tmp = $b;
        $b = $a->copy;
        $b .= $tmp;
    }
    my $mask = {
        'max' => sub { $_[0] < $_[1] },
        'min' => sub { $_[0] > $_[1] },
    }->{$criteria}->($a, $b);
    my $c = $a->copy;
    $c->where($mask) .= $b->where($mask);
    $c;
}

func pdl_maximum(PDL $a, PDL|Num $b)
{
    _pdl_compare($a, $b, 'max');
}

func pdl_minimum(PDL $a, PDL|Num $b)
{
    _pdl_compare($a, $b, 'min');
}

func mlp2()
{
    my $data = AI::MXNet::Symbol->Variable('data');
    my $out  = AI::MXNet::Symbol->FullyConnected(data=>$data, name=>'fc1', num_hidden=>1000);
    $out     = AI::MXNet::Symbol->Activation(data=>$out, act_type=>'relu');
    $out     = AI::MXNet::Symbol->FullyConnected(data=>$out, name=>'fc2', num_hidden=>10);
    return $out;
}

func conv()
{
    my $data    = AI::MXNet::Symbol->Variable('data');
    my $conv1   = AI::MXNet::Symbol->Convolution(data => $data, name=>'conv1', num_filter=>32, kernel=>[3,3], stride=>[2,2]);
    my $bn1     = AI::MXNet::Symbol->BatchNorm(data => $conv1, name=>"bn1");
    my $act1    = AI::MXNet::Symbol->Activation(data => $bn1, name=>'relu1', act_type=>"relu");
    my $mp1     = AI::MXNet::Symbol->Pooling(data => $act1, name => 'mp1', kernel=>[2,2], stride=>[2,2], pool_type=>'max');

    my $conv2   = AI::MXNet::Symbol->Convolution(data => $mp1, name=>'conv2', num_filter=>32, kernel=>[3,3], stride=>[2,2]);
    my $bn2     = AI::MXNet::Symbol->BatchNorm(data => $conv2, name=>"bn2");
    my $act2    = AI::MXNet::Symbol->Activation(data => $bn2, name=>'relu2', act_type=>"relu");
    my $mp2     = AI::MXNet::Symbol->Pooling(data => $act2, name => 'mp2', kernel=>[2,2], stride=>[2,2], pool_type=>'max');

    my $fl      = AI::MXNet::Symbol->Flatten(data => $mp2, name=>"flatten");
    my $fc2     = AI::MXNet::Symbol->FullyConnected(data => $fl, name=>'fc2', num_hidden=>10);
    my $softmax = AI::MXNet::Symbol->SoftmaxOutput(data => $fc2, name => 'sm');
    return $softmax;
}

1;