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

package AI::MXNet::KVStore;
use strict;
use warnings;
use AI::MXNet::Base;
use AI::MXNet::NDArray;
use AI::MXNet::Optimizer;
use MIME::Base64;
use Storable;
use Mouse;
use AI::MXNet::Function::Parameters;

=head1 NAME

    AI::MXNet::KVStore - Key value store interface of MXNet for parameter synchronization.
=cut
func _ctype_key_value($keys, $vals)
{
    if(ref $keys eq 'ARRAY') {
        assert(@$keys == @$vals);
        my @c_keys;
        my @c_vals;
        zip(sub {
            my($key, $val) = @_;
            my($c_key_i, $c_val_i) = _ctype_key_value($key, $val);
            push @c_keys, @$c_key_i;
            push @c_vals, @$c_key_i;
        }, $keys, $vals);
        return (\@c_keys, \@c_vals);
    }
    my @names;
    if(blessed $vals and $vals->isa('AI::MXNet::NDArray')) {
        push @names, "" . $keys;
        return (\@names, [ $vals->handle ]);
    } else {
        assert(blessed $_ and $_->isa('AI::MXNet::NDArray'))
            for @$vals;
        return ([ ($keys) x @$vals ], [ map { $_->handle } @$vals ]);
    }
}

# A wrapper for the user-defined handle.
method _updater_wrapper($updater)
{
    return sub {
        my($key, $lhs_handle, $rhs_handle) = @_;
        # ctypes function
        my $lhs = AI::MXNet::NDArray->new(handle => $lhs_handle);
        my $rhs = AI::MXNet::NDArray->new(handle => $rhs_handle);
        $updater->($key, $lhs, $rhs);
    };
}
=head1 DESCRIPTION

    A key-value store for synchronization of values, over multiple devices.
=cut

has 'handle' => (is => 'ro', isa => 'KVStoreHandle', required => 1);
has '_updater' => (is => 'rw',  isa => 'AI::MXNet::Updater');
has '_updater_func' => (is => 'rw', isa => 'CodeRef');

sub DEMOLISH
{
    check_call(AI::MXNetCAPI::KVStoreFree(shift->handle));
}

=head2  init

    Initializes a single or a sequence of key-value pairs into the store.

    For each key, one must init it before calling push or pull.
    When multiple workers invoke init for the same key, only
    the value supplied by worker with rank 0 is used. This function returns
    after data has been initialized successfully.

    Parameters
    ----------
    key : str or an array ref of str
        The keys.
    value : NDArray or an array ref of NDArray objects
        Values corresponding to the keys.

    Examples
    --------
    >>> # init a single key-value pair
    >>> $shape = [2,3]
    >>> $kv = mx->kv->create('local')
    >>> $kv->init(3, mx->nd->ones($shape)*2)
    >>> $a = mx->nd->zeros($shape)
    >>> $kv->pull(3, out=>$a)
    >>> print $a->aspdl
    [[ 2  2  2]
    [ 2  2  2]]

    >>> # init a list of key-value pairs
    >>> $keys = [5, 7, 9]
    >>> $kv->init(keys, [map { mx->nd->ones($shape) } 0..@$keys-1])
=cut

method init(
    Str|ArrayRef[Str] $key,
    AI::MXNet::NDArray|ArrayRef[AI::MXNet::NDArray]|ArrayRef[ArrayRef[AI::MXNet::NDArray]] $value
)
{
    my ($ckeys, $cvals) = _ctype_key_value($key, $value);
    check_call(AI::MXNetCAPI::KVStoreInitEx($self->handle, 0 + @$ckeys, $ckeys, $cvals));
}

=head2  push

    Pushes a single or a sequence of key-value pairs into the store.

    This function returns immediately after adding an operator to the engine.
    The actual operation is executed asynchronously after all previous push
    and pull calls for the same input key(s) are finished.
    There is no synchronization between workers. One can use _barrier()
    to sync all workers.

    Parameters
    ----------
    key : str or array ref of str
    value : NDArray or array ref of NDArray or array ref of array refs of NDArray
    priority : int, optional
        The priority of the push operation.
        Higher priority push operations are likely to be executed before
        other push actions.

    Examples
    --------
    >>> # push a single key-value pair
    >>> $kv->push(3, mx->nd->ones($shape)*8)
    >>> $kv->pull(3, out=>$a) # pull out the value
    >>> print $a->aspdl()
        [[ 8.  8.  8.]
        [ 8.  8.  8.]]

    >>> # aggregate the value and the push
    >>> $gpus = [map { mx->gpu($_) } 0..3]
    >>> $b = [map { mx->nd->ones($shape, ctx => $_) } @$gpus]
    >>> $kv->push(3, $b)
    >>> $kv->pull(3, out=>$a)
    >>> print $a->aspdl
        [[ 4.  4.  4.]
        [ 4.  4.  4.]]

    >>> # push a list of keys.
    >>> # single device
    >>> $kv->push($keys, [map { mx->nd->ones($shape) } 0..@$keys-1)
    >>> $b = [map { mx->nd->zeros(shape) } 0..@$keys-1]
    >>> $kv->pull($keys, out=>$b)
    >>> print $b->[1]->aspdl
        [[ 1.  1.  1.]
        [ 1.  1.  1.]]

    >>> # multiple devices:
    >>> $b = [map { [map { mx->nd->ones($shape, ctx => $_) } @$gpus] } @$keys-1]
    >>> $kv->push($keys, $b)
    >>> $kv->pull($keys, out=>$b)
    >>> print $b->[1][1]->aspdl()
        [[ 4.  4.  4.]
        [ 4.  4.  4.]]
=cut

method push(
    Str|ArrayRef[Str] $key,
    AI::MXNet::NDArray|ArrayRef[AI::MXNet::NDArray]|ArrayRef[ArrayRef[AI::MXNet::NDArray]] $value,
    Int :$priority=0
)
{
    my ($ckeys, $cvals) = _ctype_key_value($key, $value);
    check_call(AI::MXNetCAPI::KVStorePushEx($self->handle, 0 + @$ckeys, $ckeys, $cvals, 0 + $priority));
}

=head2 pull

    Pulls a single value or a sequence of values from the store.

    This function returns immediately after adding an operator to the engine.
    Subsequent attempts to read from the out variable will be blocked until the
    pull operation completes.

    pull is executed asynchronously after all previous push and pull calls
    for the same input key(s) are finished.

    The returned values are gauranteed to be the latest values in the store.

    For row_sparse values, please use row_sparse_pull instead.

    Parameters
    ----------
    key : str or array ref of str
        Keys
    out: NDArray or array ref of NDArray or array ref of array refs of NDArray
        Values corresponding to the keys.

    priority : int, optional
        The priority of the pull operation.
        Higher priority pull operations are likely to be executed before
        other pull actions.

    Examples
    --------
    >>> # pull a single key-value pair
    >>> $a = mx->nd->zeros($shape)
    >>> $kv->pull(3, out=>$a)
    >>> print $a->aspdl
        [[ 2.  2.  2.]
        [ 2.  2.  2.]]

    >>> # pull into multiple devices
    >>> $b = [map { mx->nd->ones($shape, $_) } @$gpus]
    >>> $kv->pull(3, out=>$b)
    >>> print $b->[1]->aspdl()
        [[ 2.  2.  2.]
        [ 2.  2.  2.]]

    >>> # pull a list of key-value pairs.
    >>> # On single device
    >>> $keys = [5, 7, 9]
    >>> $b = [map { mx->nd->zeros($shape) } 0..@$keys-1]
    >>> $kv->pull($keys, out=>$b)
    >>> print $b->[1]->aspdl()
        [[ 2.  2.  2.]
        [ 2.  2.  2.]]
    >>> # On multiple devices
    >>> $b = [map { [map { mx->nd->ones($shape, ctx => $_) } @$gpus ] } 0..@$keys-1]
    >>> $kv->pull($keys, out=>$b)
    >>> print $b->[1][1]->aspdl()
        [[ 2.  2.  2.]
        [ 2.  2.  2.]]
=cut

method pull(
    Str|ArrayRef[Str] $key,
    AI::MXNet::NDArray|ArrayRef[AI::MXNet::NDArray]|ArrayRef[ArrayRef[AI::MXNet::NDArray]] :$out,
    Int :$priority=0
)
{
    my ($ckeys, $cvals) = _ctype_key_value($key, $out);
    check_call(AI::MXNetCAPI::KVStorePullEx($self->handle, 0 + @$ckeys, $ckeys, $cvals, 0 + $priority));
}

=head2  set_optimizer

    Registers an optimizer with the kvstore.

    When using a single machine, this function updates the local optimizer.
    If using multiple machines and this operation is invoked from a worker node,
    it will serialized the optimizer with pickle and send it to all servers.
    The function returns after all servers have been updated.

    Parameters
    ----------
    optimizer : Optimizer
        The new optimizer for the store

    Examples
    --------

    >>> $kv = mx->kv->create()
    >>> $shape = [2, 2]
    >>> $weight = mx->nd->zeros($shape)
    >>> $kv->init(3, $weight)
    >>> # set the optimizer for kvstore as the default SGD optimizer
    >>> $kv->set_optimizer(mx->optimizer->SGD())
    >>> $grad = mx->nd->ones($shape)
    >>> $kv->push(3, $grad)
    >>> $kv->pull(3, out => $weight)
    >>> # weight is updated via gradient descent
    >>> print $weight->aspdl()
        [[-0.01, -0.01],
        [-0.01, -0.01]]
=cut
method set_optimizer(AI::MXNet::Optimizer $optimizer)
{
    my $is_worker = check_call(AI::MXNetCAPI::KVStoreIsWorkerNode());
    if($self->type =~ /dist/ and $is_worker)
    {
        my $optim_str = MIME::Base64::encode_base64(Storable::freeze($optimizer), "");
        $self->_send_command_to_servers(0, $optim_str);
    } else {
        $self->_set_updater(AI::MXNet::Optimizer->get_updater($optimizer));
    }
}

=head2  type

    Returns the type of this kvstore.

    Returns
    -------
    type : str
        the string type
=cut

method type()
{
    return '' . check_call(AI::MXNetCAPI::KVStoreGetType($self->handle));
}

=head2  rank

    Returns the rank of this worker node.

    Returns
    -------
    rank : int
        The rank of this node, which is in [0, get_num_workers())
=cut

method rank()
{
    return 0 + check_call(AI::MXNetCAPI::KVStoreGetRank($self->handle));
}

=head2  num_workers

    Returns the number of worker nodes

    Returns
    -------
    size :int
        The number of worker nodes
=cut

method num_workers()
{
    return 0 + check_call(AI::MXNetCAPI::KVStoreGetGroupSize($self->handle));
}

=head2 save_optimizer_states

    Saves the optimizer (updater) state to a file. This is often used when checkpointing
    the model during training.

    Parameters
    ----------
    fname : str
        Path to the output states file.
=cut

method save_optimizer_states(Str $fname)
{
    assert(defined($self->_updater), "Cannot save states for distributed training");
    IO::File->new($fname, 'wb')->print($self->_updater->get_states());
}

=head2 load_optimizer_states

    Loads the optimizer (updater) state from the file.

    Parameters
    ----------
    fname : str
        Path to input states file.
=cut

method load_optimizer_states(Str $fname)
{
    assert(defined($self->_updater), "Cannot save states for distributed training");
    $self->_updater->set_states(join '', IO::File->new($fname, 'rb')->getlines());
}

=head2 _set_updater

    Sets a push updater into the store.

    This function only changes the local store. When running on multiple machines one must
    use set_optimizer.

    Parameters
    ----------
    updater : function
        the updater function

    Examples
    --------
    >>> my $update = sub { my ($key, input, stored) = @_;
        ...     print "update on key: $key\n";
        ...     $stored += $input * 2; };
        >>> $kv->_set_updater($update)
        >>> $kv->pull(3, out=>$a)
        >>> print $a->aspdl()
        [[ 4.  4.  4.]
        [ 4.  4.  4.]]
        >>> $kv->push(3, mx->nd->ones($shape))
        update on key: 3
        >>> $kv->pull(3, out=>$a)
        >>> print $a->aspdl()
        [[ 6.  6.  6.]
        [ 6.  6.  6.]]
=cut

method _set_updater(AI::MXNet::Updater $updater)
{
    $self->_updater($updater);
    $self->_updater_func($self->_updater_wrapper($updater));
    check_call(AI::MXNetCAPI::KVStoreSetUpdater($self->handle, $self->_updater_func));
}

=head2 _barrier

    Invokes global barrier among all worker nodes.

    For example, assume there are n machines. We would like machine 0 to first
    init the values and then have all the workers pull the initialized value.
    Before pulling, we can place invoke _barrier() to guarantee that the
    initialization is finished.
=cut

method _barrier()
{
    check_call(AI::MXNetCAPI::KVStoreBarrier($self->handle));
}

=head2 _send_command_to_servers

    Sends a command to all server nodes.

    Sending command to a server node will cause that server node to invoke
    KVStoreServer->controller to execute the command.

    This function returns after the command has been executed on all server
    nodes.

    Parameters
    ----------
    head : int
        the head of the command
    body : str
        the body of the command
=cut

method _send_command_to_servers(Int $head, Str $body)
{
    check_call(AI::MXNetCAPI::KVStoreSendCommmandToServers($self->handle, $head, $body));
}

=head2 create

    Creates a new KVStore.

    For single machine training, there are two commonly used types:

    ``local``: Copies all gradients to CPU memory and updates weights there.

    ``device``: Aggregates gradients and updates weights on GPUs. With this setting,
    the KVStore also attempts to use GPU peer-to-peer communication,
    potentially accelerating the communication.

    For distributed training, KVStore also supports a number of types:

    ``dist_sync``: Behaves similarly to ``local`` but with one major difference.
    With ``dist_sync``, batch-size now means the batch size used on each machine.
    So if there are ``n`` machines and we use batch size ``b``,
    then ``dist_sync`` behaves like ``local`` with batch size ``n * b``.

    ``dist_device_sync``: Identical to ``dist_sync`` with the difference similar
    to ``device`` vs ``local``.

    ``dist_async``: Performs asynchronous updates.
    The weights are updated whenever gradients are received from any machine.
    No two updates happen on the same weight at the same time. However, the order is not
    guaranteed.

    Parameters
    ----------
    name : {'local', 'device', 'dist_sync', 'dist_device_sync', 'dist_async'}
    The type of KVStore
    Returns
    -------
    kv : KVStore
        The created AI::MXNet::KVStore
=cut

method create(Str $name='local')
{
    my $handle = check_call(AI::MXNetCAPI::KVStoreCreate($name));
    return __PACKAGE__->new(handle => $handle);
}

1;
# src: python/mxnet/kvstore.py@{0b13631} Tue Aug 22 14:56:33 2017 -0700
