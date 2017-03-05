package AI::MXNet::Executor;
use strict;
use warnings;
use AI::MXNet::Base;
use AI::MXNet::Context;
use Mouse;
use AI::MXNet::Types;
use AI::MXNet::Function::Parameters;

has 'handle'            => (is => 'ro', isa => 'ExecutorHandle', required => 1);
has 'arg_arrays'        => (is => 'rw', isa => 'Maybe[ArrayRef[AI::MXNet::NDArray]]');
has 'grad_arrays'       => (is => 'rw', isa => 'Maybe[ArrayRef[Undef|AI::MXNet::NDArray]]'); 
has 'aux_arrays'        => (is => 'rw', isa => 'Maybe[ArrayRef[AI::MXNet::NDArray]]');
has '_symbol'           => (is => 'rw', init_arg => 'symbol',    isa => 'AI::MXNet::Symbol');
has '_ctx'              => (is => 'rw', init_arg => 'ctx',       isa => 'AI::MXNet::Context' );
has '_grad_req'         => (is => 'rw', init_arg => 'grad_req',  isa => 'Maybe[Str|ArrayRef[Str]|HashRef[Str]]');
has '_group2ctx'        => (is => 'rw', init_arg => 'group2ctx', isa => 'Maybe[HashRef[AI::MXNet::Context]]');
has '_monitor_callback' => (is => 'rw', isa => 'CodeRef');
has [qw/_arg_dict
        _grad_dict
        _aux_dict
        _output_dict
        outputs
        _output_dirty/] => (is => 'rw', init_arg => undef);

=head2

    Executor is the actual executing object of MXNet.
        Constructor, used AI::MXNet::Symbol->bind and AI::MXNet::Symbol->simple_bind instead.

        Parameters
        ----------
        handle: ExecutorHandle
            ExecutorHandle generated by calling bind

        See Also
        --------
        AI::MXNet::Symbol->bind : to create executor
=cut

sub BUILD
{
    my $self = shift;
    my ($symbol, $ctx, $grad_req, $group2ctx)
        =
    ($self->_symbol, $self->_ctx, $self->_grad_req, $self->_group2ctx);
    $symbol = $symbol->deepcopy;
    $ctx    = $ctx->deepcopy;
    if(ref $grad_req)
    {
        if(ref $grad_req eq 'ARRAY')
        {
            $grad_req = [ @{ $grad_req }];
        }
        elsif(ref $grad_req eq 'HASH')
        {
            $grad_req = { %{ $grad_req } };

        }
    }
    if(ref $group2ctx)
    {
        $group2ctx = { %{ $group2ctx } };
    }
    $self->_symbol($symbol);
    $self->_ctx($ctx);
    $self->_grad_req($grad_req);
    $self->_group2ctx($group2ctx);
    $self->outputs($self->_get_outputs);
}

sub DEMOLISH
{
    check_call(AI::MXNetCAPI::ExecutorFree(shift->handle));
}

# Get the dictionary given name and ndarray pairs.
func _get_dict(
    ArrayRef[Str]                       $names,
    ArrayRef[Maybe[AI::MXNet::NDArray]] $ndarrays
)
{
    my %nset = ();
    for my $nm (@{ $names })
    {
        if(exists $nset{ $nm })
        {
            confess("Duplicate names detected, @$names")
        }
        $nset{ $nm }++;
    }
    my %ret;
    @ret{ @{ $names } } = @{ $ndarrays };
    return \%ret;
}

=head2 outputs

        list all the output ndarray

        Returns
        -------
        A list of ndarray bound to the heads of executor.
=cut

method _get_outputs()
{
    return [
            map {
                AI::MXNet::NDArray->new(handle => $_)
            }
            @{ check_call(AI::MXNetCAPI::ExecutorOutputs($self->handle)) }
    ];
}

=head2 forward
        Calculate the outputs specified by the bound symbol.

        Parameters
        ----------
        is_train: bool, optional
            whether this forward is for evaluation purpose. If True,
            a backward call is expected to follow. Otherwise following
            backward is invalid.

        **kwargs
            Additional specification of input arguments.

        Examples
        --------
        >>> # doing forward by specifying data
        >>> $texec->forward(1, data => $mydata);
        >>> # doing forward by not specifying things, but copy to the executor before hand
        >>> $mydata->copyto($texec->arg_dict->{'data'});
        >>> $texec->forward(1);
        >>> # doing forward by specifying data and get outputs
        >>> my $outputs = $texec->forward(1, data => $mydata);
        >>> print $outputs->[0]->aspdl;
=cut

method forward(Int $is_train=0, %kwargs)
{
    if(%kwargs)
    {
        my $arg_dict = $self->arg_dict;
        while (my ($name, $array) = each %kwargs)
        {
            if(not find_type_constraint('AcceptableInput')->check($array))
            {
                confess('only accept keyword argument of NDArrays/PDLs/Perl Array refs');
            }
            if(not exists $arg_dict->{ $name })
            {
                confess("unknown argument $name");
            }
            if(not blessed($array) or not $array->isa('AI::MXNet::NDArray'))
            {
                $array = AI::MXNet::NDArray->array($array);
            }
            if(join(',', @{ $arg_dict->{$name}->shape }) ne join(',', @{ $array->shape }))
            {
                my $expected = $arg_dict->{$name}->shape;
                my $got = $array->shape;
                confess("Shape not match! Argument $name, need: @$expected, received: @$got'");
            }
            $arg_dict->{ $name } .= $array;
        }
    }
    check_call(AI::MXNetCAPI::ExecutorForward(
            $self->handle,
            $is_train
        )
    );
    if($self->_output_dirty)
    {
        AI::MXNet::Logging->warning(
            "Calling forward the second time after forward(is_train=1) "
            ."without calling backward first. Is this intended?"
        );
    }
    $self->_output_dirty($is_train);
    return $self->outputs;
}

=head2 backward

        Do backward pass to get the gradient of arguments.

        Parameters
        ----------
        out_grads : NDArray or list of NDArray or dict of str to NDArray, optional
            Gradient on the outputs to be propagated back.
            This parameter is only needed when bind is called
            on outputs that are not a loss function.
=cut

method backward(Maybe[AI::MXNet::NDArray|ArrayRef[AI::MXNet::NDArray]|HashRef[AI::MXNet::NDArray]] $out_grads=)
{
    $out_grads //= [];
    if(blessed $out_grads)
    {
        $out_grads = [$out_grads];
    }
    elsif(ref $out_grads eq 'HASH')
    {
        $out_grads = [ @{ $out_grads }{ @{ $self->symbol->list_outputs() } } ];
    }
    check_call(
        AI::MXNetCAPI::ExecutorBackward(
            $self->handle,
            scalar(@{ $out_grads }),
            [map { $_->handle } @{ $out_grads }]
        )
    );
    if(not $self->_output_dirty)
    {
        AI::MXNet::Logging->warning(
            "Calling backward without calling forward(is_train=True) "
            ."first. Behavior is undefined."
        );
    }
    $self->_output_dirty(0);
}

=head2 set_monitor_callback

        Install callback.

        Parameters
        ----------
        callback : subref
            Takes a string and an NDArrayHandle.
=cut

method set_monitor_callback(CodeRef $callback)
{
    $self->_monitor_callback($callback);
    check_call(
        AI::MXNetCAPI::ExecutorSetMonitorCallback(
            $self->handle,
            $self->_monitor_callback
        )
    );
}

=head2 arg_dict

        Get dictionary representation of argument arrrays.

        Returns
        -------
        arg_dict : dict of str to NDArray
            The dictionary that maps name of arguments to NDArrays.

        Raises
        ------
        ValueError : if there are duplicated names in the arguments.
=cut

method arg_dict()
{
    if(not defined $self->_arg_dict)
    {
        $self->_arg_dict(_get_dict(
                $self->_symbol->list_arguments(),
                $self->arg_arrays
            )
        );
    }
    return $self->_arg_dict;
}

=head2 grad_dict

        Get dictionary representation of gradient arrays.

        Returns
        -------
        grad_dict : dict of str to NDArray
            The dictionary that maps name of arguments to gradient arrays.
=cut

method grad_dict()
{
    if(not defined $self->_grad_dict)
    {
        $self->_grad_dict(_get_dict(
                $self->_symbol->list_arguments(),
                $self->grad_arrays
            )
        );
    }
    return $self->_grad_dict;
}

=head2 aux_dict

        Get dictionary representation of auxiliary states arrrays.

        Returns
        -------
        aux_dict : dict of str to NDArray
            The dictionary that maps name of auxiliary states to NDArrays.

        Raises
        ------
        ValueError : if there are duplicated names in the arguments.
=cut

method aux_dict()
{
    if(not defined $self->_aux_dict)
    {
        $self->_aux_dict(_get_dict(
                $self->_symbol->list_auxiliary_states(),
                $self->aux_arrays()
            )
        );
    }
    return $self->_aux_dict;
}

=head2 output_dict

        Get dictionary representation of argument arrrays.

        Returns
        -------
        output_dict : dict of str to NDArray
            The dictionary that maps name of arguments to NDArrays.

        Raises
        ------
        ValueError : if there are duplicated names in the arguments.
=cut

method output_dict()
{
    if(not defined $self->_ouput_dict)
    {
        $self->_output_dict(_get_dict(
                $self->_symbol->list_outputs(),
                $self->outputs
            )
        );
    }
    return $self->_output_dict;
}

=head2 copy_params_from

        Copy parameters from arg_params, aux_params into executor's internal array.

        Parameters
        ----------
        arg_params : dict of str to NDArray
            Parameters, dict of name to NDArray of arguments

        aux_params : dict of str to NDArray, optional
            Parameters, dict of name to NDArray of auxiliary states.

        allow_extra_params : boolean, optional
            Whether allow extra parameters that are not needed by symbol
            If this is True, no error will be thrown when arg_params or aux_params
            contain extra parameters that is not needed by the executor.

        Raises
        ------
        ValueError
            If there is additional parameters in the dict but allow_extra_params=False
=cut

method copy_params_from(
    HashRef[AI::MXNet::NDArray]        $arg_params,
    Maybe[HashRef[AI::MXNet::NDArray]] $aux_params=,
    Maybe[Bool]                        $allow_extra_params=
)
{
    my %arg_dict = %{ $self->arg_dict };
    while (my ($name, $array) = each %{ $arg_params })
    {
        if(exists $arg_dict{ $name })
        {
            my $dst = $arg_dict{ $name };
            $array->astype($dst->dtype)->copyto($dst);
        }
        elsif(not $allow_extra_params)
        {
            confess("Found name \"$name\" that is not in the arguments");
        }
    }
    if(defined $aux_params)
    {
        my %aux_dict = %{ $self->aux_dict };
        while (my ($name, $array) = each %{ $aux_params })
        {
            if(exists $aux_dict{ $name })
            {
                my $dst = $aux_dict{ $name };
                $array->astype($dst->dtype)->copyto($dst);
            }
            elsif(not $allow_extra_params)
            {
                confess("Found name \"$name\" that is not in the arguments");
            }
        }
    }
}

=head2 reshape 

        Return a new executor with the same symbol and shared memory,
        but different input/output shapes.
        For runtime reshaping, variable length sequences, etc.
        The returned executor shares state with the current one,
        and cannot be used in parallel with it.

        Parameters
        ----------
        partial_shaping : bool
            Whether to allow changing the shape of unspecified arguments.
        allow_up_sizing : bool
            Whether to allow allocating new ndarrays that's larger than the original.
        kwargs : dict of string to tuple of int
            new shape for arguments.
        Returns
        -------
        exec : Executor
            A new executor that shares memory with self.
=cut


method reshape(HashRef[Shape] $kwargs, Int $partial_shaping=0, Int $allow_up_sizing=0)
{
    my ($arg_shapes, undef, $aux_shapes) = $self->_symbol->infer_shape(%{ $kwargs });
    confess("Insufficient argument shapes provided.") 
        unless defined $arg_shapes;
    my %new_arg_dict;
    my %new_grad_dict;
    my $i = 0;
    for my $name (@{ $self->_symbol->list_arguments() })
    {
        my $new_shape = $arg_shapes->[$i];
        my $arr       = $self->arg_arrays->[$i];
        my $darr;
        if(@{ $self->grad_arrays })
        {
            $darr = $self->grad_arrays->[$i];
        }
        if(
            $partial_shaping 
                or
            exists $kwargs->{ $name } 
                or 
            join(',', @{ $new_shape }) eq join(',', @{ $arr->shape })
        )
        { 
            if(AI::MXNet::NDArray->size($new_shape) > $arr->size)
            {
                confess(
                    "New shape of arg:$name larger than original. "
                    ."First making a big executor and then down sizing it "
                    ."is more efficient than the reverse."
                    ."If you really want to up size, set \$allow_up_sizing=1 "
                    ."to enable allocation of new arrays."
                ) unless $allow_up_sizing;
                $new_arg_dict{ $name }  = AI::MXNet::NDArray->empty(
                    $new_shape,
                    ctx => $arr->context,
                    dtype => $arr->dtype
                );
                if(defined $darr)
                {
                    $new_grad_dict{ $name } = AI::MXNet::NDArray->empty(
                        $new_shape,
                        ctx => $darr->context,
                        dtype => $arr->dtype
                    );
                }
            }
            else
            {
                $new_arg_dict{ $name } = $arr->reshape($new_shape);
                if(defined $darr)
                {
                    $new_grad_dict{ $name } = $darr->reshape($new_shape);
                }
            }
        }
        else
        {
            confess(
                    "Shape of unspecified array arg:$name changed. "
                    ."This can cause the new executor to not share parameters "
                    ."with the old one. Please check for error in network."
                    ."If this is intended, set partial_shaping=True to suppress this warning."
            );
        }
        $i++;
    }
    my %new_aux_dict;
    $i = 0;
    for my $name ($self->_symbol->list_auxiliary_states())
    {
        my $new_shape = $aux_shapes->[$i];
        my $arr = $self->aux_arrays->[$i];
        if($partial_shaping or join(',', @{ $new_shape }) eq join (',', @{ $arr->shape }))
        {
            if(AI::MXNet::NDArray->size($new_shape) > $arr->size)
            {
                confess(
                    "New shape of arg:$name larger than original. "
                    ."First making a big executor and then down sizing it "
                    ."is more efficient than the reverse."
                    ."If you really want to up size, set \$allow_up_sizing=1 "
                    ."to enable allocation of new arrays."
                ) unless $allow_up_sizing;
                $new_aux_dict{ $name }  = AI::MXNet::NDArray->empty(
                    $new_shape, 
                    ctx => $arr->context,
                    dtype => $arr->dtype
                );
            }
            else
            {
                $new_aux_dict{ $name } = $arr->reshape($new_shape);
            }
        }
        else
        {
            confess(
                "Shape of unspecified array aux:$name changed. "
                ."This can cause the new executor to not share parameters "
                ."with the old one. Please check for error in network."
                ."If this is intended, set partial_shaping=True to suppress this warning."
            );
        }
        $i++;
    }
    return $self->_symbol->bind(
                $self->_ctx,
                \%new_arg_dict,
                \%new_grad_dict,
                $self->_grad_req,
                \%new_aux_dict,
                $self->_group2ctx,
                $self
    );
}

=head2 debug_str

        Get a debug string about internal execution plan.

        Returns
        -------
        debug_str : string
            Debug string of the executor.
=cut

method debug_str()
{
    return scalar(check_call(AI::MXNetCAPI::ExecutorPrint($self->handle)));
}

1;
