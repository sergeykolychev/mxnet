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

package AI::MXNet::NDArray::Sparse::Base;
use strict;
use warnings;
use AI::MXNet::Base;
use AI::MXNet::Function::Parameters;
use Mouse;
extends 'AI::MXNet::NDArray';

=head1 NAME

    AI::MXNet::NDArray::Sparse - Sparse NDArray API of MXNet
=cut

=head1 NAME

    AI::MXNet::NDArray::Sparse::Base
=cut

=head1 DESCRIPTION

    The base class of an NDArray stored in a sparse storage format.
    See AI::MXNet::NDArray::CSR and AI::MXNet::NDArray::RowSparse for more details.
=cut

method _new_alloc_handle(
    StorageType              $stype,
    Shape                    $shape,
    AI::MXNet::Context       $ctx,
    Bool                     $delay_alloc,
    Dype                     $dtype,
    AuxTypes                 $aux_types,
    Maybe[ArrayRef[Shape]]   $aux_shapes=
)
{
    confess("only int64 is supported for aux types")
        if (grep { $_ ne 'int64' } @$aux_types);
    my $aux_type_ids = [map { DTYPE_STR_TO_MX->{$_} } @$aux_types];
    $aux_shapes //= [map { [0] } @$aux_types];
    my $aux_shape_lens = [map { scalar(@$_) } @$aux_shapes];
    @$aux_shapes = map { @$_} @$aux_shapes;
    my $num_aux = mx_uint(len(aux_types))
    my $handle = check_call(
        AI::MXNetCAPI::NDArrayCreateSparseEx(
            STORAGE_TYPE_STR_TO_ID->{$stype},
            $shape,
            scalar(@$shape),
            $ctx->device_type_id,
            $ctx->device_id,
            $delay_alloc,
            DTYPE_STR_TO_MX->{$dtype},
            scalar(@$aux_types),
            $aux_type_ids,
            $aux_shape_lens,
            $aux_shapes
        )
    );
}

method _class_name()
{
    my $class = ref $self || $self;
    $class =~ s/^.+:://;
    $class;
}

sub not_implemented { confess "Not implemented" }
use overload '""' => sub {
                        my $self = shift;
                        my $shape_info = join('x', @{ $self->shape });
                        sprintf("\n<%s, % @%s>", $self->_class_name, $shape_info, $self->context);
                     },
             '+=' => \&not_implemented,
             '-=' => \&not_implemented,
             '*=' => \&not_implemented,
             '/=' => \&not_implemented;
{
    no warnings 'redefine';
    *_sync_copyfrom = *_at = *_slice = *reshape = *size = \&not_implemented;
}

method _aux_type(Int $i)
{
    return DTYPE_MX_TO_STR->{
        check_call(
            AI::MXNetCAPI::NDArrayGetAuxType(
                $self->handle, $i
        )
    }
}

method _num_aux()
{
    return scalar(@{ STORAGE_AUX_TYPES->{ $self->stype });
}

method _aux_types()
{
    [map { $self->_aux_type($_) } 0..$self->_num_aux-1];
}

=head2 aspdl

    Return a dense PDL object with value copied from this array
=cut

method aspdl()
{
    return $self->tostype('default')->aspdl;
}

=head2 astype

        Returns a copy of the array after casting to a specified type.
        Parameters
        ----------
        dtype : Dtype
            The type of the returned array.
        Examples
        --------
        >>> $x = mx->nd->sparse->zeros('row_sparse', [2,3], dtype=>'float32')
        >>> $y = $x->astype('int32')
        >>> $y->dtype
        <type 'int32'>
=cut

method astype(Dtype $dtype)
{
    my $res = $self->zeros(
        shape => $self->shape, ctx => $self->context,
        dtype => $dtype, stype => $self->stype
    );
    $self->copyto($res);
    return $res;
}

=head2 copyto

        Copies the value of this array to another array.

        Parameters
        ----------
        other : NDArray or NDArray::CSR or NDArray::RowSparse or Context
            The destination array or context.

        Returns
        -------
        NDArray or CSRNDArray::CSR or NDArray::RowSparse
            The copied array.
=cut

method copyto(AI::MXNet::NDArray|AI::MXNet::Context $other)
{
    if($other->isa('AI::MXNet::NDArray'))
    {
        if($self->handle eq $other->handle)
        {
            Carp::cluck('You are attempting to copy an array to itself');
            return;
        }
        else
        {
            return __PACKAGE__->_copyto($self, out => $other);
        }
    }
    elsif($other->isa('AI::MXNet::Context'))
    {
        my    hret = _ndarray_cls(_new_alloc_handle(self.stype, self.shape, other,
                                                  True, self.dtype, self._aux_types))
            return _internal._copyto(self, out=hret)
    }
}

method _ndarray_cls($handle, $writable=1, $stype=STORAGE_TYPE_UNDEFINED)
{
    if($stype eq STORAGE_TYPE_UNDEFINED)
    {
        $stype = __PACKAGE__->_storage_type($handle);
    }
    if($stype eq STORAGE_TYPE_DEFAULT)
    {
        return AI::MXNet::NDArray->new(handle => $handle, writable => $writable);
    }
    elsif($stype eq STORAGE_TYPE_CSR)
    {
        return AI::MXNet::NDArray::CSR->new(handle => $handle, writable => $writable);
    }
    elsif($stype eq STORAGE_TYPE_ROW_SPARSE)
    {
        return AI::MXNet::NDArray::RowSparse->new(handle => $handle, writable => $writable);
    }
    else
    {
        confess("unknown storage type: $stype");
    }
}


=head2 check_format

        Check whether the NDArray format is valid.

        Parameters
        ----------
        full_check : bool, optional
            If `True`, rigorous check, O(N) operations. Otherwise
            basic check, O(1) operations (default True).
=cut

method check_format(Bool $full_check=1)
{
    scalar(check_call(AI::MXNetCAPI::NDArraySyncCheckFormat($self->handle, $full_check)));
}

=head2 _data

        A deep copy NDArray of the data array associated with the BaseSparseNDArray.

        This function blocks. Do not use it in performance critical code.
=cut

method _data()
{
    $self->wait_to_read;
    my $handle = check_call(AI::MXNetCAPI::NDArrayGetDataNDArray($self->handle));
    return AI::MXNet::NDArray->new(handle => $handle);
}

=head2 _aux_data

        Get a deep copy NDArray of the i-th aux data array associated with the
        AI::MXNet::NDArray::Sparse::Base

        This function blocks. Do not use it in performance critical code.
=cut

method _aux_data(Int $i)
{
    $self->wait_to_read;
    my $handle = check_call(AI::MXNetCAPI::NDArrayGetAuxNDArray($self->handle, $i));
    return AI::MXNet::NDArray->new(handle => $handle);
}

package AI::MXNet::NDArray::CSR;
use Mouse;
extends 'AI::MXNet::NDArray::Sparse::Base';

=head1 NAME

    AI::MXNet::NDArray::CSR - A sparse representation of 2D NDArray in the Compressed Sparse Row format.
=cut

=head1 DESCRIPTION

    A AI::MXNet::NDArray::CSR represents an AI::MXNet::NDArray as three separate arrays: `data`,
    `indptr` and `indices`. It uses the CSR representation where the column indices for
    row i are stored in ``indices[indptr[i]:indptr[i+1]]`` and their corresponding values are stored
    in ``data[indptr[i]:indptr[i+1]]``.

    The column indices for a given row are expected to be sorted in ascending order.
    Duplicate column entries for the same row are not allowed.

    Example
    -------
    >>> $a = mx->nd->array([[0, 1, 0], [2, 0, 0], [0, 0, 0], [0, 0, 3]]);
    >>> $a = $a->tostype('csr');
    >>> $a->data->aspdl;
    [ 1  2  3]
    >>> $a->indices->aspdl
    [1 0 2]
    >>> $a->indptr->aspdl
    [0 1 2 2 3]

    See Also
    --------
    csr_matrix: Several ways to construct a CSRNDArray
=cut

#    def __reduce__(self):
#        return CSRNDArray, (None,), super(CSRNDArray, self).__getstate__()

use overload '+=' => sub { ($_[0] + $_[1])->copyto($_[0]) },
             '-=' => sub { ($_[0] - $_[1])->copyto($_[0]) },
             '*=' => sub { ($_[0] * $_[1])->copyto($_[0]) },
             '/=' => sub { ($_[0] / $_[1])->copyto($_[0]) };

=head2 slice

        Returns a sliced view of this array.

        Parameters
        ----------
        key : int or array ref
            Indexing key.

        Examples
        --------
        >>> $indptr = [0, 2, 3, 6];
        >>> $indices = [0, 2, 2, 0, 1, 2];
        >>> $data = [1, 2, 3, 4, 5, 6];
        >>> $a = mx->nd->sparse->csr_matrix([$data, $indices, $indptr], shape=>[3, 3])
        >>> $a->aspdl
            [[ 1  0  2]
             [ 0  0  3]
             [ 4  5  6]]
        >>> $a->slice([1,2])->aspdl
        [[ 0  0  3]]
        >>> $a->slice(1)->aspdl
        [[ 0  0  3]]
        >>> $a->[-1]->aspdl
        [[ 4  5  6]]
=cut

method slice(Slice $slice)
{
    my ($begin, $end);
    if(not ref $slice)
    {
        if($slice < 0)
        {
            $begin = $self->shape->[0] + $slice;
        }
        else
        {
            $begin = $slice;
        }
        $end = $begin + 1;
    }
    else
    {
        ($begin, $end) = @{ $slice };
        $end //= $self->shape->[0];
    }
    return $self->SUPER::slice(begin => $begin, end => $end);
}

    def __setitem__(self, key, value):
        """x.__setitem__(i, y) <=> x[i]=y

        Set self[key] to value. Only slice key [:] is supported.

        Parameters
        ----------
        key : slice
            The indexing key.
        value : NDArray or CSRNDArray or numpy.ndarray
            The value to set.

        Examples
        --------
        >>> src = mx.nd.sparse.zeros('csr', (3,3))
        >>> src.asnumpy()
        array([[ 0.,  0.,  0.],
               [ 0.,  0.,  0.],
               [ 0.,  0.,  0.]], dtype=float32)
        >>> # assign CSRNDArray with same storage type
        >>> x = mx.nd.ones('row_sparse', (3,3)).tostype('csr')
        >>> x[:] = src
        >>> x.asnumpy()
        array([[ 1.,  1.,  1.],
               [ 1.,  1.,  1.],
               [ 1.,  1.,  1.]], dtype=float32)
        >>> # assign NDArray to CSRNDArray
        >>> x[:] = mx.nd.ones((3,3)) * 2
        >>> x.asnumpy()
        array([[ 2.,  2.,  2.],
               [ 2.,  2.,  2.],
               [ 2.,  2.,  2.]], dtype=float32)
        """
        if not self.writable:
            raise ValueError('Failed to assign to a readonly CSRNDArray')
        if isinstance(key, py_slice):
            if key.step is not None or key.start is not None or key.stop is not None:
                raise ValueError('Assignment with slice for CSRNDArray is not ' \
                                 'implmented yet.')
            if isinstance(value, NDArray):
                # avoid copying to itself
                if value.handle is not self.handle:
                    value.copyto(self)
            elif isinstance(value, numeric_types):
                raise ValueError("Assigning numeric types to CSRNDArray is " \
                                 "not implemented yet.")
            elif isinstance(value, (np.ndarray, np.generic)):
                # TODO(haibin/anisub) check scipy.sparse and use _sync_copy_from to
                # avoid the temporary copy
                warnings.warn('Assigning non-NDArray object to CSRNDArray is not efficient',
                              RuntimeWarning)
                tmp = _array(value)
                tmp.copyto(self)
            else:
                raise TypeError('type %s not supported' % str(type(value)))
        else:
            assert(isinstance(key, (int, tuple)))
            raise Exception('CSRNDArray only supports [:] for assignment')

    @property
    def indices(self):
        """A deep copy NDArray of the indices array of the CSRNDArray.
        This generates a deep copy of the column indices of the current `csr` matrix.

        Returns
        -------
        NDArray
            This CSRNDArray's indices array.
        """
        return self._aux_data(1)

    @property
    def indptr(self):
        """A deep copy NDArray of the indptr array of the CSRNDArray.
        This generates a deep copy of the `indptr` of the current `csr` matrix.

        Returns
        -------
        NDArray
            This CSRNDArray's indptr array.
        """
        return self._aux_data(0)

    @property
    def data(self):
        """A deep copy NDArray of the data array of the CSRNDArray.
        This generates a deep copy of the `data` of the current `csr` matrix.

        Returns
        -------
        NDArray
            This CSRNDArray's data array.
        """
        return self._data()

    @indices.setter
    def indices(self, indices):
        raise NotImplementedError()

    @indptr.setter
    def indptr(self, indptr):
        raise NotImplementedError()

    @data.setter
    def data(self, data):
        raise NotImplementedError()


    def tostype(self, stype):
        """Return a copy of the array with chosen storage type.

        Returns
        -------
        NDArray or CSRNDArray
            A copy of the array with the chosen storage stype
        """
        if stype == 'row_sparse':
            raise ValueError("cast_storage from csr to row_sparse is not supported")
        return op.cast_storage(self, stype=stype)

    def copyto(self, other):
        """Copies the value of this array to another array.

        If ``other`` is a ``NDArray`` or ``CSRNDArray`` object, then ``other.shape`` and
        ``self.shape`` should be the same. This function copies the value from
        ``self`` to ``other``.

        If ``other`` is a context, a new ``CSRNDArray`` will be first created on
        the target context, and the value of ``self`` is copied.

        Parameters
        ----------
        other : NDArray or CSRNDArray or Context
            The destination array or context.

        Returns
        -------
        NDArray or CSRNDArray
            The copied array. If ``other`` is an ``NDArray`` or ``CSRNDArray``, then the return
            value and ``other`` will point to the same ``NDArray`` or ``CSRNDArray``.
        """
        if isinstance(other, Context):
            return super(CSRNDArray, self).copyto(other)
        elif isinstance(other, NDArray):
            stype = other.stype
            if stype == 'default' or stype == 'csr':
                return super(CSRNDArray, self).copyto(other)
            else:
                raise TypeError('copyto does not support destination NDArray stype ' + str(stype))
        else:
            raise TypeError('copyto does not support type ' + str(type(other)))

    def asscipy(self):
        """Returns a ``scipy.sparse.csr.csr_matrix`` object with value copied from this array

        Examples
        --------
        >>> x = mx.nd.sparse.zeros('csr', (2,3))
        >>> y = x.asscipy()
        >>> type(y)
        <type 'scipy.sparse.csr.csr_matrix'>
        >>> y
        <2x3 sparse matrix of type '<type 'numpy.float32'>'
        with 0 stored elements in Compressed Sparse Row format>
        """
        data = self.data.asnumpy()
        indices = self.indices.asnumpy()
        indptr = self.indptr.asnumpy()
        if not spsp:
            raise ImportError("scipy is not available. \
                               Please check if the scipy python bindings are installed.")
        return spsp.csr_matrix((data, indices, indptr), shape=self.shape, dtype=self.dtype)

# pylint: disable=abstract-method
class RowSparseNDArray(BaseSparseNDArray):
    """A sparse representation of a set of NDArray row slices at given indices.

    A RowSparseNDArray represents a multidimensional NDArray using two separate arrays: `data` and
    `indices`. The number of dimensions has to be at least 2.

    - data: an NDArray of any dtype with shape [D0, D1, ..., Dn].
    - indices: a 1-D int64 NDArray with shape [D0] with values sorted in ascending order.

    The `indices` stores the indices of the row slices with non-zeros,
    while the values are stored in `data`. The corresponding NDArray ``dense``
    represented by RowSparseNDArray ``rsp`` has

    ``dense[rsp.indices[i], :, :, :, ...] = rsp.data[i, :, :, :, ...]``

        >>> dense.asnumpy()
        array([[ 1.,  2., 3.],
               [ 0.,  0., 0.],
               [ 4.,  0., 5.],
               [ 0.,  0., 0.],
               [ 0.,  0., 0.]], dtype=float32)
        >>> rsp = dense.tostype('row_sparse')
        >>> rsp.indices.asnumpy()
        array([0, 2], dtype=int64)
        >>> rsp.data.asnumpy()
        array([[ 1.,  2., 3.],
               [ 4.,  0., 5.]], dtype=float32)

    A RowSparseNDArray is typically used to represent non-zero row slices of a large NDArray
    of shape [LARGE0, D1, .. , Dn] where LARGE0 >> D0 and most row slices are zeros.

    RowSparseNDArray is used principally in the definition of gradients for operations
    that have sparse gradients (e.g. sparse dot and sparse embedding).

    See Also
    --------
    row_sparse_array: Several ways to construct a RowSparseNDArray
    """
    def __reduce__(self):
        return RowSparseNDArray, (None,), super(RowSparseNDArray, self).__getstate__()

    def __iadd__(self, other):
        (self + other).copyto(self)
        return self

    def __isub__(self, other):
        (self - other).copyto(self)
        return self

    def __imul__(self, other):
        (self * other).copyto(self)
        return self

    def __idiv__(self, other):
        (self / other).copyto(self)
        return self

    def __itruediv__(self, other):
        (self / other).copyto(self)
        return self

    def __getitem__(self, key):
        """x.__getitem__(i) <=> x[i]

        Returns a sliced view of this array.

        Parameters
        ----------
        key : slice
            Indexing key.

        Examples
        --------
        >>> x = mx.nd.sparse.zeros('row_sparse', (2, 3))
        >>> x[:].asnumpy()
        array([[ 0.,  0.,  0.],
               [ 0.,  0.,  0.]], dtype=float32)
        """
        if isinstance(key, int):
            raise Exception("__getitem__ with int key is not implemented for RowSparseNDArray yet")
        if isinstance(key, py_slice):
            if key.step is not None or key.start is not None or key.stop is not None:
                raise Exception('RowSparseNDArray only supports [:] for __getitem__')
            else:
                return self
        if isinstance(key, tuple):
            raise ValueError('Multi-dimension indexing is not supported')

    def __setitem__(self, key, value):
        """x.__setitem__(i, y) <=> x[i]=y

        Set self[key] to value. Only slice key [:] is supported.

        Parameters
        ----------
        key : slice
            The indexing key.
        value : NDArray or numpy.ndarray
            The value to set.

        Examples
        --------
        >>> src = mx.nd.row_sparse([[1, 0, 2], [4, 5, 6]], [0, 2], (3,3))
        >>> src.asnumpy()
        array([[ 1.,  0.,  2.],
               [ 0.,  0.,  0.],
               [ 4.,  5.,  6.]], dtype=float32)
        >>> # assign RowSparseNDArray with same storage type
        >>> x = mx.nd.sparse.zeros('row_sparse', (3,3))
        >>> x[:] = src
        >>> x.asnumpy()
        array([[ 1.,  0.,  2.],
               [ 0.,  0.,  0.],
               [ 4.,  5.,  6.]], dtype=float32)
        >>> # assign NDArray to RowSparseNDArray
        >>> x[:] = mx.nd.ones((3,3))
        >>> x.asnumpy()
        array([[ 1.,  1.,  1.],
               [ 1.,  1.,  1.],
               [ 1.,  1.,  1.]], dtype=float32)
        """
        if not self.writable:
            raise ValueError('Failed to assign to a readonly RowSparseNDArray')
        if isinstance(key, py_slice):
            if key.step is not None or key.start is not None or key.stop is not None:
                raise ValueError('Assignment with slice for RowSparseNDArray ' \
                                 'is not implmented yet.')
            if isinstance(value, NDArray):
                # avoid copying to itself
                if value.handle is not self.handle:
                    value.copyto(self)
            elif isinstance(value, numeric_types):
                raise ValueError("Assigning numeric types to RowSparseNDArray " \
                                 "is not implemented yet.")
            elif isinstance(value, (np.ndarray, np.generic)):
                warnings.warn('Assigning non-NDArray object to RowSparseNDArray is not efficient',
                              RuntimeWarning)
                tmp = _array(value)
                tmp.copyto(self)
            else:
                raise TypeError('type %s not supported' % str(type(value)))
        else:
            assert(isinstance(key, (int, tuple)))
            raise TypeError('RowSparseNDArray only supports [:] for assignment')

    @property
    def indices(self):
        """A deep copy NDArray of the indices array of the RowSparseNDArray.
        This generates a deep copy of the row indices of the current `row_sparse` matrix.

        Returns
        -------
        NDArray
            This RowSparseNDArray's indices array.
        """
        return self._aux_data(0)

    @property
    def data(self):
        """A deep copy NDArray of the data array of the RowSparseNDArray.
        This generates a deep copy of the `data` of the current `row_sparse` matrix.

        Returns
        -------
        NDArray
            This RowSparseNDArray's data array.
        """
        return self._data()

    @indices.setter
    def indices(self, indices):
        raise NotImplementedError()

    @data.setter
    def data(self, data):
        raise NotImplementedError()

    def tostype(self, stype):
        """Return a copy of the array with chosen storage type.

        Returns
        -------
        NDArray or RowSparseNDArray
            A copy of the array with the chosen storage stype
        """
        if stype == 'csr':
            raise ValueError("cast_storage from row_sparse to csr is not supported")
        return op.cast_storage(self, stype=stype)

    def copyto(self, other):
        """Copies the value of this array to another array.

        If ``other`` is a ``NDArray`` or ``RowSparseNDArray`` object, then ``other.shape``
        and ``self.shape`` should be the same. This function copies the value from
        ``self`` to ``other``.

        If ``other`` is a context, a new ``RowSparseNDArray`` will be first created on
        the target context, and the value of ``self`` is copied.

        Parameters
        ----------
        other : NDArray or RowSparseNDArray or Context
            The destination array or context.

        Returns
        -------
        NDArray or RowSparseNDArray
            The copied array. If ``other`` is an ``NDArray`` or ``RowSparseNDArray``, then the
            return value and ``other`` will point to the same ``NDArray`` or ``RowSparseNDArray``.
        """
        if isinstance(other, Context):
            return super(RowSparseNDArray, self).copyto(other)
        elif isinstance(other, NDArray):
            stype = other.stype
            if stype == 'default' or stype == 'row_sparse':
                return super(RowSparseNDArray, self).copyto(other)
            else:
                raise TypeError('copyto does not support destination NDArray stype ' + str(stype))
        else:
            raise TypeError('copyto does not support type ' + str(type(other)))

    def retain(self, *args, **kwargs):
        """Convenience fluent method for :py:func:`retain`.

        The arguments are the same as for :py:func:`retain`, with
        this array as data.
        """
        return retain(self, *args, **kwargs)

def _prepare_src_array(source_array, dtype):
    """Prepare `source_array` so that it can be used to construct NDArray.
    `source_array` is converted to a `np.ndarray` if it's neither an `NDArray` \
    nor an `np.ndarray`.
    """
    if not isinstance(source_array, NDArray) and not isinstance(source_array, np.ndarray):
        try:
            source_array = np.array(source_array, dtype=dtype)
        except:
            raise TypeError('values must be array like object')
    return source_array

def _prepare_default_dtype(src_array, dtype):
    """Prepare the value of dtype if `dtype` is None. If `src_array` is an NDArray, numpy.ndarray
    or scipy.sparse.csr.csr_matrix, return src_array.dtype. float32 is returned otherwise."""
    if dtype is None:
        if isinstance(src_array, (NDArray, np.ndarray)):
            dtype = src_array.dtype
        elif spsp and isinstance(src_array, spsp.csr.csr_matrix):
            dtype = src_array.dtype
        else:
            dtype = mx_real_t
    return dtype

def _check_shape(s1, s2):
    """check s1 == s2 if both are not None"""
    if s1 and s2 and s1 != s2:
        raise ValueError("Shape mismatch detected. " + str(s1) + " v.s. " + str(s2))

def csr_matrix(arg1, shape=None, ctx=None, dtype=None):
    """Creates a `CSRNDArray`, an 2D array with compressed sparse row (CSR) format.

    The CSRNDArray can be instantiated in several ways:

    - csr_matrix(D):
        to construct a CSRNDArray with a dense 2D array ``D``
            -  **D** (*array_like*) - An object exposing the array interface, an object whose \
            `__array__` method returns an array, or any (nested) sequence.
            - **ctx** (*Context, optional*) - Device context \
            (default is the current default context).
            - **dtype** (*str or numpy.dtype, optional*) - The data type of the output array. \
            The default dtype is ``D.dtype`` if ``D`` is an NDArray or numpy.ndarray, \
            float32 otherwise.

    - csr_matrix(S)
        to construct a CSRNDArray with a sparse 2D array ``S``
            -  **S** (*CSRNDArray or scipy.sparse.csr.csr_matrix*) - A sparse matrix.
            - **ctx** (*Context, optional*) - Device context \
            (default is the current default context).
            - **dtype** (*str or numpy.dtype, optional*) - The data type of the output array. \
            The default dtype is ``S.dtype``.

    - csr_matrix((M, N))
        to construct an empty CSRNDArray with shape ``(M, N)``
            -  **M** (*int*) - Number of rows in the matrix
            -  **N** (*int*) - Number of columns in the matrix
            - **ctx** (*Context, optional*) - Device context \
            (default is the current default context).
            - **dtype** (*str or numpy.dtype, optional*) - The data type of the output array. \
            The default dtype is float32.

    - csr_matrix((data, indices, indptr))
        to construct a CSRNDArray based on the definition of compressed sparse row format \
        using three separate arrays, \
        where the column indices for row i are stored in ``indices[indptr[i]:indptr[i+1]]`` \
        and their corresponding values are stored in ``data[indptr[i]:indptr[i+1]]``. \
        The column indices for a given row are expected to be **sorted in ascending order.** \
        Duplicate column entries for the same row are not allowed.
            - **data** (*array_like*) - An object exposing the array interface, which \
            holds all the non-zero entries of the matrix in row-major order.
            - **indices** (*array_like*) - An object exposing the array interface, which \
            stores the column index for each non-zero element in ``data``.
            - **indptr** (*array_like*) - An object exposing the array interface, which \
            stores the offset into ``data`` of the first non-zero element number of each \
            row of the matrix.
            - **shape** (*tuple of int, optional*) - The shape of the array. The default \
            shape is inferred from the indices and indptr arrays.
            - **ctx** (*Context, optional*) - Device context \
            (default is the current default context).
            - **dtype** (*str or numpy.dtype, optional*) - The data type of the output array. \
            The default dtype is ``data.dtype`` if ``data`` is an NDArray or numpy.ndarray, \
            float32 otherwise.

    - csr_matrix((data, (row, col)))
        to construct a CSRNDArray based on the COOrdinate format \
        using three seperate arrays, \
        where ``row[i]`` is the row index of the element, \
        ``col[i]`` is the column index of the element \
        and ``data[i]`` is the data corresponding to the element. All the missing \
        elements in the input are taken to be zeroes.
            - **data** (*array_like*) - An object exposing the array interface, which \
            holds all the non-zero entries of the matrix in COO format.
            - **row** (*array_like*) - An object exposing the array interface, which \
            stores the row index for each non zero element in ``data``.
            - **col** (*array_like*) - An object exposing the array interface, which \
            stores the col index for each non zero element in ``data``.
            - **shape** (*tuple of int, optional*) - The shape of the array. The default \
            shape is inferred from the ``row`` and ``col`` arrays.
            - **ctx** (*Context, optional*) - Device context \
            (default is the current default context).
            - **dtype** (*str or numpy.dtype, optional*) - The data type of the output array. \
            The default dtype is float32.

    Parameters
    ----------
    arg1: tuple of int, tuple of array_like, array_like, CSRNDArray, scipy.sparse.csr_matrix, \
    scipy.sparse.coo_matrix, tuple of int or tuple of array_like
        The argument to help instantiate the csr matrix. See above for further details.
    shape : tuple of int, optional
        The shape of the csr matrix.
    ctx: Context, optional
        Device context (default is the current default context).
    dtype: str or numpy.dtype, optional
        The data type of the output array.

    Returns
    -------
    CSRNDArray
        A `CSRNDArray` with the `csr` storage representation.

    Example
    -------
    >>> a = mx.nd.sparse.csr_matrix(([1, 2, 3], [1, 0, 2], [0, 1, 2, 2, 3]), shape=(4, 3))
    >>> a.asnumpy()
    array([[ 0.,  1.,  0.],
           [ 2.,  0.,  0.],
           [ 0.,  0.,  0.],
           [ 0.,  0.,  3.]], dtype=float32)

    See Also
    --------
    CSRNDArray : MXNet NDArray in compressed sparse row format.
    """
    # construct a csr matrix from (M, N) or (data, indices, indptr)
    if isinstance(arg1, tuple):
        arg_len = len(arg1)
        if arg_len == 2:
            # construct a sparse csr matrix from
            # scipy coo matrix if input format is coo
            if isinstance(arg1[1], tuple) and len(arg1[1]) == 2:
                data, (row, col) = arg1
                if isinstance(data, NDArray):
                    data = data.asnumpy()
                if isinstance(row, NDArray):
                    row = row.asnumpy()
                if isinstance(col, NDArray):
                    col = col.asnumpy()
                coo = spsp.coo_matrix((data, (row, col)), shape=shape)
                _check_shape(coo.shape, shape)
                csr = coo.tocsr()
                return array(csr, ctx=ctx, dtype=dtype)
            else:
                # empty matrix with shape
                _check_shape(arg1, shape)
                return empty('csr', arg1, ctx=ctx, dtype=dtype)
        elif arg_len == 3:
            # data, indices, indptr
            return _csr_matrix_from_definition(arg1[0], arg1[1], arg1[2], shape=shape,
                                               ctx=ctx, dtype=dtype)
        else:
            raise ValueError("Unexpected length of input tuple: " + str(arg_len))
    else:
        # construct a csr matrix from a sparse / dense one
        if isinstance(arg1, CSRNDArray) or (spsp and isinstance(arg1, spsp.csr.csr_matrix)):
            # construct a csr matrix from scipy or CSRNDArray
            _check_shape(arg1.shape, shape)
            return array(arg1, ctx=ctx, dtype=dtype)
        elif isinstance(arg1, RowSparseNDArray):
            raise ValueError("Unexpected input type: RowSparseNDArray")
        else:
            # construct a csr matrix from a dense one
            # prepare default ctx and dtype since mx.nd.array doesn't use default values
            # based on source_array
            dtype = _prepare_default_dtype(arg1, dtype)
            # create dns array with provided dtype. ctx is not passed since copy across
            # ctx requires dtype to be the same
            dns = _array(arg1, dtype=dtype)
            if ctx is not None and dns.context != ctx:
                dns = dns.as_in_context(ctx)
            _check_shape(dns.shape, shape)
            return dns.tostype('csr')

def _csr_matrix_from_definition(data, indices, indptr, shape=None, ctx=None,
                                dtype=None, indices_type=None, indptr_type=None):
    """Create a `CSRNDArray` based on data, indices and indptr"""
    storage_type = 'csr'
    # context
    ctx = Context.default_ctx if ctx is None else ctx
    # types
    dtype = _prepare_default_dtype(data, dtype)
    indptr_type = _STORAGE_AUX_TYPES[storage_type][0] if indptr_type is None else indptr_type
    indices_type = _STORAGE_AUX_TYPES[storage_type][1] if indices_type is None else indices_type
    # prepare src array and types
    data = _prepare_src_array(data, dtype)
    indptr = _prepare_src_array(indptr, indptr_type)
    indices = _prepare_src_array(indices, indices_type)

    # TODO(junwu): Convert data, indptr, and indices to mxnet NDArrays
    # if they are not for now. In the future, we should provide a c-api
    # to accept np.ndarray types to copy from to result.data and aux_data
    if not isinstance(data, NDArray):
        data = _array(data, ctx, dtype)
    if not isinstance(indptr, NDArray):
        indptr = _array(indptr, ctx, indptr_type)
    if not isinstance(indices, NDArray):
        indices = _array(indices, ctx, indices_type)
    if shape is None:
        if indices.shape[0] == 0:
            raise ValueError('invalid shape')
        shape = (len(indptr) - 1, op.max(indices).asscalar() + 1)
    # verify shapes
    aux_shapes = [indptr.shape, indices.shape]
    if data.ndim != 1 or indptr.ndim != 1 or indices.ndim != 1 or \
        indptr.shape[0] == 0 or len(shape) != 2:
        raise ValueError('invalid shape')
    result = CSRNDArray(_new_alloc_handle(storage_type, shape, ctx, False, dtype,
                                          [indptr_type, indices_type], aux_shapes))
    check_call(_LIB.MXNDArraySyncCopyFromNDArray(result.handle, data.handle, ctypes.c_int(-1)))
    check_call(_LIB.MXNDArraySyncCopyFromNDArray(result.handle, indptr.handle, ctypes.c_int(0)))
    check_call(_LIB.MXNDArraySyncCopyFromNDArray(result.handle, indices.handle, ctypes.c_int(1)))
    return result

def row_sparse_array(arg1, shape=None, ctx=None, dtype=None):
    """Creates a `RowSparseNDArray`, a multidimensional row sparse array with a set of \
    tensor slices at given indices.

    The RowSparseNDArray can be instantiated in several ways:

    - row_sparse_array(D):
        to construct a RowSparseNDArray with a dense ndarray ``D``
            -  **D** (*array_like*) - An object exposing the array interface, an object whose \
            `__array__` method returns an array, or any (nested) sequence.
            - **ctx** (*Context, optional*) - Device context \
            (default is the current default context).
            - **dtype** (*str or numpy.dtype, optional*) - The data type of the output array. \
            The default dtype is ``D.dtype`` if ``D`` is an NDArray or numpy.ndarray, \
            float32 otherwise.

    - row_sparse_array(S)
        to construct a RowSparseNDArray with a sparse ndarray ``S``
            -  **S** (*RowSparseNDArray*) - A sparse ndarray.
            - **ctx** (*Context, optional*) - Device context \
            (default is the current default context).
            - **dtype** (*str or numpy.dtype, optional*) - The data type of the output array. \
            The default dtype is ``S.dtype``.

    - row_sparse_array((D0, D1 .. Dn))
        to construct an empty RowSparseNDArray with shape ``(D0, D1, ... Dn)``
            -  **D0, D1 .. Dn** (*int*) - The shape of the ndarray
            - **ctx** (*Context, optional*) - Device context \
            (default is the current default context).
            - **dtype** (*str or numpy.dtype, optional*) - The data type of the output array. \
            The default dtype is float32.

    - row_sparse_array((data, indices))
        to construct a RowSparseNDArray based on the definition of row sparse format \
        using two separate arrays, \
        where the `indices` stores the indices of the row slices with non-zeros,
        while the values are stored in `data`. The corresponding NDArray ``dense``
        represented by RowSparseNDArray ``rsp`` has \
        ``dense[rsp.indices[i], :, :, :, ...] = rsp.data[i, :, :, :, ...]``
        The row indices for are expected to be **sorted in ascending order.** \
            - **data** (*array_like*) - An object exposing the array interface, which \
            holds all the non-zero row slices of the array.
            - **indices** (*array_like*) - An object exposing the array interface, which \
            stores the row index for each row slice with non-zero elements.
            - **shape** (*tuple of int, optional*) - The shape of the array. The default \
            shape is inferred from the indices and indptr arrays.
            - **ctx** (*Context, optional*) - Device context \
            (default is the current default context).
            - **dtype** (*str or numpy.dtype, optional*) - The data type of the output array. \
            The default dtype is float32.

    Parameters
    ----------
    arg1: NDArray, numpy.ndarray, RowSparseNDArray, tuple of int or tuple of array_like
        The argument to help instantiate the row sparse ndarray. See above for further details.
    shape : tuple of int, optional
        The shape of the row sparse ndarray.
    ctx : Context, optional
        Device context (default is the current default context).
    dtype : str or numpy.dtype, optional
        The data type of the output array.

    Returns
    -------
    RowSparseNDArray
        An `RowSparseNDArray` with the `row_sparse` storage representation.

    Example
    -------
    >>> a = mx.nd.sparse.row_sparse_array(([[1, 2], [3, 4]], [1, 4]), shape=(6, 2))
    >>> a.asnumpy()
    array([[ 0.,  0.],
           [ 1.,  2.],
           [ 0.,  0.],
           [ 0.,  0.],
           [ 3.,  4.],
           [ 0.,  0.]], dtype=float32)

    See Also
    --------
    RowSparseNDArray : MXNet NDArray in row sparse format.
    """
    # construct a row sparse array from (D0, D1 ..) or (data, indices)
    if isinstance(arg1, tuple):
        arg_len = len(arg1)
        if arg_len < 2:
            raise ValueError("Unexpected length of input tuple: " + str(arg_len))
        elif arg_len > 2:
            # empty ndarray with shape
            _check_shape(arg1, shape)
            return empty('row_sparse', arg1, ctx=ctx, dtype=dtype)
        else:
            # len(arg1) = 2, is either shape or (data, indices)
            if isinstance(arg1[0], integer_types) and isinstance(arg1[1], integer_types):
                # empty ndarray with shape
                _check_shape(arg1, shape)
                return empty('row_sparse', arg1, ctx=ctx, dtype=dtype)
            else:
                # data, indices, indptr
                return _row_sparse_ndarray_from_definition(arg1[0], arg1[1], shape=shape,
                                                           ctx=ctx, dtype=dtype)
    else:
        # construct a row sparse ndarray from a dense / sparse array
        if isinstance(arg1, RowSparseNDArray):
            # construct a row sparse ndarray from RowSparseNDArray
            _check_shape(arg1.shape, shape)
            return array(arg1, ctx=ctx, dtype=dtype)
        elif isinstance(arg1, CSRNDArray):
            raise ValueError("Unexpected input type: CSRNDArray")
        else:
            # construct a csr matrix from a dense one
            # prepare default dtype since mx.nd.array doesn't use default values
            # based on source_array
            dtype = _prepare_default_dtype(arg1, dtype)
            # create dns array with provided dtype. ctx is not passed since copy across
            # ctx requires dtype to be the same
            dns = _array(arg1, dtype=dtype)
            if ctx is not None and dns.context != ctx:
                dns = dns.as_in_context(ctx)
            _check_shape(dns.shape, shape)
            return dns.tostype('row_sparse')

def _row_sparse_ndarray_from_definition(data, indices, shape=None, ctx=None,
                                        dtype=None, indices_type=None):
    """Create a `RowSparseNDArray` based on data and indices"""
    storage_type = 'row_sparse'
    # context
    ctx = Context.default_ctx if ctx is None else ctx
    # types
    dtype = _prepare_default_dtype(data, dtype)
    indices_type = _STORAGE_AUX_TYPES[storage_type][0] if indices_type is None else indices_type
    # prepare src array and types
    data = _prepare_src_array(data, dtype)
    indices = _prepare_src_array(indices, indices_type)

    # TODO(junwu): Convert data, indptr, and indices to mxnet NDArrays
    # if they are not for now. In the future, we should provide a c-api
    # to accept np.ndarray types to copy from to result.data and aux_data
    if not isinstance(data, NDArray):
        data = _array(data, ctx, dtype)
    if not isinstance(indices, NDArray):
        indices = _array(indices, ctx, indices_type)
    if shape is None:
        num_indices = indices.shape[0]
        if num_indices == 0:
            raise ValueError('invalid shape')
        dim0 = indices[num_indices - 1].asscalar() + 1
        shape = (dim0, ) + data.shape[1:]
    # verify shapes
    if data.ndim != len(shape) or indices.ndim != 1 or np.prod(shape[1:]) == 0:
        raise ValueError("invalid shape")
    result = RowSparseNDArray(_new_alloc_handle(storage_type, shape, ctx, False, dtype,
                                                [indices_type], [indices.shape]))
    check_call(_LIB.MXNDArraySyncCopyFromNDArray(result.handle, data.handle, ctypes.c_int(-1)))
    check_call(_LIB.MXNDArraySyncCopyFromNDArray(result.handle, indices.handle, ctypes.c_int(0)))
    return result

def _ndarray_cls(handle, writable=True, stype=_STORAGE_TYPE_UNDEFINED):
    if stype == _STORAGE_TYPE_UNDEFINED:
        stype = _storage_type(handle)
    if stype == _STORAGE_TYPE_DEFAULT:
        return NDArray(handle, writable=writable)
    elif stype == _STORAGE_TYPE_CSR:
        return CSRNDArray(handle, writable=writable)
    elif stype == _STORAGE_TYPE_ROW_SPARSE:
        return RowSparseNDArray(handle, writable=writable)
    else:
        raise Exception("unknown storage type: %s"%stype)


_set_ndarray_class(_ndarray_cls)


def zeros(stype, shape, ctx=None, dtype=None, **kwargs):
    """Return a new array of given shape and type, filled with zeros.

    Parameters
    ----------
    stype: string
        The storage type of the empty array, such as 'row_sparse', 'csr', etc
    shape : int or tuple of int
        The shape of the empty array
    ctx : Context, optional
        An optional device context (default is the current default context)
    dtype : str or numpy.dtype, optional
        An optional value type (default is `float32`)

    Returns
    -------
    RowSparseNDArray or CSRNDArray
        A created array
    Examples
    --------
    >>> mx.nd.sparse.zeros('csr', (1,2))
    <CSRNDArray 1x2 @cpu(0)>
    >>> mx.nd.sparse.zeros('row_sparse', (1,2), ctx=mx.cpu(), dtype='float16').asnumpy()
    array([[ 0.,  0.]], dtype=float16)
    """
    if stype == 'default':
        return _zeros_ndarray(shape, ctx=ctx, dtype=dtype, **kwargs)
    if ctx is None:
        ctx = Context.default_ctx
    dtype = mx_real_t if dtype is None else dtype
    if stype == 'row_sparse' or stype == 'csr':
        aux_types = _STORAGE_AUX_TYPES[stype]
    else:
        raise ValueError("unknown storage type" + stype)
    out = _ndarray_cls(_new_alloc_handle(stype, shape, ctx, True, dtype, aux_types))
    return _internal._zeros(shape=shape, ctx=ctx, dtype=dtype, out=out, **kwargs)


def empty(stype, shape, ctx=None, dtype=None):
    """Returns a new array of given shape and type, without initializing entries.

    Parameters
    ----------
    stype: string
        The storage type of the empty array, such as 'row_sparse', 'csr', etc
    shape : int or tuple of int
        The shape of the empty array.
    ctx : Context, optional
        An optional device context (default is the current default context).
    dtype : str or numpy.dtype, optional
        An optional value type (default is `float32`).

    Returns
    -------
    CSRNDArray or RowSparseNDArray
        A created array.
    """
    if isinstance(shape, int):
        shape = (shape, )
    if ctx is None:
        ctx = Context.default_ctx
    if dtype is None:
        dtype = mx_real_t
    assert(stype is not None)
    if stype == 'csr' or stype == 'row_sparse':
        return zeros(stype, shape, ctx=ctx, dtype=dtype)
    else:
        raise Exception("unknown stype : " + str(stype))


def array(source_array, ctx=None, dtype=None):
    """Creates a sparse array from any object exposing the array interface.

    Parameters
    ----------
    source_array : RowSparseNDArray, CSRNDArray or scipy.sparse.csr.csr_matrix
        The source sparse array
    ctx : Context, optional
        The default context is ``source_array.context`` if ``source_array`` is an NDArray. \
        The current default context otherwise.
    dtype : str or numpy.dtype, optional
        The data type of the output array. The default dtype is ``source_array.dtype``
        if `source_array` is an `NDArray`, `numpy.ndarray` or `scipy.sparse.csr.csr_matrix`, \
        `float32` otherwise.

    Returns
    -------
    RowSparseNDArray or CSRNDArray
        An array with the same contents as the `source_array`.

    Examples
    --------
    >>> import scipy.sparse as spsp
    >>> csr = spsp.csr_matrix((2, 100))
    >>> mx.nd.sparse.array(csr)
    <CSRNDArray 2x100 @cpu(0)>
    >>> mx.nd.sparse.array(mx.nd.sparse.zeros('csr', (3, 2)))
    <CSRNDArray 3x2 @cpu(0)>
    >>> mx.nd.sparse.array(mx.nd.sparse.zeros('row_sparse', (3, 2)))
    <RowSparseNDArray 3x2 @cpu(0)>
    """
    ctx = Context.default_ctx if ctx is None else ctx
    if isinstance(source_array, NDArray):
        assert(source_array.stype != 'default'), \
               "Please use `tostype` to create RowSparseNDArray or CSRNDArray from an NDArray"
        # prepare dtype and ctx based on source_array, if not provided
        dtype = _prepare_default_dtype(source_array, dtype)
        # if both dtype and ctx are different from source_array, we cannot copy directly
        if source_array.dtype != dtype and source_array.context != ctx:
            arr = empty(source_array.stype, source_array.shape, dtype=dtype)
            arr[:] = source_array
            arr = arr.as_in_context(ctx)
        else:
            arr = empty(source_array.stype, source_array.shape, dtype=dtype, ctx=ctx)
            arr[:] = source_array
        return arr
    elif spsp and isinstance(source_array, spsp.csr.csr_matrix):
        # TODO(haibin) implement `_sync_copy_from` with scipy csr object to reduce a copy
        # preprocess scipy csr to canonical form
        csr = source_array.sorted_indices()
        csr.sum_duplicates()
        dtype = _prepare_default_dtype(source_array, dtype)
        return csr_matrix((csr.data, csr.indices, csr.indptr), shape=csr.shape, \
                          dtype=dtype, ctx=ctx)
    elif isinstance(source_array, (np.ndarray, np.generic)):
        raise ValueError("Please use mx.nd.array to create an NDArray with source_array of type ",
                         type(source_array))
    else:
        raise ValueError("Unexpected source_array type: ", type(source_array))
