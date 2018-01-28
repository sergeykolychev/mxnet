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

use lib '../lib';
use strict;
use warnings;
use Scalar::Util qw(blessed);
use Test::More 'no_plan';
use AI::MXNet qw(mx);
use AI::MXNet::TestUtils qw(zip assert enumerate same rand_shape_2d rand_shape_3d
    rand_sparse_ndarray random_arrays almost_equal rand_ndarray randint allclose);
use AI::MXNet::Base qw(pones pzeros pdl product);
$ENV{MXNET_STORAGE_FALLBACK_LOG_VERBOSE} = 0;
use Data::Dumper;
sub sparse_nd_ones
{
    my ($shape, $stype) = @_;
    return mx->nd->ones($shape)->tostype($stype);
}

sub test_sparse_nd_elemwise_add
{
    my $check_sparse_nd_elemwise_binary = sub {
        my ($shapes, $stypes, $f, $g) = @_;
        # generate inputs
        my @nds;
        enumerate(sub {
            my ($i, $stype) = @_;
            my $nd;
            if($stype eq 'row_sparse')
            {
                ($nd) = rand_sparse_ndarray($shapes->[$i], $stype);
            }
            elsif($stype eq 'default')
            {
                $nd = mx->nd->array(random_arrays($shapes->[$i]), dtype => 'float32');
            }
            else
            {
                die;
            }
            push @nds, $nd;
        }, $stypes);
        # check result
        my $test = $f->($nds[0], $nds[1]);
        ok(almost_equal($test->aspdl, $g->($nds[0]->aspdl, $nds[1]->aspdl)));
    };
    my $num_repeats = 2;
    my $g = sub { $_[0] + $_[1] };
    my $op = sub { mx->nd->elemwise_add(@_) };
    for my $i (0..$num_repeats)
    {
        my $shape = rand_shape_2d();
        $shape = [$shape, $shape];
        $check_sparse_nd_elemwise_binary->($shape, ['default', 'default'], $op, $g);
        $check_sparse_nd_elemwise_binary->($shape, ['row_sparse', 'row_sparse'], $op, $g);
    }
}

test_sparse_nd_elemwise_add();

sub test_sparse_nd_copy
{
    my $check_sparse_nd_copy = sub { my ($from_stype, $to_stype, $shape) = @_;
        my $from_nd = rand_ndarray($shape, $from_stype);
        # copy to ctx
        my $to_ctx = $from_nd->copyto(AI::MXNet::Context->current_ctx);
        # copy to stype
        my $to_nd = rand_ndarray($shape, $to_stype);
        $from_nd->copyto($to_nd);
        ok(($from_nd->aspdl != $to_ctx->aspdl)->abs->sum == 0);
        ok(($from_nd->aspdl != $to_nd->aspdl)->abs->sum == 0);
    };
    my $shape = rand_shape_2d();
    my $shape_3d = rand_shape_3d();
    my @stypes = ('row_sparse', 'csr');
    for my $stype (@stypes)
    {
        $check_sparse_nd_copy->($stype, 'default', $shape);
        $check_sparse_nd_copy->('default', $stype, $shape);
    }
    $check_sparse_nd_copy->('row_sparse', 'row_sparse', $shape_3d);
    $check_sparse_nd_copy->('row_sparse', 'default', $shape_3d);
    $check_sparse_nd_copy->('default', 'row_sparse', $shape_3d);
}

test_sparse_nd_copy();

sub test_sparse_nd_basic
{
    my $check_sparse_nd_basic_rsp = sub {
        my $storage_type = 'row_sparse';
        my $shape = rand_shape_2d();
        my ($nd) = rand_sparse_ndarray($shape, $storage_type);
        ok($nd->_num_aux == 1);
        ok($nd->indices->dtype eq 'int64');
        ok($nd->stype eq 'row_sparse');
    };
    $check_sparse_nd_basic_rsp->();
}

test_sparse_nd_basic();

sub test_sparse_nd_setitem
{
    my $check_sparse_nd_setitem = sub { my ($stype, $shape, $dst) = @_;
        my $x = mx->nd->zeros($shape, stype=>$stype);
        $x .= $dst;
        my $dst_nd = (blessed $dst and $dst->isa('PDL')) ? mx->nd->array($dst) : $dst;
        ok(($x->aspdl == (ref $dst_nd ? $dst_nd->aspdl : $dst_nd))->all);
    };

    my $shape = rand_shape_2d();
    for my $stype ('row_sparse', 'csr')
    {
        # ndarray assignment
        $check_sparse_nd_setitem->($stype, $shape, rand_ndarray($shape, 'default'));
        $check_sparse_nd_setitem->($stype, $shape, rand_ndarray($shape, $stype));
        # numpy assignment
        $check_sparse_nd_setitem->($stype, $shape, pones(reverse @{ $shape }));
    }
    # scalar assigned to row_sparse NDArray
    $check_sparse_nd_setitem->('row_sparse', $shape, 2);
}

test_sparse_nd_setitem();

sub test_sparse_nd_slice
{
    my $shape = [randint(2, 10), randint(2, 10)];
    my $stype = 'csr';
    my ($A) = rand_sparse_ndarray($shape, $stype);
    my $A2 = $A->aspdl;
    my $start = randint(0, $shape->[0] - 1);
    my $end = randint($start + 1, $shape->[0]);
    ok(same($A->slice([$start, $end])->aspdl, $A2->slice('X', [$start, $end])));
    ok(same($A->slice([$start - $shape->[0], $end])->aspdl, $A2->slice('X', [$start, $end])));
    ok(same($A->slice([$start, $shape->[0] - 1])->aspdl, $A2->slice('X', [$start, $shape->[0]-1])));
    ok(same($A->slice([0, $end])->aspdl, $A2->slice('X', [0, $end])));

    my $start_col = randint(0, $shape->[1] - 1);
    my $end_col = randint($start_col + 1, $shape->[1]);
    my $result = $A->slice(begin=>[$start, $start_col], end=>[$end, $end_col]);
    my $result_dense = mx->nd->array($A2)->slice(begin=>[$start, $start_col], end=>[$end, $end_col]);
    ok(same($result_dense->aspdl, $result->aspdl));

    $A = mx->nd->sparse->zeros('csr', $shape);
    $A2 = $A->aspdl;
    ok(same($A->slice([$start, $end])->aspdl, $A2->slice('X', [$start, $end])));
    $result = $A->slice(begin=>[$start, $start_col], end=>[$end, $end_col]);
    $result_dense = mx->nd->array($A2)->slice(begin=>[$start, $start_col], end=>[$end, $end_col]);
    ok(same($result_dense->aspdl, $result->aspdl));

    my $check_slice_nd_csr_fallback = sub { my ($shape) = @_;
        my $stype = 'csr';
        my ($A) = rand_sparse_ndarray($shape, $stype);
        my $A2 = $A->aspdl;
        my $start = randint(0, $shape->[0] - 1);
        my $end = randint($start + 1, $shape->[0]);

        # non-trivial step should fallback to dense slice op
        my $result = $A->slice(begin=>[$start], end=>[$end+1], step=>[2]);
        my $result_dense = mx->nd->array($A2)->slice(begin=>[$start], end=>[$end + 1], step=>[2]);
        ok(same($result_dense->aspdl, $result->aspdl));
    };
    $shape = [randint(2, 10), randint(1, 10)];
    $check_slice_nd_csr_fallback->($shape);
}

test_sparse_nd_slice();

sub test_sparse_nd_equal
{
    for my $stype ('row_sparse', 'csr')
    {
        my $shape = rand_shape_2d();
        my $x = mx->nd->zeros($shape, stype=>$stype);
        my $y = sparse_nd_ones($shape, $stype);
        my $z = $x == $y;
        ok(($z->aspdl == pzeros(reverse @{ $shape }))->all);
        $z = 0 == $x;
        ok(($z->aspdl == pones(reverse @{ $shape }))->all);
    }
}

test_sparse_nd_equal();

sub test_sparse_nd_not_equal
{
    for my $stype ('row_sparse', 'csr')
    {
        my $shape = rand_shape_2d();
        my $x = mx->nd->zeros($shape, stype=>$stype);
        my $y = sparse_nd_ones($shape, $stype);
        my $z = $x != $y;
        ok(($z->aspdl == pones(reverse @{ $shape }))->all);
        $z = 0 != $x;
        ok(($z->aspdl == pzeros(reverse @{ $shape }))->all);
    }
}

test_sparse_nd_not_equal();

sub test_sparse_nd_greater
{
    for my $stype ('row_sparse', 'csr')
    {
        my $shape = rand_shape_2d();
        my $x = mx->nd->zeros($shape, stype=>$stype);
        my $y = sparse_nd_ones($shape, $stype);
        my $z = $x > $y;
        ok(($z->aspdl == pzeros(reverse @{ $shape }))->all);
        $z = $y > 0;
        ok(($z->aspdl == pones(reverse @{ $shape }))->all);
        $z = 0 > $y;
        ok(($z->aspdl == pzeros(reverse @{ $shape }))->all);
    }
}

test_sparse_nd_greater();

sub test_sparse_nd_greater_equal
{
    for my $stype ('row_sparse', 'csr')
    {
        my $shape = rand_shape_2d();
        my $x = mx->nd->zeros($shape, stype=>$stype);
        my $y = sparse_nd_ones($shape, $stype);
        my $z = $x >= $y;
        ok(($z->aspdl == pzeros(reverse @{ $shape }))->all);
        $z = $y >= 0;
        ok(($z->aspdl == pones(reverse @{ $shape }))->all);
        $z = 0 >= $y;
        ok(($z->aspdl == pzeros(reverse @{ $shape }))->all);
        $z = $y >= 1;
        ok(($z->aspdl == pones(reverse @{ $shape }))->all);
    }
}

test_sparse_nd_greater_equal();

sub test_sparse_nd_lesser
{
    for my $stype ('row_sparse', 'csr')
    {
        my $shape = rand_shape_2d();
        my $x = mx->nd->zeros($shape, stype=>$stype);
        my $y = sparse_nd_ones($shape, $stype);
        my $z = $y < $x;
        ok(($z->aspdl == pzeros(reverse @{ $shape }))->all);
        $z = 0 < $y;
        ok(($z->aspdl == pones(reverse @{ $shape }))->all);
        $z = $y < 0;
        ok(($z->aspdl == pzeros(reverse @{ $shape }))->all);
    }
}

test_sparse_nd_lesser();

sub test_sparse_nd_lesser_equal
{
    for my $stype ('row_sparse', 'csr')
    {
        my $shape = rand_shape_2d();
        my $x = mx->nd->zeros($shape, stype=>$stype);
        my $y = sparse_nd_ones($shape, $stype);
        my $z = $y <= $x;
        ok(($z->aspdl == pzeros(reverse @{ $shape }))->all);
        $z = 0 <= $y;
        ok(($z->aspdl == pones(reverse @{ $shape }))->all);
        $z = $y <= 0;
        ok(($z->aspdl == pzeros(reverse @{ $shape }))->all);
        $z = 1 <= $y;
        ok(($z->aspdl == pones(reverse @{ $shape }))->all);
    }
}

test_sparse_nd_lesser_equal();

sub test_sparse_nd_binary
{
    my $N = 2;
    my $check_binary = sub { my ($fn, $stype) = @_;
        for (0 .. 2)
        {
            my $ndim = 2;
            my $oshape = [map { randint(1, 6) } 1..$ndim];
            my $bdim = 2;
            my @lshape = @$oshape;
            # one for broadcast op, another for elemwise op
            my @rshape = @lshape[($ndim-$bdim)..@lshape-1];
            for my $i (0..$bdim-1)
            {
                my $sep = mx->nd->random->uniform(0, 1)->asscalar;
                if($sep < 0.33)
                {
                    $lshape[$ndim-$i-1] = 1;
                }
                elsif($sep < 0.66)
                {
                    $rshape[$bdim-$i-1] = 1;
                }
            }
            my $lhs = mx->nd->random->uniform(0, 1, shape=>\@lshape)->aspdl;
            my $rhs = mx->nd->random->uniform(0, 1, shape=>\@rshape)->aspdl;
            my $lhs_nd = mx->nd->array($lhs)->tostype($stype);
            my $rhs_nd = mx->nd->array($rhs)->tostype($stype);
            ok(allclose($fn->($lhs, $rhs), $fn->($lhs_nd, $rhs_nd)->aspdl, 1e-4));
        }
    };
    for my $stype ('row_sparse', 'csr')
    {
        $check_binary->(sub { $_[0] +  $_[1] }, $stype);
        $check_binary->(sub { $_[0] -  $_[1] }, $stype);
        $check_binary->(sub { $_[0] *  $_[1] }, $stype);
        $check_binary->(sub { $_[0] /  $_[1] }, $stype);
        $check_binary->(sub { $_[0] ** $_[1] }, $stype);
        $check_binary->(sub { $_[0] >  $_[1] }, $stype);
        $check_binary->(sub { $_[0] <  $_[1] }, $stype);
        $check_binary->(sub { $_[0] >= $_[1] }, $stype);
        $check_binary->(sub { $_[0] <= $_[1] }, $stype);
        $check_binary->(sub { $_[0] == $_[1] }, $stype);
    }
}

test_sparse_nd_binary();

sub test_sparse_nd_binary_scalar_op
{
    my $N = 3;
    my $check = sub { my ($fn, $stype) = @_;
        for (1..$N)
        {
            my $ndim = 2;
            my $shape = [map { randint(1, 6) } 1..$ndim];
            my $npy = mx->nd->random->normal(0, 1, shape=>$shape)->aspdl;
            my $nd = mx->nd->array($npy)->tostype($stype);
            ok(allclose($fn->($npy), $fn->($nd)->aspdl, 1e-4));
        }
    };
    for my $stype ('row_sparse', 'csr')
    {
        $check->(sub { 1 +    $_[0] }, $stype);
        $check->(sub { 1 -    $_[0] }, $stype);
        $check->(sub { 1 *    $_[0] }, $stype);
        $check->(sub { 1 /    $_[0] }, $stype);
        $check->(sub { 2 **   $_[0] }, $stype);
        $check->(sub { 1 >    $_[0] }, $stype);
        $check->(sub { 0.5 >  $_[0] }, $stype);
        $check->(sub { 0.5 <  $_[0] }, $stype);
        $check->(sub { 0.5 >= $_[0] }, $stype);
        $check->(sub { 0.5 <= $_[0] }, $stype);
        $check->(sub { 0.5 == $_[0] }, $stype);
        $check->(sub { $_[0] / 2    }, $stype);
    }
}

test_sparse_nd_binary_scalar_op();

sub test_sparse_nd_binary_iop
{
    my $N = 3;
    my $check_binary = sub { my ($fn, $stype) = @_;
        for (1..$N)
        {
            my $ndim = 2;
            my $oshape = [map { randint(1, 6) } 1..$ndim];
            my $lhs = mx->nd->random->uniform(0, 1, shape => $oshape)->aspdl;
            my $rhs = mx->nd->random->uniform(0, 1, shape => $oshape)->aspdl;
            my $lhs_nd = mx->nd->array($lhs)->tostype($stype);
            my $rhs_nd = mx->nd->array($rhs)->tostype($stype);
            ok(
                allclose(
                    $fn->($lhs, $rhs),
                    $fn->($lhs_nd, $rhs_nd)->aspdl,
                    1e-4
                )
            );
        }
    };

    my $inplace_add = sub { my ($x, $y) = @_;
        $x += $y;
        return $x
    };
    my $inplace_mul = sub { my ($x, $y) = @_;
        $x *= $y;
        return $x
    };
    my @stypes = ('csr', 'row_sparse');
    my @fns = ($inplace_add, $inplace_mul);
    for my $stype (@stypes)
    {
        for my $fn (@fns)
        {
            $check_binary->($fn, $stype);
        }
    }
}

test_sparse_nd_binary_iop();

sub test_sparse_nd_negate
{
    my $check_sparse_nd_negate = sub { my ($shape, $stype) = @_;
        my $npy = mx->nd->random->uniform(-10, 10, shape => rand_shape_2d())->aspdl;
        my $arr = mx->nd->array($npy)->tostype($stype);
        ok(almost_equal($npy, $arr->aspdl));
        ok(almost_equal(-$npy, (-$arr)->aspdl));

        # a final check to make sure the negation (-) is not implemented
        # as inplace operation, so the contents of arr does not change after
        # we compute (-arr)
        ok(almost_equal($npy, $arr->aspdl));
    };
    my $shape = rand_shape_2d();
    my @stypes = ('csr', 'row_sparse');
    for my $stype (@stypes)
    {
        $check_sparse_nd_negate->($shape, $stype);
    }
}

test_sparse_nd_negate();

sub test_sparse_nd_broadcast
{
    my $sample_num = 10; # TODO 1000
    my $test_broadcast_to = sub { my ($stype) = @_;
        for (1..$sample_num)
        {
            my $ndim = 2;
            my $target_shape = [map { randint(1, 11) } 1..$ndim];
            my $shape = \@{ $target_shape };
            my $axis_flags = [map { randint(0, 2) } 1..$ndim];
            my $axes = [];
            enumerate(sub {
                my ($axis, $flag) = @_;
                if($flag)
                {
                    $shape->[$axis] = 1;
                }
            }, $axis_flags);
            my $dat = mx->nd->random->uniform(0, 1, shape => $shape)->aspdl - 0.5;
            my $pdl_ret = $dat;
            my $ndarray = mx->nd->array($dat)->tostype($stype);
            my $ndarray_ret = $ndarray->broadcast_to($target_shape);
            ok((pdl($ndarray_ret->shape) == pdl($target_shape))->all);
            my $err = (($ndarray_ret->aspdl - $pdl_ret)**2)->avg;
            ok($err < 1E-8);
        }
    };
    my @stypes = ('csr', 'row_sparse');
    for my $stype (@stypes)
    {
        $test_broadcast_to->($stype);
    }
}

test_sparse_nd_broadcast();

sub test_sparse_nd_transpose
{
    my $npy = mx->nd->random->uniform(-10, 10, shape => rand_shape_2d())->aspdl;
    my @stypes = ('csr', 'row_sparse');
    for my $stype (@stypes)
    {
        my $nd = mx->nd->array($npy)->tostype($stype);
        ok(almost_equal($npy->transpose, ($nd->T)->aspdl));
    }
}

test_sparse_nd_transpose();

sub test_sparse_nd_storage_fallback
{
    my $check_output_fallback = sub { my ($shape) = @_;
        my $ones = mx->nd->ones($shape);
        my $out = mx->nd->zeros($shape, stype=>'csr');
        mx->nd->broadcast_add($ones, $ones * 2, out=>$out);
        ok(($out->aspdl - 3)->sum == 0);
    };

    my $check_input_fallback = sub { my ($shape) = @_;
        my $ones = mx->nd->ones($shape);
        my $out = mx->nd->broadcast_add($ones->tostype('csr'), $ones->tostype('row_sparse'));
        ok(($out->aspdl - 2)->sum == 0);
    };

    my $check_fallback_with_temp_resource = sub { my ($shape) = @_;
        my $ones = mx->nd->ones($shape);
        my $out = mx->nd->sum($ones);
        ok($out->asscalar == product(@{ $shape }));
    };

    my $shape = rand_shape_2d();
    $check_output_fallback->($shape);
    $check_input_fallback->($shape);
    $check_fallback_with_temp_resource->($shape);
}

test_sparse_nd_storage_fallback();

sub test_sparse_nd_astype
{
    my @stypes = ('row_sparse', 'csr');
    for my $stype (@stypes)
    {
        my $x = mx->nd->zeros(rand_shape_2d(), stype => $stype, dtype => 'float32');
        my $y = $x->astype('int32');
        ok($y->dtype eq 'int32');
    }
}

test_sparse_nd_astype();

__END__

def test_sparse_nd_pickle():
    np.random.seed(0)
    repeat = 1
    dim0 = 40
    dim1 = 40
    stypes = ['row_sparse', 'csr']
    densities = [0, 0.5]
    stype_dict = {'row_sparse': RowSparseNDArray, 'csr': CSRNDArray}
    for _ in range(repeat):
        shape = rand_shape_2d(dim0, dim1)
        for stype in stypes:
            for density in densities:
                a, _ = rand_sparse_ndarray(shape, stype, density)
                assert isinstance(a, stype_dict[stype])
                data = pkl.dumps(a)
                b = pkl.loads(data)
                assert isinstance(b, stype_dict[stype])
                assert same(a.asnumpy(), b.asnumpy())


def test_sparse_nd_save_load():
    np.random.seed(0)
    repeat = 1
    stypes = ['default', 'row_sparse', 'csr']
    stype_dict = {'default': NDArray, 'row_sparse': RowSparseNDArray, 'csr': CSRNDArray}
    num_data = 20
    densities = [0, 0.5]
    fname = 'tmp_list.bin'
    for _ in range(repeat):
        data_list1 = []
        for i in range(num_data):
            stype = stypes[np.random.randint(0, len(stypes))]
            shape = rand_shape_2d(dim0=40, dim1=40)
            density = densities[np.random.randint(0, len(densities))]
            data_list1.append(rand_ndarray(shape, stype, density))
            assert isinstance(data_list1[-1], stype_dict[stype])
        mx.nd.save(fname, data_list1)

        data_list2 = mx.nd.load(fname)
        assert len(data_list1) == len(data_list2)
        for x, y in zip(data_list1, data_list2):
            assert same(x.asnumpy(), y.asnumpy())

        data_map1 = {'ndarray xx %s' % i: x for i, x in enumerate(data_list1)}
        mx.nd.save(fname, data_map1)
        data_map2 = mx.nd.load(fname)
        assert len(data_map1) == len(data_map2)
        for k, x in data_map1.items():
            y = data_map2[k]
            assert same(x.asnumpy(), y.asnumpy())
    os.remove(fname)

def test_sparse_nd_unsupported():
    nd = mx.nd.zeros((2,2), stype='row_sparse')
    fn_slice = lambda x: x._slice(None, None)
    fn_at = lambda x: x._at(None)
    fn_reshape = lambda x: x.reshape(None)
    fns = [fn_slice, fn_at, fn_reshape]
    for fn in fns:
        try:
            fn(nd)
            assert(False)
        except:
            pass

def test_create_csr():
    def check_create_csr_from_nd(shape, density, dtype):
        matrix = rand_ndarray(shape, 'csr', density)
        # create data array with provided dtype and ctx
        data = mx.nd.array(matrix.data.asnumpy(), dtype=dtype)
        indptr = matrix.indptr
        indices = matrix.indices
        csr_created = mx.nd.sparse.csr_matrix((data, indices, indptr), shape=shape)
        assert csr_created.stype == 'csr'
        assert same(csr_created.data.asnumpy(), data.asnumpy())
        assert same(csr_created.indptr.asnumpy(), indptr.asnumpy())
        assert same(csr_created.indices.asnumpy(), indices.asnumpy())
        # verify csr matrix dtype and ctx is consistent from the ones provided
        assert csr_created.dtype == dtype, (csr_created, dtype)
        assert csr_created.data.dtype == dtype, (csr_created.data.dtype, dtype)
        assert csr_created.context == Context.default_ctx, (csr_created.context, Context.default_ctx)
        csr_copy = mx.nd.array(csr_created)
        assert(same(csr_copy.asnumpy(), csr_created.asnumpy()))

    def check_create_csr_from_coo(shape, density, dtype):
        matrix = rand_ndarray(shape, 'csr', density)
        sp_csr = matrix.asscipy()
        sp_coo = sp_csr.tocoo()
        csr_created = mx.nd.sparse.csr_matrix((sp_coo.data, (sp_coo.row, sp_coo.col)), shape=shape, dtype=dtype)
        assert csr_created.stype == 'csr'
        assert same(csr_created.data.asnumpy(), sp_csr.data)
        assert same(csr_created.indptr.asnumpy(), sp_csr.indptr)
        assert same(csr_created.indices.asnumpy(), sp_csr.indices)
        csr_copy = mx.nd.array(csr_created)
        assert(same(csr_copy.asnumpy(), csr_created.asnumpy()))
        # verify csr matrix dtype and ctx is consistent
        assert csr_created.dtype == dtype, (csr_created.dtype, dtype)
        assert csr_created.data.dtype == dtype, (csr_created.data.dtype, dtype)
        assert csr_created.context == Context.default_ctx, (csr_created.context, Context.default_ctx)

    def check_create_csr_from_scipy(shape, density, f):
        def assert_csr_almost_equal(nd, sp):
            assert_almost_equal(nd.data.asnumpy(), sp.data)
            assert_almost_equal(nd.indptr.asnumpy(), sp.indptr)
            assert_almost_equal(nd.indices.asnumpy(), sp.indices)
            sp_csr = nd.asscipy()
            assert_almost_equal(sp_csr.data, sp.data)
            assert_almost_equal(sp_csr.indptr, sp.indptr)
            assert_almost_equal(sp_csr.indices, sp.indices)
            assert(sp.dtype == sp_csr.dtype), (sp.dtype, sp_csr.dtype)

        try:
            import scipy.sparse as spsp
            # random canonical csr
            csr_sp = spsp.rand(shape[0], shape[1], density, format="csr")
            csr_nd = f(csr_sp)
            assert_csr_almost_equal(csr_nd, csr_sp)
            # non-canonical csr which contains duplicates and unsorted indices
            indptr = np.array([0, 2, 3, 7])
            indices = np.array([0, 2, 2, 0, 1, 2, 1])
            data = np.array([1, 2, 3, 4, 5, 6, 1])
            non_canonical_csr = spsp.csr_matrix((data, indices, indptr), shape=(3, 3), dtype=csr_nd.dtype)
            canonical_csr_nd = f(non_canonical_csr, dtype=csr_nd.dtype)
            canonical_csr_sp = non_canonical_csr.copy()
            canonical_csr_sp.sum_duplicates()
            canonical_csr_sp.sort_indices()
            assert_csr_almost_equal(canonical_csr_nd, canonical_csr_sp)
        except ImportError:
            print("Could not import scipy.sparse. Skipping unit tests for scipy csr creation")

    dim0 = 20
    dim1 = 20
    densities = [0, 0.5]
    dtype = np.float64
    for density in densities:
        shape = rand_shape_2d(dim0, dim1)
        check_create_csr_from_nd(shape, density, dtype)
        check_create_csr_from_coo(shape, density, dtype)
        check_create_csr_from_scipy(shape, density, mx.nd.sparse.array)
        check_create_csr_from_scipy(shape, density, mx.nd.array)

def test_create_row_sparse():
    dim0 = 50
    dim1 = 50
    densities = [0, 0.5, 1]
    for density in densities:
        shape = rand_shape_2d(dim0, dim1)
        matrix = rand_ndarray(shape, 'row_sparse', density)
        data = matrix.data
        indices = matrix.indices
        rsp_created = mx.nd.sparse.row_sparse_array((data, indices), shape=shape)
        assert rsp_created.stype == 'row_sparse'
        assert same(rsp_created.data.asnumpy(), data.asnumpy())
        assert same(rsp_created.indices.asnumpy(), indices.asnumpy())
        rsp_copy = mx.nd.array(rsp_created)
        assert(same(rsp_copy.asnumpy(), rsp_created.asnumpy()))

def test_create_sparse_nd_infer_shape():
    def check_create_csr_infer_shape(shape, density, dtype):
        try:
            matrix = rand_ndarray(shape, 'csr', density=density)
            data = matrix.data
            indptr = matrix.indptr
            indices = matrix.indices
            nd = mx.nd.sparse.csr_matrix((data, indices, indptr), dtype=dtype)
            num_rows, num_cols = nd.shape
            assert(num_rows == len(indptr) - 1)
            assert(indices.shape[0] > 0), indices
            assert(np.sum((num_cols <= indices).asnumpy()) == 0)
            assert(nd.dtype == dtype), (nd.dtype, dtype)
        # cannot infer on invalid shape
        except ValueError:
            pass

    def check_create_rsp_infer_shape(shape, density, dtype):
        try:
            array = rand_ndarray(shape, 'row_sparse', density=density)
            data = array.data
            indices = array.indices
            nd = mx.nd.sparse.row_sparse_array((data, indices), dtype=dtype)
            inferred_shape = nd.shape
            assert(inferred_shape[1:] == data.shape[1:])
            assert(indices.ndim > 0)
            assert(nd.dtype == dtype)
            if indices.shape[0] > 0:
                assert(np.sum((inferred_shape[0] <= indices).asnumpy()) == 0)
        # cannot infer on invalid shape
        except ValueError:
            pass

    dtype = np.int32
    shape = rand_shape_2d()
    shape_3d = rand_shape_3d()
    densities = [0, 0.5, 1]
    for density in densities:
        check_create_csr_infer_shape(shape, density, dtype)
        check_create_rsp_infer_shape(shape, density, dtype)
        check_create_rsp_infer_shape(shape_3d, density, dtype)

def test_create_sparse_nd_from_dense():
    def check_create_from_dns(shape, f, dense_arr, dtype, default_dtype, ctx):
        arr = f(dense_arr, dtype=dtype, ctx=ctx)
        assert(same(arr.asnumpy(), np.ones(shape)))
        assert(arr.dtype == dtype)
        assert(arr.context == ctx)
        # verify the default dtype inferred from dense arr
        arr2 = f(dense_arr)
        assert(arr2.dtype == default_dtype)
        assert(arr2.context == Context.default_ctx)
    shape = rand_shape_2d()
    dtype = np.int32
    src_dtype = np.float64
    ctx = mx.cpu(1)
    dense_arrs = [mx.nd.ones(shape, dtype=src_dtype), np.ones(shape, dtype=src_dtype), \
                  np.ones(shape, dtype=src_dtype).tolist()]
    for f in [mx.nd.sparse.csr_matrix, mx.nd.sparse.row_sparse_array]:
        for dense_arr in dense_arrs:
            default_dtype = dense_arr.dtype if isinstance(dense_arr, (NDArray, np.ndarray)) \
                            else np.float32
            check_create_from_dns(shape, f, dense_arr, dtype, default_dtype, ctx)

def test_create_sparse_nd_from_sparse():
    def check_create_from_sp(shape, f, sp_arr, dtype, src_dtype, ctx):
        arr = f(sp_arr, dtype=dtype, ctx=ctx)
        assert(same(arr.asnumpy(), np.ones(shape)))
        assert(arr.dtype == dtype)
        assert(arr.context == ctx)
        # verify the default dtype inferred from dense arr
        arr2 = f(sp_arr)
        assert(arr2.dtype == src_dtype)
        assert(arr2.context == Context.default_ctx)

    shape = rand_shape_2d()
    src_dtype = np.float64
    dtype = np.int32
    ctx = mx.cpu(1)
    ones = mx.nd.ones(shape, dtype=src_dtype)
    csr_arrs = [ones.tostype('csr')]
    rsp_arrs = [ones.tostype('row_sparse')]
    try:
        import scipy.sparse as spsp
        csr_sp = spsp.csr_matrix(np.ones(shape, dtype=src_dtype))
        csr_arrs.append(csr_sp)
    except ImportError:
        print("Could not import scipy.sparse. Skipping unit tests for scipy csr creation")
    f_csr = mx.nd.sparse.csr_matrix
    f_rsp = mx.nd.sparse.row_sparse_array
    for sp_arr in csr_arrs:
        check_create_from_sp(shape, f_csr, sp_arr, dtype, src_dtype, ctx)
    for sp_arr in rsp_arrs:
        check_create_from_sp(shape, f_rsp, sp_arr, dtype, src_dtype, ctx)

def test_create_sparse_nd_empty():
    def check_empty(shape, stype):
        arr = mx.nd.empty(shape, stype=stype)
        assert(arr.stype == stype)
        assert same(arr.asnumpy(), np.zeros(shape))

    def check_csr_empty(shape, dtype, ctx):
        arr = mx.nd.sparse.csr_matrix(shape, dtype=dtype, ctx=ctx)
        assert(arr.stype == 'csr')
        assert(arr.dtype == dtype)
        assert(arr.context == ctx)
        assert same(arr.asnumpy(), np.zeros(shape))
        # check the default value for dtype and ctx
        arr = mx.nd.sparse.csr_matrix(shape)
        assert(arr.dtype == np.float32)
        assert(arr.context == Context.default_ctx)

    def check_rsp_empty(shape, dtype, ctx):
        arr = mx.nd.sparse.row_sparse_array(shape, dtype=dtype, ctx=ctx)
        assert(arr.stype == 'row_sparse')
        assert(arr.dtype == dtype)
        assert(arr.context == ctx)
        assert same(arr.asnumpy(), np.zeros(shape))
        # check the default value for dtype and ctx
        arr = mx.nd.sparse.row_sparse_array(shape)
        assert(arr.dtype == np.float32)
        assert(arr.context == Context.default_ctx)

    stypes = ['csr', 'row_sparse']
    shape = rand_shape_2d()
    shape_3d = rand_shape_3d()
    dtype = np.int32
    ctx = mx.cpu(1)
    for stype in stypes:
        check_empty(shape, stype)
    check_csr_empty(shape, dtype, ctx)
    check_rsp_empty(shape, dtype, ctx)
    check_rsp_empty(shape_3d, dtype, ctx)

def test_synthetic_dataset_generator():
    def test_powerlaw_generator(csr_arr, final_row=1):
        """Test power law distribution
        Total Elements: 32000, Number of zeros: 3200
        Every row has 2 * non zero elements of the previous row.
        Also since (2047 < 3200 < 4095) this will be true till 10th row"""
        indices = csr_arr.indices.asnumpy()
        indptr = csr_arr.indptr.asnumpy()
        for row in range(1, final_row + 1):
            nextrow = row + 1
            current_row_nnz = indices[indptr[row] - 1] + 1
            next_row_nnz = indices[indptr[nextrow] - 1] + 1
            assert next_row_nnz == 2 * current_row_nnz

    # Test if density is preserved
    csr_arr_cols, _ = rand_sparse_ndarray(shape=(32, 10000), stype="csr",
                                          density=0.01, distribution="powerlaw")

    csr_arr_small, _ = rand_sparse_ndarray(shape=(5, 5), stype="csr",
                                           density=0.5, distribution="powerlaw")

    csr_arr_big, _ = rand_sparse_ndarray(shape=(32, 1000000), stype="csr",
                                         density=0.4, distribution="powerlaw")

    csr_arr_square, _ = rand_sparse_ndarray(shape=(1600, 1600), stype="csr",
                                            density=0.5, distribution="powerlaw")
    assert len(csr_arr_cols.data) == 3200
    test_powerlaw_generator(csr_arr_cols, final_row=9)
    test_powerlaw_generator(csr_arr_small, final_row=1)
    test_powerlaw_generator(csr_arr_big, final_row=4)
    test_powerlaw_generator(csr_arr_square, final_row=6)

def test_sparse_nd_fluent():
    def check_fluent_regular(stype, func, kwargs, shape=(5, 17), equal_nan=False):
        with mx.name.NameManager():
            data = mx.nd.random_uniform(shape=shape, ctx=default_context()).tostype(stype)
            regular = getattr(mx.ndarray, func)(data, **kwargs)
            fluent = getattr(data, func)(**kwargs)
            if isinstance(regular, list):
                for r, f in zip(regular, fluent):
                    assert almost_equal(r.asnumpy(), f.asnumpy(), equal_nan=equal_nan)
            else:
                assert almost_equal(regular.asnumpy(), fluent.asnumpy(), equal_nan=equal_nan)

    common_func = ['zeros_like', 'square']
    rsp_func = ['round', 'rint', 'fix', 'floor', 'ceil', 'trunc',
                'abs', 'sign', 'sin', 'degrees', 'radians', 'expm1']
    for func in common_func:
        check_fluent_regular('csr', func, {})
    for func in common_func + rsp_func:
        check_fluent_regular('row_sparse', func, {})

    rsp_func = ['arcsin', 'arctan', 'tan', 'sinh', 'tanh',
                'arcsinh', 'arctanh', 'log1p', 'sqrt', 'relu']
    for func in rsp_func:
        check_fluent_regular('row_sparse', func, {}, equal_nan=True)

    check_fluent_regular('csr', 'slice', {'begin': (2, 5), 'end': (4, 7)}, shape=(5, 17))
    check_fluent_regular('row_sparse', 'clip', {'a_min': -0.25, 'a_max': 0.75})

    for func in ['sum', 'mean']:
        check_fluent_regular('csr', func, {'axis': 0})


def test_sparse_nd_exception():
    """ test invalid sparse operator will throw a exception """
    a = mx.nd.ones((2,2))
    assertRaises(mx.base.MXNetError, mx.nd.sparse.retain, a, invalid_arg="garbage_value")
    assertRaises(ValueError, mx.nd.sparse.csr_matrix, a, shape=(3,2))
    assertRaises(ValueError, mx.nd.sparse.csr_matrix, (2,2), shape=(3,2))
    assertRaises(ValueError, mx.nd.sparse.row_sparse_array, (2,2), shape=(3,2))
    assertRaises(ValueError, mx.nd.sparse.zeros, "invalid_stype", (2,2))

def test_sparse_nd_check_format():
    """ test check_format for sparse ndarray """
    shape = rand_shape_2d()
    stypes = ["csr", "row_sparse"]
    for stype in stypes:
        arr, _ = rand_sparse_ndarray(shape, stype)
        arr.check_format()
        arr = mx.nd.sparse.zeros(stype, shape)
        arr.check_format()
    # CSR format index pointer array should be less than the number of rows
    shape = (3, 4)
    data_list = [7, 8, 9]
    indices_list = [0, 2, 1]
    indptr_list = [0, 5, 2, 3]
    a = mx.nd.sparse.csr_matrix((data_list, indices_list, indptr_list), shape=shape)
    assertRaises(mx.base.MXNetError, a.check_format)
    # CSR format indices should be in ascending order per row
    indices_list = [2, 1, 1]
    indptr_list = [0, 2, 2, 3]
    a = mx.nd.sparse.csr_matrix((data_list, indices_list, indptr_list), shape=shape)
    assertRaises(mx.base.MXNetError, a.check_format)
    # CSR format indptr should end with value equal with size of indices
    indices_list = [1, 2, 1]
    indptr_list = [0, 2, 2, 4]
    a = mx.nd.sparse.csr_matrix((data_list, indices_list, indptr_list), shape=shape)
    assertRaises(mx.base.MXNetError, a.check_format)
    # CSR format indices should not be negative
    indices_list = [0, 2, 1]
    indptr_list = [0, -2, 2, 3]
    a = mx.nd.sparse.csr_matrix((data_list, indices_list, indptr_list), shape=shape)
    assertRaises(mx.base.MXNetError, a.check_format)
    # Row Sparse format indices should be less than the number of rows
    shape = (3, 2)
    data_list = [[1, 2], [3, 4]]
    indices_list = [1, 4]
    a = mx.nd.sparse.row_sparse_array((data_list, indices_list), shape=shape)
    assertRaises(mx.base.MXNetError, a.check_format)
    # Row Sparse format indices should be in ascending order
    indices_list = [1, 0]
    a = mx.nd.sparse.row_sparse_array((data_list, indices_list), shape=shape)
    assertRaises(mx.base.MXNetError, a.check_format)
    # Row Sparse format indices should not be negative
    indices_list = [1, -2]
    a = mx.nd.sparse.row_sparse_array((data_list, indices_list), shape=shape)
    assertRaises(mx.base.MXNetError, a.check_format)


if __name__ == '__main__':
    import nose
    nose.runmodule()
