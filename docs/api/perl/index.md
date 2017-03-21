# MXNet - Perl API

See the [MXNet Perl API Documentation](http://mxnet.io/api/perl/docs/index.html).

MXNet supports the Perl programming language. The MXNet Perl package brings flexible and efficient GPU
computing and state-of-art deep learning to Perl. It enables you to write seamless tensor/matrix computation with multiple GPUs in Perl.
It also lets you construct and customize the state-of-art deep learning models in Perl,
  and apply them to tasks, such as image classification and data science challenges.

You can perform tensor or matrix computation in Perl with AI::MXNet:

```perl
   pdl> use AI::MXNet qw(mx); # creates 'mx' module on the fly with the interface close to the Python's API

   pdl> print $arr = mx->nd->ones([2, 3])
   <AI::MXNet::NDArray 2x3 @cpu(0)>

   pdl> print Data::Dumper::Dumper($arr->shape)
   $VAR1 = [
          2,
          3
        ];

   pdl> print (($arr*2)->aspdl) ## converts AI::MXNet::NDArray object to PDL object

   [
    [2 2 2]
    [2 2 2]
   ]

   pdl> print $arr = mx->nd->array([[1,2],[3,4]]) ## init the NDArray from Perl array ref given in PDL::pdl constructor format
   <AI::MXNet::NDArray 2x2 @cpu(0)>
   pdl> print $arr->aspdl

   [
    [1 2]
    [3 4]
   ]

   pdl> print mx->nd->array(sequence(2,3))->aspdl ## init the NDArray from PDL but be aware that PDL methods expect the dimensions order in column major format
        ## the dimensions ordered in the column major format, and NDArray is row major

   [
    [0 1]
    [2 3]
    [4 5]
   ]
```

 ## Perl API Reference
 * [Module API](module.md) is a flexible high-level interface for training neural networks.
 * [Symbolic API](symbol.md) performs operations on NDArrays to assemble neural networks from layers.
 * [IO Data Loading API](io.md) performs parsing and data loading.
 * [NDArray API](ndarray.md) performs vector/matrix/tensor operations.
 * [KVStore API](kvstore.md) performs multi-GPU and multi-host distributed training.


## Resources

* [MXNet Perl API Documentation](http://mxnet.io/api/perl/docs/index.html)
* [Handwritten Digit Classification in Perl](http://mxnet.io/tutorials/perl/mnist.html)
* [Neural Style in Scala on MXNet](https://github.com/dmlc/mxnet/blob/master/scala-package/examples/src/main/scala/ml/dmlc/mxnet/examples/neuralstyle/NeuralStyle.scala)
* [More Scala Examples](https://github.com/dmlc/mxnet/tree/master/scala-package/examples/src/main/scala/ml/dmlc/mxnet/examples)
