# pylint: disable=no-member, too-many-lines, redefined-builtin, protected-access, unused-import, invalid-name
# pylint: disable=too-many-arguments, too-many-locals, no-name-in-module, too-many-branches, too-many-statements
"""Read invidual image files and perform augmentations."""

from __future__ import absolute_import, print_function

import os
import random
import logging
import numpy as np
from .base import numeric_types

try:
    import cv2
except ImportError:
    cv2 = None

from . import ndarray as nd
from . import _ndarray_internal as _internal
from ._ndarray_internal import _cvimresize as imresize
from ._ndarray_internal import _cvcopyMakeBorder as copyMakeBorder
from . import io
from . import recordio
package AI::MXNet:Image;
use strict;
use warnings;
use Scalar::Util qw(blessed);
use AI::MXNet::Base;
use AI::MXNet::Function::Parameters;

=head1 NAME

AI::MXNet:Image - Read invidual image files and perform augmentations.
=cut

=head2 imdecode

Decode an image from string. Requires OpenCV to work.

Parameters
----------
$buf : str, array ref, pdl, ndarray
    Binary image data.
:$flag : int
    0 for grayscale. 1 for colored.
:$to_rgb : int
    0 for BGR format (OpenCV default). 1 for RGB format (MXNet default).
:$out : NDArray
    Output buffer. Do not specify for automatic allocation.
=cut

method imdecode(Str|PDL $buf, Int :$flag=1, Int :$to_rgb=1, Maybe[AI::MXNet::NDArray] :$out=)
{
    if(not ref $buf)
    {
        my $pdl_type = PDL::Type->new(DTYPE_MX_TO_PDL->{'uint8'});
        my $len; { use bytes; $len = length $buf; }
        my $pdl = PDL->new_from_specification($pdl_type, $len);
        ${$pdl->get_dataref} = $buf;
        $pdl->upd_data;
        $buf = $pdl;
    }
    if(not (blessed $buf and $buf->isa('AI::MXNet::NDArray')))
    {
        $buf = AI::MXNet::NDArray->array($buf, dtype=>'uint8');
    }
    return AI::MXNet::NDArray->_cvimdecode($buf, { flag => $flag, to_rgb => $to_rgb, ($out ? (out => $out) : ()) });
}

=head2 scale_down

Scale down crop size if it's bigger than image size

Parameters:
Shape $src_size
Shape $size

Returns:
($w, $h)
=cut

method scale_down(Shape $src_size, Shape $size)
{
    my ($w, $h = @{ $size };
    my ($sw, $sh) = @{ $src_size };
    if($sh < $h)
    {
        ($w, $h) = (($w*$sh)/$h, $sh);
    }
    if($sw < $w)
    {
        ($w, $h) = ($sw, ($h*$sw)/$w);
    }
    return (int($w), int($h));
}

=head2 resize_short

Resize shorter edge to size

Parameters:
AI::MXNet::NDArray $src
Int                $size
Int                $interp=2

Returns:
AI::MXNet::NDArray $resized_image
=cut

method resize_short(AI::MXNet::NDArray $src, Int $size, Int $interp=2)
{
    my ($new_h, $new_w);
    my ($h, $w) = @{ $src->shape };
    if($h > $w)
    {
        ($new_h, $new_w) = ($size*$h/$w, $size);
    }
    else
    {
        ($new_h, $new_w) = ($size, $size*$w/$h);
    }
    return AI::MXNet::NDArray->_cvimresize($src, $new_w, $new_h, { interp=>$interp });
}

=head2 fixed_crop

Crop src at fixed location, and (optionally) resize it to size

Parameters:
AI::MXNet::NDArray $src
Int                $x0
Int                $y0
Int                $w
Int                $h
Maybe[Shape]       $size=
Int                $interp=2

Returns:
AI::MXNet::NDArray $cropped_image
=cut

method fixed_crop(AI::MXNet::NDArray $src, Int $x0, Int $y0, Int $w, Int $h, Maybe[Shape] $size=, Int $interp=2)
{
    my $out = AI::MXNet::NDArray->crop($src, { begin=>[$y0, $x0, 0], end=>[$y0+$h, $x0+$w, $src->shape->[2]] });
    if(defined $size and join(',', $w, $h) ne join(',', @{ $size }))
    {
        $out = AI::MXNet::NDArray->_cvimresize($out, @{ $size }, { interp=>$interp });
    }
    return $out;
}

=head2 random_crop

Randomly crop src with size. Upsample result if src is smaller than size

Parameters:
AI::MXNet::NDArray $src
Shape              $size=
Int                $interp=2

Returns:
($cropped_image, [$x0, $y0, $new_w, $new_h])
=cut

method random_crop(AI::MXNet::NDArray $src, Shape $size, Int $interp=2)
{
    my ($mh, $w) = @{ $src->shape };
    my ($new_w, $new_h) = __PACKAGE__->scale_down([$w, $h], $size);

    my $x0 = int(rand($w - $new_w + 1));
    my $y0 = int(rand($h - $new_h + 1));

    my $out = __PACKAGE__->fixed_crop($src, $x0, $y0, $new_w, $new_h, $size, $interp);
    return ($out, [$x0, $y0, $new_w, $new_h]);
}

=head2 center_crop

Randomly crop src with size around the center. Upsample result if src is smaller than size

Parameters:
AI::MXNet::NDArray $src
Shape              $size=
Int                $interp=2

Returns:
($cropped_image, [$x0, $y0, $new_w, $new_h])
=cut

method center_crop(AI::MXNet::NDArray $src, Shape $size, Int $interp=2)
{
    my ($h, $w) = @{ $src->shape };
    my ($new_w, $new_h) = __PACKAGE__->scale_down([$w, $h], $size);

    my $x0 = int(($w - $new_w)/2);
    my $y0 = int(($h - $new_h)/2);

    my $out = __PACKAGE__->fixed_crop($src, $x0, $y0, $new_w, $new_h, $size, $interp);
    return ($out, [$x0, $y0, $new_w, $new_h]);
}

=head2 color_normalize

Normalize src with mean and std

Parameter:
AI::MXNet::NDArray $src
Num|AI::MXNet::NDArray $mean
Maybe[Num|AI::MXNet::NDArray] $std=
Int $interp=2

Returns:
AI::MXNet::NDArray $normalized_image
=cut

method color_normalize(AI::MXNet::NDArray $src, Num|AI::MXNet::NDArray $mean, Maybe[Num|AI::MXNet::NDArray] $std=)
{
    $src -= mean;
    if(defined $std)
    {
        $src /= $std;
    }
    return $src;
}

=head2 random_size_crop

Randomly crop src with size. Randomize area and aspect ratio

Parameters:
AI::MXNet::NDArray $src
Shape              $size
Int                $min_area
ArrayRef[Int]      [$from, $to] # $ratio
Maybe[Int]         $interp=2

Returns:
($cropped_image, [$x0, $y0, $new_w, $new_h])
=cut

method random_size_crop(AI::MXNet::NDArray $src, Shape $size, Int $min_area, ArrayRef[Int] $ratio, Maybe[Int] $interp=2)
{
    my ($h, $w) = @{ $src->shape };
    my ($from, $to) = @{ $ratio };
    my $new_ratio = $from + ($to-$from) * rand;
    my $max_area;
    if($new_ratio * $h > $w)
    {
        $max_area = $w*int($w/$new_ratio);
    }
    else
    {
        $max_area = $h*int($h*$new_ratio);
    }

    $min_area *= $h*$w;
    if($max_area < $min_area)
    {
        return __PACKAGE__->random_crop($src, $size, $interp);
    }
    my $new_area = $min_area + ($max_area-$min_area) * rand;
    my $new_w = int(sqrt($new_area*$new_ratio));
    my $new_h = $new_w;

    assert($new_w <= $w and $new_h <= $h);
    my $x0 = int(rand($w - $new_w + 1));
    my $y0 = int(rand($h - $new_h + 1));

    my $out = __PACKAGE__->fixed_crop($src, $x0, $y0, $new_w, $new_h, $size, $interp);
    return [$out, ($x0, $y0, $new_w, $new_h]);
}

=head2 ResizeAug

Makes "resize shorter edge to size augumenter" closure

Parameters:
Shape              $size
Int                $interp=2

Returns:
CodeRef that accepts AI::MXNet::NDArray $src as input
and returns [__PACKAGE__->resize_short($src, $size, $interp)]
=cut

method ResizeAug(Shape $size, Int $interp=2)
{
    my $aug = sub {
        my $src = shift;
        return [__PACKAGE__->resize_short($src, $size, $interp)];
    };
    return $aug;
}

=head2 RandomCropAug

Makes "random crop augumenter" closure

Parameters:
Shape              $size
Int                $interp=2

Returns:
CodeRef that accepts AI::MXNet::NDArray $src as input
and returns [(__PACKAGE__->random_crop($src, $size, $interp))[0]]
=cut

method RandomCropAug(Shape $size, Int $interp=2)
{
    my $aug = sub {
        my $src = shift;
        return [(__PACKAGE__->random_crop($src, $size, $interp))[0]];
    };
    return $aug;
}

=head2 RandomSizedCropAug

Makes "random crop augumenter" closure

Parameters:
Shape              $size
Int                $min_area
ArrayRef[Int]      $ratio
Int                $interp=2

Returns:
CodeRef that accepts AI::MXNet::NDArray $src as input
and returns [(__PACKAGE__->random_size_crop($src, $size, $min_area, $ratio, $interp))[0]]
=cut

method RandomSizedCropAug(Shape $size, Int $min_area, ArrayRef[Int] $ratio, Int $interp=2)
{
    my $aug = sub {
        my $src = shift;
        return [(__PACKAGE__->random_size_crop($src, $size, $min_area, $ratio, $interp))[0]];
    };
    return $aug;
}

=head2 CenterCropAug

Makes "center crop augumenter" closure

Parameters:
Shape              $size
Int                $interp=2

Returns:
CodeRef that accepts AI::MXNet::NDArray $src as input
and returns [(__PACKAGE__->center_crop($src, $size, $interp))[0]]
=cut

method CenterCropAug(Shape $size, Int $interp=2)
{
    my $aug = sub {
        my $src = shift;
        return [(__PACKAGE__->center_crop($src, $size, $interp))[0]];
    };
    return $aug;
}

=head2 RandomOrderAug

Makes "Apply list of augmenters in random order" closure

Parameters:
ArrayRef[CodeRef]  $ts

Returns:
CodeRef that accepts AI::MXNet::NDArray $src as input
and returns ArrayRef[AI::MXNet::NDArray]
=cut

method RandomOrderAug(ArrayRef[CodeRef] $ts)
{
    my $aug = sub {
        my $src = shift;
        my @ts = List::Util::shuffle(@{ $ts })
        my @tmp;
        for my $t (@ts)
        {
            push @tmp, &{$t}($src);
        }
        return \@tmp;
    };
    return $aug;
}

=head2 RandomOrderAug

Makes "Apply random brightness, contrast and saturation jitter in random order" closure

Parameters:
Num $brightness
Num $contrast
Num $saturation

Returns:
CodeRef that accepts AI::MXNet::NDArray $src as input
and returns ArrayRef[AI::MXNet::NDArray]
=cut

method ColorJitterAug(Num $brightness, Num $contrast, Num $saturation)
{
    my @ts;
    my $coef = AI::MXNet::NDArray->array([[[0.299, 0.587, 0.114]]]);
    if($brightness > 0)
    {
        my $baug = sub { my $src = shift;
            my $alpha = 1 + -$brightness + 2 * $brightness * rand;
            $src *= $alpha
            return [$src];
        };
        push @ts, $baug;
    }

    if($contrast > 0)
    {
        my $caug = sub { my $src = shift;
            my $alpha = 1 + -$contrast + 2 * $contrast * rand;
            my $gray  = $src*$coef;
            $gray = (3.0*(1.0-$alpha)/$gray->size)*$gray->sum;
            $src *= $alpha
            $src += $gray;
            return [$src];
        };
        push @ts, $caug;
    }

    if($saturation > 0)
    {
        my $saug = sub { my $src = shift;
            my $alpha = 1 + -$saturation + 2 * $saturation * rand;
            my $gray  = $src*$coef;
            $gray = AI::MXNet::NDArray->sum($gray, { axis=>2, keepdims =>1 });
            $gray *= (1.0-$alpha);
            $src *= $alpha;
            $src += $gray;
            return [$src];
        };
        push @ts, $saug;
    }

    return __PACKAGE__->RandomOrderAug(\@ts);
}

=head2 LightingAug

Makes "Add PCA based noise" closure

Parameters:
Num $alphastd
PDL $eigval
PDL $eigvec

Returns:
CodeRef that accepts AI::MXNet::NDArray $src as input
and returns ArrayRef[AI::MXNet::NDArray]
=cut

def LightingAug(Num $alphastd, PDL $eigval, PDL $eigvec)
    my $aug = sub { my $src = shift;
        my $alpha = AI::MXNet::NDArray->zeros([3]);
        AI::MXNet::Random->normal(0, $alphastd, { out => $alpha });
        my $rgb = ($eigvec*$alpha->aspdl) x $eigval;
        $src += AI::MXNet::NDArray->array($rgb);
        return [$src]
    };
    return $aug
}

=head2 ColorNormalizeAug

Makes "Mean and std normalization" closure

Parameters:
PDL $mean
PDL $std

Returns:
CodeRef that accepts AI::MXNet::NDArray $src as input
and returns [__PACKAGE__->color_normalize($src, $mean, $std)]
=cut

method ColorNormalizeAug(PDL $mean, PDL $std)
{
    $mean = AI::MXNet::NDArray->array($mean);
    $std = AI::MXNet::NDArray->array($std);
    my $aug = sub { my $src = shift;
        return [__PACKAGE__->color_normalize($src, $mean, $std)]
    };
    return $aug;
}

=head2 HorizontalFlipAug

Makes "Random horizontal flipping" closure

Parameters:
Num $p < 1

Returns:
CodeRef that accepts AI::MXNet::NDArray $src as input
and returns [$p > rand ? AI::MXNet::NDArray->flip($src, axis=1>) : $src]
=cut

method HorizontalFlipAug(Num $p)
{
    my $aug = sub { my $src = shift;
        return [$p > rand ? AI::MXNet::NDArray->flip($src, axis=1>) : $src]
    };
    return $aug;
}

=head2 CastAug

Makes "Cast to float32" closure

Parameters:
Num $p < 1

Returns:
CodeRef that accepts AI::MXNet::NDArray $src as input
and returns [$src->astype('float32')]
=cut

method CastAug()
{
    my $aug = sub { my $src = shift;
        return [$src->astype('float32')]
    };
    return $aug;
}

=head2 CreateAugmenter

Create augumenter list

Parameters:
Shape          :$data_shape,
Bool           :$resize=0,
Bool           :$rand_crop=0,
Bool           :$rand_resize=0,
Bool           :$rand_mirror=0,
Maybe[Num|PDL] :$mean=,
Maybe[Num|PDL] :$std=,
Num            :$brightness=0,
Num            :$contrast=0,
Num            :$saturation=0,
Num            :$pca_noise=0,
Int            :$inter_method=2
=cut

method CreateAugmenter(
Shape          :$data_shape,
Bool           :$resize=0,
Bool           :$rand_crop=0,
Bool           :$rand_resize=0,
Bool           :$rand_mirror=0,
Maybe[Num|PDL] :$mean=,
Maybe[Num|PDL] :$std=,
Num            :$brightness=0,
Num            :$contrast=0,
Num            :$saturation=0,
Num            :$pca_noise=0,
Int            :$inter_method=2
)
{
    my @auglist;
    if($resize > 0)
    {
        push @auglist, __PACKAGE__->ResizeAug($resize, $inter_method);
    }

    my $crop_size = [$data_shape->[2], $data_shape->[1]];
    if($rand_resize)
    {
        assert($rand_crop);
        push @auglist, __PACKAGE__->RandomSizedCropAug($crop_size, 0.3, (3.0/4.0, 4.0/3.0), $inter_method);
    }
    elsif($rand_crop)
    {
        push @auglist, __PACKAGE__->RandomCropAug($crop_size, $inter_method);
    }
    else
    {
        push @auglist, __PACKAGE__->CenterCropAug($crop_size, $inter_method);
    }

    if($rand_mirror)
    {
        push @auglist, __PACKAGE__->HorizontalFlipAug(0.5);
    }

    push @auglist, __PACKAGE_->CastAug;

    if($brightness or $contrast or $saturation)
    {
        push @auglist, __PACKAGE__->ColorJitterAug($brightness, $contrast, $saturation);
    }
    if($pca_noise > 0)
    {
        my $eigval = AI::MXNet::NDArray->array([55.46, 4.794, 1.148])->aspdl;
        my $eigvec = AI::MXNet::NDArray->array([[-0.5675, 0.7192, 0.4009],
                           [-0.5808, -0.0045, -0.8140],
                           [-0.5836, -0.6948, 0.4203]])->aspdl;
        push @auglist, __PACKAGE__->LightingAug($pca_noise, $eigval, $eigvec);
    }

    if($mean)
    {
        $mean = AI::MXNet::NDArray->array([123.68, 116.28, 103.53])->aspdl;
    }
    if($std)
    {
        $std = AI::MXNet::NDArray->array([58.395, 57.12, 57.375])->aspdl;
    }
    if(defined $mean)
    {
        assert(defined $std);
        push @auglist, __PACKAGE__->ColorNormalizeAug($mean, $std);
    }

    return \@auglist;
}


package AI::MXNet::ImageIter;
use Mouse;
extends 'AI::MXNet::DataIter';

=head1 NAME

AI::MXNet::ImageIter - Image data iterator
=cut

=head1 DESCRIPTION


Image data iterator with a large number of augumentation choices.
Supports reading from both .rec files and raw image files with image list.

To load from .rec files, please specify path_imgrec. Also specify path_imgidx
to use data partition (for distributed training) or shuffling.

To load from raw image files, specify path_imglist and path_root.

Parameters
----------
batch_size : Int
    Number of examples per batch
data_shape : Shape
    Data shape in (channels, height, width).
    For now, only RGB image with 3 channels is supported.
label_width : Int
    dimension of label
path_imgrec : str
    path to image record file (.rec).
    Created with tools/im2rec.py or bin/im2rec
path_imglist : str
    path to image list (.lst)
    Created with tools/im2rec.py or with custom script.
    Format: index\t[one or more label separated by \t]\trelative_path_from_root
imglist: array ref
    a list of image with the label(s)
    each item is a list [imagelabel: float or list of float, imgpath]
path_root : str
    Root folder of image files
path_imgidx : str
    Path to image index file. Needed for partition and shuffling when using .rec source.
shuffle : bool
    Whether to shuffle all images at the start of each iteration.
    Can be slow for HDD.
part_index : int
    Partition index
num_parts : int
    Total number of partitions.
kwargs : ...
=cut

has 'batch_size'  => (is => 'ro', isa => 'Int',   required => 1);
has 'data_shape'  => (is => 'ro', isa => 'Shape', required => 1);
has 'label_width' => (is => 'ro', isa => 'Int',   default  => 1);
has [qw/path_imgrec
        path_imglist
        path_root
        path_imgidx
    /]            => (is => 'ro', isa => 'Str');

    def __init__(self, batch_size, data_shape, label_width=1,
                 path_imgrec=None, path_imglist=None, path_root=None, path_imgidx=None,
                 shuffle=False, part_index=0, num_parts=1, aug_list=None, imglist=None, **kwargs):
        super(ImageIter, self).__init__()
        assert(path_imgrec or path_imglist or (isinstance(imglist, list)))
        if path_imgrec:
            print('loading recordio...')
            if path_imgidx:
                self.imgrec = recordio.MXIndexedRecordIO(path_imgidx, path_imgrec, 'r')
                self.imgidx = list(self.imgrec.keys)
            else:
                self.imgrec = recordio.MXRecordIO(path_imgrec, 'r') # pylint: disable=redefined-variable-type
                self.imgidx = None
        else:
            self.imgrec = None

        if path_imglist:
            print('loading image list...')
            with open(path_imglist) as fin:
                imglist = {}
                imgkeys = []
                for line in iter(fin.readline, ''):
                    line = line.strip().split('\t')
                    label = nd.array([float(i) for i in line[1:-1]])
                    key = int(line[0])
                    imglist[key] = (label, line[-1])
                    imgkeys.append(key)
                self.imglist = imglist
        elif isinstance(imglist, list):
            print('loading image list...')
            result = {}
            imgkeys = []
            index = 1
            for img in imglist:
                key = str(index) # pylint: disable=redefined-variable-type
                index += 1
                if isinstance(img[0], numeric_types):
                    label = nd.array([img[0]])
                else:
                    label = nd.array(img[0])
                result[key] = (label, img[1])
                imgkeys.append(str(key))
            self.imglist = result
        else:
            self.imglist = None
        self.path_root = path_root

        assert len(data_shape) == 3 and data_shape[0] == 3
        self.provide_data = [('data', (batch_size,) + data_shape)]
        if label_width > 1:
            self.provide_label = [('softmax_label', (batch_size, label_width))]
        else:
            self.provide_label = [('softmax_label', (batch_size,))]
        self.batch_size = batch_size
        self.data_shape = data_shape
        self.label_width = label_width

        self.shuffle = shuffle
        if self.imgrec is None:
            self.seq = imgkeys
        elif shuffle or num_parts > 1:
            assert self.imgidx is not None
            self.seq = self.imgidx
        else:
            self.seq = None

        if num_parts > 1:
            assert part_index < num_parts
            N = len(self.seq)
            C = N/num_parts
            self.seq = self.seq[part_index*C:(part_index+1)*C]
        if aug_list is None:
            self.auglist = CreateAugmenter(data_shape, **kwargs)
        else:
            self.auglist = aug_list
        self.cur = 0
        self.reset()

    def reset(self):
        if self.shuffle:
            random.shuffle(self.seq)
        if self.imgrec is not None:
            self.imgrec.reset()
        self.cur = 0

    def next_sample(self):
        """helper function for reading in next sample"""
        if self.seq is not None:
            if self.cur >= len(self.seq):
                raise StopIteration
            idx = self.seq[self.cur]
            self.cur += 1
            if self.imgrec is not None:
                s = self.imgrec.read_idx(idx)
                header, img = recordio.unpack(s)
                if self.imglist is None:
                    return header.label, img
                else:
                    return self.imglist[idx][0], img
            else:
                label, fname = self.imglist[idx]
                if self.imgrec is None:
                    with open(os.path.join(self.path_root, fname), 'rb') as fin:
                        img = fin.read()
                return label, img
        else:
            s = self.imgrec.read()
            if s is None:
                raise StopIteration
            header, img = recordio.unpack(s)
            return header.label, img

    def next(self):
        batch_size = self.batch_size
        c, h, w = self.data_shape
        batch_data = nd.empty((batch_size, c, h, w))
        batch_label = nd.empty(self.provide_label[0][1])
        i = 0
        try:
            while i < batch_size:
                label, s = self.next_sample()
                data = [imdecode(s)]
                if len(data[0].shape) == 0:
                    logging.debug('Invalid image, skipping.')
                    continue
                for aug in self.auglist:
                    data = [ret for src in data for ret in aug(src)]
                for d in data:
                    assert i < batch_size, 'Batch size must be multiples of augmenter output length'
                    batch_data[i][:] = nd.transpose(d, axes=(2, 0, 1))
                    batch_label[i][:] = label
                    i += 1
        except StopIteration:
            if not i:
                raise StopIteration

        return io.DataBatch([batch_data], [batch_label], batch_size-i)
