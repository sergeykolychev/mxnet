package AI::MXNet::RecordIO;
use strict;
use warnings;
use AI::MXNet::Function::Parameters;
use AI::MXNet::Types;
use AI::MXNet::Base;
use Mouse;

=head1 NAME

    AI::MXNet::Function::Parameters - Read/write RecordIO format data
=cut

=head2 new

Parameters
----------
uri : Str
        uri path to recordIO file.
flag: Str
        "r" for reading or "w" writing.
=cut

has 'uri'         => (is => 'ro', isa => 'Str', required => 1);
has 'flag'        => (is => 'ro', isa => enum([qw/r w/]), required => 1);
has 'handle'      => (is => 'rw', isa => 'RecordIOHandle');
has [qw/writable 
        is_open/] => (is => 'rw', isa => 'Bool');

sub BUILD
{
    my $self = shift;
    $self->is_open(0);
    $self->open();
}

sub DEMOLISH
{
    shift->close;
}

=head2 open

Open record file
=cut

method open()
{
    my $handle;
    if($self->flag eq 'w')
    {
        $handle = check_call(AI::MXNetCAPI::RecordIOWriterCreate($self->uri));
        $self->writable(1);
    }
    else
    {
        $handle = check_call(AI::MXNetCAPI::RecordIOReaderCreate($self->uri));
        $self->writable(0);
    }
    $self->handle($handle);
    $self->is_open(1);
}

=head2 close

Close record file
=cut

method close()
{
    return if not $self->is_open;
    if($self->writable)
    {
        check_call(AI::MXNetCAPI::RecordIOWriterFree($self->handle));
    }
    else
    {
        check_call(AI::MXNetCAPI::RecordIOReaderFree($self->handle));
    }
    $self->is_open(0);
}

=head2 reset

Reset pointer to first item. If record is opened with 'w',
this will truncate the file to empty.
=cut

method reset()
{
    $self->close;
    $self->open;
}

=head2 write

Write a string buffer as a record

Parameters
----------
$buf : buffer to write.
=cut

method write(Str $buf)
{
    assert($self->writable);
    check_call(
        AI::MXNetCAPI::RecordIOWriterWriteRecord(
            $self->handle,
            $buf,
            length($buf)
        )
    );
}

=head2 read

Read a record as string

Returns
----------
$buf : string
=cut

method read()
{
    assert(not $self->writable);
    return scalar(check_call(
        AI::MXNetCAPI::RecordIOReaderReadRecord(
            $self->handle,
        )
    ));
}

package AI::MXNet::IndexedRecordIO;
use Mouse;
extends 'AI::MXNet::RecordIO';

=head1 NAME

AI::MXNet::IndexedRecordIO - Read/write RecordIO format data supporting random access.
=cut

=head2 new

Parameters
----------
idx_path : str
    Path to index file
uri : str
    Path to record file. Only support file types that are seekable.
flag : str
    'w' for write or 'r' for read
=cut

has 'idx_path'  => (is => 'ro', isa => 'Str', required => 1);
has [qw/idx
    keys fidx/] => (is => 'rw', init_arg => undef);

sub BUILD
{
    my $self = shift;
    $self->idx({});
    $self->keys([]);
}

method open()
{
    $self->SUPER::open();
    $self->idx({});
    $self->keys([]);
    open(my $f, $self->flag eq 'r' ? '<' : '>', $self->idx_path);
    $self->fidx($f);
    if(not $self->writable)
    {
        while(<$f>)
        {
            chomp;
            my ($key, $val) = split(/\t/);
            push @{ $self->keys }, $key;
            $self->idx->{$key} = $val;
        }
    }
}

method close()
{
    return if not $self->is_open;
    $self->SUPER::close();
    $self->fidx(undef);
}

=head2 seek

Query current read head position
=cut

method seek(Int $idx)
{
    assert(not $self->writable);
    my $pos = $self->idx->{$idx};
    check_call(AI::RecordIOReaderSeek($self->handle, $pos));
}

=head2 tell

Query current write head position
=cut

method tell()
{
    assert($self->writable);
    return scalar(check_call(AI::MXNetCAPI::RecordIOWriterTell($self->handle)));
}

=head2 read_idx

Read record with index

Parameters:
$idx
=cut

method read_idx(Int $idx)
{
    $self->seek($idx);
    return $self->read();
}

=head2 write_idx

Write record with index

Parameters:
$idx, $buf
=cut

method write_idx(Int $idx, Str $buf)
{
    my $pos = $self->tell();
    $self->write($buf);
    my $f = $self->fidx;
    print $f "$idx\t$pos\n";
    $self->idx->{$idx} = $pos;
    push @{ $self->keys }, $idx;
}

1;
