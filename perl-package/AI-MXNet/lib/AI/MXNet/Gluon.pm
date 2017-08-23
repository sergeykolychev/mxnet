package AI::MXNet::Gluon;
use strict;
use warnings;
use AI::MXNet::Gluon::Loss;
use AI::MXNet::Gluon::Trainer;
use AI::MXNet::Gluon::Utils;
use AI::MXNet::Gluon::Data;

sub import
{
    my ($class, $short_name) = @_;
    if($short_name)
    {
        $short_name =~ s/[^\w:]//g;
        if(length $short_name)
        {
            my $short_name_package =<<"EOP";
            package $short_name;
            sub data { 'AI::MXNet::Gluon::Data' }
            sub loss { 'AI::MXNet::Gluon::Loss' }
            sub utils { 'AI::MXNet::Gluon::Utils' }
            sub Trainer { shift; AI::MXNet::Gluon::Trainer->new(\@_); }
            1;
EOP
            eval $short_name_package;
        }
    }
}

1;