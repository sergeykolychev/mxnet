package AI::MXNet::Gluon::RNN;
use strict;
use warnings;
use AI::MXNet::Gluon::RNN::Layer;

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
            use parent 'AI::MXNet::Gluon::RNN';
            1;
EOP
            eval $short_name_package;
        }
    }
}

1;