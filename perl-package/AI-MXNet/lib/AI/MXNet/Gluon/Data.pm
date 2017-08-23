package AI::MXNet::Gluon::Data;
use strict;
use warnings;
use AI::MXNet::Gluon::Data::Set;
use AI::MXNet::Gluon::Data::Sampler;
use AI::MXNet::Gluon::Data::Loader;
use AI::MXNet::Gluon::Data::Vision;
use AI::MXNet::Function::Parameters;
sub vision { 'AI::MXNet::Gluon::Data::Vision' }
method DataLoader(@args) { AI::MXNet::Gluon::Data::Loader->new(@args) }

1;


