package common::util;
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

use strict;
use warnings;
use File::Path ();

sub download_file{ my($url, $local_fname, $force_write) = @_;
    # requests is not default installed
    require LWP::UserAgent;
    $local_fname //= (split '/', $url)[-1];
    return $local_fname
        if not $force_write and -e $local_fname;

    my($dir_name) = $local_fname =~ m{^(.*)/};

    File::Path::make_path($dir_name) if $dir_name and not -d $dir_name;

    my $requests = CORE::state $ua = LWP::UserAgent->new(show_progress => 1);
    my $r = $requests->get($url, ':content_file' => $local_fname);
    die sprintf("failed to open %s", $url) unless $r->code == 200;

    return $local_fname;
}

sub get_gpus {
    # return a list of GPUs
    my $i = grep { /GPU/ } `nvidia-smi -L`;
    return(0 .. $i - 1);
}

1;
