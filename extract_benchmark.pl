#!/bin/perl

use strict;
use warnings;

my @files = @ARGV;
my $total_throughput = 0;
my $count = 0;

sub usage() {
    print STDERR "Usage: \$perl extract_benchmark.pl <log_files...>\n"
}

if (@ARGV == 0 && -t STDIN && -t STDERR) { 
    usage();
    exit();
}

foreach my $file (@files) {
    open(my $fh, '<', $file) or die "Cannot open file $file: $!";
    while (my $line = <$fh>) {
        if ($line =~ /RuntimeError: ([\w\s]+)/) {
            print("File $count: RuntimeError: $1\n");
            last;
        }

        if ($line =~ /Throughput per token including tokenize: ([\d.]+) msecs/) {
            $count++;
            print "File $count: Throughput per token including tokenize: ", $1, " msecs\n";
            $total_throughput += $1;
        }
        elsif ($line =~ /Start to ready to generate: ([\d.]+) secs/) {
            print "File $count: Start to ready to generate: ", $1, " secs\n";
        }
        elsif ($line =~ /Tokenize and generate (\d+) \(bs=\d+\) tokens: ([\d.]+) secs/) {
            print "File $count: Tokenize and generate $1 tokens: ", $2, " secs\n";
        }
        elsif ($line =~ /Start to finish: ([\d.]+) secs/) {
            print "File $count: Start to finish: ", $1, " secs\n";
        }
    }
    close($fh);
}

print "***\n";
if ($count > 0) {
    print "Average Throughput per token including tokenize: ", $total_throughput / $count, " msecs\n";
} else {
    print "No performance stats found in the provided files.\n";
}
