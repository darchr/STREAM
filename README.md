# STREAM

This is a modified version of STREAM and provides a compiler+runtime option to a memory 
mapped file as the storage medium for the throughput arrays instead of DRAM.

Compiling this program as normal yields the standard STREAM program. 
```
gcc -march=native -mtune=native -mcmodel=large -DSTREAM_ARRAY_SIZE=100000000 -O3 -fopenmp stream.c -o stream
```

To program with the memory mapped file option, use the command below.
```
gcc -march=native -mtune=native -mcmodel=large -DSTREAM_ARRAY_SIZE=100000000 -O3 -fopenmp -DUSE_MMAP stream.c -o stream
```
Then, run the program.
```
./stream <path-to-mmap-file>
```

To run this program on specific numa nodes (helpful when using NVDIMMs), use `numactl`:
```
numactl --cpunodebind=0 --membind=0 ./stream [<path-to-mmap-file>]
```

## Stream Repeated

The original STREAM benchmark loops over the entire test suite. The program 
`stream_repeated.c` instead loops over each individual test, which may yield different
results depending on which system is being used.

## Non-Temporal Stores

If you look in the `cpp/` folder, you might find yet ANOTHER version of STREAM, this one
using non-temporal store intrinsics which esentially bypass the cache. This is a more
specialized benchmark meant to measure copy times from DRAM to persistent memory.

### Setup

I'm using this on a system with two NUMA nodes and with a single persistent memory region
on each node. Follow the instructions below to replicate. NOTE: I'm using NUMA node 1 so
none of the test code is running on CPU 0.
```
sudo ndctl destroy-namespace --force namespace1.0
sudo ipmctl create -goal MemoryMode=0
sudo reboot
# after reboot
sudo ndctl create-namespace --type=pmem --mode=fsdax --region=region1
sudo mkfs.xfs -f /dev/pmem1
sudo mount -o dax /dev/pmem1 /mnt
# make some directories and files to be used by STREAM
sudo mkdir /mnt/public
sudo chmod 777 /mnt/public
```

I use the following to compile:
```
g++ -march=native -mtune=native -mcmodel=large -DSTREAM_ARRAY_SIZE=50000000 -O3 -fopenmp -DUSE_MMAP stream.cpp -lpmem -o stream
```
Finally, I use the following command to run
```
numactl --cpunodebind=1 --membind=1 ./stream /mnt/public/file.pmem
```
Finally, you can run with fewer CPUs using:
```
numactl --physcpubind=24-27 --membind=1 ./stream /mnt/public/file.pmem
```

## Original README

```
===============================================

STREAM is the de facto industry standard benchmark
for measuring sustained memory bandwidth.

Documentation for STREAM is on the web at:
   http://www.cs.virginia.edu/stream/ref.html

===============================================
NEWS
===============================================
UPDATE: October 28 2014:

"stream_mpi.c" released in the Versions directory.

Based on Version 5.10 of stream.c, stream_mpi.c
brings the following new features:
* MPI implementation that *distributes* the arrays
  across all MPI ranks. (The older Fortran version
  of STREAM in MPI *replicates* the arrays across
  all MPI ranks.)
* Data is allocated using "posix_memalign" 
  rather than using static arrays.  Different
  compiler flags may be needed for both portability
  and optimization.
  See the READ.ME file in the Versions directory
  for more details.
* Error checking and timing done by all ranks and
  gathered by rank 0 for processing and output.
* Timing code uses barriers to ensure correct
  operation even when multiple MPI ranks run on
  shared memory systems.

NOTE: MPI is not a preferred implementation for
  STREAM, which is intended to measure memory
  bandwidth in shared-memory systems.  In stream_mpi,
  the MPI calls are only used to properly synchronize
  the timers (using MPI_Barrier) and to gather
  timing and error data, so the performance should 
  scale linearly with the size of the cluster.
  But it may be useful, and was an interesting 
  exercise to develop and debug.

===============================================
UPDATE: January 17 2013:

Version 5.10 of stream.c is finally available!

There are no changes to what is being measured, but
a number of long-awaited improvements have been made:

* Updated validation code does not suffer from 
  accumulated roundoff error for large arrays.
* Defining the preprocessor variable "VERBOSE"
  when compiling will (1) cause the code to print the
  measured average relative absolute error (rather than
  simply printing "Solution Validates", and (2) print
  the first 10 array entries with relative error exceeding
  the error tolerance.
* Array index variables have been upgraded from
  "int" to "ssize_t" to allow arrays with more
  than 2 billion elements on 64-bit systems.
* Substantial improvements to the comments in 
  the source on how to configure/compile/run the
  benchmark.
* The proprocessor variable controlling the array
  size has been changed from "N" to "STREAM_ARRAY_SIZE".
* A new preprocessor variable "STREAM_TYPE" can be
  used to override the data type from the default
  "double" to "float".
  This mechanism could also be used to change to 
  non-floating-point types, but several "printf"
  statements would need to have their formats changed
  to accomodate the modified data type.
* Some small changes in output, including printing
  array sizes is GiB as well as MiB.
* Change to the default output format to print fewer
  decimals for the bandwidth and more decimals for
  the min/max/avg execution times.


===============================================
UPDATE: February 19 2009:

The most recent "official" versions have been renamed
"stream.f" and "stream.c" -- all other versions have
been moved to the "Versions" subdirectory and should be
considered obsolete.

The "official" timer (was "second_wall.c") has been
renamed "mysecond.c".   This is embedded in the C version
("stream.c"), but still needs to be externally linked to
the FORTRAN version ("stream.f").  The new version defines
entry points both with and without trailing underscores,
so it *should* link automagically with any Fortran compiler.

===============================================

STREAM is a project of "Dr. Bandwidth":
	John D. McCalpin, Ph.D.
	john@mccalpin.com

===============================================

The STREAM web and ftp sites are currently hosted at
the Department of Computer Science at the University of
Virginia under the generous sponsorship of Professor Bill
Wulf and Professor Alan Batson.

===============================================
```
