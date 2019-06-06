/* Define this macro to get the O_DIRECT flag */
#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif

#include <stdio.h>
#include <unistd.h>
#include <math.h>
#include <float.h>
#include <limits.h>
#include <sys/time.h>
#include <omp.h>
#include <iostream>

/* For MMAP */
#include "libpmem.h"

// IMM intrinsics
#include <immintrin.h>

#ifndef STREAM_ARRAY_SIZE
#   define STREAM_ARRAY_SIZE	10000000
#endif

#ifndef NTIMES
#   define NTIMES	10
#endif

#ifndef STREAM_TYPE
#define STREAM_TYPE __m512i
#endif

// Static arrays for generating tests
static STREAM_TYPE A[STREAM_ARRAY_SIZE];

#ifndef USE_MMAP
static STREAM_TYPE B[STREAM_ARRAY_SIZE];
#else
static STREAM_TYPE *B;
#endif

// For doing pointer alignment
#define ALIGNMENT 4096
size_t roundup(size_t a, size_t b)
{ 
    return (1 + (a - 1) / b) * b; 
}

#define NTESTS 6
static double avgtime[NTESTS] = {0};
static double maxtime[NTESTS] = {0};
static double mintime[NTESTS] = {FLT_MAX, FLT_MAX, FLT_MAX, FLT_MAX, FLT_MAX, FLT_MAX};
extern int omp_get_num_threads();

static std::string label[NTESTS] = {
    "stream_to_remote: ",
    "stream_to_local:  ",
    "copy_to_remote:   ",
    "copy_to_local::   ",
    "Read:             ",
    "Write:            "
};

static double	bytes[NTESTS] = {
    sizeof(STREAM_TYPE) * STREAM_ARRAY_SIZE,
    sizeof(STREAM_TYPE) * STREAM_ARRAY_SIZE,
    sizeof(STREAM_TYPE) * STREAM_ARRAY_SIZE,
    sizeof(STREAM_TYPE) * STREAM_ARRAY_SIZE,
    sizeof(STREAM_TYPE) * STREAM_ARRAY_SIZE,
    sizeof(STREAM_TYPE) * STREAM_ARRAY_SIZE
    };

extern double mysecond();

// Defeat the optimizer!!
static void escape(void *p) {
    asm volatile("" : : "g"(p) : "memory");
}

static void clobber() {
    asm volatile("" : : : "memory");
}


/////
///// Define Kernel Functions
/////
void copy_to_remote_stream()
{
#pragma omp parallel for schedule(static)
    for(size_t j = 0; j < STREAM_ARRAY_SIZE; j++)
    {
        STREAM_TYPE x = _mm512_stream_load_si512(&A[j]);
        _mm512_stream_si512(&B[j], x);
    }
}

void copy_to_remote()
{
#pragma omp parallel for schedule(static)
    for(size_t j = 0; j < STREAM_ARRAY_SIZE; j++)
    {
        STREAM_TYPE x = _mm512_load_si512(&A[j]);
        _mm512_store_si512(&B[j], x);
    }
}

void copy_to_local_stream()
{
#pragma omp parallel for schedule(static)
    for(size_t j = 0; j < STREAM_ARRAY_SIZE; j++)
    {
        STREAM_TYPE x = _mm512_stream_load_si512(&B[j]);
        _mm512_stream_si512(&A[j], x);
    }
}

void copy_to_local()
{
#pragma omp parallel for schedule(static)
    for(size_t j = 0; j < STREAM_ARRAY_SIZE; j++)
    {
        STREAM_TYPE x = _mm512_load_si512(&B[j]);
        _mm512_store_si512(&A[j], x);
    }
}

void do_accumulate()
{
int s = 0;
#pragma omp parallel for schedule(static)
    for(size_t j=0; j < STREAM_ARRAY_SIZE; j++)
    {
        s += _mm512_reduce_add_epi32(B[j]); 
    }
    // Escape the int - keep it from being optimized away
    escape(&s);
}

void do_zero()
{
STREAM_TYPE s = {0};
#pragma omp parallel for schedule(static)
    for(size_t j=0; j < STREAM_ARRAY_SIZE; j++)
    {
        _mm512_stream_si512(&B[j], s); 
    }
}



int main( int argc, char * argv[] )
{
    std::cout << "Size of STREAM_TYPE = " << sizeof(STREAM_TYPE) << std::endl;

    int			quantum, checktick();
    int			BytesPerWord;
    int			k;
    ssize_t		j;
    STREAM_TYPE		scalar;
    double		t, times[NTESTS][NTIMES];

    BytesPerWord = sizeof(STREAM_TYPE);

    // Report number of threads requested
#pragma omp parallel
    {
#pragma omp master
	{
	    k = omp_get_num_threads();
	    printf ("Number of Threads requested = %i\n",k);
        }
    }

    // Report number of threads actually given
	k = 0;
#pragma omp parallel
#pragma omp atomic
		k++;
    printf ("Number of Threads counted = %i\n",k);

    /* Potentially Create the arrays A, B, and C */
#ifdef USE_MMAP
    // Check if a file name was passed
    char* file;
    if (argc != 2) {
        printf("Usage: ./stream <file-to-mmap>\n");
        return 0;
    } else {
        file = argv[1];
    }

    size_t mmap_size  = sizeof(STREAM_TYPE) * (STREAM_ARRAY_SIZE + 2 * ALIGNMENT);
    size_t mapped_lenp;
    int is_pmemp;

    void* mmap_ptr = pmem_map_file(
        file, 
        mmap_size,
        PMEM_FILE_CREATE, 
        0666, 
        &mapped_lenp, 
        &is_pmemp
    );

    B = (STREAM_TYPE*) roundup((size_t)(STREAM_TYPE*) mmap_ptr, ALIGNMENT);

    std::cout << "Mapped Len: " << mapped_lenp << std::endl;
    std::cout << "Is PMEM: " << is_pmemp << std::endl;
    std::cout << "MMap Pointer: " << mmap_ptr << std::endl;
    std::cout << "MMap length: " << mmap_size << std::endl;
    std::cout << "B Pointer: " << B << std::endl;
    std::cout << "Stream Array Size: " << STREAM_ARRAY_SIZE << std::endl;
    printf("MMap: pmem_is_pmem: %i\n", pmem_is_pmem(mmap_ptr, mmap_size));
    printf("B: pmem_is_pmem: %i\n", pmem_is_pmem(B, STREAM_ARRAY_SIZE));

#endif

    /////
    ///// Main Testing Loop
    /////
    for (size_t k = 0; k < NTIMES; k++)
    {
        // Read Test
        times[0][k] = mysecond();
        copy_to_remote_stream();
        times[0][k] = mysecond() - times[0][k];

        // Write Test
        times[1][k] = mysecond();
        copy_to_local_stream();
        times[1][k] = mysecond() - times[1][k];

        times[2][k] = mysecond();
        copy_to_remote();
        times[2][k] = mysecond() - times[2][k];

        times[3][k] = mysecond();
        copy_to_local();
        times[3][k] = mysecond() - times[3][k];

        // Read Test
        times[4][k] = mysecond();
        do_accumulate();
        times[4][k] = mysecond() - times[4][k];

        // Write Test
        times[5][k] = mysecond();
        do_zero();
        times[5][k] = mysecond() - times[5][k];
    }


    /////
    ///// Print Results
    /////

    for (k=1; k<NTIMES; k++) /* note -- skip first iteration */
	{
	for (j=0; j<NTESTS; j++)
	    {
	    avgtime[j] = avgtime[j] + times[j][k];
	    mintime[j] = std::min(mintime[j], times[j][k]);
	    maxtime[j] = std::max(maxtime[j], times[j][k]);
	    }
	}

    printf("DATA\n");
    printf("Function    Best Rate MB/s  Avg time     Min time     Max time\n");
    for (j=0; j<NTESTS; j++) {
		avgtime[j] = avgtime[j]/(double)(NTIMES-1);

		printf("%s%12.1f  %11.6f  %11.6f  %11.6f\n", label[j].c_str(),
	       1.0E-06 * bytes[j]/mintime[j],
	       avgtime[j],
	       mintime[j],
	       maxtime[j]);
    }

    return 0;
};

/////
///// Misc Helper Functions
/////

# define	M	20

int checktick()
{
    int		i, minDelta, Delta;
    double	t1, t2, timesfound[M];

/*  Collect a sequence of M unique time values from the system. */

    for (i = 0; i < M; i++)
    {
        t1 = mysecond();
        while( ((t2=mysecond()) - t1) < 1.0E-6 );

        timesfound[i] = t1 = t2;
	}

/*
 * Determine the minimum difference between these M values.
 * This result will be our estimate (in microseconds) for the
 * clock granularity.
 */

    minDelta = 1000000;
    for (i = 1; i < M; i++)
    {
        Delta = (int)( 1.0E6 * (timesfound[i]-timesfound[i-1]));
        minDelta = std::min(minDelta, std::max(Delta,0));
	}

    return(minDelta);
}

#include <sys/time.h>

double mysecond()
{
    struct timeval tp;
    struct timezone tzp;
    int i;

    i = gettimeofday(&tp,&tzp);
    return ( (double) tp.tv_sec + (double) tp.tv_usec * 1.e-6 );
}
