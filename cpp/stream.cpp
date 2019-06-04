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
//#include <sys/mman.h>
//#include <fcntl.h>
//#include <sys/types.h>
//#include <sys/stat.h>
#include "libpmem.h"

// IMM intrinsics
#include <immintrin.h>

#ifndef STREAM_ARRAY_SIZE
#   define STREAM_ARRAY_SIZE	10000000
#endif
// Generally just want 4 threads running
# define WRITE_CHUNK_SIZE STREAM_ARRAY_SIZE / 4

#ifndef NTIMES
#   define NTIMES	10
#endif

#ifndef STREAM_TYPE
#define STREAM_TYPE __m512i
#endif

# ifndef MIN
# define MIN(x,y) ((x)<(y)?(x):(y))
# endif
# ifndef MAX
# define MAX(x,y) ((x)>(y)?(x):(y))
# endif

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

#define NTESTS 2
static double avgtime[NTESTS] = {0};
static double maxtime[NTESTS] = {0};
static double mintime[NTESTS] = {FLT_MAX, FLT_MAX};
extern int omp_get_num_threads();

static char	*label[NTESTS] = {
    "Read:      ",
    "Write:     "};

static double	bytes[NTESTS] = {
    sizeof(STREAM_TYPE) * STREAM_ARRAY_SIZE,
    sizeof(STREAM_TYPE) * STREAM_ARRAY_SIZE
    };

extern double mysecond();

/////
///// Define Kernel Functions
/////
void do_write()
{
#pragma omp parallel for schedule(static)
    for(size_t j = 0; j < STREAM_ARRAY_SIZE; j++)
    {
        STREAM_TYPE x = _mm512_stream_load_si512(&A[j]);
        _mm512_stream_si512(&B[j], x);
    }
}

void do_read()
{
#pragma omp parallel for schedule(static)
    for(size_t j = 0; j < STREAM_ARRAY_SIZE; j++)
    {
        STREAM_TYPE x = _mm512_stream_load_si512(&B[j]);
        _mm512_stream_si512(&A[j], x);
    }
}


int main( int argc, char * argv[] )
{
    printf ("Size of STREAM_TYPE = %u\n", sizeof(STREAM_TYPE));

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

    //int fd = open(file, O_RDWR | O_DIRECT, S_IRWXU);
    //void* mmap_ptr = mmap(
    //    NULL,
    //    3 * sizeof(STREAM_TYPE) * (STREAM_ARRAY_SIZE + 2 * ALIGNMENT),
    //    PROT_WRITE | PROT_READ,
    //    MAP_SHARED,
    //    fd,
    //    0
    //);
    //close(fd);
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
        do_read();
        times[0][k] = mysecond() - times[0][k];

        // Write Test
        times[1][k] = mysecond();
        do_write();
        times[1][k] = mysecond() - times[1][k];
    }


    /////
    ///// Print Results
    /////

    for (k=1; k<NTIMES; k++) /* note -- skip first iteration */
	{
	for (j=0; j<NTESTS; j++)
	    {
	    avgtime[j] = avgtime[j] + times[j][k];
	    mintime[j] = MIN(mintime[j], times[j][k]);
	    maxtime[j] = MAX(maxtime[j], times[j][k]);
	    }
	}

    printf("Function    Best Rate MB/s  Avg time     Min time     Max time\n");
    for (j=0; j<NTESTS; j++) {
		avgtime[j] = avgtime[j]/(double)(NTIMES-1);

		printf("%s%12.1f  %11.6f  %11.6f  %11.6f\n", label[j],
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
        while( ((t2=mysecond()) - t1) < 1.0E-6 )
            ;

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
        minDelta = MIN(minDelta, MAX(Delta,0));
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
