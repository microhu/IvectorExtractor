// pplhelpers.h -- some helpers for PPL library
//
// F. Seide, Nov 2010
//
// $Log: /Speech_To_Speech_Translation/latgen/hapivitelib/pplhelpers.h $
// 
// 11    10/08/11 10:23 Fseide
// new accessor function get_cores()
// 
// 10    6/10/11 8:04 Fseide
// (fixed two compiler warnings about 'static' global functions)
// 
// 9     1/05/11 7:36p Fseide
// added a wrapper for parallel_for() to allow to test single-threaded
// execution, and removed 'using Concurrency'
// 
// 8     1/05/11 12:13p Fseide
// (for_all_numa_nodes_approximately() is its new name)
// 
// 7     1/04/11 10:44p Fseide
// for_all_cores_approximately(): increased number of items, to increase
// chance of hitting all cores... :(
// 
// 6     1/04/11 10:14p Fseide
// new method for_all_cores_approximately(), which is VERY approximate...
// 
// 5     11/26/10 14:24 Fseide
// changed rounding in foreach_index_block() towards bigger blocks/less
// loop overhead
// 
// 4     11/22/10 2:11p Fseide
// (commented out a message)
// 
// 3     11/22/10 13:39 Fseide
// number of cores is now a global setting for foreach_index_block()
// 
// 2     11/22/10 13:32 Fseide
// number of cores is now a global variable
// 
// 1     11/22/10 10:49 Fseide
// factored out from main.cpp

#pragma once

#include <ppl.h>

namespace msra { namespace parallel {

// ===========================================================================
// helpers related to multiprocessing and NUMA
// ===========================================================================

// determine number of CPU cores on this machine
static inline size_t determine_num_cores()
{
    SYSTEM_INFO sysInfo;
    GetSystemInfo (&sysInfo);
    return sysInfo.dwNumberOfProcessors;
}

extern size_t ppl_cores;    // number of cores to run on as requested by user

static inline void set_cores (size_t cores)
{
    ppl_cores = cores;
}

static inline size_t get_cores()    // if returns 1 then no parallelization will be done
{
    return ppl_cores;
}

// wrapper around Concurrency::parallel_for() to allow disabling parallelization altogether
template <typename FUNCTION> void parallel_for (size_t begin, size_t end, size_t step, const FUNCTION & f)
{
    const size_t cores = ppl_cores;
    if (cores > 1)  // parallel computation (regular)
    {
        //fprintf (stderr, "foreach_index_block: computing %d blocks of %d frames on %d cores\n", nblocks, nfwd, determine_num_cores());
        Concurrency::parallel_for (begin, end, step, f);
    }
    else            // for comparison: single-threaded (this also documents what the above means)
    {
        //fprintf (stderr, "foreach_index_block: computing %d blocks of %d frames on a single thread\n", nblocks, nfwd);
        for (size_t j0 = begin; j0 < end; j0 += step) f (j0);
    }
}

// execute a function 'body (j0, j1)' for j = [0..n) in chunks of ~targetstep in 'cores' cores
// Very similar to parallel_for() except that body function also takes end index,
// and the 'targetsteps' gets rounded a little to better map to 'cores.'
// ... TODO: Currently, 'cores' does not limit the number of threads in parallel_for() (not so critical, fix later or never)
template <typename FUNCTION> void foreach_index_block (size_t n, size_t targetstep, size_t targetalignment, const FUNCTION & body)
{
    const size_t cores = ppl_cores;
    const size_t maxnfwd = 2 * targetstep;
    size_t nblocks = (n + targetstep / 2) / targetstep;
    if (nblocks == 0) nblocks = 1;
    // round to a multiple of the number of cores
    if (nblocks < cores)    // less than # cores -> round up
        nblocks = (1+(nblocks-1)/cores) * cores;
    else                    // more: round down (reduce overhead)
        nblocks = nblocks / cores * cores;
    size_t nfwd = 1 + (n - 1) / nblocks;
    assert (nfwd * nblocks >= n);
    if (nfwd > maxnfwd) nfwd = maxnfwd; // limit to allocated memory just in case
    // ... TODO: does the above actually do anything/significant? nfwd != targetstep?

    // enforce alignment
    nfwd = (1 + (nfwd -1) / targetalignment) * targetalignment;

    // execute it!
    parallel_for (0, n, nfwd, [&](size_t j0)
    {
        size_t j1 = min (j0 + nfwd, n);
        body (j0, j1);
    });
}

};};
