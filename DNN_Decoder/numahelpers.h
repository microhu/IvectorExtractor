// numahelpers.h -- some helpers with NUMA
//
// F. Seide, Nov 2010
//
// $Log: /Speech_To_Speech_Translation/latgen/hapivitelib/numahelpers.h $
// 
// 28    7/11/11 8:02 Fseide
// (fixed a compiler error, no idea why it never surfaced before)
// 
// 27    6/19/11 7:14 Fseide
// added a few log messages to track down a memory issue
// 
// 26    6/10/11 8:04 Fseide
// added a missing #include;
// fixed compiler warnings
// 
// 25    2/15/11 15:15 Fseide
// malloc() now calls VirtualAllocExNuma() through a function pointer so
// we can run on pre-Vista OS
// 
// 24    1/28/11 11:06 Fseide
// class numalocaldatacache #ifdef-0'd out, since it is no longer used and
// will not
// 
// 23    1/14/11 6:12p Fseide
// new function getmostspaciousnumanode()
// 
// 22    1/13/11 10:08a Fseide
// (finetuned a parameter)
// 
// 21    1/05/11 7:49p Fseide
// (documented a bug)
// 
// 20    1/05/11 7:36p Fseide
// added a wrapper for parallel_for() to allow to test single-threaded
// execution, and removed 'using Concurrency'
// 
// 19    1/05/11 6:16p Fseide
// parallel_for_on_each_numa_node() now takes extra parameter
// 
// 18    1/05/11 4:53p Fseide
// foreach_node() renamed to foreach_node_single_threaded() and now only
// used in ok-to-be-slow constructors;
// reclone() now goes through parallel_for_on_each_numa_node() in prep of
// parallelization
// 
// 17    1/05/11 3:58p Fseide
// numalocaldatacache streamlined;
// numalocaldatacache now distinguishes between DATATYPE and CACHEDTYPE,
// so we can use it for the accumulator
// 
// 16    1/05/11 12:13p Fseide
// added a missing #include
// 
// 15    1/05/11 12:12p Fseide
// new function parallel_for_on_each_numa_node();
// used in process()
// 
// 14    1/04/11 10:43p Fseide
// foreach_node() more robust, but not perfect yet (good enough for now)
// 
// 13    1/04/11 10:13p Fseide
// changed to allocation-free recloning
// 
// 12    1/04/11 8:43p Fseide
// NUMA malloc() changed to allocate at least 1 MB, as smaller allocations
// get coalesced in the heap and likely land on the wrong NUMA node
// 
// 11    1/04/11 19:25 Fseide
// new method showavailablememory()
// 
// 10    11/26/10 15:54 Fseide
// (added a comment)
// 
// 9     11/25/10 7:33 Fseide
// added a non-NUMA version (#ifdef-0'd out) to allow building a version
// for older Windows Server versions that some of our server farms still
// use...
// 
// 8     11/23/10 10:54a Fseide
// (bug fix in clone())
// 
// 7     11/23/10 9:53a Fseide
// getclone() now no longer creates the data on the fly, but rather has to
// be pre-computed. This is to avoid cores to stall while waiting for the
// other core to finish the pre-computation
// 
// 6     11/23/10 8:54 Fseide
// getclone() now takes an optional post-processing function for preparing
// pre-cached/pre-computed data
// 
// 5     11/22/10 2:23p Fseide
// new method reset()
// 
// 4     11/19/10 14:44 Fseide
// changed to VirtualFree() per MSDN sample source code
// 
// 3     11/18/10 11:19 Fseide
// abstracted out the NUMA-local model cache
// 
// 2     11/15/10 19:46 Fseide
// commented
// 
// 1     11/15/10 19:44 Fseide
// created by moving functions here

#pragma once

#include <Windows.h>
#include <stdexcept>
#include "pplhelpers.h"
#include "simple_checked_arrays.h"
#include "basetypes.h"  // for FormatWin32Error

namespace msra { namespace numa {

// ... TODO: this can be a 'static', as it should only be set during foreach_node but not outside
extern int node_override;   // -1 = normal operation; >= 0: force a specific NUMA node

// force a specific NUMA node (only do this during single-threading!)
static inline void overridenode (int n = -1)
{
    node_override = n;
}

// get the number of NUMA nodes we would like to distinguish
static inline size_t getnumnodes()
{
    ULONG n;
    if (!GetNumaHighestNodeNumber (&n)) return 1;
    return n +1;
}

// execute body (node, i, n), i in [0,n) on all NUMA nodes in small chunks
template <typename FUNCTION> void parallel_for_on_each_numa_node (bool multistep, const FUNCTION & body)
{
    // get our configuration
	const size_t cores = msra::parallel::ppl_cores;
    assert (cores > 0);
    const size_t nodes = getnumnodes();
    const size_t corespernode = (cores -1) / nodes + 1;
    // break into 8 steps per thread
    const size_t stepspernode = multistep ? 16 : 1;
    const size_t steps = corespernode * stepspernode;
    // now run on many threads, hoping to hit all NUMA nodes, until we are done
    hardcoded_array<LONG/*unsigned int*/,256> nextstepcounters;    // next block to run for a NUMA node
    if (nodes > nextstepcounters.size())
        throw std::logic_error ("parallel_for_on_each_numa_node: nextstepcounters buffer too small, need to increase hard-coded size");
    for (size_t k = 0; k < nodes; k++) nextstepcounters[k] = 0;
    overridenode();
    //unsigned int totalloops = 0;    // for debugging only, can be removed later
    msra::parallel::parallel_for (0, nodes * steps /*execute each step on each NUMA node*/, 1, [&](size_t /*dummy*/)
    {
        const size_t numanodeid = getcurrentnode();
        // find a node that still has work left, preferring our own node
        // Towards the end we will run on wrong nodes, but what can we do.
        for (size_t node1 = numanodeid; node1 < numanodeid + nodes; node1++)
        {
            const size_t node = node1 % nodes;
            const unsigned int step = InterlockedIncrement (&nextstepcounters[node]) -1;  // grab this step
            if (step >= steps)  // if done then counter has exceeded the required number of steps
                continue;       // so try next NUMA node
            // found one: execute and terminate loop
            body (node, step, steps);
            //InterlockedIncrement (&totalloops);
            return; // done
        }
        // oops??
        throw std::logic_error ("parallel_for_on_each_numa_node: no left-over block found--should not get here!!");
    });
    //assert (totalloops == nodes * steps);
}

// execute a passed function once for each NUMA node
// This must be run from the main thread only.
// ... TODO: honor ppl_cores == 1 for comparative measurements against single threads.
template<typename FUNCTION>
static void foreach_node_single_threaded (const FUNCTION & f)
{
    const size_t n = getnumnodes();
    for (size_t i = 0; i < n; i++)
    {
        overridenode ((int) i);
        f();
    }
    overridenode (-1);
}

// get the current NUMA node
static inline size_t getcurrentnode()
{
    // we can force it to be a certain node, for use in initializations
    if (node_override >= 0)
        return (size_t) node_override;
    // actually use current node
    DWORD i = GetCurrentProcessorNumber();  // note: need to change for >63 processors
    UCHAR n;
    if (!GetNumaProcessorNode ((UCHAR) i, &n)) return 0;
    if (n == 0xff)
        throw std::logic_error ("GetNumaProcessorNode() failed to determine NUMA node for GetCurrentProcessorNumber()??");
    return n;
}

// allocate memory
// Allocation seems to be at least on a 512-byte boundary. We nevertheless verify alignment requirements.
typedef LPVOID (WINAPI *VirtualAllocExNuma_t) (HANDLE,LPVOID,SIZE_T,DWORD,DWORD,DWORD);
static VirtualAllocExNuma_t VirtualAllocExNuma = (VirtualAllocExNuma_t)-1;

static inline void * malloc (size_t n, size_t align)
{
    // VirtualAllocExNuma() only exists on Vista+, so go through an explicit function pointer
    if (VirtualAllocExNuma == (VirtualAllocExNuma_t)-1)
    {
        VirtualAllocExNuma = (VirtualAllocExNuma_t) GetProcAddress (GetModuleHandle ( TEXT ("kernel32.dll")), "VirtualAllocExNuma");
        fprintf (stderr, "VirtualAllocExNuma = %x\n", VirtualAllocExNuma);
    }

    // if we have the function then do a NUMA-aware allocation
    void * p;
    if (VirtualAllocExNuma != NULL)
    {
        size_t node = getcurrentnode();
        // "all Win32 heap allocations that are 1 MB or greater are forwarded directly to NtAllocateVirtualMemory
        // when they are allocated and passed directly to NtFreeVirtualMemory when they are freed" Greg Colombo, 2010/11/17
        if (n < 1024*1024)
            n = 1024*1024;	// -> brings NUMA-optimized code back to Node Interleave level (slightly faster)
        p = VirtualAllocExNuma (GetCurrentProcess(), NULL, n, MEM_COMMIT | MEM_RESERVE, PAGE_READWRITE, (DWORD) node);
    }
    else    // on old OS call no-NUMA version
    {
        p = VirtualAllocEx (GetCurrentProcess(), NULL, n, MEM_COMMIT | MEM_RESERVE, PAGE_READWRITE);
    }
    if (p == NULL)
        fprintf (stderr, "numa::malloc: failed allocating %d bytes with alignment %d\n", n, align);
    if (((size_t) p) % align != 0)
        throw std::logic_error ("VirtualAllocExNuma() returned an address that does not match the alignment requirement");
    return p;
}

// free memory allocated with numa::malloc()
static inline void free (void * p)
{
    assert (p != NULL);
    if (!VirtualFree (p, 0, MEM_RELEASE))
        throw std::logic_error ("VirtualFreeEx failure");
}

// dump memory allocation
static inline void showavailablememory (const char * what)
{
    size_t n = getnumnodes();
    for (size_t i = 0; i < n; i++)
    {
        ULONGLONG availbytes = 0;
        BOOL rc = GetNumaAvailableMemoryNode ((UCHAR) i, &availbytes);
        const double availmb = availbytes / (1024.0*1024.0);
        if (rc)
            fprintf (stderr, "%s: %8.2f MB available on NUMA node %d\n", what, availmb, i);
        else
            fprintf (stderr, "%s: error '%S' for getting available memory on NUMA node %d\n", what, FormatWin32Error (::GetLastError()).c_str(), i);
    }
}

// determine NUMA node with most memory available
static inline size_t getmostspaciousnumanode()
{
    size_t n = getnumnodes();
    size_t bestnode = 0;
    ULONGLONG bestavailbytes = 0;
    for (size_t i = 0; i < n; i++)
    {
        ULONGLONG availbytes = 0;
        GetNumaAvailableMemoryNode ((UCHAR) i, &availbytes);
        if (availbytes > bestavailbytes)
        {
            bestavailbytes = availbytes;
            bestnode = i;
        }
    }
    return bestnode;
}

#if 0   // this is no longer used (we now parallelize the big matrix products directly)
// class to manage multiple copies of data on local NUMA nodes
template<class DATATYPE,class CACHEDTYPE> class numalocaldatacache
{
    numalocaldatacache (const numalocaldatacache&); numalocaldatacache & operator= (const numalocaldatacache&);

    // the data set we associate to
    const DATATYPE & data;

    // cached copies of the models for NUMA
    vector<unique_ptr<CACHEDTYPE>> cache;

    // get the pointer to the clone for the NUMA node of the current thread (must exist)
    CACHEDTYPE * getcloneptr()
    {
        return cache[getcurrentnode()].get();
    }
public:
    numalocaldatacache (const DATATYPE & data) : data (data), cache (getnumnodes())
    {
        foreach_node_single_threaded ([&]()
        {
            cache[getcurrentnode()].reset (new CACHEDTYPE (data));
        });
    }

    // this takes the cached versions of the parent model
    template<typename ARGTYPE1,typename ARGTYPE2,typename ARGTYPE3>
    numalocaldatacache (numalocaldatacache<DATATYPE,DATATYPE> & parentcache, const ARGTYPE1 & arg1, const ARGTYPE2 & arg2, const ARGTYPE3 & arg3) : data (*(DATATYPE*)nullptr), cache (getnumnodes())
    {
        foreach_node_single_threaded ([&]()
        {
            const DATATYPE & parent = parentcache.getclone();
            size_t numanodeid = getcurrentnode();
            cache[numanodeid].reset (new CACHEDTYPE (parent, arg1, arg2, arg3));
        });
    }

    // re-clone --update clones from the cached 'data' reference
    // This is only valid if CACHEDTYPE==DATATYPE.
    // ... parallelize this!
    void reclone()
    {
        parallel_for_on_each_numa_node (true, [&] (size_t numanodeid, size_t step, size_t steps)
        {
            if (step != 0)
                return;     // ... TODO: tell parallel_for_on_each_numa_node() to only have one step, or parallelize
            cache[numanodeid].get()->copyfrom (data);	// copy it all over
        });
    }

    // post-process all clones
    // 'numanodeid' is ideally the current NUMA node most of the time, but not required.
    template<typename POSTPROCFUNC>
    void process (const POSTPROCFUNC & postprocess)
    {
        parallel_for_on_each_numa_node (true, [&] (size_t numanodeid, size_t step, size_t steps)
        {
            postprocess (*cache[numanodeid].get(), step, steps);
        });
    }

    // a thread calls this to get the data pre-cloned for its optimal NUMA node
    // (only works for memory allocated through msra::numa::malloc())
    const CACHEDTYPE & getclone() const { return *getcloneptr(); }
    CACHEDTYPE & getclone()             { return *getcloneptr(); }
};
#endif
};};
