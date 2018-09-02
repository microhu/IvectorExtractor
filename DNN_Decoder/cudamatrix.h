// cudamatrix.h -- matrix with CUDA execution
//
// F. Seide, Jan 2011
//
// $Log: /Speech_To_Speech_Translation/dbn/cudamatrix/cudamatrix.h $
// 
// 53    8/07/12 9:42 Fseide
// now defines CopyFlags even if NOCUDA, to make some code easier
// 
// 52    8/02/12 12:24p F-gli
// changed check-in on 7/17/12 by Adame outside NOCUDA, because it is not
// related to latgen build
// 
// 51    7/27/12 2:51p V-hansu
// encoding in GB2312, I do not know why vs2010 ask me to change it again
// and again. Acutally nothing has been changed.
// 
// 50    7/17/12 5:31p Adame
// Update for no-sync framework
// async copy fixes
// 
// 49    6/24/12 9:27p V-xieche
// switch code into a work point(an old version as well).
// 
// 47    6/08/12 8:36p V-xieche
// add a flag to decide to use async copy or sync copy. Need to improve it
// later.
// 
// 46    4/05/12 9:52p V-xieche
// add functions for posteriorstats in striped toplayer pipeline training.
// not finished yet.
// 
// 45    4/01/12 2:05p Fseide
// seterrorsignal now takes an offset parameter so that it can work for
// vertical stripes
// 
// 44    4/01/12 2:00p V-xieche
// code for striped seterror signal
// 
// 43    4/01/12 11:24a V-xieche
// add code for striped softmax computation in 2 gpu.
// 
// 42    3/31/12 19:16 Fseide
// new method assign() from another matrix
// 
// 41    2/26/12 6:58p V-xieche
// Add codes for coping date between CUDA device.
// 
// 40    2/25/12 5:24p V-xieche
// Add helpler function for coping date in CUDA device
// 
// 39    1/01/12 10:33a Fseide
// (added a comment)
// 
// 38    12/06/11 5:47p Dongyu
// #include <stdexcept>
// 
// 37    11/28/11 5:56p Dongyu
// added reshapecolumnproduct to support backprop in dtnn
// 
// 36    11/23/11 4:03p Dongyu
// add reshape and KhatriRaoProduct
// 
// 35    11/04/11 14:54 Fseide
// new parameter for addrowsum()
// 
// 34    10/25/11 5:17p Dongyu
// Implemented weight difference (L2 relative to a refmodel) based
// regularization, KL divergence (relative to a refmodel) based
// regularization, CL (only change large weight) and CS (only change small
// weight) based regularization for conservative adaptation. 
// 
// Right now I branched some of the functions. These functions can be
// combined to reduce redundency in the future.
// 
// 33    10/06/11 5:16p Dongyu
// added support to allow adapting weights whose absolute value is above
// or below a threshold controlled by --nochangeifaboveorbelow switch.
// 
// 32    6/21/11 13:40 Fseide
// added frame for new function patchasblockdiagonal(), but inner loop not
// implemented yet
// 
// 31    6/10/11 7:46 Fseide
// removed explicit #undef NOCUDA so we can #define it inside the CPP file
// 
// 30    3/02/11 9:35a Dongyu
// add setto0ifabsbelow definition
// 
// 29    2/26/11 4:31p Fseide
// new method softmax()
// 
// 28    2/25/11 5:55p Fseide
// new method synchronize();
// assign(0 and fetch() now take a parameter to run sync or async
// 
// 27    2/11/11 1:50p Fseide
// rolled back previous check-in
// 
// 26    2/11/11 1:47p Fseide
// 
// 25    2/10/11 1:14p Fseide
// new method posteriorstats()
// 
// 24    2/10/11 11:21a Fseide
// new method mulbydsigm
// 
// 23    2/10/11 10:56a Fseide
// new method setbackpropagationerrorsignal()
// 
// 22    2/07/11 9:34p Fseide
// new method llstats()
// 
// 21    2/07/11 7:03p Fseide
// new method addtoallcolumns()
// 
// 20    2/07/11 6:38p Fseide
// new method samplebinary()
// 
// 19    2/07/11 6:13p Fseide
// new method sigmoid()
// 
// 18    2/07/11 6:04p Fseide
// new method addrowsum()
// 
// 17    2/05/11 8:55p Fseide
// new method patch()
// 
// 16    2/02/11 8:03a Fseide
// gemm() now allows B also to be transposed
// 
// 15    2/01/11 4:52p Fseide
// deleted addcol()
// 
// 14    2/01/11 15:32 Fseide
// new CUDA method addcol for column-wise addition (to add bias)
// 
// 13    2/01/11 14:55 Fseide
// replaced dummy operator+= by method gems()
// 
// 12    2/01/11 13:52 Fseide
// added NOCUDA compilation mode
// 
// 11    1/31/11 3:31p Fseide
// (forgot to make operator+= pure virtual)
// 
// 10    1/31/11 2:47p Fseide
// matrix is now an interface
// 
// 9     1/31/11 8:38a Fseide
// (added a test() function)
// 
// 8     1/30/11 11:44p Fseide
// renamed numdevices() to getnumdevices() as it seemed to have conflicted
// with the other declaration
// 
// 7     1/30/11 11:37p Fseide
// fixed wrong #pragma
// 
// 6     1/30/11 11:30p Fseide
// now references the cudamatrix DLL
// 
// 5     1/30/11 11:21p Fseide
// added numdevices() to msra::cuda in cudamatrix.h
// 
// 4     1/30/11 11:19p Fseide
// changed to DLL-export cudamatrix instead of cudalib
// 
// 3     1/30/11 17:54 Fseide
// updated the #include
// 
// 2     1/30/11 16:37 Fseide
// added missing #pragma once
// 
// 1     1/30/11 16:29 Fseide
// CUDA-related source files added (currently empty placeholders)

#pragma once
#include <stdexcept>    // (for NOCUDA version only)

#define NOCUDA      // define this to skip CUDA components (will act as if no CUDA device)

namespace msra { namespace cuda {

struct/*interface*/ matrix
{
    virtual ~matrix() { }
    virtual void setdevice (size_t deviceid) = 0;
    virtual void allocate (size_t n, size_t m) = 0;
    virtual size_t rows() const throw() = 0;
    virtual size_t cols() const throw() = 0;
    virtual void reshape(const size_t newrows, const size_t newcols) = 0;
    virtual void KhatriRaoProduct(const matrix & m1, const matrix & m2) = 0;
    virtual matrix * patch (size_t i0, size_t i1, size_t j0, size_t j1) = 0;
    // transfer
    virtual void assign (size_t i0, size_t i1, size_t j0, size_t j1, const float * pi0j0, size_t colstride, bool synchronize) = 0;
    virtual void fetch (size_t i0, size_t i1, size_t j0, size_t j1, float * pi0j0, size_t colstride, bool synchronize) const = 0;
    // sync --consider all functions asynchronous, call this if needed for data access or time measurement
    virtual void synchronize() const = 0;
    // CUBLAS functions
    virtual void gemm (float beta, const matrix & A, bool Aistransposed, const matrix & B, bool Bistransposed, float alpha) = 0;
    virtual void gems (float beta, const matrix & other, float alpha) = 0;
    // additional specialized helpers with our own kernels
    virtual void setto0ifabsbelow (float threshold) = 0;
    virtual void setto0ifabsbelow2 (matrix &  ref, float threshold)=0;
    virtual void setto0ifabsabove2 (matrix &  ref, float threshold)=0;
    virtual void patchasblockdiagonal (size_t diagblocks, bool averageblocks, size_t firstcol) = 0;
    virtual void addrowsum (float beta, const matrix & othercols, float alpha) = 0;
    virtual void sigmoid() = 0;
    virtual void samplebinary (const matrix & P, unsigned int randomseed) = 0;
    virtual void addtoallcolumns (const matrix & other) = 0;
    virtual void llstats (const matrix & v1, matrix & logllsums, bool gaussian) const = 0;
    virtual void softmax() = 0;
    virtual void setbackpropagationerrorsignal (const matrix & uids, const matrix & Pu, size_t i0) = 0;
    virtual void setbackpropagationerrorsignalwithklreg (const matrix & uids, const matrix & Pu, const matrix & refPu, const float alpha) = 0;
    virtual void mulbydsigm (const matrix & sigm) = 0;
    virtual void posteriorstats (const matrix & Pu, matrix & logpps, matrix & pps, matrix & fcors) const = 0;
    // posteriorstats in striped mode. [v-xieche]
    virtual void stripedposteriorstats (const matrix & Pu, matrix & logpps, matrix & pps, matrix & fcors, size_t i0) const = 0;
    virtual void reshapecolumnproduct (const matrix & eh, const matrix & h, const bool isehtransposed) = 0;
     // transfer data between matrices, potentially across devices
    virtual void assign (matrix & other, float * pi0j0/*CPU buffer in case it's needed*/, size_t colstride, bool synchronize, int copyFlags) = 0;
#if  1   // striped softmax function. only for 2 gpu now.[v-xieche]
    virtual void stripedsoftmaxstep1 (matrix &partialsumvectors) = 0;
    virtual void stripedsoftmaxstep2 (matrix &partialsumvectors) = 0;
#endif
};

enum CopyFlags
{
    copySync = 0,	        // use synchronous copy
    copyAsync = 1,	        // use asynchronous copy
    copyUsePassedBuffer = 2,	// use the passed buffer for Async
    copyDirect = 4,             // use universal addressing (UA), or peer to peer copy
    copyUseDestinationBuffers = 8, // for async copy we usually use the CPU buffers associated with the source GPU
                                // this flag swaps the usage to the destination buffers instead
};

static inline size_t getnumdevices() { return 0; }
static inline matrix * newmatrix() { throw std::runtime_error ("should not be here"); }
static inline void test() {}

};};
