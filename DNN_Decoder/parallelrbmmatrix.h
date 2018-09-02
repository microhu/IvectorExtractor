// parallelrbmmatrix.h -- parallelized implementation of matrix functions required for RBMs
//
// F. Seide, Jan 2011
//
// $Log: /Speech_To_Speech_Translation/dbn/dbn/parallelrbmmatrix.h $
// 
// 212   8/15/12 10:17a V-hansu
// change some indentation
// 
// 211   8/06/12 20:51 Fseide
// samplebinary() now works in-place
// 
// 210   7/23/12 10:57a V-hansu
// modify setblockdiagonal function to make it compatible with second top
// layer adaptation
// 
// 209   7/17/12 5:32p Adame
// Update for no-sync framework
// async copy fixes
// 
// 208   7/11/12 7:31p V-hansu
// modify setblockdiagonal to make it compatible with round up mode of
// adaptation
// 
// 207   7/06/12 9:15p V-hansu
// modify  setblockdiagonal to make it compatible with new adaptation
// method
// 
// 206   6/30/12 2:28p V-hansu
// add a function to get colstride
// 
// 205   6/08/12 9:32p V-xieche
// delete code related to delayupdate.
// 
// 204   5/27/12 3:37p V-xieche
// modify the funciton onedevicedim for striped mode, consider the
// situation when number of toplayer's nodes divide cuda devices used on
// toplayer is not an integer.
// 
// 203   5/15/12 8:40p V-xieche
// enable code defined by COMPACTTRAINER, MULTICUDA, OPTPIPELINETRAIN and
// STRIPEDTOPLAYER to make the code has all functions defined for pipeline
// training. remove these MACRO later.
// 
// 202   5/10/12 6:46p V-xieche
// add code to make the MACRO MULTICUDA be compatible plain BP training as
// well.
// 
// 200   5/08/12 9:49p V-xieche
// Add macro SIMPLIFYCODE used for cleanup the code.
// 
// 199   4/18/12 4:37p V-xieche
// clean up the code related to macro DELAYUPDATE
// 
// 198   4/18/12 4:01p V-xieche
// clean up all code related to target propagation.
// 
// 197   4/18/12 2:06p V-xieche
// clean up the code in block of #ifdef TARGETBP #endif 
// 
// 196   4/11/12 5:28p V-xieche
// add DBNFASTTRAIN_NORMALBPTRAIN and DBNFASTTRAIN_PIPELINETRAIN, to debug
// the normal BP training and pipeline training for class dbnfasttrain in
// dbnfasttrain.h
// 
// 195   4/10/12 7:17p V-xieche
// add a temp macro DBNFASTTRAIN, to test the new class in dbnfasttrain.h.
// need to delete it after verification.
// 
// 194   4/08/12 9:29p V-xieche
// modify code, use function instead of previous pointer.
// 
// 193   4/06/12 6:26p V-xieche
// Add codes for posteriorstats function for striped top layer. not
// finished yet.
// 
// 192   4/05/12 9:51p V-xieche
// add code for accumulate prior and posteriorstats in striped toplayer
// pipeline training. not finished yet.
// 
// 191   4/03/12 8:41p V-xieche
// check in all the code for pipeline training. stripe top layer on two
// devices. need to add comments and adjust the code make it easy to read.
//
// 190   4/01/12 2:10p Fseide
// adapted for new i0 argument to set error signal
// 
// 189   3/27/12 2:18a V-xieche
// delete a function don't use any more.
// 
// 188   3/27/12 1:18a V-xieche
// add code for pipeline training with multi cuda devices. Need to add
// comments later. 
// 
// 187   3/16/12 2:11a V-xieche
// use fetch function to get thereference from cuda, the time used for
// training in compacttrainer is correct now.
// 
// 186   3/14/12 12:45a V-xieche
// add the MACRO to output time stats in compact trainer.
// 
// 185   3/11/12 7:05p V-xieche
// add code for a compact trainer. make it run in CUDA directly.
// 
// 184   3/08/12 10:33p V-xieche
// add code to make forward and backward prop do in CUDA directly.
// verified the training is correct, while speed faster than previous.
// need to debug it.
// 
// 183   3/05/12 9:10p V-xieche
// Add code for compact trainer to simplify the implmentation of DNN
// trainer(MACRO COMPACTTRAINER), to make it purely on CUDA and prepare
// for pipeline training in multiply CUDA device.
// 
// 182   2/26/12 8:45p V-xieche
// Add macro COPYINCUDA_FORDELAYUPDATE_V2, copy all data in CUDA device
// directly now.
// 
// 181   2/26/12 6:57p V-xieche
// Add codes for copy date between CUDA.
// 
// 180   2/25/12 5:23p V-xieche
// modify code for copy data in CUDA device. not completed.
// 
// 179   2/24/12 11:16p V-xieche
// Add code to assign value in CUDA directly for delayupdate training. not
// finished yet.
// 
// 178   2/23/12 5:47p V-xieche
// fix bugs exist in previous code for delay update mode.
// 
// 177   1/05/12 7:34p Fseide
// (editorial)
// 
// 176   1/04/12 5:41p Fseide
// bug fix in dropframes(), now uses correct lock function
// 
// 175   1/04/12 4:59p Fseide
// new method dropframes()
// 
// 174   12/20/11 3:14p Dongyu
// move KhatriRaoProduct and reshapecolumnproduct to class
// rbmstatevectorsbase
// 
// 173   12/09/11 2:02p F-gli
// add comments
// 
// 172   12/09/11 2:01p F-gli
// share cudamatrix.h to current folder because latgen,
// TranscriptorService does not have cudamatrix project
// 
// 171   12/07/11 4:26p Dongyu
// fixed stripping errors in reshapecolumnproduct
// 
// 170   12/06/11 5:44p Dongyu
// fixed bugs in reshapecolumnproduct
// 
// 169   11/28/11 5:56p Dongyu
// added reshapecolumnproduct to support backprop in dtnn
// 
// 168   11/23/11 4:33p Dongyu
// add reshape and KhatriRaoProduct
// 
// 167   11/16/11 11:55p V-xieche
// add macro DELAYUPDATE_V2_SWAPTEST for swap on delay update model.
// 
// 166   11/15/11 8:46p V-xieche
// add swap function for acceleratedmatrixbase class. Swap date both in
// CPU and CUDA if at cudamode. need to test it.
// 
// 165   11/14/11 4:01p V-xieche
// add micro LOADSTEEPERSIGMOIDMODEL for training the model from
// intermediate model for steeper or flatter sigmoid model.
// 
// 164   11/05/11 8:11p V-xieche
// add code for delay update model in code block DELAYUPDATE_V2
// 
// 163   11/04/11 16:27 Fseide
// scaleandaddallcols() now implements 'otherweight' for CUDA mode (but
// not yet NUMA mode)
// 
// 162   11/04/11 14:22 Fseide
// (incorrect comment fixed)
// 
// 161   10/31/11 9:01p V-xieche
// add code for simple experiment of delay update models.
// 
// 160   10/28/11 15:37 Fseide
// (minor fix)
// 
// 159   10/28/11 15:35 Fseide
// towards allowing scaled update to parameters, for better handling of
// momentum
// 
// 158   10/28/11 8:23 Fseide
// fixed another embarrassing bug in the efficiency "fix" for
// scaleandaddmatprod_numa()
// 
// 157   10/27/11 19:35 Fseide
// fixed incorrect 'fix' of scaleandaddmatprod_numa() NUMA inefficiency
// 
// 156   10/26/11 11:24a Dongyu
// removed debugging code for regularized adaptation.
// 
// 155   10/25/11 5:18p Dongyu
// Implemented weight difference (L2 relative to a refmodel) based
// regularization, KL divergence (relative to a refmodel) based
// regularization, CL (only change large weight) and CS (only change small
// weight) based regularization for conservative adaptation. 
// 
// Right now I branched some of the functions. These functions can be
// combined to reduce redundency in the future.
// 
// 154   10/18/11 9:07p V-xieche
// modify the code to implement a true steeper or flat sigmoid function.
// i.e. scale the bias as well
// 
// 153   10/11/11 3:34p V-xieche
// undefine SPARSENESSOUTPUTOFHIDDENLAYER. fix a minor argument.
// 
// 152   10/11/11 3:22p V-xieche
// modify the code for setting output of hidden layer below specific value
// to zero.
// 
// 151   10/11/11 12:09p V-xieche
// fix a minor bug for sparse experiment of hidden layer output
// 
// 150   10/11/11 11:38a V-xieche
// add code for setto0ifbelow for the output of hidden layer.
// 
// 149   10/11/11 8:24 Fseide
// fixed a compiler warning
// 
// 148   10/08/11 14:36 Fseide
// enabled NUMA fix
// 
// 147   10/08/11 10:22 Fseide
// fixed inefficieny of scaleandaddmatprod_numa() when not actually
// running in NUMA mode (to be tested);
// new special-purpose method peek()
// 
// 146   10/06/11 5:17p Dongyu
// added support to allow adapting weights whose absolute value is above
// or below a threshold controlled by --nochangeifaboveorbelow switch.
// 
// 145   10/03/11 14:35 Fseide
// (fixed a compiler warning)
// 
// 144   9/29/11 10:05p V-xieche
// Add CLUSTERSTATE definition to cluster state by monophone.
// 
// 143   9/28/11 10:05p V-xieche
// 
// 142   9/26/11 8:43p V-xieche
// Add some codes for log(sigmoid + epison) experiment.
// 
// 141   9/22/11 9:24p V-xieche
// fix a bug for deeper sigmoid experiment.
// 
// 140   9/20/11 2:46p V-xieche
// fix a minor bug for steeper sigmoid experiment
// 
// 139   9/19/11 10:54p V-xieche
// delete some debug code.
// 
// 138   9/19/11 10:49p V-xieche
// get and set weight matrix for temp experiment.
// 
// 137   9/17/11 12:27p F-gli
// comment out one useless line to make it build
// 
// 136   8/24/11 9:06p V-xieche
// add some log infomation for adding margin term.
// 
// 135   8/23/11 7:57p V-xieche
// add margin-based training code for dbn according to Heigold's thesis.
// 
// 134   8/22/11 4:06p V-xieche
// add some code for target propagation v5
// 
// 133   8/21/11 4:57p V-xieche
// add some code for target propagation version 5, it try to modify the
// weight according the normal BP algorithm, to see whether it works.
// 
// 132   8/18/11 11:03p V-xieche
// add some comment and log information.
// 
// 131   8/17/11 10:29p V-xieche
// add some code for targetpropagation version 3 to use label as target
// feature
// 
// 130   8/16/11 10:36p V-xieche
// fix a minor bug
// 
// 129   8/16/11 10:34p V-xieche
// add target propagation version 4 code. which used to verify the valid
// of target propagation. It do the same thing as the normal
// backpropagation for 2kx1 model only updating the bottom layer.
// 
// 128   8/15/11 10:58p V-xieche
// fix a minor bug and add some code for statistic the correct ratio for
// weight vector lies in decision region experiment
// 
// 127   8/15/11 10:30p V-xieche
// add code to statistic the ratio the top layer weight matrix lies in
// their class decision region
// 
// 126   8/13/11 9:42p V-xieche
// fix a minor bug
// 
// 125   8/13/11 5:29p V-xieche
// Add target propagationv3 function for the experiment of B=M*h
// experiment, which considerate all b, not only the label class.
// 
// 124   8/02/11 10:48p V-xieche
// correct a commant
// 
// 123   8/02/11 12:49a V-xieche
// undefine TARGETBP for a check-in code
// 
// 122   8/02/11 12:47a V-xieche
// must add the minus direction, i.e times a -2
// 
// 121   8/02/11 12:31a V-xieche
// add function settargetbpv2errorsignal() to implement targetpropagation
// version2. b=w*h
// 
// 120   7/29/11 5:58p V-xieche
// add some debug code to verify the target feature is correct. add code
// to verify target propagation could decrease the square error when
// updating bottom layer
// 
// 119   7/29/11 12:07a V-xieche
// add  err.lockforreading and unlock function and modify 2 to -2
// according to formulas.
// 
// 118   7/28/11 8:38p V-xieche
// add updatetargetfeatstats function for get and update target feature
// 
// 117   7/27/11 9:23p V-xieche
// Add the code for target propagation(to be debugged). in the #ifdef
// TARGEBP #endif block. 
// 
// 116   7/26/11 8:51a V-xieche
// fix a bug in bianry function
// 
// 115   7/25/11 9:54p V-xieche
// fix a bug in binary function
// 
// 114   7/25/11 8:45p V-xieche
// fix the bug in binarize function and rename the original function to
// binarize
// 
// 113   7/25/11 1:15p V-xieche
// Modify the setvalue function considering CUDA mode
// 
// 112   7/25/11 10:15a V-xieche
// Modify quantization and setvalue with considerate the cuda model, add
// lockforreadwrite and unlock funtion in them.
// 
// 111   7/23/11 5:52p V-xieche
// Add a function for set a to a fixed value. Often for set bias to 0 for
// experiment purpose
// 
// 110   7/15/11 11:38 Fseide
// added a comment on potential perf improvement
// 
// 109   7/13/11 19:02 Fseide
// new method matprod_col_mtm() for supporting on-demand LL evaluation
// 
// 108   7/07/11 8:46a V-xieche
// Modify a bug exists in the first check in code
// 
// 107   7/06/11 11:29p V-xieche
// add some code in the #if 0 #endif block for the histgoram stats.
// 
// 106   6/30/11 1:46p V-xieche
// Modeify a bug for the Nopooled-Diag matrix adaption in the setdiagblock
// function
// 
// 105   6/23/11 10:49a V-xieche
// delete some unneccessary bracket in the setblockdiagonal function.
// also add an else to throw exception if it is not a square matrix or an
// array.
// 
// 104   6/22/11 5:11p V-xieche
// modify the setblockdiagonal function also support the array for the
// pool of a.
// 
// 103   6/21/11 9:25p V-xieche
// set poolblocks to be true for generate the pooled-diag matrix
// 
// 102   6/21/11 13:46 Fseide
// first step towards CUDA implementation of setblockdiagonal
// 
// 101   6/21/11 1:20p V-xieche
// Modify the part of pooled-diagonal matrix caulation in setblockdiagonal
// function, make it more general
// 
// 100   6/21/11 9:52a V-xieche
// modify the segblockdiagonal function avoiding some bound check and some
// additional  memory load.
// 
// 99    6/21/11 7:56 Fseide
// (added TODOs, fixed TAB/indentation)
// 
// 98    6/20/11 10:18p V-xieche
// implement the function setblockdiagonal for the diagonal matrix
// adaptation and pooled diagonal matrix adaptation.
// 
// 97    6/20/11 7:43 Fseide
// new method setblockdiagonal()
// 
// 96    6/10/11 15:46 Fseide
// (fixed a spelling error in a log message)
// 
// 95    6/10/11 8:04 Fseide
// (fixed a few compiler warnings about unused function arguments)
// 
// 94    5/17/11 1:57p Fseide
// (minor edit to disabled pruning mode in seterrorsignal())
// 
// 93    4/22/11 10:14 Fseide
// added experimental pruning to seterrorsignal()
// 
// 92    3/14/11 11:15 Fseide
// documented seterrorsignal()
// 
// 91    3/05/11 8:29p Fseide
// added a 'const' modifier to write()
// 
// 90    3/03/11 8:16a Dongyu
// added weight sparseness support in training.
// 
// 89    2/26/11 6:03p Fseide
// minor fix in entercomputation(), now again working in CPU-only mode
// 
// 88    2/26/11 4:57p Fseide
// moved softmax() to GPU--reduces runtime by 1/3
// 
// 87    2/26/11 4:12p Fseide
// transited BP functions also to multi-GPU mode
// 
// 86    2/25/11 9:38p Fseide
// updated sumacrossdevices() to allow for parallel data transfer from
// different devices
// 
// 85    2/25/11 7:51p Fseide
// added explicit synchronization control to syncfromcuda()/synctocuda()
// 
// 84    2/25/11 6:04p Fseide
// changed synchronization--assign() and fetch() are now bulk-launched
// across devices, and synchronized after all have been kicked off, to
// allow for full parallelization (we don't know if it actually does it,
// though)
// 
// 83    2/25/11 5:41p Fseide
// bug fix in sumacrossdevices()
// 
// 82    2/25/11 10:03a Fseide
// sumacrossdevices() implemented, but still stuck due to gems()
// implementation
// 
// 81    2/24/11 11:16p Fseide
// (fixed a warning)
// 
// 80    2/24/11 11:15p Fseide
// (minor change of default 'simulateddevices' so we can test on the
// dual-Tesla machine before we really support 2 devices)
// 
// 79    2/24/11 11:00p Fseide
// (minor change to multi-device simulation)
// 
// 78    2/24/11 10:07p Fseide
// debugged and fixed syncto/fromcuda();
// added debugging facility to fake 2 cards (temporarily enabled)
// 
// 77    2/24/11 8:07p Fseide
// llstats() now parallelizes across multiple devices--completed
// parallelization of pretraining (except for sumacrossdevices())
// 
// 76    2/24/11 7:57p Fseide
// bug fix: cudadistributedmatrix::validstriping must be a reference to
// the master copy, it now is;
// new method sumacrossdevices() to reduce split matrix products
// 
// 75    2/24/11 6:06p Fseide
// few more steps towards multi-CUDA on the way
// 
// 74    2/24/11 3:00p Fseide
// added the on-the-fly view, but 'validstriping' not yet correctly
// supported in syncing--should we?
// 
// 73    2/24/11 9:33a Fseide
// removed viewedstriping in favor of an on-the-fly view
// 
// 72    2/23/11 6:08p Fseide
// steps towards multiple views for state vectors, still kind of messy
// 
// 71    2/23/11 4:04p Fseide
// misc. first-round bug fixes discovered during stepping-through;
// entercomputation() now sets the striping mode;
// checkcudastripingmode() now actually just checks rather than lazy
// initialization since that is no longer necessary
// 
// 70    2/23/11 1:51p Fseide
// finished change to cudadistributedmatrix, except for actual usage in
// math, where a compat mode was added for now
// 
// 69    2/19/11 16:52 Fseide
// baby coming--need to check in
// 
// 68    2/19/11 16:47 Fseide
// infrastructure for striped CUDA laid, but not completed yet as now all
// operations need to be updated (currently not compiling)
// 
// 67    2/17/11 18:13 Fseide
// now actually moves data to multiple GPU devices (but not used there
// yet)--not tested, could fail with assertions for old mode
// 
// 66    2/17/11 14:52 Fseide
// added more code towards multiple devices, not called yet
// 
// 65    2/16/11 16:49 Fseide
// (towards multi-GPUs)
// 
// 64    2/16/11 15:18 Fseide
// added design notes for multi-CUDA version
// 
// 63    2/15/11 16:28 Fseide
// rbmstatevectors() failed in non-CUDA mode;
// new method cudaptr() which returns a NULL in non-CUDA mode
// 
// 62    2/15/11 15:28 Fseide
// hascuda() now catches a DLL-load exception when calling
// getnumdevices(), i.e. we can run (in NUMA mode) if cudamatrix.dll or
// the CUDA DLLs are missing
// 
// 61    2/10/11 3:21p Fseide
// (fixed a variable spelling error in an assertion)
// 
// 60    2/10/11 1:54p Fseide
// switched to CUDA mode
// 
// 59    2/10/11 1:13p Fseide
// posteriorstats() change of logic
// 
// 58    2/10/11 12:59p Fseide
// posteriorstats() now uses vector mode, ready for CUDA version
// 
// 57    2/10/11 12:37p Fseide
// posteriorstats() factored into rbmstatevectorsref
// 
// 56    2/10/11 11:33a Fseide
// mulbydsigm() switched over to CUDA  --cuts over 15% runtime of
// backpropagationstats()
// 
// 55    2/10/11 11:18a Fseide
// seterrorsignal() switched to CUDA implementation
// 
// 54    2/10/11 10:53a Fseide
// seterrorsignal() working now
// 
// 53    2/10/11 10:32a Fseide
// moved error-signal computation to rbmstatevectors, for future CUDA
// implementation
// 
// 52    2/10/11 10:01a Fseide
// fetch() function gone;
// one unnecessary argument from scaleandaddallcols() gone;
// new method accumulate()
// 
// 51    2/09/11 10:10p Fseide
// added some #if-0'ed out experimental code (add +0.1 to the derivative)
// 
// 50    2/09/11 12:23a Fseide
// added some test code, but #if-ed out
// 
// 49    2/08/11 9:40p Fseide
// acceleratedmatrixbase::cudamatrix made private;
// acceleratedmatrixbase::operator msra::cuda::matrix*() changed to
// forcuda() & dealt with fallout;
// cachedmatrixbase no longer knows CUDA (getting ready for being folded
// into acceleratedmatrixbase)
// 
// 48    2/08/11 8:37p Fseide
// bug fix: alloccuda() should not not do anything if empty (that was an
// outdated condition)
// 
// 47    2/08/11 4:23p Fseide
// moved three resizeonce() calls from updatedeltas() to inside their NUMA
// counterparts (they are not used in CUDA, so no need to allocate them)
// 
// 46    2/08/11 2:50p Fseide
// made checkcudadims() const
// 
// 45    2/08/11 2:20p Fseide
// added code to verify runtime dimensions of CUDA matrix
// 
// 44    2/07/11 9:53p Fseide
// llstats() now uses CUDA implementation
// 
// 43    2/07/11 9:31p Fseide
// moved llstats() into rbmstatevectorsref, to allow acceleration by CUDA
// 
// 42    2/07/11 7:08p Fseide
// matprod_mt?m() now uses addtoallcolumns()
// 
// 41    2/07/11 6:52p Fseide
// samplebinary() now implemented in CUDA
// 
// 40    2/07/11 6:32p Fseide
// (moved up samplebinary())
// 
// 39    2/07/11 6:28p Fseide
// sigmoid() and addrowsum() now in CUDA
// 
// 38    2/07/11 5:24p Fseide
// rbmstatevector now going live in CUDA mode --data living in CUDA space;
// rbmmodelmatrixbase now implements operations in CUDA (so far not faster
// because too much happening on CPU side still)
// 
// 37    2/07/11 4:36p Fseide
// (removed two unused functions)
// 
// 36    2/07/11 4:29p Fseide
// hasnan() now requires locked mode
// 
// 35    2/07/11 4:11p Fseide
// bug fix: stripe() now longer returns a && but just the object itself
// (because it was just created inside);
// lock state changed to independent read and write state;
// lock state implemented in all rbmstatevectorsref functions, so they
// theoreticallt do work now even in case of CUDA
// 
// 34    2/07/11 3:25p Fseide
// moved the whole locking business from rbmstatevectorsbase to
// rbmstatevectorsrefbase
// 
// 33    2/07/11 2:26p Fseide
// rbmstatevectorsrefbase now derived from acceleratedmatrixbase, to
// reduce code duplication in managing CUDA stuff
// 
// 32    2/07/11 2:01p Fseide
// acceleratedmatrixbase moved down w.r.t. its template argument, which
// can now be matrix (with allocation) or matrixstriperef (no allocation)
// 
// 31    2/07/11 1:51p Fseide
// new method rbmstatevectors::stripe(), which required some reorganizing
// of things
// 
// 30    2/06/11 3:23p Fseide
// (minor cleanup)
// 
// 29    2/05/11 9:26p Fseide
// moved 'computing' state from acceleratedmatrixbase to derived class
// rbmmodelmatrixbase
// 
// 28    2/05/11 8:24p Fseide
// added mechanism for "locking" for direct access to the CPU-side
// rbmstatevectorsbase matrix, which takes care of moving from/to CUDA RAM
// 
// 27    2/05/11 7:00p Fseide
// factored out syncto/fromcuda();
// moved mulbydsigm() and samplebinary() to rbmstatevectorsref
// 
// 26    2/03/11 9:33p Fseide
// entercomputation() and fetch() now no longer fail if the matrix is
// empty (used to hit an assertion in matrix(i,j))
// 
// 25    2/02/11 11:29a Fseide
// added comments for next steps of CUDA transition
// 
// 24    2/02/11 11:23a Fseide
// moved sigmoid() and softmax() to here from original matrixbase class
// (which is now a mere typedef)
// 
// 23    2/02/11 11:14a Fseide
// rbmmodelmatrixbase now takes all state inputs as rbmmodelvectors;
// rbmmodelvectorsref fixed w.r.t. types for down-stream calls
// 
// 22    2/02/11 10:49a Fseide
// moved rbmstatevector before rbmmodelmatrix because the latter takes
// inputs of the type of the former
// 
// 21    2/02/11 10:47a Fseide
// added compat stub for hasnan()
// 
// 20    2/02/11 10:43a Fseide
// added some feed-through functions, to be replaced by abstracting the
// functions that call them here
// 
// 19    2/02/11 10:24a Fseide
// dummy implementations of rbmstatevectorsbase and rbmstatevectorsrefbase
// 
// 18    2/02/11 9:22a Fseide
// started new class rbmstatevectorsbase
// 
// 17    2/02/11 8:57a Fseide
// (added a comment)
// 
// 16    2/02/11 8:55a Fseide
// split rbmmodelmatrixbase out from acceleratedmatrixbase (we will later
// also have an rbmstatematrixbase)
// 
// 15    2/02/11 8:22a Fseide
// pushed some math ops on updatedeltas() down to acceleratedmatrix, for
// further CUDA optimization
// 
// 14    2/01/11 7:11p Fseide
// replaced addition of biases by a CPU-side function, because it leads to
// more accurate results (??)
// 
// 13    2/01/11 4:57p Fseide
// make_ones() compiles now--time to test!
// 
// 12    2/01/11 4:54p Fseide
// replaced addcol() by a dyadic matrix product--because cublas cannot do
// otherwise
// 
// 11    2/01/11 15:32 Fseide
// new CUDA method addcol for column-wise addition (to add bias)
// 
// 10    2/01/11 15:28 Fseide
// cuda version implemented
// 
// 9     2/01/11 15:00 Fseide
// matprod_m*m() functions now take one additional cache object for moving
// data to/from CUDA
// 
// 8     2/01/11 14:57 Fseide
// added stub if statements for cuda mode
// 
// 7     2/01/11 11:49a Fseide
// reactivated acceleratedmatrixbase::operator= (const) for use during
// computation state in updatedeltas()
// 
// 6     1/30/11 11:45p Fseide
// renamed numdevices() to getnumdevices()
// 
// 5     1/30/11 19:01 Fseide
// first steps towards CUDA mode--detect CUDA
// 
// 4     1/30/11 17:53 Fseide
// added #include "cudamatrix.h"
// 
// 3     1/30/11 16:37 Fseide
// added missing #pragma once
// 
// 2     1/30/11 16:33 Fseide
// acceleratedmatrixbase and cachedmatrixbase moved to parallelrbmmatrix.h
// 
// 1     1/30/11 16:30 Fseide
// parallelrbmmatrix.h added

#pragma once

#include "numahelpers.h"
#include "pplhelpers.h"
#include "cudamatrix.h"		// share cudamatrix.h to current folder because latgen, TranscriptorService does not have cudamatrix project

// #define STEEPERSIGMOID // using a more steeper sigmoid function in hidden layer. [v-xieche]
// #define LOADSTEEPERSIGMOIDMODEL // continued to train steeper or flatter model, need to divide the scale before training when loading model.
// #define UPDATEWEIGHTFORSPSM  // when using steepersigmoid, updated weight matrix in hidden layer as well. i.e. multiply the scale on hidden layer.[v-xieche]
//#define SCALEBIASFORSS       // also add a scale on the bias in the hidden layer when using a steeper and flat sigmoid.[v-xieche]
//#define AMPNUM 0.7     // the scale of the sigmoid function [v-xieche]
// #define LOGINSIGMOID  // the log in output of sigmoid function in hidden layer. i.e. log (sigmoid(z) + epison). [v-xieche]
// #define EPISONFORLOG  0.5 // the epison used for log function, avoid of numeric problem.[v-xieche]
// #define CLUSTERSTATE   // cluster the monophone as a class, to analysis the histogram table. only used in mltrain model now[v-xieche]
// #define SPARSENESSOUTPUTOFHIDDENLAYER   // test the experiment of sparseness of output of hidden layer. [v-xieche]

//#define COMPACTTRAINER       //for no cuda//for a compact DNN trainer and for fast and pure train on CUDA. [v-xieche]
// #define PIPELINETRAIN        // implement the code for pipeline training. [v-xieche]
//#define MULTICUDA          //for  no cuda  // for multi cuda device and pipeline trianing on them. [v-xieche]
#define OPTPIPELINETRAIN     //for no cuda // for top layer, Forward, Backward then Update. For other layers, Backward, Update, then Forward. [v-xieche]
//#define STRIPEDTOPLAYER    // for no cuda  // striped top layer.[v-xieche]
// #define TIMESTATS            //statistical the time distribution in each part of each layer. [v-xieche]
// #define DEBUGINFO_PIPELINETRAIN // output the debug infomation for pipeline training [v-xieche]

namespace msra { namespace dbn {

// helper to check CUDA state
static size_t numcudadevices()
{
    static int cudadevices = -1;    // -1 = unknown yet
    if (cudadevices == -1)
    {
        __try
        {
            cudadevices = (int) msra::cuda::getnumdevices();
            if (cudadevices == 0)
                fprintf (stderr, "numcudadevices: NUMA mode (no CUDA device found)\n");
            else
                fprintf (stderr, "numcudadevices: CUDA mode (%d CUDA devices found)\n", cudadevices);
        }
        __except (EXCEPTION_EXECUTE_HANDLER)
        {
            cudadevices = 0;
            fprintf (stderr, "numcudadevices: NUMA mode (cudamatrix.dll or underlying CUDA DLLs not installed)\n");
        }
    }
    return (size_t) cudadevices;
}

static bool hascuda() { return numcudadevices() > 0; }

// notes on parallelization across N CUDA cards
//
// Key matrix operations (v and h are frames stacked into matrix columns):
//  - h = W' v + a  |  sigmoid          // forwardprop()
//  - v = W h + b   |  sigmoid          // CD, backpropagationstats()
//  - dW += v * h'                      // updatedeltas()
//
// Striping assumption:
//  - N vertical stripes of W (that's N horiontal stripes of W')
//  - N horizontal stripes of a
//  - N horizontal stripes of b
//  - accordingly for dW, da, db
//
// Consequence:
//  - v is needed full-copy format
//  - h is needed horizontal stripes
//  - as things propagate, h becomes v and vice versa, potentially requiring conversion
//
// Operation: h = W' v + a  |  sigmoid  // forwardprop(), i.e. used everywhere
//  - input: full copy of v       --N times distribution overhead
//     - full copy needed after filling from host
//     - at end of CD, v is in horizontal stripes
//  - on each card compute horizontal stripe of (W'v+a) -> horizontal stripe of h
//  - apply sigmoid to horizontal stripes of h in-place   --may need to push down sigmoid through interface
//  - output: horizontal stripes of h
//
// Operation: v = W h + b  |  sigmoid   // pre-training, BP
//  - input: horizontal stripes of h            --no overhead
//  - compute N partial results of W h
//  - aggregate partial results and b through binary merge
//     - move half of data to other half of GPUs
//     - half goes from upper to lower, other half goes from lower to upper
//     - then merge (fully N-way parallel)
//     - in last merge add b (CD version only)
//     - now v is in horizontal stripes
//     - total moving cost: (N/2 reads + N/2 writes) * log N * size of v   --cheaper than distribution of v
//  - then take sigmoid
//  - output: horizontal stripes of v
//  - conversion:
//  - htov version (CD; has b) -> need to convert to full copy for forwardprop() and updatedeltas()
//  - ehtoev version (BP; no b) -> this is already in the format needed for next layer of BP
//
// Operation: dW += v * h'              // updatedeltas()
//  - input: full copy of v
//  - input: horizontal stripes of h
//  - compute vertical stripes of dW, keep them separate
//  - both inputs happen to be in correct format in all use cases (v is h. stripes in CD but turned to copies for forwardprop())
//
// Additional operations:
//  - sigmoid
//     - seems always on horizontal stripes, i.e. no overhead
//  - random sampling of h
//     - input will be in horizontal stripes
//     - required right before W h, i.e. in horizontal stripes
//     - causes issue with rand() (not compatible)
//  - mulbydsigm
//     - operates on horizontal stripes of h and eh   --no overhead
//  - scaleandaddallcols  (in updatedeltas())
//     - operates on da and db
//     - horizontal stripes are suitable, would need h and v in h. stripes
//        - v will be in full copy, which is a superset of stripes
//  - softmax   --to be parallelized as well
//     - would need vertical stripes... an otherwise never needed format
//
// Overall approach:
//  - state vectors
//     - two formats
//        - a full copy   (for v)
//        - a horizontal stripe (for h, v (temp during CD), eh, ev)
//           - this is consistent with higher-level striping, which is in time dimension=columns
//     - allocate full memory, but only use stripe
//        - that's a waste! And can be big. Optimize later.
//     - keep a 'valid range' variable (full memory or only stripe) for assertions only
//  - models
//     - two formats
//        - a horizontal stripe (for a, b and deltas)
//        - a vertical stripe (for W and delta)
//     - store the sub-range,reorigined to 0; functions know hard-coded whether it is horizontal or vertical
//     - keep a variable to store the actual patch offset
//  - conversion
//     - conversion only happens at entry of forwardprop() (horizontal stripes -> full copy)
//     - updatedeltas() will find it in the right format
//     - all functions are now well-defined in their input/output formats and know what to do, only checks needed, no lazy conversion


// class to hold a matrix possibly distributed over multiple CUDA devices
// Its methods may only be called in cudamode.

class cudadistributedmatrix
{
    cudadistributedmatrix (const cudadistributedmatrix &); void operator= (const cudadistributedmatrix &);

protected:
    const bool cudamode;                        // true if has CUDA hardware
    enum cudastriping_t
    {
        invalidstriping = -1,  // not determined yet
        notstriped = 0,        // maintains a full copy on each device
        stripedwrtrows = 1,    // striped w.r.t. first coordinate (row index)
        stripedwrtcols = 2     // striped w.r.t. second coordinate (col index)
    };
private:
    cudastriping_t cudastriping;                // how are the CUDA matrices striped
    cudastriping_t thisvalidstriping;           // 'notstriped' can be partially valid
    cudastriping_t & validstriping;             // we go through a reference so we get the right variable in a stripe
    size_t numrows, numcols;                    // overall dimensions of the underlying matrix
    std::vector<unique_ptr<msra::cuda::matrix> > cudamatrices;    // copies in CUDA space; validity not managed/checked in this class
    void checkcudastripingset() const { if (cudastriping == invalidstriping) throw std::logic_error ("checkcudastripingset: no CUDA striping set yet"); }
    void checkdeviceid (size_t deviceid) const { if (deviceid >= numcudadevices()) throw std::logic_error ("checkdeviceid: invalid CUDA device id"); }
    void alloccudamatrices() { if (cudamode) cudamatrices.resize (msra::dbn::numcudadevices());}
public:

    // needed for model matrices
    cudadistributedmatrix() : cudamode (hascuda()), cudastriping (invalidstriping), validstriping (thisvalidstriping), numrows (0), numcols (0) { alloccudamatrices(); }

    // construct from rvalue reference  --used when creating a stripe into an acceleratedmatrix
    cudadistributedmatrix (cudadistributedmatrix && other)
        : cudamode (hascuda()), cudastriping (other.cudastriping),
        thisvalidstriping ((&other.validstriping == &other.thisvalidstriping) ? other.validstriping : invalidstriping),
        validstriping ((&other.validstriping == &other.thisvalidstriping) ? thisvalidstriping : other.validstriping),   // keep external reference if it is one
        numrows (other.numrows), numcols (other.numcols), cudamatrices (std::move (other.cudamatrices))
    {
        assert (cudamode == other.cudamode);
        // 'other' will be destructed right after this, so no value in resetting the scalar values in it; cudamatrices[] is already cleared
    }

    // constructor for a column stripe (standalone; pushed into an acceleratedmatrix object by move constructor above)
    cudadistributedmatrix (cudadistributedmatrix & other, size_t firstframe, size_t numframes)
        : cudamode (hascuda()), cudastriping (other.cudastriping), thisvalidstriping (invalidstriping),
        validstriping (other.validstriping),    // this is a reference--we keep the reference to the input one
        numrows (other.numrows), numcols (numframes)
    {
        assert (cudamode == other.cudamode);
        if (!cudamode)
            return;

        // copy over striping and allocate devices' matrices
        alloccudamatrices();

        // set up stripes
        if (cudastriping == stripedwrtcols)
            throw std::logic_error ("cudadistributedmatrix: cannot construct a column stripe from a column-striped distributed matrix");
        foreach_index (i, cudamatrices) // note: patch copies the device as well
            cudamatrices[i].reset (other.cudamatrices[i]->patch (0, other.cudamatrices[i]->rows(), firstframe, numframes + firstframe));
    }

    // this can only be set once
    void setcudastriping (cudastriping_t s)
    {
        if (cudastriping != invalidstriping && cudastriping != s)
            throw std::logic_error ("setcudastriping: cannot change striping mode of a matrix once set");
        if (s == invalidstriping)
            throw std::logic_error ("setcudastriping: attempted to change striping mode to invalid");
        if (!cudamode)
            throw std::logic_error ("setcudastriping: cannot set striping mode if no CUDA device");
        if (cudamatrices.empty())
            throw std::logic_error ("setcudastriping: no CUDA device??");
        // TODO: strong exception guarantee: use local var + swap
        foreach_index (deviceid, cudamatrices)
        {
            cudamatrices[deviceid].reset (msra::cuda::newmatrix());
            cudamatrices[deviceid]->setdevice (deviceid);
        }
        cudastriping = s;
        validstriping = cudastriping;
    }

    // functions call this to verify striping
    void checkvalidstriping (cudastriping_t s) const
    {
        if (validstriping != s)
            throw std::logic_error ("checkvalidstriping: wrong striping mode or partially valid");
    }

    // verify that underlying striping is 's' and that it is valid
    void checkvalidcudastriping (cudastriping_t s) const
    {
        if (cudastriping != s)      // must have this type
            throw std::logic_error ("checkvalidcudastriping: wrong striping mode");
        checkvalidstriping (s);     // and be fully valid
    }

    // convert the striping mode if needed
    // Use for upgrading from row striping to full copy.
    // Do not use if you need a downgraded view only, use setinputstriping().
    template<class MATRIX> void makeinputstriping (cudastriping_t s, MATRIX & buffer)
    {
        // check compatibility
        if (s == notstriped && cudastriping != notstriped)
            throw std::logic_error ("makeinputstriping: attempted to upgrade valid striping mode for mismatching cudastriping mode");
        if (s != notstriped && cudastriping != s && cudastriping != notstriped)
            throw std::logic_error ("makeinputstriping: attempted to downgrade valid striping mode for mismatching cudastriping mode");
        // if upgrading then we actually do something
        const bool upgrading = (s == notstriped && validstriping != notstriped);
        const bool upgradingandneedtoactuallydosomething = upgrading && (numcudadevices() != 1);
#ifndef MULTICUDA1 // tmp code, need to be removed later.[v-xieche], here is for striped mode, think a smart way to handle it!!
        if (upgradingandneedtoactuallydosomething)
#else
        validstriping = stripedwrtrows;
#endif
        syncfromcuda (buffer, true);          // copy to CPU space (this copies stripes)
        validstriping = s;
#ifndef MULTICUDA1 // tmp code for striped on toplayer. need to be removed later. [v-xieche]
        if (upgradingandneedtoactuallydosomething)
#endif
            synctocuda (buffer, false);            // and copy back (this copies to all)
    }

    // sum up all device copies
    // Each device copy contains a partial matrix product of full dimension.
    // All devices' content needs to be summed up.
    template<class MATRIX> void sumacrossdevices (cudastriping_t s, MATRIX & buffer)
    {
        if (validstriping != notstriped)
            throw std::logic_error ("sumacrossdevices: can only be applied to 'notstriped' matrices");
        if (s == notstriped)
            throw std::logic_error ("sumacrossdevices: output format must be striped");
        if (numcudadevices() != 1)  // only one device: nothing to do
        {
            // TODO: Too bad, we cannot do without an additional second buffer.
            // This should be preallocated, but for now we do it locally here.

            // We process stripe by stripe and linearly add all other partial sums into the stripe's one.
            // Loop complexity: O((n-1)^2)    n=number of devices
            // Data complexity: O(n)

            // allocate a temp buffer in all of the devices and get a stripe view on each target
            std::vector<unique_ptr<msra::cuda::matrix> > ms (numcudadevices());
            std::vector<unique_ptr<msra::cuda::matrix> > targets (numcudadevices());
            for (size_t targetdevid = 0; targetdevid < numcudadevices(); targetdevid++)
            {
                // get size of this stripe
                size_t frdummy, fcdummy, nr, nc;
                devicedim (targetdevid, s, frdummy, fcdummy, nr, nc);

                // allocate a CUDA-side matrix to move partial sums from other devices into
                unique_ptr<msra::cuda::matrix> & m = ms[targetdevid];
                m.reset (msra::cuda::newmatrix());
                m->setdevice (targetdevid);
                m->allocate (nr, nc);

                // get local (0-based) views on this stripe
                targets[targetdevid] = stripeforcudadevice (targetdevid, s);
            }
            // now accumulate all stripes
            // get stripe from device (targetdevid + relrevid
            for (size_t reldevid = 1; reldevid < numcudadevices(); reldevid++)
            {
                // for each stripe, accumulate from device reldevid devices away
                // Target stripes live in different devices as well.

                // first get the respective data to accumulate
                // These are all in different devices.
                for (size_t targetdevid = 0; targetdevid < numcudadevices(); targetdevid++)
                {
                    // get patch coordinates of this stripe
                    size_t fr, fc, nr, nc;
                    devicedim (targetdevid, s, fr, fc, nr, nc);

                    const size_t sourcedevid = (targetdevid + reldevid) % numcudadevices();

                    // get the stripe into our local buffer variable
                    unique_ptr<msra::cuda::matrix> partial = stripeforcudadevice (cudamatrices[sourcedevid], targetdevid, s);
                    assert (buffer.rows() >= fr + nr && buffer.cols() >= fc + nc);
                    partial->fetch (0, nr, 0, nc, &buffer(fr,fc), buffer.getcolstride(), false);   // async
                }

                // accumulate stripe into device
                for (size_t targetdevid = 0; targetdevid < numcudadevices(); targetdevid++)
                {
                    // get a local (0-based) view on this stripe
                    unique_ptr<msra::cuda::matrix> & target = targets[targetdevid];

                    // get patch coordinates of this stripe
                    size_t fr, fc, nr, nc;
                    devicedim (targetdevid, s, fr, fc, nr, nc);

                    const size_t sourcedevid = (targetdevid + reldevid) % numcudadevices();

                    // our local buffer variable in the target device
                    unique_ptr<msra::cuda::matrix> & m = ms[targetdevid];

                    // wait until incoming transfer is done
                    cudamatrices[sourcedevid]->synchronize();	// this is where it came from

                    // move it to target device
                    m->assign (0, nr, 0, nc, &buffer(fr,fc), buffer.getcolstride(), false);

                    // accumulate it up
                    target->gems (1.0f, *m, 1.0f);
                }
            }
            // TODO: does free() at the end cause a sync?
        }
        // our output is now in stripe format
        setoutputstriping (stripedwrtrows);
    }

    // notify of partial validity
    // Use this to set the result type of an operation.
    void setoutputstriping (cudastriping_t s)
    {
        // check compatibility
        if (s == notstriped && cudastriping != notstriped)
            throw std::logic_error ("setoutputstriping: attempted to upgrade valid striping mode for mismatching cudastriping mode");
        if (s != notstriped && cudastriping != s && cudastriping != notstriped)
            throw std::logic_error ("setoutputstriping: attempted to downgrade valid striping mode for mismatching cudastriping mode");
        validstriping = s;
    }

    // for all per-element operations, mode must match and be non-disjunct
    void checkmatchingdisjunctcudastriping (const cudadistributedmatrix & othercols) const
    {
        if (cudastriping != othercols.cudastriping)
            throw std::logic_error ("checkmatchingdisjunctcudastriping: mismatching striping modes");
        checkdisjunctcudastriping();
    }

    // check if non-overlapping striping
    void checkdisjunctcudastriping() const
    {
        checkcudastripingset();
        if (cudastriping == notstriped && numcudadevices() > 1) // single device is OK as a compat mode; later remove that condition
            throw std::logic_error ("checkdisjunctcudastriping: an operation was used that is invalid for overlapping striping");
    }

    // allocate all parts  --note: empty matrix possible  --TODO: is resize() possible?
    void alloccuda (size_t n, size_t m)
    {
        checkcudastripingset();
        // TODO: exception guarantee?
        numrows = n;
        numcols = m;
        foreach_index (deviceid, cudamatrices)
        {
            size_t fr, fc, nr, nc;  // coordinates in CPU-side matrix
            devicedim (deviceid, cudastriping, fr, fc, nr, nc);
            cudamatrices[deviceid]->allocate (nr, nc);
        }
    }

    // determine the coordinate range of a stripe; or full if not striped
    void onedevicedim (const size_t deviceid, const bool isstriped, const size_t dim, size_t & first, size_t & subdim) const
    {
        if (isstriped)
        {
            const size_t n = numcudadevices();
            first = dim * deviceid / n;
            const size_t next = dim * (deviceid+1) / n;
            if (next > dim)
                throw std::logic_error ("onedevicedim: deviceid out of range");
            subdim = next - first;
        }
        else
        {
            first = 0;
            subdim = dim;
        }
    }
    // determine the patch coordinates into the full matrix for a given device and striping mode
    void devicedim (size_t deviceid, cudastriping_t s, size_t & fr, size_t & fc, size_t & nr, size_t & nc) const
    {
        onedevicedim (deviceid, s == stripedwrtrows, numrows, fr, nr);
        onedevicedim (deviceid, s == stripedwrtcols, numcols, fc, nc);
    }

    void synchronize (size_t deviceid)
    {
        cudamatrices[deviceid]->synchronize ();
    }

    void synchronize (std::vector<size_t> &deviceids) const
    {
        foreach_index (i, deviceids)
            cudamatrices[deviceids[i]]->synchronize();
    }

    // synchronize all devices after kicking off asynchronous data transfers with multiple
    void synchronize() const
    {
        foreach_index (deviceid, cudamatrices)
            cudamatrices[deviceid]->synchronize();
    }

    // copy a matrix to the distributed setup
    // This may copy in stripes or make full copies.
    // This operates on the valid striping.
    template<class MATRIX> void synctocuda (const MATRIX & m, bool synchronous)
    {
        checkcudastripingset();
        assert (numrows == m.rows() && numcols == m.cols());    // must have been allocated
        // copy to all devices
        // Depending on the mode, this can be overlapping or non-overlapping.
        foreach_index (deviceid, cudamatrices)
        {
            size_t cfr, cfc, cnr, cnc;  // stored portion of CPU-side matrix
            devicedim (deviceid, cudastriping, cfr, cfc, cnr, cnc);
            size_t vfr, vfc, vnr, vnc;  // valid portion in CPU-side matrix
            devicedim (deviceid, validstriping, vfr, vfc, vnr, vnc);
            if (vnr > 0 && vnc > 0)  // (if empty then m(.,.) may be invalid)
                cudamatrices[deviceid]->assign (vfr - cfr, vfr - cfr + vnr, vfc - cfc, vfc - cfc + vnc, &m(vfr,vfc), m.getcolstride(), false);
        }
        // wait until all transfers have completed (we hope they are in parallel)
        if (synchronous)
            synchronize();
    }

    // copy from devices
    // If 'notstriped' we assume all copies are identical and just copy the first.
    // This operates on the valid striping.
    template<class MATRIX> void syncfromcuda (MATRIX & m, bool synchronous) const
    {
        assert (numrows == m.rows() && numcols == m.cols());    // must have been allocated
        // if not striped then we have multiple copies--get the first one
        if (validstriping == notstriped)
            cudamatrices[0]->fetch (0, numrows, 0, numcols, const_cast<float*> (&m(0,0)), m.getcolstride(), false);
        // copy from all devices
        else foreach_index (deviceid, cudamatrices)
        {
            size_t cfr, cfc, cnr, cnc;  // stored portion of CPU-side matrix
            devicedim (deviceid, cudastriping, cfr, cfc, cnr, cnc);
            size_t vfr, vfc, vnr, vnc;  // valid portion in CPU-side matrix
            devicedim (deviceid, validstriping, vfr, vfc, vnr, vnc);
            if (vnr > 0 && vnc > 0)  // (if empty then m(.,.) may be invalid)
            {

                cudamatrices[deviceid]->fetch (vfr - cfr, vfr - cfr + vnr, vfc - cfc, vfc - cfc + vnc, const_cast<float*> (&m(vfr,vfc)), m.getcolstride(), false);
            }
        }
        // wait until all transfers have completed (we hope they are in parallel)
        if (synchronous)
            synchronize();
    }


    template<class MATRIX> void syncfromcuda (MATRIX &m, bool synchronous, std::vector<size_t> &deviceids) const
    {
        assert (numrows == m.rows() && numcols == m.cols());    // must have been allocated
        // if not striped then we have multiple copies--get the first one
        if (validstriping == notstriped)
        {
            fprintf (stderr, "Could touch to here ? debug point[v-xieche]!\n");
            cudamatrices[deviceids[0]]->fetch (0, numrows, 0, numcols, const_cast<float*> (&m(0,0)), m.getcolstride(), false);
        }
        // copy from all devices
        else foreach_index (i, deviceids)
        {
            size_t cfr, cfc, cnr, cnc;  // stored portion of CPU-side matrix
            devicedim (deviceids[i], cudastriping, cfr, cfc, cnr, cnc, deviceids.size());
            size_t vfr, vfc, vnr, vnc;  // valid portion in CPU-side matrix
            devicedim (deviceids[i], validstriping, vfr, vfc, vnr, vnc, deviceids.size());
            if (vnr > 0 && vnc > 0)  // (if empty then m(.,.) may be invalid)
                cudamatrices[deviceids[i]]->fetch (vfr - cfr, vfr - cfr + vnr, vfc - cfc, vfc - cfc + vnc, const_cast<float*> (&m(vfr,vfc)), m.getcolstride(), false);
        }
        // wait until all transfers have completed (we hope they are in parallel)
        // wait until all transfers have completed (we hope they are in parallel
        if (synchronous)
            synchronize();

    }

    template<class MATRIX> void synctocuda (const MATRIX & m, bool synchronous, std::vector<size_t> &deviceids)
    {
        checkcudastripingset();
        assert (numrows == m.rows() && numcols == m.cols());    // must have been allocated
        // copy to all devices
        // Depending on the mode, this can be overlapping or non-overlapping.
        foreach_index (i, deviceids)
        {
            size_t cfr, cfc, cnr, cnc;  // stored portion of CPU-side matrix
            devicedim (i, cudastriping, cfr, cfc, cnr, cnc);
            size_t vfr, vfc, vnr, vnc;  // valid portion in CPU-side matrix
            devicedim (i, validstriping, vfr, vfc, vnr, vnc);
            if (vnr > 0 && vnc > 0)  // (if empty then m(.,.) may be invalid)
                cudamatrices[deviceids[i]]->assign (vfr - cfr, vfr - cfr + vnr, vfc - cfc, vfc - cfc + vnc, &m(vfr,vfc), m.getcolstride(), false);
        }
        // wait until all transfers have completed (we hope they are in parallel)
        if (synchronous)
            synchronize();
    }

    // used for compacttrainer, we don't need striped type and each cuda device keeps a full copy data.
    template<class MATRIX> void syncfromcuda (MATRIX & m, bool synchronous, size_t deviceid) const
    {
        assert (numrows == m.rows() && numcols == m.cols());    // must have been allocated
        // if not striped then we have multiple copies--get the first one
        cudamatrices[deviceid]->fetch (0, numrows, 0, numcols, const_cast<float*> (&m(0,0)), m.getcolstride(), false);
        // wait until all transfers have completed (we hope they are in parallel)
        //if (synchronous)
        //    synchronize();
        cudamatrices[deviceid]->synchronize ();
    }

    template<class MATRIX> void synctocuda (const MATRIX & m, bool synchronous, size_t deviceid)
    {
        assert (numrows == m.rows() && numcols == m.cols());    // must have been allocated
        // copy to all devices
        // Depending on the mode, this can be overlapping or non-overlapping.
        if (numrows > 0 && numcols > 0)
            cudamatrices[deviceid]->assign (0, numrows, 0, numcols, &m(0,0), m.getcolstride(), false);
        // wait until all transfers have completed (we hope they are in parallel)
        if (synchronous)
            synchronize(deviceid);
    }


    // access to the stripes
    size_t numcudadevices() const { checkcudastripingset(); return cudamatrices.size(); }

    // CUDA matrix/matrix stripe for a specific device (if 'notstriped' then these are overlapping)
    // TODO: this should return a patch if our view is different from the base type
    // ... two functions: one operating on the valid view, and one operating on a sub-view if requested (returning a unique_ptr with a patch--done!)
    // get stripe for specific device in its base form (must be fully valid)
    // Takes a cudastriping_t which is checked but will not trigger any transform.
    msra::cuda::matrix &       forcudadevice (size_t deviceid, cudastriping_t s)       { checkvalidcudastriping (s); checkdeviceid (deviceid); return *cudamatrices[deviceid].get(); }
    const msra::cuda::matrix & forcudadevice (size_t deviceid, cudastriping_t s) const { checkvalidcudastriping (s); checkdeviceid (deviceid); return *cudamatrices[deviceid].get(); }

    // no striping parameter --view in its base form (must be fully valid). Used for models, where all that matters is consistent striping.
    msra::cuda::matrix &       forcudadevice (size_t deviceid)       { checkvalidstriping (cudastriping); checkdeviceid (deviceid); return *cudamatrices[deviceid].get(); }
    const msra::cuda::matrix & forcudadevice (size_t deviceid) const { checkvalidstriping (cudastriping); checkdeviceid (deviceid); return *cudamatrices[deviceid].get(); }

    // get stripe view for specific device, where the base type may be 'notstriped'
    // These return a newly created patch of the underlying matrix. It works for all combinations, although really intended for viewing a 'notstriped' as a striped matrix.
    //msra::cuda::matrix * makeselfpatch (msra::cuda::matrix * m) { return m->patch (0, m->rows(), 0, m->cols()); }
    unique_ptr<msra::cuda::matrix> stripeforcudadevice (unique_ptr<msra::cuda::matrix> & m, size_t deviceid, cudastriping_t s)
    {
        assert (s != invalidstriping);
        checkdeviceid (deviceid);
        if (validstriping != s && validstriping != notstriped)
            throw std::logic_error ("stripeforcudadevice: invalid striping mode");
        // view == full view (either full view or a stripe)
        if (s == cudastriping)
            return unique_ptr<msra::cuda::matrix> (m->patch (0, m->rows(), 0, m->cols()));  // full view
        // view is sub-view  --base format must be 'notstriped'
        assert (cudastriping == notstriped && s != notstriped);
        size_t fr, fc, nr, nc;  // coordinates into full matrix
        devicedim (deviceid, s, fr, fc, nr, nc);
        return unique_ptr<msra::cuda::matrix> (m->patch (fr, fr + nr, fc, fc + nc));
    }
    unique_ptr<msra::cuda::matrix> stripeforcudadevice (size_t deviceid, cudastriping_t s)
    {
        return stripeforcudadevice (cudamatrices[deviceid], deviceid, s);
    }
    const unique_ptr<msra::cuda::matrix> stripeforcudadevice (size_t deviceid, cudastriping_t s) const { return const_cast<cudadistributedmatrix*> (this)->stripeforcudadevice (deviceid, s); }

    // compat mode  --delete, then see what fails and fix it
    //msra::cuda::matrix &       forcuda()       { checkcudastripingset(); if (numcudadevices() != 1) throw std::runtime_error ("forcuda: compat mode only allowed for a single GPU"); return *cudamatrices[0].get(); }
    //const msra::cuda::matrix & forcuda() const { checkcudastripingset(); if (numcudadevices() != 1) throw std::runtime_error ("forcuda: compat mode only allowed for a single GPU"); return *cudamatrices[0].get(); }

    // (diagnostics only)
    std::string cudastripingtostr (bool wantvalid) const
    {
        switch (wantvalid ? validstriping : cudastriping)
        {
        case invalidstriping: return "invalidstriping";
        case notstriped: return "notstriped";
        case stripedwrtrows: return "stripedwrtrows";
        case stripedwrtcols: return "stripedwrtcols";
        default: return "(cudastripingtostr: invalid striping value--oops?)";
        }
    }
};


// a matrix wrapper for accelerated matrix computation
//  - CUDA support:
//    - provides access to an array of CUDA matrices, one per device; can be a sub-stripe or full
//    - provides functions for moving data back and forth; they know to handle stripes
//    - provides a function to view a full copy as a stripe (it's a superset)
//    - restricts access to underlying CPU-side functions to classes that derive from this
//    - no caching or state tracking in this class, i.e. caller must avoid redundant data moves
//  - NUMA support:
//     - TODO: absorb cachedmatrix class into here entirely
//  - template argument can be matrix (with allocation) or matrixstriperef (no allocation)
template<class matrixbase> class acceleratedmatrixbase : protected matrixbase, public cudadistributedmatrix
{
protected:
	
    // move data to/from CUDA if in CUDA mode
    void alloccuda()
    {
        if (cudamode)               // allocate matrix memory in CUDA space (note: can be an empty matrix)
            cudadistributedmatrix::alloccuda (rows(), cols());
    }
    friend class cudadistributedmatrix; // needs to access operator() and getcolstride() for syncto/fromcuda()
    void synctocuda (bool synchronously)
    {
        if (cudamode)
        {
            alloccuda();            // dimensions may have changed
            cudadistributedmatrix::synctocuda (*this, synchronously);
        }
    }
    void syncfromcuda (bool synchronously) const       // consider CPU-side copy mutable
    {
        if (cudamode)
            cudadistributedmatrix::syncfromcuda (*this, synchronously);
    }

    // (see cudadistributedmatrix::makeinputstriping() for description)
    void makeinputstriping (cudastriping_t s) { cudadistributedmatrix::makeinputstriping (s, *this); }    // we pass our CPU-side matrix as the buffer
    // merge partial sums (see cudadistributedmatrix for description)
    void sumacrossdevices (cudastriping_t s) { cudadistributedmatrix::sumacrossdevices (s, *this); }    // we pass our CPU-side matrix as the buffer
	
    // default constructor used in model matrix
    acceleratedmatrixbase() {}

    // used during construction only
    void resize (size_t n, size_t m)
    {
        matrix::resize (n, m);  // (only resizes CPU-side object)
        alloccuda();            // allocate CUDA-side object if in CUDA mode
    }
    // move constructor used with stripe()
    acceleratedmatrixbase (acceleratedmatrixbase && other) : matrixbase (std::move (other))/*, cudadistributedmatrix (std::move (other))*/ {}

    // constructor for use with stripe()
	
    acceleratedmatrixbase (matrixbase && otherm, cudadistributedmatrix && otherc)
        : matrixbase (std::move (otherm)), cudadistributedmatrix (std::move (otherc)) {}
		
public:
    size_t rows() const { return matrixbase::rows(); }
    size_t cols() const { return matrixbase::cols(); }
    size_t colstride() const { return matrixbase::getcolstride(); }
    void reshape(const size_t newrows, const size_t newcols) { matrixbase::reshape(newrows, newcols);};
    bool empty() const { return matrixbase::empty(); }


    // partial matrix dump 
    void glimpse() const
    {
        const auto & us = *this;
        for (size_t i = 0; i < rows(); i++)
        {
            if (rows() > 6 && i == 3)
            {
                i = rows() - 3;
                fprintf (stderr, "  ...\n");
            }
            fprintf (stderr, "%c", i == 0 ? '(' : ' ');
            for (size_t j = 0; j < cols(); j++)
            {
                if (cols() > 6 && j == 4)
                {
                    j = cols() - 3;
                    fprintf (stderr, " ...");
                }
                fprintf (stderr, " %7.3f", us(i,j));
            }
            if (i == rows() -1)
                fprintf (stderr, " )");
            fprintf (stderr, "\n");
        }
    }

    // debug function to dump the value of the matrix out
    // Only dumps the first and last 3 elements in each dimension.
    // Note that this interferes in that it syncs the matrix back. If this changes results, then we also learned something.
    void glimpse (const char * name) const
    {
        fprintf (stderr, "### glimps at %s:     %d devices, valid=%s\n", name, numcudadevices(), cudastripingtostr (true).c_str());
        syncfromcuda();
        matrixbase::glimpse();
    }
};

// a reference RBM state vectors (including a sub-range) that reside in a rbmstatevectorsbase object
//  - implements all high-level operations needed inside the RBM class
//     - CUDA mode: these objects live entirely on the CUDA side; all operations happen inside CUDA
//     - NUMA mode: optimized parallelized (multi-threaded) matrix product implemented in this class
// TODO: this is a non-CUDA 'compatibility' implementation that exposes the underlying non-accelerated stripe
// TODO: move ALL computation to CUDA (!!)
// TODO: Also ensure that calls to fornuma() are only used when entering actual CPU-side computation.
// TODO: All other fornuma() calls (which are in CUDA mode) must be replaced by CUDA-side computation.
// TODO: Then we are done with CUDA-side computation.

template<class matrixbase> class rbmstatevectorsrefbase : public acceleratedmatrixbase<msra::math::ssematrixstriperef<matrixbase>>
{
    typedef acceleratedmatrixbase<msra::math::ssematrixstriperef<matrixbase>> acceleratedmatrix;
    rbmstatevectorsrefbase(){}

    mutable bool lockedforreading, lockedforwriting;
    static void failwithbadstate() { throw std::logic_error ("checklockstate: a rbmstatevectorsbase function was called in wrong state"); }
    static void checklockstate (bool lockstate) { if (!lockstate) failwithbadstate(); }
    static void checknotlockstate (bool lockstate) { if (lockstate) failwithbadstate(); }
    void checklockedforwriting() const { checklockstate (lockedforwriting); }
    void checknotlockedforwriting() const { checknotlockstate (lockedforwriting); }
    void checklockedforreading() const { checklockstate (lockedforreading); }
    void checklocked() const { checklockstate (lockedforreading || lockedforwriting); }
    void checkunlocked() const { checknotlockstate (lockedforreading || lockedforwriting); }
protected:
#if 1  // just for target propagation to use data in cuda device [v-xieche]
public:
#endif
    // lock/unlock for direct access of CPU-side ssematrix object
    // This is used to control the moving of data from/to CUDA.
    // Call lock() to be allowed to access operator(), and unlock() when done.
    void lockforreading() const // lockstate is mutable  --this allows this call on 'const' objects
    {
        checkunlocked();
        lockedforreading = true;
        syncfromcuda (true);

    }
    void lockforwriting()
    {
        checkunlocked();
        lockedforwriting = true;
        synchronize();	// make sure there is no copy-back action ongoing that we could interfere with
        // we will create this object --so no need to copy it from CUDA
    }

    void lockforreadwrite()
    {
        lockforreading();
        lockedforwriting = true;    // causes it to sync back at the end
    }

    void unlock()
    {
        checklocked();
        if (lockedforwriting)       // if locked for writing then need to copy data to CUDA
        synctocuda (false);

        lockedforreading = false;
        lockedforwriting = false;
    }
    void unlock() const
    {
        checklockedforreading();
        checknotlockedforwriting();
        lockedforreading = false;
    }
public:
    // to allow construction from rbmstatevectors::stripe()
    rbmstatevectorsrefbase (rbmstatevectorsrefbase && other) : acceleratedmatrix (std::move (other)), lockedforreading (false), lockedforwriting (false) { other.checkunlocked(); }

    // constructor for rbmstatevectors::stripe()
    // note: input matrix may be empty -> returns an empty matrix (will fail if ever accessed)
	
    rbmstatevectorsrefbase (msra::math::ssematrixstriperef<matrixbase> && stripem, cudadistributedmatrix && stripec)
        : acceleratedmatrix (std::move (stripem), std::move (stripec)), lockedforreading (false), lockedforwriting (false) { }
	
    // get underlying CPU-side matrixbase object, for NUMA operation (don't use with CUDA mode)
    msra::math::ssematrixstriperef<matrixbase> &       fornuma()       { checkunlocked(); return *this; }
    const msra::math::ssematrixstriperef<matrixbase> & fornuma() const { checkunlocked(); return *this; }

    // temp compat functions
    msra::math::ssematrixstriperef<matrixbase> &       fromcuda()       { checkunlocked(); checkcudadims(); syncfromcuda(); return *this; }
    const msra::math::ssematrixstriperef<matrixbase> & fromcuda() const { checkunlocked(); checkcudadims(); syncfromcuda(); return *this; }
    void tocuda() { checkunlocked(); checkcudadims(); synctocuda(); }

    // get underlying CUDA object
    //msra::cuda::matrix &       forcuda()       { checkunlocked(); return acceleratedmatrix::forcuda(); }
    //const msra::cuda::matrix & forcuda() const { checkunlocked(); return acceleratedmatrix::forcuda(); }

    // (see cudadistributedmatrix::makeinputstriping() for description)
	
    void makeinputstriping (cudastriping_t s)  { acceleratedmatrixbase::makeinputstriping (s); }
    void makeinputstriping (cudastriping_t s) const
    {
        // TODO: decide what is const and what is mutable...
        rbmstatevectorsrefbase * us = const_cast<rbmstatevectorsrefbase*> (this);
        us->makeinputstriping (s);
    }
	
    // merge partial sums
	
    void sumacrossdevices (cudastriping_t s) { acceleratedmatrixbase::sumacrossdevices (s); }
	
    // get underlying NUMA-cached object
    // TODO

    float &       operator() (size_t i, size_t j)       { checklockedforwriting(); return acceleratedmatrixbase::operator() (i, j); }
    const float & operator() (size_t i, size_t j) const { checklocked(); return acceleratedmatrixbase::operator() (i, j); }

    // operations
    bool hasnan (const char * name) const { checklocked(); return acceleratedmatrix::hasnan (name); }

    // this = sigmoid (this)
    void sigmoid()
    {
            auto & us = *this;
            us.lockforreadwrite();
            foreach_coord (i, j, us)
            {
                float exponent = us(i,j);
                if (exponent < -30.0f)
                    us(i,j) = 0.0f;
                else
                    us(i,j) = 1.0f / (1.0f + expf (-exponent));
            }
            checknan (us);
            us.unlock();
        
    }

    // bianarize the matrix using 0.5 as the threshold [v-xieche]
    void binarize ()
    {
        auto & us = *this;
        us.lockforreadwrite ();
        fprintf (stderr, "bianarize the output of hidden layer when updating top layer..\n");
        foreach_coord (i, j, us)
        {
            if(us(i, j) > 0.5)  us(i, j) = 1.0;
            else   us(i, j) = 0.0;
        }
        us.unlock ();
    }

    void setto0ifbelow (const float value)
    {
        auto & us = *this;
        us.lockforreadwrite ();
        fprintf (stderr, "set the output of hidden layer when value below to %.4f..\n", value);
        size_t counter = 0;
        size_t hitnum = 0;
        foreach_coord (i, j, us)
        {
            counter ++;
            if (us(i,j) < value) 
            {
                us(i,j) = 0;
                hitnum ++;
            }
        }
        float ratio = float(100.0) * hitnum / counter;
        fprintf (stderr, "%.2f%% set to be zero.\n", ratio);  
        us.unlock ();
    }


    // add epision in the output of sigmoid, then exert log funtion on it. log (us + epison). [v-xieche]
    void addepisonlog ()  
    {
        auto & us = *this;
        us.lockforreadwrite ();
        foreach_coord (i, j, us)
            us (i, j) = (float) log (us(i,j) + EPISONFORLOG);
        us.unlock ();
    }

    void getorisigmoid ()  // calculate the sigmoid function from log ( epison + s(z) ). [v-xieche]
    {
        auto & us = *this;
        us.lockforreadwrite ();
        foreach_coord (i, j, us)
            us(i, j) = (float) (exp(us(i, j)) - EPISONFORLOG);
        us.unlock();
    }


    // special function for RBM pre-training: sample binary values according to probability
    // P is the probability that the bit should be 1.
    void samplebinary (const rbmstatevectorsrefbase & P, unsigned int randomseed)
    {
            auto & res = *this;
            if (&P != &res)
            {
                P.lockforreading();
                res.lockforwriting();
            }
            else
                res.lockforreadwrite();     // in-place
            foreach_column (t, res)
            {
                srand (randomseed + (unsigned int) t);  // (note: srand() is thread-safe)
                //fprintf (stderr, "samplebinary: seed = %d\n", randomseed + (unsigned int) t);
                foreach_row (i, res)
                {
                    float randval = rand() / (float) RAND_MAX;
                    float bit = randval < P(i,t) ? 1.0f : 0.0f;
                    res(i,t) = bit;
                }
            }
            res.unlock();
            if (&P != &res) // (otherwise both are the same, we read-write lock only once)
                P.unlock();
        
    }

private:

    double colvecsum()  // sum over all elements of a column vector
    {
        const rbmstatevectorsrefbase & us = *this;
        assert (us.cols() == 1);
        us.lockforreading();
        double sum = 0.0;
        foreach_row (i, us)
            sum += us[i];
        us.unlock();
        return sum;
    }
    double rowvecsum()  // sum over all elements of a row vector
    {
        const rbmstatevectorsrefbase & us = *this;
        assert (us.rows() == 1);
        us.lockforreading();
        double sum = 0.0;
        foreach_column (j, us)
            sum += us(0,j);
        us.unlock();
        return sum;
    }

    size_t numpositive() // count number of elements > 0 in a matrix
    {
        const rbmstatevectorsrefbase & us = *this;
        us.lockforreading();
        size_t sum = 0;
        foreach_coord (i, j, us)
            if (us(i,j) > 0)
                sum++;
        us.unlock();
        return sum;
    }

public:

    // statistics for pre-training
    // Measures reconstruction likelihood of 'this' against 'v1' (v1=reconstructed)
    // This uses temp vectors glogllsums and logllsums to compute it independently for each node.
    // We compute both in all cases because Dong's original metric was the Gaussian one in either case.
    // TODO: Reduce to one metric, and take a 'gaussian' flag.
    // Then it sums up the values. This is to allow the main computation to run in CUDA space.
    // It did otherwise consume the same amount of time as all the remaining computation!
    // TODO: change return values from -sum to av-
    void llstats (const rbmstatevectorsrefbase & v1, rbmstatevectorsrefbase & glogllsums, rbmstatevectorsrefbase & logllsums, double & /*out*/glogllsum, double & /*out*/logllsum) const
    {
        const rbmstatevectorsrefbase & v = *this;

        assert (v.rows() == v1.rows() && v.cols() == v1.cols());
        assert (glogllsums.rows() == v.rows() && glogllsums.cols() == 1);
        assert (logllsums.rows() == v.rows() && logllsums.cols() == 1);

        // first compute it per node, summing up over frames
        // (This strange split is done to allow for CUDA acceleration.)
            v.lockforreading();
            v1.lockforreading();
            logllsums.lockforwriting();
            glogllsums.lockforwriting();

            checknan (v);
            checknan (v1);

            foreach_column (t, v)
            {
                // gaussian: we compute the Gaussian metric also for binary units, as it seems to be common...
                foreach_row (i, v)
                {
                    if (t == 0)
                        glogllsums[i] = 0.0;
                    double diff = v(i,t) - v1(i,t);
                    double glogll = -0.5 * diff * diff;         // note that we assume unit variance
                    // We normalize against the 'perfect reconstruction' hypothesis (diff == 0)
                    // thus the Gaussian normalization factor (1/sqrt (2.0 * M_PI)) cancels out.
                    glogllsums[i] += (float) glogll;
                }
                //fprintf (stderr, "llstat: glogll[%3d] = %7.4f\t   ->\t%7.4f\t%7.4f\t%7.4f\t  vs.\t%7.4f\t%7.4f\t%7.4f ...\n", t, glogll, v(0,t), v(1,t), v(2,t), v1(0,t), v1(1,t), v1(2,t));

                // binary: expected log prob of reconstruction, expectation over input data
                foreach_row (i, v)
                {
                    if (t == 0)
                        logllsums[i] = 0.0;
                    double Pv = v(i,t);     // prob of v being 1
                    if (Pv < 0.000001) Pv = 0.000001;   // to be sure (not observed)
                    if (Pv > 0.999999) Pv = 0.999999;
                    double Pv1 = v1(i,t);   // prob of v1 being 1
                    if (Pv1 < 0.000001) Pv1 = 0.000001;   // we do see 1.0
                    if (Pv1 > 0.999999) Pv1 = 0.999999;
                    double logll = Pv * log (Pv1) + (1 - Pv) * log (1 - Pv1);
                    // normalize against perfect reconstruction hypothesis for better readability
                    logll -= Pv * log (Pv) + (1 - Pv) * log (1 - Pv);
                    logllsums(i,0) += (float) logll;
                }
            }
            glogllsums.unlock();
            logllsums.unlock();
            v.unlock();
            v1.unlock();
        

        // compute the sum
        logllsum = logllsums.colvecsum();
        glogllsum = glogllsums.colvecsum(); // Gaussian (also for binary units, for diagnostics)
    }

    // special function for backprop: multiply error vector by derivative of sigmoid function.
    //   err = eh .* h .* (1 - h)
    //   h .* (1 - h) = derivative of sigmoid
    // where err and eh are the same variable (updated in place)
    // We leverage that the derivative can be computed from values of the sigmoid function in 'sigm' cheaply.
    void mulbydsigm (const rbmstatevectorsrefbase & sigm)
    {
        if (cudamode)
        {
            this->setoutputstriping (stripedwrtrows);
            // process stripe by stripe
            for (size_t deviceid = 0; deviceid < numcudadevices(); deviceid++)
                this->stripeforcudadevice (deviceid, stripedwrtrows)->mulbydsigm (*sigm.stripeforcudadevice (deviceid, stripedwrtrows));
        }
        else
        {
            auto & eh = *this;
            eh.lockforreadwrite();
            sigm.lockforreading();
            // suggestion by Fahlman (cf. Quickprop algorithm): add 0.1 to derivative to avoid flat tails
            // This does not seem to converge faster, but rather more slowly. Maybe I need to let it run longer.
#if 0
            foreach_coord (i, t, eh)
                eh(i,t) *= (0.1f + 0.9f * sigm(i,t) * (1.0f - sigm(i,t)));
#else
            foreach_coord (i, t, eh)
                eh(i,t) *= sigm(i,t) * (1.0f - sigm(i,t));
#endif
            sigm.unlock();
            eh.unlock();
        }
    }

    void KhatriRaoProduct(const rbmstatevectorsrefbase & m1, const rbmstatevectorsrefbase & m2)
    {
        assert(m1.cols() == m2.cols() && cols() == m1.cols());

        if (cudamode)
        {
            this->setoutputstriping (stripedwrtcols);
            // process stripe by stripe
            for (size_t deviceid = 0; deviceid < numcudadevices(); deviceid++)
                this->stripeforcudadevice (deviceid, stripedwrtcols)->KhatriRaoProduct (*m1.stripeforcudadevice (deviceid, stripedwrtrows), *m2.stripeforcudadevice (deviceid, stripedwrtrows));
        }
        else
        {
            auto & us = *this;
            us.lockforreadwrite();
            m1.lockforreading();
            m2.lockforreading();

            matrixbase::KhatriRaoProduct(m1, m2);

            m1.unlock();
            m2.unlock();
            us.unlock();
        }
    }
    //   this = reshape each column of eh from (K1xK2,1) to (K1, K2) and times each column of h (K2, frames).
    //   the output is a (K1, frames) matrix
    //   eh can be transposed.
    //   used for tensor DNN
    void reshapecolumnproduct (const rbmstatevectorsrefbase & eh, const rbmstatevectorsrefbase & h, const bool isehtransposed)
    {
        assert(eh.cols() == h.cols() && cols() == h.cols());
        assert (eh.rows() == h.rows() * rows());

        if (cudamode)
        {
            this->setoutputstriping (stripedwrtcols);
            // process stripe by stripe
            for (size_t deviceid = 0; deviceid < numcudadevices(); deviceid++)
                this->stripeforcudadevice (deviceid, stripedwrtcols)->reshapecolumnproduct (*eh.stripeforcudadevice (deviceid, stripedwrtrows), *h.stripeforcudadevice (deviceid, stripedwrtrows), isehtransposed);
        }
        else
        {
            auto & hnew = *this;
            hnew.lockforreadwrite();
            eh.lockforreading();
            h.lockforreading();

            matrixbase::reshapecolumnproduct(eh, h, isehtransposed);

            h.unlock();
            eh.unlock();
            hnew.unlock();
        }
    }

    // function: us = us / log (h + sigmoid) [v-xieche]
    void divideaddsigmoid (const rbmstatevectorsrefbase & sigm)
    {
        auto & us = *this;
        us.lockforreadwrite ();
        sigm.lockforreading ();
        foreach_coord (i, t, us)   
            us (i,t) /= (float)  (EPISONFORLOG + sigm(i, t));   // it now be s(z) + epison, don't need to add log again.
        us.unlock ();
        sigm.unlock ();
    }

    // this = softmax (this)
    void softmax()
    {
            auto & us = *this;
            us.lockforreadwrite();
            foreach_column (j, us)
            {
                // find max (to bring exp() into comfortable range)
                float colmax = 0.0f;
                foreach_row (i, us)
                {
                    float usij = us(i,j);
                    if (usij > colmax)
                        colmax = usij;
                }
                // sum
                // we divide by exp (colmax), which will cancel out when we normalize below
                double sum = 0.0;
                foreach_row (i, us)
                {
                    float usexp = exp (us(i,j)-colmax);
                    us(i,j) = usexp;
                    sum += usexp;
                }
                // normalize
                float sumf = float (sum);
                foreach_row (i, us)
                {
                    us(i,j) /= sumf;
                }
            }
            checknan (us);
            us.unlock();
        
    }
	//[v-wenh] added for rectifier linear 
	float leakyroot(float x, size_t rootorder, float leakiness)
	{
		float ax = fabs(x);
		float y = pow(ax + 1, 1.0f / rootorder) - 1;
		if (x <= 0.0f)
			y *= -leakiness;
		return y;
	}
	float dleakyroot(float y, size_t rootorder, float leakiness)
	{
		float ay = y;           // reconstruct the positive y (before applying sign)
		if (y <= 0.0f)
			ay /= -leakiness;
		// derivative of f(x) = (x+1)^1/r is
		// df/dx = 1/r * (x+1)^(1/r-1)
#if 0
		float x = pow(ay + 1, rootorder) - 1;   // now we know the x, and it is non-negative; we can compute the derivative now
		float d = pow(x + 1, 1.0f / rootorder - 1.0f) / rootorder;
#else
		float d = pow(ay + 1, 1.0f - rootorder) / rootorder;
#endif
		if (y <= 0.0f)
			d *= leakiness;
		return d;
	}

	void leakyroot(size_t rootorder, float leakiness)
	{
		
		auto & us = *this;
		us.lockforreadwrite();
		foreach_coord(i, t, us)
			us(i, t) = leakyroot(us(i, t), rootorder, leakiness);
		checknan(us);
		us.unlock();
		
	}


	void mulbydleakyroot(const rbmstatevectorsrefbase & lruvals, size_t rootorder, float leakiness)
	{
		auto & eh = *this;
		eh.lockforreadwrite();
		lruvals.lockforreading();
		foreach_coord(i, t, eh)
			eh(i, t) = eh(i, t) * dleakyroot(lruvals(i, t), rootorder, leakiness);  // err = eh .* derivative
		lruvals.unlock();
		eh.unlock();
	}

	void mulbydlru(const rbmstatevectorsrefbase & reluz)
	{
		
		auto & eh = *this;
		eh.lockforreadwrite();
		reluz.lockforreading();
		foreach_coord(i, t, eh)
			if (reluz(i, t) <= 0.0f)  // err = eh .* (h > 0) in place
				eh(i, t) = 0.0f;
		reluz.unlock();
		eh.unlock();
		
	}

    // this = delta((Pu(i,t)==uids[t]) - Pu
    // What this is: (log softmax)'_j(z) = delta(s(t)==j)-softmax_j(z)
    // This is the error of the top layer, the signal being back-propagated.
    void seterrorsignal (const rbmstatevectorsrefbase & uids, const rbmstatevectorsrefbase & Pu)
    {
        // experimental pruning mode (disabled by default)  --note: pruning currently hurts bigtime. Need to normalize?
        float pruningbeam = 0.0f;   // off the best (not considering correct state)
            auto & us = *this;
            assert (cols() == uids.cols() && cols() == Pu.cols());
            assert (rows() == Pu.rows() && uids.rows() == 1);

            us.lockforwriting();
            uids.lockforreading();
            Pu.lockforreading();

            size_t numpruned = 0;
            foreach_column (t, us)
            {
                // pruning: we determine the maximum Pu(i,t), not considering the correct state.
                float maxPu_t = 0.0f;
                if (pruningbeam > 0)
                {
                    foreach_row (i, us)
                    {
                        const size_t uid = (size_t) uids(0,t);
                        if (i == uid)   // we only consider competitors
                            continue;
                        if (Pu(i,t) > maxPu_t)
                            maxPu_t = Pu(i,t);
                    }
                }

                // set the target weights for this frame
                // Each row of W will be nudged towards v(t) by the target weight.
                // For the correct state, the target weight is 1.0-P, otherwise -P, i.e.
                // for incorrect states, rows will be gently nudged away from v(t).
                // Setting the target weight to 0 will leave the row of W unchanged.
                foreach_row (i, us)
                {
                    const size_t uid = (size_t) uids(0,t);
                    if ((float) uid != uids(0,t))
                        throw std::runtime_error ("seterrorsignal: uids not integer!");
                    const float utarget_it = (i == uid) ? 1.0f : 0.0f;
                    us(i,t) = utarget_it - Pu(i,t);

                    // do the pruning
                    // We always nudge towards the correct state, but exclude low-scoring competitors.
                    if (i != uid && Pu(i,t) < maxPu_t * pruningbeam)
                    {
                        us(i,t) = 0.0f; // setting it to 0 will exclude it from having an effect
                        numpruned++;
                    }
                }
            }

            if (numpruned > 0)
                fprintf (stderr, "seterrorsignal: %d out of %d pruned (%.2f%%) (pruningbeam param = %.2f)\n",
                numpruned, us.rows()*us.cols(), 100.0 * numpruned / (us.rows()*us.cols()), pruningbeam);

            Pu.unlock();
            uids.unlock();
            checknan (us);
            us.unlock();
        
    }

    // this = (1-alpha)* delta((Pu(i,t)==uids[t]) + alpha*refPu(i,t) - Pu
    // Same as seterrorsignal(), but replacing 1/0 reference by linear interpolation of 1/0 reference and the posterior from the reference model.
    // I.e. keep it a little similar to the reference model.
    void seterrorsignalwithklreg (const rbmstatevectorsrefbase & uids, const rbmstatevectorsrefbase & Pu, const rbmstatevectorsrefbase & refPu, const float alpha)
    {
        // experimental pruning mode (disabled by default)  --note: pruning currently hurts bigtime. Need to normalize?
        float pruningbeam = 0.0f;   // off the best (not considering correct state)
            auto & us = *this;
            assert (cols() == uids.cols() && cols() == Pu.cols());
            assert (rows() == Pu.rows() && uids.rows() == 1);

            us.lockforwriting();
            uids.lockforreading();
            Pu.lockforreading();
            refPu.lockforreading();

            size_t numpruned = 0;
            const float oneminusalpha=1-alpha;
            foreach_column (t, us)
            {
                // pruning: we determine the maximum Pu(i,t), not considering the correct state.
                float maxPu_t = 0.0f;
                if (pruningbeam > 0)
                {
                    foreach_row (i, us)
                    {
                        const size_t uid = (size_t) uids(0,t);
                        if (i == uid)   // we only consider competitors
                            continue;
                        if (Pu(i,t) > maxPu_t)
                            maxPu_t = Pu(i,t);
                    }
                }

                // set the target weights for this frame
                // Each row of W will be nudged towards v(t) by the target weight.
                // For the correct state, the target weight is 1.0-P, otherwise -P, i.e.
                // for incorrect states, rows will be gently nudged away from v(t).
                // Setting the target weight to 0 will leave the row of W unchanged.
                foreach_row (i, us)
                {
                    const size_t uid = (size_t) uids(0,t);
                    if ((float) uid != uids(0,t))
                        throw std::runtime_error ("seterrorsignal: uids not integer!");
                    float utarget_it = (i == uid) ? 1.0f : 0.0f;
                    if (alpha>0) utarget_it = oneminusalpha*utarget_it + alpha*refPu(i,t);
                    us(i,t) = utarget_it - Pu(i,t);

                    // do the pruning
                    // We always nudge towards the correct state, but exclude low-scoring competitors.
                    if (i != uid && Pu(i,t) < maxPu_t * pruningbeam)
                    {
                        us(i,t) = 0.0f; // setting it to 0 will exclude it from having an effect
                        numpruned++;
                    }
                }
            }

            if (numpruned > 0)
                fprintf (stderr, "seterrorsignal: %d out of %d pruned (%.2f%%) (pruningbeam param = %.2f)\n",
                numpruned, us.rows()*us.cols(), 100.0 * numpruned / (us.rows()*us.cols()), pruningbeam);

            refPu.unlock();
            Pu.unlock();
            uids.unlock();
            checknan (us);
            us.unlock();
        
    }

    // drop frames by consolidating surviving frames at the start of the slice
    // A frame is dropped if keepsampleflags[t] is 0.0.
    void dropframes (const rbmstatevectorsrefbase & keepsampleflags)
    {
            auto & us = *this;
            assert (cols() == keepsampleflags.cols() && keepsampleflags.rows() == 1);

            us.lockforreadwrite();
            keepsampleflags.lockforreading();

            size_t t1 = 0;
            foreach_column (t, us)
            {
                if (keepsampleflags(0,t) == 0.0f)
                    continue;
                if (t1 < t)
                    us.col(t1).assign (us.col(t));
                t1++;
            }

            keepsampleflags.unlock();
            us.unlock();
        
    }

    // statistics for tracking progress of backpropagation
    // called as: fu.posteriorstats (Pu, sumlogpps_stripe, sumpps_stripe, sumfcors_stripe, avlogpp, avpp, avfcor);
    // Measures av. log PP, av. PP, and av. frames correct against ground truth ('this' as a row vector of indices).
    // This uses temp row vectors to compute it independently for each time point (using CUDA), and then sums up the result.
    void posteriorstats (const rbmstatevectorsrefbase & Pu, rbmstatevectorsrefbase & logpps, rbmstatevectorsrefbase & pps, rbmstatevectorsrefbase & fcors, double & /*out*/avlogpp, double & /*out*/avpp, double & /*out*/avfcor) const
    {
        const auto & uids = *this;
        const size_t n = uids.cols();
  // This is a weirdly inefficient implemetenation. We keep it as it mirrors the CUDA version.
            uids.lockforreading();
            Pu.lockforreading();
            logpps.lockforwriting();
            pps.lockforwriting();
            fcors.lockforwriting();

            assert (Pu.cols() == n);
            assert (uids.rows() == 1);
            assert (logpps.cols() == n && pps.cols() == n && fcors.cols() == n);
            assert (logpps.rows() == 1 && pps.rows() == 1 && fcors.rows() == 1);
            checknan (Pu);
            foreach_column (t, uids)
            {
                const size_t clsid = (size_t) uids(0,t);
                assert ((float) clsid == uids(0,t));
                const float pp = Pu(clsid,t);
                pps(0,t) = pp;
                logpps(0,t) = logf (max (pp, 0.000001f));   // (avoid underflow if prob has been rounded to 0)
                // which is the max?
                fcors(0,t) = pp;            // non-null indicates assumption that it is correct
                foreach_row (i, Pu)
                    if (i != clsid && Pu(i,t) >= pp)
                        fcors(0,t) = 0.0f;  // assumption was wrong
            }

            fcors.unlock();
            pps.unlock();
            logpps.unlock();
            Pu.unlock();
            uids.unlock();
        
        assert (n > 0);
        avlogpp = logpps.rowvecsum() / n;
        avpp = pps.rowvecsum() / n;
        avfcor = fcors.numpositive() / (double) n;
    
};
};

// matrix to hold RBM state vectors (input and activations; multiple vectors in time sequence)
//  - CUDA mode: stored entirely in CUDA memory, no outside computation allowed (except set input and get posteriors)
//  - DBN-level operations are implemented on this
//  - RBM-internal operations are implemented through rbmstatevectorsrefbase
//     - user get a ref first
//     - then execute operation on that object
// TODO: lock mechanism should return the ref
template<class matrixbase> class rbmstatevectorsbase : public acceleratedmatrixbase<msra::math::ssematrix<matrixbase>>
{
    typedef rbmstatevectorsrefbase<matrixbase> rbmstatevectorsref;
public:
    // get a stripe without lock, for passing stripes from DBN to RBM
    rbmstatevectorsref stripe (size_t firstframe, size_t numframes)
    {
        auto stripem = msra::math::ssematrixstriperef<matrixbase> (*this, firstframe, numframes);
        auto stripec = cudadistributedmatrix (*this, firstframe, numframes);
        return rbmstatevectorsref (std::move (stripem), std::move (stripec));
    }
    /*const*/ rbmstatevectorsref stripe (size_t firstframe, size_t numframes) const
    {
        auto res = const_cast<rbmstatevectorsbase*> (this)->stripe (firstframe, numframes);
        return res;
    }

    // get a stripe with a lock, for initializing and getting results out in DBN
    class lockforwriting : public rbmstatevectorsrefbase<matrixbase>
    {
    public:
        lockforwriting (rbmstatevectorsbase & m, size_t firstframe, size_t numframes) : rbmstatevectorsrefbase<matrixbase> (m.stripe (firstframe, numframes)) { rbmstatevectorsrefbase::lockforwriting(); }

        ~lockforwriting() { unlock(); }
    };
    class lockforreading : public rbmstatevectorsrefbase<matrixbase>
    {
    public:
        lockforreading (const rbmstatevectorsbase & m, size_t firstframe, size_t numframes) : rbmstatevectorsrefbase<matrixbase> (m.stripe (firstframe, numframes)) {/*setDeviceId(m.getDeviceId());*/  rbmstatevectorsrefbase::lockforreading(); }
        ~lockforreading() { unlock(); }
    };

    // these two functions need to be there to allow vector<>::resize() and the likes, but we may never call those
    rbmstatevectorsbase (const rbmstatevectorsbase & other)   // will not copy, just construct acceleratedmatrixbase empty
    {
        if (!empty())
            throw std::logic_error ("rbmstatevectorsbase: cannot assign");
        if (cudamode)
        {
            other.checkvalidstriping (notstriped);
            setcudastriping (notstriped);
        }
    }
    void operator= (rbmstatevectorsbase &&) { throw std::logic_error ("rbmstatevectorsbase: cannot assign"); }

    // needed for the layerstate/errorstate vectors' initial resize()
    rbmstatevectorsbase() { if (cudamode) setcudastriping (notstriped); }

    // use during construction only
    void resize (size_t n, size_t m) { acceleratedmatrixbase::resize (n, m); }
};

// temporary storage for matrices used in accelerated computation
//  - NUMA mode: NUMA-node local copies
//  - CUDA mode: no longer used for this
// Although it is temporary, we keep the memory around across calls
// (that's what 'cached' in the name is supposed to indicate).
// TODO: better name... bufferedmatrix?
// Note: the CUDA side of this is not in here. Can we merge this with acceleratedmatrixbase?? Do it when I run non-CUDA again.
template<class matrixbase> class cachedmatrixbase   // TODO: move this inside acceleratedmatrixbase
{
    typedef msra::math::ssematrix<matrixbase> matrix;
    // NUMA version
    std::vector<matrix> numacopies;  // [numanodeid]
public:
    // NUMA version only
    // note: these are only supposed to be called from acceleratedmatrixbase
    // TODO: we should really make this a class inside there
    void allocate_numa (size_t n, size_t m)
    {
        // NUMA version
        size_t numnodes = msra::numa::getnumnodes();    // NUMA nodes
        numacopies.resize (numnodes);
        msra::numa::foreach_node_single_threaded ([&]()
        {
            size_t numanode = msra::numa::getcurrentnode();
            numacopies[numanode].resizeonce (n, m);
        });
    }
    // NUMA version only  --can this be abstracted better?
    matrix & operator[] (size_t numanodeid) { return numacopies[numanodeid]; }
};

// matrix to hold RBM model parameters
//  - purpose: in CUDA mode, hold models in CUDA RAM except outside computation (that is, during model load/save)
//     - 'computing' state controls whether data currently lives in CPU or CUDA side
//     - provides underlying CUDA storage if in CUDA mode
//     - future: will handle multiple CUDA devices
//  - provides access to a controlled subset of matrix operations
//  - has all high-level operations needed for manipulating models
//     - CUDA mode: execution using CUDA copies wherever possible
//     - NUMA mode: optimized parallelized (multi-threaded) matrix product implemented in this class
template<class matrixbase> class rbmmodelmatrixbase : public acceleratedmatrixbase<msra::math::ssematrix<matrixbase>>
{
    typedef rbmstatevectorsrefbase<matrixbase> rbmstatevectorsref;

    bool computing;                             // entercomputation() called <=> data lives in CUDA side?
    void checknotcomputing() const { if (computing) throw std::logic_error ("acceleratedmatrixbase: function called while in 'computing' state, forbidden"); }
    void checkcomputing() const { if (!computing) throw std::logic_error ("acceleratedmatrixbase: function called while not in 'computing' state, forbidden"); }
public:
    typedef cachedmatrixbase<matrixbase> cachedmatrix;  // TODO: move cachedmatrix inside here

    rbmmodelmatrixbase() : computing (false) {}

    // controlling 'computing' state
    void entercomputation()
    {
        checknotcomputing();
        // TODO: some badly encapsulated knowledge here
        synctocuda (false);
        computing = true;           // matrix now owned by CUDA space; our CPU copy can only be used for reading dimensions etc.
    }
    void exitcomputation()
    {
        checkcomputing();
        syncfromcuda (true);        // claim back matrix from CUDA space
        computing = false;
    }

    // outside computation
    void operator= (matrix && other) { checknotcomputing(); matrix::operator= (std::move (other)); }
    // operator= used in doublenodes() and model loading (all move semantics)
    // outside computation
    void resize (size_t n, size_t m) { checknotcomputing(); matrix::resize (n, m); }    // CPU-side (CUDA in entercomputation()); used by constructors only
    void read  (FILE * f, const char * name, const string &begintag=std::string())       { checknotcomputing(); matrix::read (f, name,begintag); }
    void write (FILE * f, const char * name) const { checknotcomputing(); matrix::write (f, name); }
    float &       operator() (size_t i, size_t j)       {  checknotcomputing(); return matrix::operator() (i, j); }  // doublenodes()
    const float & operator() (size_t i, size_t j) const {  checknotcomputing(); return matrix::operator() (i, j); }  // initrandom(), doublenodes();
    float &       operator[] (size_t i)       { checknotcomputing(); return matrix::operator[] (i); }  // doublenodes()
    const float & operator[] (size_t i) const { checknotcomputing(); return matrix::operator[] (i); }  // initrandom(), doublenodes();

    // special-purpose access, e.g. used to for model quantization
    const matrix & peek() const { checknotcomputing(); return (const matrix &) *this; }

    // during computation
    // For these, we must be carefully choose what is where (CUDA or CPU memory). Should only be in one place at a time if at all possible (avoid cached copies).
    // For all public methods below, arguments passed in with a 'cachedmatrix' object are unique to this frame
    // and therefore must be moved to CUDA each time. The 'cachedmatrix' objects will hold pre-allocted memory for that.
    // All outputs are also frame-unique (unless same as input, and in CUDA memory).
    // Note that scaleandaddmatprod_numa, however, exists in two usage scenarios, one with A and one with C in CUDA memory.

    // this = this * thisscale + rowsum(othercols) * otherweight, 'othercols' is state memory i.e. temp per frame
    // The rowsum is computed into othertowsumtmp, which must have been allocated to correct size already.
    // If scale==0, we know to just assign.
    void scaleandaddallcols (const float thisscale, const rbmstatevectorsref & othercols, const float otherweight, msra::math::ssematrix<matrixbase> & otherrowsumtmp)  // used by updatedeltas()
    {
        // compute the row sum
        checkcomputing();
      
       
            //if (otherweight != 1.0f) throw logic_error ("scaleandaddallcols: cannot yet scale the summand--implement this");   // TODO: implement this
            otherrowsumtmp.resizeonce (othercols.rows(), 1);
            othercols.fornuma().rowsum (otherrowsumtmp, otherweight);
            const matrixbase & other = otherrowsumtmp;  // the vector to add
            if (thisscale == 0.0f)  // this is an assignment (original content may be invalid, e.g. NaN)
                matrixbase::assign (other);
            else
                matrix::scaleandadd (thisscale, other);
        
    }

    // accumulator += this, where other lives in CPU space
    // Actually it seems this makes no measurable runtime difference at all.
    template<class VECTOR> void accumulate (VECTOR & accumulator) // const
    {
        checkcomputing();
        syncfromcuda (true);     // bring it into CPU space

        assert (accumulator.size() == rows() && cols() == 1);
        foreach_index (i, accumulator)
            accumulator[i] += matrix::operator() (i,0);
    }


    // this += other * weight, both in accelerated memory
    // This is used for model update.
    void addweighted (const rbmmodelmatrixbase & other, float weight)         // adddeltas()
    {
        checkcomputing();
        matrix::addweighted (other, weight);
    }

    // set the value to zero if less than threshold
    // This is used for model update.
    void setto0ifabsbelow (float threshold) 
    {
        checkcomputing();
        matrix::setto0ifabsbelow (threshold);
    }

    void setto0ifabsbelow2 (rbmmodelmatrixbase &  ref, float threshold) 
    {
        checkcomputing();
        matrix::setto0ifabsbelow2 (ref, threshold);
    }

    void setto0ifabsabove2 (rbmmodelmatrixbase &  ref, float threshold) 
    {
        checkcomputing();
        matrix::setto0ifabsabove2 (ref, threshold);
    }

    void KhatriRaoProduct(const rbmstatevectorsref & m1, const rbmstatevectorsref & m2)
    {
        checkcomputing();
     
        matrix::KhatriRaoProduct(m1, m2);
    }

    void reshapecolumnproduct (const rbmstatevectorsref & eh, const rbmstatevectorsref & h, const bool isehtransposed)
    {
        checkcomputing();
      
        matrix::reshapecolumnproduct(eh, h, isehtransposed);
    }


    // set elements of bias to value [v-xieche]
    void setvalue (float value)
    {
        checkcomputing();
        matrix & us = *this;
        if (us.cols() == 1)   // for a
            foreach_index (i, us)
            us[i] = value;
        
    }

    // get weight matrix from W. [v-xieche]
    template <class AType> void getweightmatrix (AType & weightbuf)
    {
     
        matrix & us = * this;
        foreach_coord (i, j, us)
            weightbuf (i,j) = us (i, j);
       
    }

    // assign weight matrix from W. [v-xieche]
    template <class AType> void assignweightmatrix (AType & weightbuf)
    {
        
        matrix & us = * this;
        foreach_coord (i, j, us)
            us (i, j) = (float)weightbuf (i,j);
        
    }


    // multiply the W with n. used for temp experiment to see what happend when sigmoid become steeper. [v-xieche]
    void multiplywith (float n)
    {
        matrix & us = *this;
      
        foreach_coord (i, j, us)
            us (i, j) *= n;
      
    }


    // set a matrix to a block-diagonal structure, by setting off-elements to 0
    // If 'poolblocks' then each block gets replaced by the average over all blocks.
    // This is intended to support input-layer transforms when inputs are augmented by neighbor frames.
    void setblockdiagonal (size_t diagblocks, bool poolblocks, const size_t & numofclasses, const size_t & roundupunit, const bool setidentity)     // modified by Hang Su adaptation
    {
        checkcomputing();
        if (diagblocks == 1)        // nothing to do
            return;
            fprintf (stderr, "setblockdigonal : diagblocks = %d, poolblock = %d\n", diagblocks, poolblocks);
            matrix & us = *this;
            // TO BE TESTED
            //All info is available in diagblocks and us.cols()/us.rows()

            if (setidentity)
            {
                foreach_coord(i,j,us)
                {
                    if ( i == j )
                        us(i,j) = 1;
                    else
                        us(i,j) = 0;
                }
            }
            else
            {
                const size_t uscols = (us.rows() + roundupunit) * numofclasses - roundupunit;     // the relationship between us.cols() and us.rows() -roundupunit because the last class is not rounded up
                if(us.cols() !=  uscols && us.cols() != 1)       throw std::logic_error ("setblockdiagonal:  the size of matrix is not a correct matrix for linear network or an array ");
                if(us.cols() == uscols)  // execute it only for rounded up adaptation matrix
                {
                    size_t feadim = us.rows() / diagblocks;      // it should be 39 in the normal conditioin
                    size_t classdim = us.cols() / numofclasses;
                    if(us.rows() != diagblocks * feadim )    throw std::logic_error ("setblockgiagonal: the row of matrix can't divided by diagblocks");
                    for (size_t j = 0; j < classdim - roundupunit; j++)         // set elements that are not in block to zeros
                    {
                        for (size_t i = 0; i < us.rows(); i++)
                        {
                            if(size_t(i / feadim) != size_t(j / feadim))
                            {
                                for ( size_t k = 0; k < numofclasses; k++)
                                    us(i, j + classdim *k) = 0;
                            }
                        }
                    }
                    for (size_t j = 0; j < roundupunit; j++)                    // set round up units in the matrix to zeros
                    {
                        for (size_t i = 0; i < us.rows(); i++)
                        {
                            for (size_t k = 0; k < numofclasses - 1; k++)       // -1 because the last class is not blowed up
                            {
                                us(i, j + us.rows() + classdim *k) = 0;
                            }
                        }
                    }
                    if(poolblocks)				//need to calculate the average from the blocks. 
                    {
                        for (size_t classid = 0; classid < numofclasses; classid++)     //modify adaptation matrix for each class 
                        {
                            for(size_t i = 0; i < feadim; i ++) for(size_t j = 0; j < feadim; j ++)  // first calculate the sum of the point corresponding at every position.
                                for(size_t k = 1; k < diagblocks; k ++)
                                    us(i, j + classdim * classid) += us(k*feadim + i, k*feadim + j + classdim * classid);
                            for(size_t i = 0; i < feadim; i ++)  for(size_t j = 0; j < feadim; j ++)  // assign W as the average of the blocks.
                            {
                                us(i, j + classdim * classid) = us(i, j + classdim * classid) / diagblocks;
                                for(size_t k = 1; k < diagblocks; k ++)
                                    us(k*feadim + i, k*feadim + j + classdim * classid) = us(i, j + classdim * classid);
                            }
                        }
                    }
                }
                else if(us.cols() == 1)   // for a
                {
                    size_t classdim = (us.rows() + roundupunit)/ numofclasses;
                    size_t feadim = ((us.rows() + roundupunit)/ numofclasses - roundupunit) /  diagblocks;      // it should be 39 in the normal conditioin
                    if((us.rows() + roundupunit)/ numofclasses - roundupunit != diagblocks * feadim )    throw std::logic_error ("setblockgiagonal: the row of matrix can't divided by diagblocks");
                    for (size_t classid = 0; classid < numofclasses - 1; classid ++)        // set roundup units to zeros
                    {
                        for (size_t i = 0; i < roundupunit; i++)
                            us(i + classdim - roundupunit + classid * classdim, 0) = 0;
                    }
                    if(poolblocks)
                    {
                        for (size_t classid = 0; classid < numofclasses; classid ++)
                        {
                            for(size_t i = 0; i < feadim; i ++)
                                for(size_t k = 1; k < diagblocks; k ++)
                                    us(i + classid * classdim, 0) += us(k*feadim + i + classid * classdim, 0);
                            for(size_t i = 0; i < feadim; i ++)
                            {
                                us(i + classid * classdim, 0) = us(i + classid * classdim, 0) / diagblocks;
                                for(size_t k = 1; k < diagblocks; k ++)
                                    us(k*feadim + i + classid * classdim, 0) = us(i + classid * classdim, 0);
                            }
                        }
                    }
                }
                else  throw std::logic_error ("setblockdiagonal : The input matrix is not a square or an array. can't be processed");

            }
    }



    // NUMA-localized matrix product dW = dW * scale + v h', dW = 'this'
    // h is transposed locally here into httmp. httmp must have been allocated at correct dimensions already.
    // v and h are per-frame unique and thus live in CUDA space.
    void scaleandaddmatprod (float thisscale, const rbmstatevectorsref & v, const rbmstatevectorsref & h, const float vhscale,
        msra::math::ssematrix<matrixbase> & httmp, cachedmatrix & cachedvs, cachedmatrix & cachedhts)
    {   // used in updatedeltas(): dW.scaleandaddmatprod_numa (momentum, v, ht, cachedvts, cachedhts);
        checkcomputing();
       
            // if (vhscale != 1.0f) throw logic_error ("scaleandaddallcols: cannot yet scale the summand--implement this");   // TODO: implement this
            // transpose h -> httmp
            httmp.resizeonce (h.cols(), h.rows());
            h.fornuma().transpose (httmp);
            const matrixbase & ht = httmp;  // 'h' no longer used below, only 'httmp'
            scaleandaddmatprod_numa (thisscale, v.fornuma(), false, ht, cachedvs, cachedhts, *this);
        
    }

    // special-purpose function for on-demand LL evaluation
    // This computes only one row, row 'i', of the result matrix.
    // It assumes that the copy of the model parameters in CPU RAM are valid.
    void matprod_col_mtm (const msra::math::ssematrixstriperef<matrixbase> & v, msra::math::ssematrixstriperef<matrixbase> & h, const rbmmodelmatrixbase & a, size_t i) const
    {
        assert (h.rows() == 1); // only one result row
        checkcomputing();

        // h = us_i' v + a_i
        auto wi = matrixstripe (const_cast<matrixbase &> ((matrixbase &) *this), i, 1);
        h.matprod_mtm (wi, v);
        foreach_column (j,h)
            h(0,j) += ((const matrixbase &)a)[i];
    }

    // NUMA-localized matrix product h = W' v + a, W' is the transpose of 'this', a is added to all columns
    // 'this' and 'a' are model parameters in CUDA RAM.
    // v is frame-unique input, and h frame-unique output.
    void matprod_mtm (const rbmstatevectorsref & v, cachedmatrix & cachedWs, cachedmatrix & cachedvs, rbmstatevectorsref & h, cachedmatrix &/*cachedhs*/, const rbmmodelmatrixbase & a, cachedmatrix &/*cacheda1s*/) const
    {   // used in vtoh(): W.matprod_mtm_numa (v, cachedWs, cachedvs, h, a);     // h = W' v + a
        checkcomputing();
        matprod_mtm_numa (*this, v.fornuma(), cachedWs, cachedvs, h.fornuma());
         h.fornuma() += a;
        
    }


    // NUMA-localized matrix product ev = W eh, W = 'this' is not transposed
    void matprod_mm (const rbmstatevectorsref & eh, cachedmatrix & cachedWts, cachedmatrix & cachedehs, rbmstatevectorsref & ev, cachedmatrix &/*cachedevs*/, const float vscale=0.0f) const
    {   // used in ehtoev(): W.matprod_mm_numa (eh, cachedWts, cachedhs, ev);   // v = W h
        checkcomputing();
        matprod_mm_numa (*this, eh.fornuma(), cachedWts, cachedehs, ev.fornuma(), vscale);
    }

    // NUMA-localized matrix product v = W h + b, W = 'this' not transposed, b is added to all columns
    void matprod_mm (const rbmstatevectorsref & h, cachedmatrix & cachedWts, cachedmatrix & cachedhs, rbmstatevectorsref & v, cachedmatrix &/*cachedvs*/, 
        const rbmmodelmatrixbase & b, cachedmatrix &/*cachedb1s*/, const float vscale=0.0f) const
    {   // used in htov(): W.matprod_mm_numa (h, cachedWts, cachedhs, v, b);     // v = W h + b
        checkcomputing();
        matprod_mm_numa (*this, h.fornuma(), cachedWts, cachedhs, v.fornuma(), vscale);
        v.fornuma() += b;
        
    }

private:
    typedef msra::math::ssematrixstriperef<matrixbase> matrixstripe;

    // NUMA-localized matrix product C = C * scale + A B
    // where A is passed as A' (i.e. we compute At' B) if 'Aistransposed' (which is faster, so do it if you can)
    // Uses class-local NUMA-local memory that is kept allocated for efficiency.
    static void scaleandaddmatprod_numa (float thisscale, const matrixbase & A, bool Aistransposed, const matrixbase & B,
        cachedmatrix & cachedAts, cachedmatrix & cachedBs, matrixbase & C)
    {
        const size_t Atrows = Aistransposed ? A.cols() : A.rows();
        const size_t Atcols = Aistransposed ? A.rows() : A.cols();
        // we do NUMA-local copies if necessary (i.e. if we are running with >1 NUMA node)
        const bool donuma = (msra::numa::getnumnodes() > 1) && (msra::parallel::get_cores() > 1);
#if 1   // print a message  --remove this once this seems to be working
        {
            static bool f = false;
            if (!f)
            {
                fprintf (stderr, "scaleandaddmatprod_numa: donuma = %s\n", donuma ? "true" : "false");
                f = true;
            }
        }
#endif
        // ensure memory is allocated as required
        if (donuma)
            cachedBs.allocate_numa (B.rows(), B.cols());
        if (donuma || !Aistransposed)
            cachedAts.allocate_numa (Atcols, Atrows);   // cachedAts also used if A is not transposed yet
        // copy B into all cachedBs[]
        if (donuma)
        {
            msra::numa::parallel_for_on_each_numa_node (true, [&] (size_t numanode, size_t i, size_t n)
            {
                matrixbase & cachedB = cachedBs[numanode];
                cachedB.assign (B, i, n);
            });
        }
        // perform product--row stripes of A will be copied locally if >1 NUMA node
        msra::parallel::foreach_index_block (Atrows, Atrows, 4, [&] (size_t i0, size_t i1)
        {
            const size_t numanode = msra::numa::getcurrentnode();
            // get the cached B
            const matrixbase & cachedB = donuma ? cachedBs[numanode] : B;
            //cachedB.checkequal (B);
            // copy over row stripe of A that belongs to this loop iteration
            matrixbase & cachedAt = (donuma || !Aistransposed) ? cachedAts[numanode] : const_cast<matrixbase &> (A);
            if (!Aistransposed) // copy and transpose a row stripe
            {
                // transpose a row stripe of A into a col stripe of At
                A.transposerows (cachedAt, i0, i1);
            }
            else if (donuma)    // copy a row stripe --it's in column form, so copy the range of columns
            {
                // rows [i0,i1) are columns [i0,i1) of A', so we can use a matrixstripe
                const matrixstripe src (const_cast<matrixbase &> (A), i0, i1 - i0);    // (it is 'const')
                matrixstripe dst (cachedAt, i0, i1 - i0);
                // TODO: This point is hit with 40% prob when randomly breaking into the debugger.
                //  --> need to avoid the copy if not NUMA
                dst.assign (src);  // only copy the bits that are needed
            }
            // perform operation from NUMA-local copies
            if (thisscale == 0.0f)
                C.matprod_mtm (cachedAt, i0, i1, cachedB);
            else
                C.scaleandaddmatprod_mtm (thisscale, cachedAt, i0, i1, cachedB);
        });
    }

    // NUMA-localized matrix product C = A B, A is transposed
    // see scaleandaddmatprod_numa
    static void matprod_mtm_numa (const matrixbase & At, const matrixbase & B, cachedmatrix & cachedAts, cachedmatrix & cachedBs, matrixbase & C)
    {
        scaleandaddmatprod_numa (0.0f, At, true, B, cachedAts, cachedBs, C);
    }

    // NUMA-localized matrix product C = A B, A is not transposed
    // see scaleandaddmatprod_numa
    static void matprod_mm_numa (const matrixbase & A, const matrixbase & B, cachedmatrix & cachedAts, cachedmatrix & cachedBs, matrixbase & C, const float vscale = 0.0f)
    {
        scaleandaddmatprod_numa (vscale, A, false, B, cachedAts, cachedBs, C);
    }

};

};};