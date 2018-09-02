// rbm.h -- implementation of Hinton's Restricted Boltzmann Machine
//
// F. Seide, Nov 2010 based on code provided by Yu Dong, MSR Speech Research Group
//
// $Log: /Speech_To_Speech_Translation/dbn/dbn/rbm.h $
// 
// 210   8/08/12 11:11 Fseide
// disabled momentum for deferupdate mode (its implementation was wrong,
// and for such huge blocks, it will have little impact anyway)
// 
// 209   8/07/12 18:14 Fseide
// completed implementation of deferupdate flag (still to be tested)
// 
// 208   8/07/12 17:46 Fseide
// new option to backpropagationmodelupdate(): deferupdate, used to
// implement batches of batches, for an MMI experiment
// 
// 207   8/07/12 5:04p V-hansu
// delete the ROUNDUPMODEL and INJECTTOPSECONDLAYER macro
// 
// 206   8/07/12 9:15 Fseide
// added Frank's weird sampling experiment
// 
// 205   7/23/12 10:54a V-hansu
// add macro INJECTTOPSECONDLAYER to do adaptation using second top layer
// 
// 204   7/19/12 11:46a V-hansu
// modify "roundup" related sentences
// 
// 203   7/06/12 9:17p V-hansu
// add numstream and numroundup in rbmbase so as to record the blowup
// information
// 
// 202   7/05/12 8:01p V-hansu
// chang the interface of blow up to let it able to return roundup unit
// 
// 201   7/02/12 4:26p V-hansu
// add function setlinearlayerweight to use GMM adaptation matrix to
// initialize
// 
// 200   6/30/12 2:29p V-hansu
// modify blowup function
// 
// 199   6/27/12 9:24p V-hansu
// add print(FILE *f) for debugging
// 
// 198   6/27/12 10:41a V-hansu
// modify dumplayer function for debugging
// 
// 197   6/26/12 3:01p V-hansu
// modify blowup function
// 
// 196   6/22/12 2:14p V-hansu
// complete the blowup function and did some change to previous interface
// of blowup
// 
// 195   6/05/12 4:21p V-hansu
// change the blowup function of class rbmbase, rbm and perceptron, not
// complete yet
// 
// 194   6/05/12 2:02p V-hansu
// add blowup function to several classes, not fully complete yet
// 
// 193   5/31/12 10:54p V-xieche
// fix a bug in exitcomputation function for more than 2 cuda devices on
// top layer
// 
// 192   5/13/12 11:00p V-xieche
// add initial code to make toplayer support more than 2 cuda devices in
// pipeline training. not finish yet
// 
// 191   5/09/12 4:25p F-gli
// 
// 190   5/09/12 4:24p F-gli
// 
// 189   4/18/12 4:01p V-xieche
// clean up all code related to target propagation and margin term.
// 
// 188   4/04/12 10:08p V-xieche
// add some commend and delete some debug code and old code won't use
// anymore.
// 
// 187   4/03/12 8:39p V-xieche
// check in all the code for pipeline training, stripe top layer and lies
// them on two cuda devices. need to add comments and adjust the code make
// it easy to read.
// 
// 186   3/27/12 1:14a V-xieche
// Add code for pipeline training with multi cuda devices
// 
// 185   3/11/12 7:05p V-xieche
// add code for a compact trainer. make it run in CUDA directly.
// 
// 184   3/08/12 10:34p V-xieche
// add code to make forward and backward prop do in CUDA directly.
// verified the training is correct, while speed faster than previous.
// need to debug it.
// 
// 183   3/06/12 10:51p V-xieche
// add code for compact trainer. Not finished.
// 
// 182   3/01/12 7:27p V-xieche
// add virtual function in dtnn class to make the code compilable for
// flatten sigmoid training.
// 
// 181   2/07/12 2:29p Dongyu
// fixed momentum invalid problem when learning rate is 0
// 
// 180   1/04/12 7:08p Fseide
// now handles momentum == 0
// 
// 179   11/29/11 5:20p F-gli
// implement peekweightmatrix() and peekbias() to Iannlayer derived class
// 
// 178   11/29/11 11:01a F-gli
// add peekweightmatrix() peekbias() forwardpropwithoutnonlinearity()
// implementation to Iannlayer derived class
// 
// 177   11/23/11 4:30p Dongyu
// refactorize rbmbase to support dtnn and other layer types. added
// Iannlayer as the major interface. May still need to update the
// definition of Iannlayer to fully support dtnn and other layer types.
// 
// 176   11/04/11 16:26 Fseide
// gradient scaling fixed
// 
// 175   11/04/11 14:46 Fseide
// (added a comment)
// 
// 174   11/04/11 14:22 Fseide
// refactored gradient weighting to allow for eliminating the gradient
// scaling by 1/(1-momentum)
// 
// 173   11/04/11 13:49 Fseide
// (editorial)
// 
// 172   11/03/11 15:18 Fseide
// momentum now passed down to network-update functions as momentum per
// sample, in prep for also taking the scaling out of the gradient
// 
// 171   10/28/11 14:51 Fseide
// formal change to use the new otherweight parameters in model update
// (but currently passing 1.0)
// 
// 170   10/28/11 13:36 Fseide
// changed 'momentum' to 'double' in prep of pushing in the scaling
// 
// 169   10/25/11 5:19p Dongyu
// Implemented weight difference (L2 relative to a refmodel) based
// regularization, KL divergence (relative to a refmodel) based
// regularization, CL (only change large weight) and CS (only change small
// weight) based regularization for conservative adaptation. 
// 
// Right now I branched some of the functions. These functions can be
// combined to reduce redundency in the future.
// 
// 168   10/18/11 9:06p V-xieche
// modify the code to implement a true steeper or flat sigmoid function.
// i.e. scale the bias as well
// 
// 167   10/08/11 10:25 Fseide
// new special-purpose access methods peekweightmatrix() and peekbias()
// 
// 166   10/06/11 5:18p Dongyu
// added support to allow adapting weights whose absolute value is above
// or below a threshold controlled by --nochangeifaboveorbelow switch.
// 
// 165   9/26/11 8:43p V-xieche
// Add some codes for log(sigmoid + epison) experiment.
// 
// 164   9/20/11 2:46p V-xieche
// fix a minor bug for steeper sigmoid experiment
// 
// 163   9/19/11 10:47p V-xieche
// Add two function to get and set weight matrix when computing for tmp
// experiment
// 
// 162   8/24/11 9:07p V-xieche
// remove a log infomation for adding margin term.
// 
// 161   8/23/11 7:57p V-xieche
// add margin-based training code for dbn according to Heigold's thesis.
// 
// 160   8/16/11 10:36p V-xieche
// add code for targetpropagation v4
// 
// 159   8/02/11 12:30a V-xieche
// add function to implement targetpropagation using b=w*h instead of h
// 
// 158   7/28/11 2:34p V-xieche
// add some indicatioin and comments modified by v-xieche
// 
// 157   7/26/11 1:07p V-xieche
// fix some TAB format
// 
// 156   7/25/11 10:17a V-xieche
// Put the setvalue() function into the #if #else block.
// 
// 155   7/23/11 5:13p V-xieche
// Add getweight and getbias function to get value from a specific
// location of a specific layer.
// 
// 154   7/20/11 4:02p V-xieche
// Add the dumplayer function and delete the unused variable layer in the
// creat function
// 
// 152   7/13/11 19:02 Fseide
// new method forwardpropwithoutnonlinearity() for a single target
// dimension, intended for use for a specific state index
// 
// 151   7/11/11 11:16a V-xieche
// Add the function for cheating experiment on hidden layer. Add a creat
// function to only create a layer
// 
// 150   7/08/11 11:07 Fseide
// documented the fact that dW, da, and db are SCALED versions of the
// low-pass filtered gradient, which is compensated for when calling
// adddeltas()
// 
// 149   7/07/11 12:11 Fseide
// fixed the momentum bug in the refactoring that was the bug fix for
// 1-frame minibatches
// 
// 148   7/06/11 14:14 Fseide
// added comments and documented a potential bug in momentum handling
// 
// 147   7/06/11 14:03 Fseide
// linearnetwork::backpropagationmodelupdate() now just calls the base
// class and post-processes, to avoid code duplication
// 
// 146   7/06/11 13:59 Fseide
// further cleanup w.r.t. momentumfiltergain
// 
// 145   7/06/11 13:56 Fseide
// (some factoring w.r.t. momentumfiltergain)
// 
// 144   7/06/11 13:52 Fseide
// pushed weighting of learning rate through to rbm
// 
// 143   6/30/11 8:04a Fseide
// added a log message in linearnetwork constructor
// 
// 142   6/22/11 5:10p V-xieche
// put the setblockdiagonal function after adddeltas to make sure the
// out-of-diag is 0.
// also setblockdiagonal function for a in the pooled situation.
// 
// 141   6/21/11 5:20p V-xieche
// just execute the setblockdiagonal functuion(previous comment it).
// 
// 140   6/20/11 10:19p V-xieche
// comment the setblockdiagonal just for temporary test purpose
// 
// 139   6/20/11 12:33p V-xieche
// No need to initial b. remove it
// 
// 138   6/20/11 7:51 Fseide
// changed backpropagationmodelupdate() to be a virtual function;
// added an override to backpropagationmodelupdate() to implement the
// block-diagonal structure
// 
// 137   6/20/11 7:25 Fseide
// factored network construction by type string out from dbn.h into a
// factory class in rbm.h where it belongs;
// moved linearnetwork::initial() out from linearnetwork to rbmbase next
// to initrandom() since it structurally seems to belong (it makes no
// assumption on linearnetwork) there although it is only used by
// linearnetwork
// 
// 136   6/19/11 3:42p V-xieche
// Initial b in the linear network also
// 
// 135   6/19/11 2:43p V-xieche
// Initial the linearnetwork, W to be a identity matrix and A to be a zero
// matrix.
// 
// 134   6/18/11 16:51 Fseide
// (renamed a variable)
// 
// 133   6/17/11 11:21 Fseide
// added class members and respective reading/writing code to
// lineartransform
// 
// 132   6/17/11 11:07 Fseide
// added comments and renamed a variable
// 
// 131   6/16/11 18:42 Fseide
// (comments)
// 
// 130   6/14/11 11:01 Fseide
// added new class 'linearnetwork'
// 
// 129   6/12/11 18:48 Fseide
// new method forwardpropwithoutnonlinearity() to support bottleneck
// features
// 
// 128   5/10/11 7:41a Fseide
// (refined logging of stats)
// 
// 127   5/09/11 15:23 Fseide
// temporarily made 'a' public for a hacked analysis tool
// 
// 126   4/11/11 3:18p Fseide
// (fixed a compiler warning)
// 
// 125   3/23/11 11:50a Fseide
// new method setweights()
// 
// 124   3/13/11 20:59 Fseide
// (a minor bug commented)
// 
// 123   3/05/11 8:30p Fseide
// printmatvaluedistribution() and checkmodel() now compute/print the
// overall number of non-null parameters in aggregate
// 
// 122   3/04/11 6:17a Dongyu
// added model weight distribution analysis and dumping functionality
// through the "checkmodel" switch
// 
// 121   3/03/11 8:16a Dongyu
// added weight sparseness support in training.
// 
// 120   2/10/11 10:02a Fseide
// scaleandaddallcols() prototype was simplified;
// documented and partially fixed spelling error 'rmbmodelmatrix' --oops!
// 
// 119   2/08/11 5:33p Fseide
// bug fix in adddeltas(): now no longer adds db in backprop mode (b is
// unused at this point anyway, but it was wrong nevertheless)
// 
// 118   2/08/11 4:23p Fseide
// moved three resizeonce() calls from updatedeltas() to inside their NUMA
// counterparts (they are not used in CUDA, so no need to allocate them)
// 
// 117   2/08/11 2:19p Fseide
// (an outdated comment deleted)
// 
// 116   2/07/11 4:29p Fseide
// removed a few checknan() that do not play well with the new
// architecture
// 
// 115   2/07/11 3:25p Fseide
// added typedefs for rbmstatevectorsrefread/writing
// 
// 114   2/05/11 8:23p Fseide
// fixed an incorrect assertion in updatedeltas()
// 
// 113   2/05/11 7:23p Fseide
// fixed an assertion in updatedeltas()
// 
// 112   2/05/11 7:00p Fseide
// moved mulbydsigm() and samplebinary() to rbmstatevectorsref
// 
// 111   2/02/11 11:22a Fseide
// moved sigmoid() and softmax() to rbmstatevectors;
// replace the now empty matrixbase class by a typedef
// 
// 110   2/02/11 10:48a Fseide
// switched all matrixbase & to rbmstatevectorsref & (not tested yet
// because underlying classes still cannot handle it)
// 
// 109   2/02/11 10:27a Fseide
// switched over from matrix/matrixbase to dummy implementations of
// rbmstatevectorsbase/rbmstatevectorsrefbase
// 
// 108   2/02/11 10:23a Fseide
// added typedef matrixstripe rbmstatevectorsref
// 
// 107   2/02/11 9:25a Fseide
// defined rbmstatevectors, but currently identical to 'matrix', need to
// solve the stripe problem first
// 
// 106   2/02/11 8:38a Fseide
// changed model parameter types from acceleratedmatrix to rbmmodelmatrix
// 
// 105   2/02/11 8:24a Fseide
// (added a comment)
// 
// 104   2/02/11 8:22a Fseide
// pushed some math ops on updatedeltas() down to acceleratedmatrix, for
// further CUDA optimization
// 
// 103   2/01/11 7:53p Fseide
// added performance comments
// 
// 102   2/01/11 6:44p Fseide
// (added a comment)
// 
// 101   2/01/11 4:53p Fseide
// added one more cache
// 
// 100   2/01/11 15:24 Fseide
// now gets cachedmatrix from inside acceleratedmatrix
// 
// 99    2/01/11 15:00 Fseide
// matprod_m*m() functions now take one additional cache object for moving
// data to/from CUDA
// 
// 98    2/01/11 14:57 Fseide
// stratified interface to acceleratedmatrix a little
// 
// 97    2/01/11 11:47a Fseide
// fixed entercomputation() protocol w.r.t. allocation of deltas and
// updatedeltas()
// 
// 96    1/30/11 16:37 Fseide
// (added a comment)
// 
// 95    1/30/11 16:33 Fseide
// (added a comment)
// 
// 94    1/30/11 16:33 Fseide
// acceleratedmatrixbase and cachedmatrixbase moved to parallelrbmmatrix.h
// 
// 93    1/30/11 16:28 Fseide
// changed acceleratedmatrix and cachedmatrix to class templates, so we
// can move them to a separate header
// 
// 92    1/30/11 15:56 Fseide
// further abstraction of cachedmatrix, ready to be reused for CUDA
// version
// 
// 91    1/28/11 17:13 Fseide
// commented the four key matrix functions in acceleratedmatrix, which are
// to be adapted to CUDA
// 
// 90    1/28/11 16:54 Fseide
// comments on what happens where
// 
// 89    1/28/11 16:38 Fseide
// changed acceleratedmatrix to derive from 'matrix' protected to make all
// calls into 'matrix' explicit;
// added call-through methods to acceleratedmatrix for all calls into
// 'matrix';
// parallelized matrix product moved inside acceleratedmatrix
// 
// 88    1/28/11 15:36 Fseide
// changed model parameters to acceleratedmatrix (first step)
// 
// 87    1/28/11 15:16 Fseide
// changed the various rbmXXX(W,a,b) constructors to take rvalue
// references
// 
// 86    1/28/11 14:43 Fseide
// further tidying-up, clean-up, moving-around, commenting as prep for
// CUDA transition
// 
// 85    1/28/11 11:41 Fseide
// new data type cachedmatrix as a first step to abstract out NUMA/CUDA
// stuff from rbmbase
// 
// 84    1/28/11 11:37 Fseide
// moved matrix-product functions out from rbmbase, in prep of CUDA
// version
// 
// 83    1/28/11 11:24 Fseide
// (removed some unused code)
// 
// 82    1/28/11 11:23 Fseide
// moved functions around in prep for modularization for CUDA
// 
// 81    1/28/11 11:11 Fseide
// removed residuals of ZMSIGM experiment
// 
// 80    1/28/11 11:09 Fseide
// removed copy construction and clone()
// 
// 79    1/28/11 10:54 Fseide
// removed cachedWt and all that depends on it (no longer needed, we clone
// and transpose on the fly)
// 
// 78    1/28/11 10:48 Fseide
// added enter/exitcomputation();
// deleted some old code related to old, frame-wise parallelization
// 
// 77    1/24/11 12:21p Fseide
// (added some #if-0'ed out debug code)
// 
// 76    1/19/11 16:34 Fseide
// (added comments to scaleandaddmatprod_numa() towards transition to a
// more standard GEMM call)
// 
// 75    1/19/11 10:05a Fseide
// added checks for matprod to check whether parallelized version is
// correct
// 
// 74    1/19/11 8:38a Fseide
// changed updatedeltas() from initializing da/db by move to initializing
// it by assignment (which requires prior allocation since this is the
// matrixbase type which cannot allocate)
// 
// 73    1/14/11 10:20p Fseide
// disabled the "speed-up" hacks, they don't seem to work just like they
// are now, something still wrong
// 
// 72    1/14/11 9:25p Fseide
// added the "optimizations" according to the "BP tricks" document (need
// to find the author and correct title!)
// 
// 71    1/14/11 6:03p Fseide
// removed a log message
// 
// 70    1/14/11 5:45p Fseide
// (cosmetic change to a log message)
// 
// 69    1/14/11 5:44p Fseide
// eliminated vt from updatedeltas() because we can now directly operate
// on the untransposed matrix
// 
// 68    1/14/11 5:36p Fseide
// scaleandaddmatprod_mtm_numa() renamed to scaleandaddmatprod_numa();
// it now implements Aistransposed flag
// 
// 67    1/14/11 5:01p Fseide
// (renamed a function--no longer needed in the future anyway)
// 
// 66    1/14/11 4:47p Fseide
// preparation for scaleandaddmatprod_numa() towards implicit
// transposition
// 
// 65    1/13/11 10:40a Fseide
// (added a diagnostics message to scaleandaddmatprod_numa())
// 
// 64    1/13/11 10:08a Fseide
// scaleandaddmatprod_numa() now parallelizes distribution of the
// input matrix
// 
// 63    1/12/11 12:30p Fseide
// towards more local parallelization of fprop/bprop
// 
// 62    1/12/11 10:48a Fseide
// some refactoring towards new parallelization of prop functions
// 
// 61    1/10/11 16:31 Fseide
// (added a comment)
// 
// 60    1/05/11 9:59p Fseide
// updatedeltas() now resizes again to account for the last block
// 
// 59    1/05/11 9:35p Fseide
// updatedeltas() now keeps its memory allocated (in a member variable)
// 
// 58    1/05/11 6:37p Fseide
// updatedeltas() now only copying portion of vt locally that is needed
// 
// 57    1/05/11 6:12p Fseide
// NUMA-optimized the model update--significant gain compared to
// non-optimized (NUMA-bad) version
// 
// 56    1/05/11 4:51p Fseide
// some tidying-up in prep for NUMA-parallelizing update function;
// backpropagationmodelupdate() no longer virtual (the same for all types)
// 
// 55    1/05/11 12:11p Fseide
// backpropagateprepare now operating striped for optimal NUMA performance
// 
// 54    1/05/11 8:37a Fseide
// copyfrom() no longer copies deltas, and only allocates cachedWt (not
// copying)
// 
// 53    1/04/11 10:20p Fseide
// changed needWt() to not doing transpose in parallel, due to new
// architecture
// 
// 52    1/04/11 9:45p Fseide
// new method copyfrom() to reclone without mem allocation (for NUMA)
// 
// 51    12/21/10 18:54 Fseide
// bug fix for top layer (which has no 'b')
// 
// 50    12/21/10 18:37 Fseide
// added experimental functionality to "split" a hidden layer by doubling
// its number of hidden nodes
// 
// 49    12/09/10 9:07p Fseide
// added several checknan() calls
// 
// 48    12/09/10 12:32 Fseide
// removed an assert() that was incorrect when running single-threaded
// 
// 47    12/08/10 3:24p Fseide
// added an overflow check (remove later)
// 
// 46    12/08/10 3:07p Fseide
// softmax() now using normalization
// 
// 45    12/06/10 15:23 Fseide
// removed 'negate' flag from updatedeltas() (was always 'false')
// 
// 44    11/30/10 1:11p Fseide
// initrandom() changed init val for a to 0 from -4 (later we probably
// want to distinguish bp and pt)
// 
// 43    11/30/10 11:22a Fseide
// switched to new implementation of backpropagationupdateshared() that
// shares code with pretraining
// 
// 42    11/30/10 9:12 Fseide
// pretrainingprepare() now separate from backpropagationprepare()
// (although doing the same)
// 
// 41    11/30/10 7:31a Fseide
// now using a little trick in pretrainingmodelupdate() for the negation
// 
// 40    11/30/10 7:01a Fseide
// (added a typecast for 64-bit correctness)
// 
// 39    11/29/10 17:00 Fseide
// several updates/fixes to updatedeltas()
// 
// 38    11/29/10 16:10 Fseide
// new virtual method type()
// 
// 37    11/29/10 15:42 Fseide
// added constructors to construct fresh RBMs from scratch, with random
// initialization
// 
// 36    11/29/10 15:06 Fseide
// (fixed a comment)
// 
// 35    11/29/10 13:17 Fseide
// added pretraining code, but untested so far
// 
// 34    11/26/10 17:12 Fseide
// (added a comment)
// 
// 33    11/26/10 16:30 Fseide
// (minor further change to the same function)
// 
// 32    11/26/10 16:15 Fseide
// (continued to refactor bp shared function, #if 0-ed out)
// 
// 31    11/26/10 16:06 Fseide
// (started to factor some bp update code for pretraining)
// 
// 30    11/25/10 17:04 Fseide
// using parallel matprod now for first (non-momentum) bp update
// 
// 29    11/25/10 15:07 Fseide
// backpropagateupdate() now implements momentum (to be tested)
// 
// 28    11/24/10 7:23 Fseide
// added functions for file I/O
// 
// 27    11/23/10 11:30a Fseide
// now using parallel_transpose() in update... no big difference
// 
// 26    11/23/10 11:22a Fseide
// rbmbase() copy constructor now copies cachedWt
// 
// 25    11/23/10 8:54 Fseide
// (added a comment)
// 
// 24    11/22/10 2:12p Fseide
// backpropagationstats() now calls mulbydsigm()
// 
// 23    11/22/10 13:36 Fseide
// removed a __forceinline as it tripped up the optimizer
// 
// 22    11/22/10 13:07 Fseide
// back prop update now operating in parallel --but slows down back-prop
// error??
// 
// 21    11/22/10 11:02a Fseide
// (added ability to switch back to single-threaded in
// parallel_transpose())
// 
// 20    11/22/10 10:50 Fseide
// backpropagationprepare() now calls parallel_transpose() (but it does
// not seem to help)
// 
// 19    11/19/10 19:11 Fseide
// (minor optimization)
// 
// 18    11/19/10 17:30 Fseide
// added a comment
// 
// 17    11/19/10 16:40 Fseide
// backpropagationupdateshared() changed to use the transpose() function
// 
// 16    11/19/10 16:07 Fseide
// added cachedWt for faster computation
// 
// 15    11/19/10 15:25 Fseide
// (added a comment)
// 
// 14    11/19/10 15:22 Fseide
// basic back-propagation training seems now complete (without momentum)
// 
// 13    11/19/10 12:48 Fseide
// fixed bug in backpropagationupdateshared()
// 
// 12    11/19/10 12:33 Fseide
// (documented a bug, not fixed yet)
// 
// 11    11/19/10 10:56 Fseide
// redesigned interface to back-propagation to avoid locks
// 
// 10    11/19/10 7:43 Fseide
// renamed 'toprbm' to 'perceptron' which is more yet not fully accurate
// 
// 9     11/19/10 7:28 Fseide
// (minor refactoring)
// 
// 8     11/19/10 7:18 Fseide
// (added 2 comments)
// 
// 7     11/18/10 17:00 Fseide
// back-propagation implemented (not tested)
// 
// 6     11/17/10 14:44 Fseide
// cleanup of backpropagationstats()
// 
// 5     11/17/10 13:02 Fseide
// implemented backwardprob();
// new method moveaccumulator() for parallelized training
// 
// 4     11/17/10 12:46 Fseide
// implemented explicit rbmbase copying constructor to allow for the
// non-assignable CCritSec object
// 
// 3     11/17/10 12:29 Fseide
// steps towards training (back-propagation for now)
// 
// 2     11/15/10 18:40 Fseide
// added the ability to clone a model, for use in NUMA-local computation
// 
// 1     11/12/10 11:38 Fseide
// RBM and DBN factored into separate header files

#if 0               //add by Hang Su to set aside code and conments
#endif
#pragma once

#include "ssematrix.h"          // for basic matrix type
#include "parallelrbmmatrix.h"  // for parallel accelerated matrix operations (NUMA, CUDA)
#include <string>
#include <stdexcept>

namespace msra { namespace dbn {

// ===========================================================================
// matrix, vector types for use in the networks
// ===========================================================================

typedef msra::math::ssematrixbase matrixbase;

// CPU-side matrices and vectors for intermediate CPU-side computation
typedef msra::math::ssematrix<matrixbase> matrix;
typedef msra::math::ssematrixstriperef<matrixbase> matrixstripe;
typedef msra::math::ssematrix<matrixbase> vector;

// model matrices that live in CUDA during computation
typedef rbmmodelmatrixbase<matrixbase> rbmmodelmatrix;
// TODO: fix the spelling error rmb->rbm --oops
typedef rbmmodelmatrixbase<matrixbase> rmbmodelmatrix;  // spelling error!
typedef rbmmodelmatrix rmbmodelvector;                  // spelling error!
typedef rbmmodelmatrix::cachedmatrix cachedmatrix;

// network state (input and activations) that lives in CUDA
// ... This is not implemented yet.
typedef rbmstatevectorsbase<matrixbase> rbmstatevectors;
typedef rbmstatevectorsrefbase<matrixbase> rbmstatevectorsref;
typedef rbmstatevectors::lockforreading rbmstatevectorsrefreading;
typedef rbmstatevectors::lockforwriting rbmstatevectorsrefwriting;

enum regularizationtype
{
    regNone,   //no regularization
    regL2,   //||w-w_ref||
    regKL,   //KL between ref posterior distributions and new model's posterior distributions
    regCL,   //change large weights only
    regCS   //change small weights only
};

class Iannlayer;

struct modelupdateinfo
{
    regularizationtype regtype;
    float sparsethreshold;
    float nochangeifaboveorbelow;
    Iannlayer * preflayer;
    float alpha;
};

// ===========================================================================
// Iannlayer -- interface of all network layers
// ===========================================================================
class Iannlayer
{
private:
    Iannlayer (const Iannlayer &);
    void operator= (const Iannlayer &);

protected:
   // Iannlayer() { };
	Iannlayer():flag(false) { };  // Jian changed for svd
public:
	char flag;  // Jian added for svd
    virtual string type() const = 0;
    virtual void print() const = 0;
    virtual void print(FILE *f) const = 0;
    virtual void write (FILE * f) const = 0;
    virtual void dumplayer() const = 0;
    virtual pair<unsigned int,unsigned int> printvaluedistribution (const string & tag) const = 0;
    
    virtual const matrix & peekweightmatrix() const = 0;
    virtual const vector & peekbias() const = 0;
    virtual void forwardpropwithoutnonlinearity (const matrixstripe & v, matrixstripe & u, size_t i) const = 0;

    virtual size_t vdim() const = 0;
    virtual size_t hdim() const = 0;  //return the overall hidden layer size (e.g., h1*h2 in dtnn case).
    virtual std::vector<size_t> hdims() const = 0;  //return each individual hidden layer size (for dtnn).

    virtual void entercomputation (int type) = 0;
    virtual void exitcomputation() = 0;

    virtual void doublenodes (bool out) = 0;
        
    virtual void validatedims() const = 0;
    virtual void initrandom (unsigned int randomseed) = 0;

    virtual void forwardprop (const rbmstatevectorsref & v, rbmstatevectorsref & h, const bool linearonly = false) const = 0;

    //TODO: ideally we should combine backpropagationstats and backpropagationmodelupdate to just backwardprop

     // compute the error signal for a group of training frames (multiple columns)
    //  - in 'h' are the activation probabilities from the preceding forwardprop() step
    //  - in 'eh' is the error signal from layer above
    //  - out 'eh' is the error signal to be used for model update (updated in-place)
    //  - out 'ev' is the error signal to be passed on to layer below
    virtual void backpropagationstats (rbmstatevectorsref & /*in/out*/ eh, const rbmstatevectorsref & h, rbmstatevectorsref & /*out*/ ev) const = 0;
    virtual void backpropagationmodelupdate (const rbmstatevectorsref & ehxs, const rbmstatevectorsref & v,
                                             float learningratepersample, double momentumpersample, bool resetmomentum, int deferupdate, modelupdateinfo & bpinfo) = 0;

    //TODO: ideally we should combine pretrainingstats and pretrainingmodelupdate to just pretrain
    //v1 and h1 below are CD 1 (only useful for RBM) on minibatch
    virtual void pretrainingstats (const rbmstatevectorsref & Ph, rbmstatevectorsref & v1, rbmstatevectorsref & h1, unsigned int randomseed) const = 0;
    virtual void pretrainingmodelupdate (const rbmstatevectorsref & v, const rbmstatevectorsref & h, rbmstatevectorsref & v1, rbmstatevectorsref & h1,
                                         float learningrate, double momentumpersample, bool resetmomentum) = 0;

    virtual size_t getnumberofweightsets() const = 0;
    virtual pair<size_t, size_t> getweightsetdims(const size_t weightsetindex) const = 0;

    virtual void blowup(const size_t blowupfactor) = 0;        //added by Hang Su adaptation
    virtual void blowup(const size_t blowupfactor, const std::vector<size_t> & statemap) = 0;
    virtual void setlinearlayerweight(const matrix & adaptmatrix) = 0;
    
    //TODO: we cannot make template funciton a virtual function. but we do want to have it virtual. do we really need tempalte?
    template<class WTYPE, class ATYPE> void setweights (const WTYPE & newW, const ATYPE & newa, const size_t weightsetindex) 
    {throw std::logic_error ("setweights: not implemented");};
};

// ===========================================================================
// rbmbase -- base class for all networks used here
// ===========================================================================
enum nonlinearitykind_t     // we support multiple non-linearities
{
	linearkind,
	sigmoidkind,
	relukind,
	softmaxkind,
	leakyrootkind,          // hack
	softpluskind
};
// abstract base class to allow unified operations on the network
class rbmbase : public Iannlayer
{
    rbmbase (const rbmbase &);
    void operator= (const rbmbase &);
protected:
    // model: E(v,h) = v'Wh + v'b + a'h ; p(v,h) = exp -E(v,h)
    // Note: Hinton's publications are not consistent in the use of a and b (which is which).
    rmbmodelmatrix W;   // for v'Wh
public: // hack for analysis
    rmbmodelvector a;   // for a'h
protected:
    rmbmodelvector b;   // for v'b  --this is unused by top level; we keep it here to keep everything together
    size_t vdimnumroundup;  // for adaptation roundup  by Hang Su adaptation
    size_t hdimnumroundup;
    size_t numstream;   // for recording num of class
	nonlinearitykind_t nonlinearitykind;
    // TODO: remove scaling by momentum; instead scale by learning rate
    rmbmodelmatrix dW;  // derivative at last call to -update(); used for momentum
    rmbmodelvector da;  // Note that these deltas are scaled by 1/(1-momentum), for code simplicity.
    rmbmodelvector db;  // This must be corrected when adding these to the model parameters (multiply by (1-momentum)).

    static void malformed (std::string msg) { throw std::runtime_error ("rbmbase: invalid model file: " + msg); }
    void validatedims() const   // check if dimensions match
    {
        if (b.rows() != W.rows() || W.cols() != a.rows())
            malformed ("invalid model file--W matrix dimensions mismatch bias dimensions");
    }
    rbmbase() { }
    //rbmbase (const rbmbase & other) { copyfrom (other); }
	struct networktypedesc_t
    {
        nonlinearitykind_t nonlinearitykind;
        // More fields can be added in the future without breaking file compat:
        // Just initialize them to default in the constructor to allow for reading old files with shorter structs.

        networktypedesc_t()
        {
            memset (this, 0, sizeof (*this));   // to allow memcmp() below
            nonlinearitykind = sigmoidkind;
        }
        bool operator!= (const networktypedesc_t & other) { return memcmp (this, &other, sizeof (*this)) != 0; }
        std::string tostring() const    // TODO: unify with nonlinearitykindtostring() which is nearly redundant to this
        {
            switch (nonlinearitykind)
            {
            case linearkind:    return "linear";
            case sigmoidkind:   return "sigmoid";
            case relukind:      return "relu";
            case leakyrootkind: return "leakyroot";
            case softmaxkind:   return "softmax";
            case softpluskind:  return "softplus";
            default:            throw std::logic_error ("tostring: invalid nonlinearitykind value");
            }
        }
    };
    rbmbase (FILE * f)  // constructor from file
    {
		string tag = fgetTag(f);
		if (tag == "BTYP")                              // network type descriptor
		{
			networktypedesc_t desc;
			size_t size = fgetint(f);
			if (size > sizeof(desc))
				throw runtime_error("rbmbase: malformed BTYP item");
			freadOrDie(&desc, size, 1, f);
			fcheckTag(f, "ETYP");
			nonlinearitykind = desc.nonlinearitykind;   // copy over all fields
			fprintf(stderr, "rbmbase: reading model with non-linearity kind '%s'\n", desc.tostring().c_str());
			tag = fgetTag(f);                          // and advance the tag
		}
		else
		{
			nonlinearitykind = sigmoidkind;             // default if BTYP missing (compat with old files)
		}
		// another messy bit: mean/variance
		if (tag == "BMVN")
		{
			//mean.read(f, "mean");
			//var.read(f, "var");
			fcheckTag(f, "EMVN");
			tag = fgetTag(f);                          // and advance the tag
		}
        W.read (f, "W",tag);
        a.read (f, "a");
        b.read (f, "b");
        vdimnumroundup = 0;
        hdimnumroundup = 0;
        numstream = 1;
    }

    // helper for constructors --reset W, a, and b with random values
    void initrandom (unsigned int randomseed)
    {
        srand (randomseed);
        foreach_coord (i, j, W)
            W(i,j) = (rand() * 0.1f / RAND_MAX) - 0.05f;
        foreach_row (i, a)
            a[i] = 0.0f;
            //a[i] = -4.0f;   // per recommendation in guideTR.pdf
        if (!b.empty())
            foreach_row (j, b)
                b[j] = 0.0f;
    }

    // helper for constructors --reset W, a, and b with identity transform (used by linearnetwork)
    void initidentity()
    {
        foreach_coord (i, j, W)
        {
            if(i == j)  W(i, j) = 1.0;
            else        W(i, j) = 0.0;
        }
    }
    void setbiaszero()
    {
        foreach_row (i, a)
            a[i] = 0.0f;
        if (!b.empty())
            foreach_row (j, b)
                b[j] = 0.0f;
    }

    void initidentity(const matrix & adaptmatrix)
    {
        if (adaptmatrix.cols() != W.cols() || adaptmatrix.rows() != W.rows())
            throw runtime_error ("initidentity: intput adaptation matrix does not match with current matrix");
        foreach_coord (i, j, W)
        {
            W(i,j) = adaptmatrix(i,j);
        }
    }

public:

    // special-purpose accessors (it's a research project after all...)
    const matrix & peekweightmatrix() const { return W.peek(); }
    const vector & peekbias() const { return a.peek(); }

    // get the row = c, col = l value from the weight matrix [v-xieche]
    float getweightvalue(size_t c, size_t l) const
    {
        return W(c, l);
    }
    // get weight matrix from W. [v-xieche]
    template <class AType> void getweightmatrix (AType & weightbuf)
    {
        W.getweightmatrix (weightbuf);
    }

    // assign weight matrix to W. [v-xieche]
    template <class AType> void assignweightmatrix (AType & weightbuf)
    {
        W.assignweightmatrix (weightbuf);
    }

    // get the m-th elemeent of the bias [v-xieche]
    float getbiasvalue(size_t m) const
    {
        return a[m];
    }

    virtual std::vector<size_t> hdims() const
    {
        std::vector<size_t> hdimsvec(1);
        hdimsvec[0] = hdim();
        return hdimsvec;
    }

    void print() const
    {
        printmat(W);
        printmat(a);
        printmat(b);
    }

    void print(FILE *f) const
    {
        printmatfile(W,f);
        printmatfile(a,f);
        printmatfile(b,f);
    }

    // dump the layer element to stdout as mat format [v-xieche]
    void dumplayer() const
    {
        fprintf(stderr, "W:[ ");
        foreach_coord(i, j, W)
        {
            if(i == 0 && j > 0)   fprintf(stderr, ";\n");
            fprintf(stderr, "%.4f ", W(i, j));
        }
        fprintf(stderr, ";]\n");

        fprintf(stderr, "a:[ ");
        foreach_coord(i, j, a)
        {
            if(i == 0 && j > 0)   fprintf(stderr, ";\n");
            fprintf(stderr, "%.4f ", a(i, j));
        }
        fprintf(stderr, ";]\n");

        fprintf(stderr, "b:[ ");
        foreach_coord(i, j, b)
        {
            if(i == 0 && j > 0)   fprintf(stderr, ";\n");
            fprintf(stderr, "%.4f ", b(i, j));
        }
        fprintf(stderr, ";]\n");
    }


    // print model stats
    // Returns a pair (total model params, total non-null model params).
    pair<unsigned int,unsigned int> printvaluedistribution (const string & tag) const
    {
        auto Wstats = msra::math::printmatvaluedistributionf (("W " + tag).c_str(), W);
        auto astats = msra::math::printmatvaluedistributionf (("a " + tag).c_str(), a);
        auto bstats = msra::math::printmatvaluedistributionf (("b " + tag).c_str(), b);
        return make_pair (Wstats.first + astats.first + bstats.first, Wstats.second + astats.second + bstats.second);
    }

    // I/O
    // This is virtual to allow networks to save network-type specific data. Currently used for 'lineartransform'.
    // 'Overridden' reading is done in the constructor from FILE *.
    virtual void write (FILE * f) const
    {
        W.write (f, "W");
        a.write (f, "a");
        b.write (f, "b");
    }

    // get the dimensions
    size_t vdim() const { return W.rows(); }
    size_t hdim() const { return W.cols(); }

    // self-identification of the type of the model; used for saving
    virtual string type() const = 0;

    // do necessary preparations to start any computation with the model
    // 'type'can be:
    //  -1 -> backpropagation
    //  +1 -> pretraining
    //   0 -> evaluation
    // With CUDA, this loads the model into the CUDA RAM.
    void entercomputation (int type)
    {
        W.entercomputation(); a.entercomputation(); b.entercomputation();
        // this is a good time to lazily allocate the delta matrices
        if (type != 0)
        {
            da.resize (a.rows(), a.cols()); // (a.cols()==1, it's a vector)
            if (!b.empty())
                db.resize (b.rows(), b.cols());
            dW.resize (W.rows(), W.cols());
            // note: the first time, entercomputation() calls below will copy garbage
        }
        dW.entercomputation(); da.entercomputation(); db.entercomputation();
    }
    // same do necessary finalization, e.g. in case of CUDA, copy updated models back to CPU RAM
    void exitcomputation()
    {
        W.exitcomputation(); a.exitcomputation(); b.exitcomputation();
        dW.exitcomputation(); da.exitcomputation(); db.exitcomputation();
    }


    // map momentum from per-sample to mbsize
    static float scalemomentum (double momentumpersample, size_t mbsize)
    {
        if (momentumpersample > 0.0)
            return (float) exp (log (momentumpersample) * mbsize);
        else
            return 0.0f;
    }

    // compute the error signal for a group of training frames (multiple columns)
    //  - in 'h' are the activation probabilities from the preceding forwardprop() step
    //  - in 'eh' is the error signal from layer above
    //  - out 'eh' is the error signal to be used for model update (updated in-place)
    //  - out 'ev' is the error signal to be passed on to layer below

    // perform model update (back-propagation)
    // First time pass 'resetmomentum' to initialize the momentum state.
    // 'ehxs' is the error signal multiplied with the sigmoid' (except for linear fDLR layer)
    // This is the default implementation. 'linearnetwork' has its own, hence the virtual function.
    // A special mode is 'deferupdate'--it will only update into the deltas but not the model. This is to simulate large minibatches (which we want to try for MMI).
    //  - deferupdate > 0: only accumulate gradient into deltas (momentum down-scaling is deferred until after update)
    //  - deferupdate < 0: this is the actual update (this also performs momentum down-scaling)
    virtual void backpropagationmodelupdate (const rbmstatevectorsref & ehxs,  const rbmstatevectorsref & v,
                                             float learningratepersample, double momentumpersample, bool resetmomentum, int deferupdate, modelupdateinfo & bpinfo /*float sparsethreshold, float nochangeifaboveorbelow*/)
    {
        if (deferupdate)                // TEMPORARY FIX --momentum does not work for batches of batches since it uses the wrong mbsize (from last mb only)
            momentumpersample = 0.0f;

        const size_t mbsize = ehxs.cols(); assert (v.cols() == mbsize);
        const float momentum = scalemomentum (momentumpersample, mbsize);  // map momentum to actual mb size  --compatible mode; will change

#define NEWGRADIENTSCALING // TODO: remove this #ifdef in the future
#ifdef NEWGRADIENTSCALING  // TODO: remove this #ifdef in the future
        const float gradientscaling = learningratepersample;    // learning rate is applied to gradients before momentum smoothing for consistency
        static bool f = false;
        if (!f)
        {
            f = true;
            fprintf (stderr, "backpropagationmodelupdate: new gradient scaling (by learning rate) enabled\n");
        }
#else   // old gradient scaling
        const float gradientscaling = 1.0f / (1.0f - momentum);                 // gradients are scaled by this
#endif
        const float inputweight = (1.0f - momentum) * gradientscaling;
        const float gradientweight = gradientscaling<1e-30? 1:learningratepersample / gradientscaling;

        if (deferupdate)
        {
            // special mode for big batching --we just add to the deltas without weighting it down
            updatedeltas (resetmomentum ? 0.0f : 1.0f, v, ehxs, inputweight, false/*updateb*/, bpinfo);    // just add
            if (deferupdate > 0)        // > 0 means just accumulate
                return;
            // perform the update (< 0 means that)
            adddeltas (gradientweight, false/*updateb*/);
            // and now scale it down for the next update (adding the current mb once again but with weight 0)
            updatedeltas (momentum, v, ehxs, 0.0f, false/*updateb*/, bpinfo);
        }
        else
        {
            // compute the deltas; keep previous deltas as "momentum" (unless 'resetmomentum')
            // Note: smoothed gradients are scaled by 1/(1-momentum).
            updatedeltas (resetmomentum ? 0.0f : momentum, v, ehxs, inputweight, false/*updateb*/, bpinfo);

            // Note: (1-momentum) is to unscale the scaled smoothed gradients, see above.
            adddeltas (gradientweight, false/*updateb*/);
        }

        if (bpinfo.sparsethreshold > 0)                // make weights sparse
            sparsifyweights (bpinfo.sparsethreshold);
    }

    // split (double-up) nodes
    // 'out' true means double the output nodes, else double the input nodes.
    // This is only defined for the forward direction. b is just updated in terms of dimension.
    // So far identical for all nodes. Change to virtual if not.
    void doublenodes (bool out)
    {
        srand ((unsigned int) W.rows());
        // double output nodes: perturb a little
        if (out)
        {
            // h = W' v + a
            matrix newW (W.rows(), 2 * W.cols());
            vector newa (2 * W.cols());
            foreach_column (j, W)
            {
                // Note: 'eps' might have to depend on the actual scale of the column,
                // which impacts the slope of the sigmoid. Now I just try to make it 'small'.
                foreach_row (i, W)
                {
                    const float eps = rand() * 0.01f / RAND_MAX;
                    newW(i,2*j)   = W(i,j) + eps;
                    newW(i,2*j+1) = W(i,j) - eps;
                }
                newa[2*j]   = a[j];
                newa[2*j+1] = a[j];
            }
            W = std::move (newW);
            a = std::move (newa);
        }
        // double input nodes: half the weights
        // Weights are halved because each input now exists twice (except small perturbance)
        else
        {
            // v = W h + b
            matrix newW (2 * W.rows(), W.cols());
            vector newb (2 * W.rows());
            foreach_row (i, W)
            {
                foreach_column (j, W)
                {
                    newW(2*i,j)   = W(i,j) * 0.5f;
                    newW(2*i+1,j) = W(i,j) * 0.5f;
                }
                // b is only updated as a formality; but it has no meaning, don't use it
                if (!b.empty())
                {
                    newb[2 * i]     = b[i];
                    newb[2 * i + 1] = b[i];
                }
            }
            W = std::move (newW);
            if (!b.empty())
                b = std::move (newb);
        }
    }

    virtual size_t getnumberofweightsets() const {return 1;}
    virtual pair<size_t, size_t> getweightsetdims(const size_t weightsetindex) const 
    {
        assert (weightsetindex < getnumberofweightsets());

        pair<size_t, size_t> p(W.rows(), W.cols());
        return p;
    }

    // set weights (this is to support hack experiments)
    template<class WTYPE, class ATYPE>
    void setweights (const WTYPE & newW, const ATYPE & newa, const size_t weightsetindex)
    {
        assert (newW.rows() == W.rows() && newW.cols() == W.cols());
        assert ((size_t) newa.size() == a.rows());
        assert (weightsetindex < getnumberofweightsets());

        foreach_coord (i, j, W)
            W(i,j) = (float) newW(i,j);
        foreach_index (i, newa)
            a[i] = (float) newa[i];
    }

protected:

    mutable cachedmatrix cachedWs;
    mutable cachedmatrix cachedWts;
    mutable cachedmatrix cachedvs;
    mutable cachedmatrix cachedhs;
    mutable cachedmatrix cacheda1s;
    mutable cachedmatrix cachedb1s;

    // apply the weight matrix to v plus bias to get h
    // ... TODO: The naming is bad. This is the input to the sigmoid, not h.
    // This function is shared across all types.
    void vtoh (const rbmstatevectorsref & v, rbmstatevectorsref & h) const
    {
        W.matprod_mtm (v, cachedWs, cachedvs, h, cachedhs, a, cacheda1s);     // h = W' v + a
    }

    void vtoh (const matrixstripe & v, matrixstripe& h, size_t i) const
    {
        W.matprod_col_mtm (v, h, a, i);    // h = W_i' v + a_i
    }

    // apply weights in reverse direction (reconstruction)
    void htov (const rbmstatevectorsref & h, rbmstatevectorsref & v) const
    {
        assert (h.cols() == v.cols());
        W.matprod_mm (h, cachedWts, cachedhs, v, cachedvs, b, cachedb1s);     // v = W h + b
    }

    // apply weights to error signal in reverse direction for error back-propagation
    // Difference to htov() is that no bias is added as this deals with error signals.
    void ehtoev (const rbmstatevectorsref & eh, rbmstatevectorsref & ev) const
    {
        assert (eh.cols() == ev.cols());
        W.matprod_mm (eh, cachedWts, cachedhs, ev, cachedvs);   // v = W h
    }

    // storage for updatedeltas() below
    // This means one instance of the model can only train once at the same time. Makes sense.
    vector sumhtmp;
    vector sumvtmp;
    matrix httmp;
    mutable cachedmatrix cachedhts;
    mutable cachedmatrix cachedvts;

    // update the deltas, with momentum
    //  - momentum is a 1st-order IIR low-pass, with non-unity gain G  --TODO: clean up these comments w.r.t. weight
    //    y(t+1) = momentum * y(t) + (1-momentum) * x(t) * G
    //    G=1/(1-momentum)
    //     - previous deltas are weighted down by feedbackweight, e.g. 0.9
    //     - gradient is added to it
    //       NOTE: gradient is weighted by (1-momentum) * G = 1, i.e. not weighted --we save a multiplication.
    //       Thus, the smoothed gradients are all too large by G=1/(1-momentum).
    //       So, remember to correct that when adding the smoothed gradients to the model parameters.
    //  - feedbackweight == 0 is used to reset the momentum (equivalent to x(t) = 0 for t < 0)
    // used for:
    //  - pretraining, second summand (negative) comes first: pass -momentum to flip sign
    //  - pretraining, first summand (positive): pass feedbackweight=-1.0 to flip sign once more
    //  - finetuning: also applies momentum, h is actually the error signal for h multiplied with the sigmoid' (except for linear fDLR layer)
    // Call this only on a single thread. Relies on keeping storage variables above around for avoiding mem alloc operations.
    void updatedeltas (const float feedbackweight,
                       const rbmstatevectorsref & v, const rbmstatevectorsref & h, const float inputweight, bool updateb)
    {
        assert (h.cols() == v.cols());  // cols = frames

        // da <- da * feedbackweight + sum (h)      * inputweight
        // dW <- dW * feedbackweight + sum (v * h') * inputweight
        // where h and v are matrices with columns = frames.
        // the momentum filter is implemented as:
        //  - feedbackweight = momentum
        //  - inputweight    = (1-momentum)

        // bias vectors
        assert (!da.empty() && !dW.empty());
        assert (!db.empty() || !updateb);
        da.scaleandaddallcols (feedbackweight, h, inputweight, sumhtmp);
        if (updateb)    // true for pre-training, false for back-propagation
            db.scaleandaddallcols (feedbackweight, v, inputweight, sumvtmp);

        // the matrix
        dW.scaleandaddmatprod (feedbackweight, v, h, inputweight, httmp, cachedvts, cachedhts);
    }

    void updatedeltas (const float feedbackweight,
                    const rbmstatevectorsref & v, const rbmstatevectorsref & h, const float inputweight,
                    bool updateb, modelupdateinfo & bpinfo)
    {
        updatedeltas (feedbackweight, v, h, inputweight, updateb);

        //apply regularization
        if (bpinfo.regtype == regCL || bpinfo.regtype == regCS)
        {
            const float threshold = fabs (bpinfo.nochangeifaboveorbelow);
            if (threshold > 0)
            {
                // TODO: interplay with momentum not clear
                if (bpinfo.nochangeifaboveorbelow > 0) 
                    dW.setto0ifabsabove2 (W, threshold);
                else
                    dW.setto0ifabsbelow2 (W, threshold);
            }
        }
        else if (bpinfo.regtype == regL2)
        {
            const rbmbase & rbmlayer = dynamic_cast<const rbmbase &> (*bpinfo.preflayer);
            const rmbmodelmatrix & Wref = rbmlayer.W;
            const rmbmodelmatrix & aref = rbmlayer.a;
            const float alpha = bpinfo.alpha * v.cols() * inputweight;  //adjust it based on number of frames

            assert(W.rows() == Wref.rows());
            assert(W.cols() == Wref.cols());
            assert(a.rows() == aref.rows());
            assert(a.cols() == aref.cols());

            // dW += alpha * (Wref - Wcur)
            // TODO: alpha interplays with scaling of gradient --ensure it is correct
            dW.addweighted (W, -alpha);
            dW.addweighted (Wref, alpha);
            da.addweighted (a, -alpha);
            da.addweighted (aref, alpha);
        }
    }

    // add deltas to parameters
    //  - deltas are already summed over the frames of the minibatch
    //  - 'gradientweight' parameter combines
    //     - per-frame learning rate
    //     - momentum complement (1-momentum) since smoothed gradients are too large by 1/(1-momentum),
    //       see comment at updatedeltas
    void adddeltas (float gradientweight, bool updateb)
    {
        // W += dW * learning rate
        // a += da * learning rate
        // b += db * learning rate  if 'updateb'

        W.addweighted (dW, gradientweight);
        a.addweighted (da, gradientweight);

        assert (!db.empty() || !updateb);
        if (updateb)
            b.addweighted (db, gradientweight);
    }

    // force sparseness to parameters
    void sparsifyweights (float threshold) 
    {
        W.setto0ifabsbelow (threshold);
    }
    virtual void blowup(const size_t blowupfactor) = 0;     //added by Hang Su adaptation
    void setlinearlayerweight(const matrix & adaptmatrix)
    {
        throw std::logic_error ("setlinearlayerweight: only accessible for linear layer");
    }
    virtual void blowup(const size_t blowupfactor, const std::vector<size_t> & statemap) = 0;
    void dumplayerpart() const                          //added by Hang Su adaptation
    {
        FILE *filetowrite = fopen ("dumpedlayer.txt" , "w");
        fprintfOrDie(filetowrite, "W: [ ");
        fflush(filetowrite);
        for (size_t j = 0; j < 1/*W.cols()*/; j++)
        {
            for (size_t i = 0; i < W.rows(); i++)
            {
                if(i == 0 && j > 0)   
                    fprintf(filetowrite, ";\n");
                fprintf(filetowrite, "%.4f ", W(i, j));
            }
            fprintf(filetowrite, ";]\n");
            fflush(filetowrite);
        }
        for (size_t j = W.cols()/2; j < W.cols()/2+1; j++)
        {
            for (size_t i = 0; i < W.rows(); i++)
            {
                if(i == 0 && j > 0)   
                    fprintf(filetowrite, ";\n");
                fprintf(filetowrite, "%.4f ", W(i, j));
            }
            fprintf(filetowrite, ";]\n");
            fflush(filetowrite);
        }
        fflush(filetowrite);
        fprintf(filetowrite, "a: [ ");
        for (size_t j = 0; j < a.cols(); j++)
        {
            for (size_t i = 0; i < a.rows(); i++)
            {
                if(i == 0 && j > 0)   
                    fprintf(filetowrite, ";\n");
                fprintf(filetowrite, "%.4f ", a(i, j));
            }
            fprintf(filetowrite, ";]\n");
        }
        fflush(filetowrite);
        fclose(filetowrite);
    }
};

// ===========================================================================
// RBM -- implementation of a Restricted Boltzman Machine
// ===========================================================================

class rbm : public rbmbase
{
public:
    // note: constructor swaps matrices in, i.e. is destructive (to save memory)
    // ... TODO: change to std::move()
    rbm (matrix && pW, vector && pa, vector && pb)
    {
        W = std::move (pW);
        a = std::move (pa);
        b = std::move (pb);
    }
    rbm (size_t vdim, size_t hdim, unsigned int randomseed)
    {
        W.resize (vdim, hdim);
        a.resize (hdim, 1);
        b.resize (vdim, 1);
        validatedims();
        initrandom (randomseed);
    }
    rbm (FILE * f) : rbmbase (f) { validatedims(); }

    // forward propagation
    // This code is shared between Gaussian-Bernoulli and Bernoulli-Bernoulli networks
    // v and Ph are blocks of column vectors
    virtual void forwardprop (const rbmstatevectorsref & v, rbmstatevectorsref & Ph, const bool linearonly = false) const
    {
        vtoh (v, Ph);   // P(h=1|v) = W' v + a
        if ((!linearonly)&&(!flag)) Ph.sigmoid();
#ifdef SAMPLING_EXPERIMENT
        static unsigned int randomseed = 0;
        Ph.samplebinary (Ph, randomseed);
        randomseed++;
#endif
    }


protected:

    // perform CD-1 statistics for pretraining (mostly identical for Gauss and Bernoulli case)
    // Ph is the probability for binary h value being 1
    void pretrainingcd1 (const rbmstatevectorsref & Ph, rbmstatevectorsref & v1, rbmstatevectorsref & h1, unsigned int randomseed,
                               bool dovsigmoid) const
    {
        // randomly sample binary h according to distribution (binary)
        // We use h1 temporarily as a buffer (will be overwritten below).
        // This takes ~7% of the runtime.
        h1.samplebinary (Ph, randomseed);

        // reconstruct v = W h + b  (use probability per Section 3.2 of guideTR.pdf)
        htov (h1, v1);

        // and v <- sigmoid (v) except for first (Gaussian) layer
        if (dovsigmoid)
            v1.sigmoid();

        // compute output probabilities h = sigmoid (v' W + a)
        forwardprop (v1, h1);
    }


    /* steps for pretraining
    # steps: (see Section 2 in guideTR.pdf)
    #  - normal forward propagation
    #  - add the current h, v, and v h' to deltas (positive term in Eq. (5) in guideTR.pdf)
    #    also apply momentum to current deltas
    #  - randomly sample binary h according to distribution (binary)
    #  - reconstruct v = W h + a  (use probability per Section 3.2 of guideTR.pdf)
    #    and v <- sigmoid (v) except for first (Gaussian) layer
    #  - compute output probabilities h = sigmoid (v' W + b)
    #  - subtract the new h, v, and v h' from deltas (negative term)
    */
    // update models for pretraining
    // ('virtual' only because top layer does not implement this--it throws)
    virtual void pretrainingmodelupdate (const rbmstatevectorsref & v, const rbmstatevectorsref & h, rbmstatevectorsref & v1, rbmstatevectorsref & h1,
                                         float learningratepersample, double momentumpersample, bool resetmomentum)
    {
        const size_t mbsize = v.cols(); assert (h.cols() == mbsize);
        const float momentum = scalemomentum (momentumpersample, mbsize);  // map momentum to actual mb size  --compatible mode; will change

#ifdef NEWGRADIENTSCALING  // TODO: remove this #ifdef in the future
        const float gradientscaling = learningratepersample;    // learning rate is applied to gradients before momentum smoothing for consistency
#else   // old gradient scaling
        const float gradientscaling = 1.0f / (1.0f - momentum);                 // gradients are scaled by this
#endif
        const float inputweight = (1.0f - momentum) * gradientscaling;
        const float gradientweight = learningratepersample / gradientscaling;

        // compute the deltas; keep previous deltas as "momentum" (unless 'resetmomentum')

        // We compute this with a little trick:
        //  dX = (momentum * dX) + Xpos - Xneg
        //     = -((-momentum * dX) + Xneg) + Xpos

        // the negative summand
        //checknan (v1); checknan (h1);
        // Note: smoothed gradients are scaled by 1/(1-momentum).
        updatedeltas (resetmomentum ? 0.0f : -momentum, v1, h1, inputweight, true/*updateb*/);

        // the positive summand
        updatedeltas (-1.0f, v, h, inputweight, true/*updateb*/);

        adddeltas (gradientweight, true/*updateb*/);
    }

    // backward error propagation
    virtual void backpropagationstats (rbmstatevectorsref & /*in/out*/ eh, const rbmstatevectorsref & h, rbmstatevectorsref & /*out*/ ev) const
    {
        assert (eh.cols() == h.cols() && eh.rows() == h.rows());

        // compute 'dW' = [ dh/d(w(i,j)) ] and 'da' = [ dh/d(a[i]) ]
        // eh = hdesired - h
        // err = eh .* h .* (1 - h)
        // h .* (1 - h) = derivative of sigmoid
        // update 'eh' in place for later use in accumulation

        // multiply by derivative
        // This is done in place because we need the very same product later in the model update.
        // Note that eh no longer corresponds to e^l in the paper after this operation.
       if(!flag) eh.mulbydsigm (h);

        // divided the log (us + epison) for exerting log function on hidden layer. [v-xieche]
#ifdef LOGINSIGMOID
        eh.divideaddsigmoid (h);
#endif
        // return value 'ev' is error back-propagated through network, to pass to next lower layer
        if (!ev.empty())    // (bottom level does not need this--pass an empty matrix)
            ehtoev (eh, ev);  // ev = W eh  (eh is the updated one)
    }

    virtual void blowup(const size_t blowupfactor)        // added by Hang Su adaptation
    {
//        dumplayerpart();            //do a check
        const size_t wrowsori = W.rows();
        const size_t wcolsori = W.cols();
        numstream = blowupfactor;
        vdimnumroundup = W.colstride() - wrowsori;
        rmbmodelmatrix Wbackup;
        Wbackup.resize (wrowsori , wcolsori);
        foreach_coord(i,j,W)  Wbackup(i,j) = W(i,j);
        const size_t wrowblowed = (wrowsori + vdimnumroundup) * blowupfactor - vdimnumroundup;  // the last stream does not need round up
        const size_t wcolblowed = wcolsori * blowupfactor;
        W.resize(wrowblowed, wcolblowed);
        foreach_coord(i,j,W)  W(i,j) = 0;

        const size_t arowsori = a.rows();
        const size_t acolsori = a.cols();
        rmbmodelmatrix abackup;
        abackup.resize (arowsori , acolsori);
        foreach_coord(i,j,a)  abackup(i,j) = a(i,j);
        a.resize( arowsori * blowupfactor , acolsori );

        const size_t browsori = b.rows();
        const size_t bcolsori = b.cols();
        rmbmodelmatrix bbackup;
        bbackup.resize (browsori , bcolsori);
        foreach_coord(i,j,b)  bbackup(i,j) = b(i,j);
        b.resize( (browsori + vdimnumroundup) * blowupfactor - vdimnumroundup, bcolsori);
        foreach_coord(i,j,b)   b(i,j) = 0;

        for (size_t blockindex = 0; blockindex < blowupfactor; blockindex++)
        {
            for (size_t i = 0 ; i < wrowsori; i++)
            {
                for (size_t j = 0 ; j < wcolsori; j++)
                    W(i + blockindex * (vdimnumroundup + wrowsori),j + blockindex * wcolsori) = Wbackup(i,j);
            }
            for (size_t i = 0; i < arowsori; i++ )
                a(i + blockindex * arowsori, 0) = abackup(i, 0);
            for (size_t i = 0; i < browsori; i++ )
                b(i + blockindex * browsori, 0) = bbackup(i, 0);
        }
        validatedims();
//        dumplayerpart();            //do a check
    }

    virtual void blowup(const size_t blowupfactor, const std::vector<size_t> & statemap)  //added by Hang Su adaptation
    {
        throw std::logic_error ("blowup: rbm layer shall not use this function with statemap");
    }
};

// ===========================================================================
// rbmgaussbernoulli -- RBM with continuous input (Gaussian)
// ===========================================================================

class rbmgaussbernoulli : public rbm
{
public:
    rbmgaussbernoulli (matrix && W, vector && a, vector && b) : rbm (std::move (W), std::move (a), std::move (b)) {}
    rbmgaussbernoulli (size_t vdim, size_t hdim, unsigned int randomseed) : rbm (vdim, hdim, randomseed) {}
	rbmgaussbernoulli (FILE * f, const char islinear) : rbm (f) {flag = islinear;} // Jian changed for svd re-training
    rbmgaussbernoulli (FILE * f) : rbm (f) {}
	//rbmbernoullibernoulli (const HANDLE f) : rbm (f) {}
    //virtual rbmbase * clone() const { return new rbmgaussbernoulli (*this); }
    virtual string type() const { return "rbmgaussbernoulli"; }

    // perform CD-1 statistics for pretraining
    // Ph is the probability for binary h value being 1
    virtual void pretrainingstats (const rbmstatevectorsref & Ph, rbmstatevectorsref & v1, rbmstatevectorsref & h1, unsigned int randomseed) const
    {
        pretrainingcd1 (Ph, v1, h1, randomseed, false/*no sigmoid for v for Gauss*/);
    }

    // TODO: these should be moved to the base class [fseide]
    virtual const matrix & peekweightmatrix() const
    {
        return W.peek();
    }

    virtual const vector & peekbias() const 
    {
        return a.peek();
    }

    virtual void forwardpropwithoutnonlinearity (const matrixstripe & v, matrixstripe & u, size_t i) const 
    {
        throw std::logic_error ("forwardpropwithoutnonlinearity: not implemented");
    }
};

// ===========================================================================
// rbmbernoullibernoulli -- RBM with binary input
// ===========================================================================

class rbmbernoullibernoulli : public rbm
{
public:
    rbmbernoullibernoulli (matrix && W, vector && a, vector && b) : rbm (std::move (W), std::move (a), std::move (b)) {}
    rbmbernoullibernoulli (size_t vdim, size_t hdim, unsigned int randomseed) : rbm (vdim, hdim, randomseed) {}
    rbmbernoullibernoulli (FILE * f) : rbm (f) {}
	rbmbernoullibernoulli (FILE * f,const char islinear) : rbm (f) {flag=islinear;} // Jian changed for low-rank re-training
	/* not sure whether it is really needed for only forward progation 
	rbmbernoullibernoulli (std::vector<std::vector<float>> &v, int dimm, int dimn): rbm (dimm, dimn, (unsigned int) time(NULL)){  // Jian added for svd decomposition
		flag=1;
		W.setvalue(v);

		a.initialize();
		b.initialize();
	}
	*/
    //virtual rbmbase * clone() const { return new rbmbernoullibernoulli (*this); }
    virtual string type() const { return "rbmbernoullibernoulli"; }

    // perform CD-1 statistics for pretraining
    // Ph is the probability for binary h value being 1
    virtual void pretrainingstats (const rbmstatevectorsref & Ph, rbmstatevectorsref & v1, rbmstatevectorsref & h1, unsigned int randomseed) const
    {
        pretrainingcd1 (Ph, v1, h1, randomseed, true/*sigmoid for v for Bernoulli*/);
    }
    
    // TODO: these should be moved to the base class [fseide]
    virtual const matrix & peekweightmatrix() const
    {
        return W.peek();
    }

    virtual const vector & peekbias() const 
    {
        return a.peek();
    }

    virtual void forwardpropwithoutnonlinearity (const matrixstripe & v, matrixstripe & u, size_t i) const 
    {
        throw std::logic_error ("forwardpropwithoutnonlinearity: not implemented");
    }
};

// ===========================================================================
// perceptron -- for top layer
// TODO: This name is not accurate--a perceptron has no non-linearity.
// ===========================================================================

class perceptron : public rbmbase
{
    void validatedims() const   // check if dimensions match
    {
        if (W.cols() != a.rows())
            malformed ("invalid model file--W matrix dimensions mismatch bias dimensions");
    }
public:
    perceptron (matrix && pW, vector && pa)
    {
        W = std::move (pW);
        a = std::move (pa);
        validatedims();
    }
    perceptron (size_t vdim, size_t hdim, unsigned int randomseed)
    {
        W.resize (vdim, hdim);
        a.resize (hdim, 1);
        validatedims();
        initrandom (randomseed);
    }
    perceptron (FILE * f) : rbmbase (f) { validatedims(); }
    //virtual rbmbase * clone() const { return new perceptron (*this); }
    virtual string type() const { return "perceptron"; }

    // forward propagation
    virtual void forwardprop (const rbmstatevectorsref & v, rbmstatevectorsref & Pu, const bool linearonly=false) const
    {
        if (linearonly) throw std::logic_error ("forwardprop: linear only is not implemented yet for top layer");

        vtoh (v, Pu);   // h = W' v + a
#ifdef MULTI_LOGISTIC_REGRESSION
		Pu.sigmoid();
#else
		#ifdef MCE_LINEAR
		//linear transform
		#else
			Pu.softmax();
		#endif
#endif
    }

    // this computes only component i of the output vector
    // Special function supposed to be used for on-demand LL evaluation.
    virtual void forwardpropwithoutnonlinearity (const matrixstripe & v, matrixstripe & u, size_t i) const
    {
        vtoh (v, u, i);   // u = w_i' v + a_i
    }

    virtual void pretrainingstats (const rbmstatevectorsref & Ph, rbmstatevectorsref & v1, rbmstatevectorsref & h1, unsigned int randomseed) const
    {
        Ph; v1; h1; randomseed;
        throw std::logic_error ("pretrainingstats: cannot be called on top layer");
    }

    virtual void pretrainingmodelupdate (const rbmstatevectorsref & v, const rbmstatevectorsref & h, rbmstatevectorsref & v1, rbmstatevectorsref & h1,
                                         float learningratepersample, double momentumpersample, bool resetmomentum)
    {
        v; h; v1; h1; learningratepersample; momentumpersample; resetmomentum;
        throw std::logic_error ("pretrainingmodelupdate: cannot be called on top layer");
    }

    // backward error propagation
    virtual void backpropagationstats (rbmstatevectorsref & /*in/out*/ eh, const rbmstatevectorsref & h, rbmstatevectorsref & /*out*/ ev) const
    {
        h;  // not used here
        // and for this type of model, 'eh' does not get modified
        // return value 'ev' is error back-propagated through network, to pass to next lower layer
        // BUGBUG: why do we need to check for empty? The bottom level is never a perceptron!
        if (!ev.empty())    // (bottom level does not need this--pass an empty matrix)
            ehtoev (eh, ev);  // ev = W eh
    }
    virtual void blowup(const size_t blowupfactor)
    {
        throw std::logic_error ("blowup: perceptron layer shall not use this function without statemap");
    }
    void blowup(const size_t blowupfactor, const std::vector<size_t> & statemap)        // added by Hang Su adaptation, not completed, shall make use of state mapping
    {
        const size_t wrowsori = W.rows();
        const size_t wcolsori = W.cols();
        vdimnumroundup = 0;
        numstream = blowupfactor;
        rmbmodelmatrix Wbackup;
        Wbackup.resize (wrowsori , wcolsori);
        for( size_t j = 0 ; j < W.cols(); j++)
        {
            for ( size_t i = 0 ; i < W.rows(); i++)
            {
                Wbackup(i,j) = W(i,j);
            }
        }
        W.resize( wrowsori * blowupfactor , wcolsori);

        const size_t browsori = b.rows();
        const size_t bcolsori = b.cols();
        b.resize( browsori * blowupfactor , bcolsori);

        // there is no need to blow up "a"
        for (size_t blockindex = 0; blockindex < blowupfactor; blockindex++)
        {
            for (size_t j = 0 ; j < wcolsori; j++)
            {
                for (size_t i = 0 ; i < wrowsori; i++)
                {
                    if( statemap[j] == blockindex )
                        W(i + blockindex * wrowsori,j) = Wbackup(i,j);
                    else
                        W(i + blockindex * wrowsori,j) = 0;
                }
            }
            for (size_t i = 0; i < browsori; i++)
                b(i + blockindex*browsori , 0) = 0;
        }
    }
};

// ===========================================================================
// linear transformation -- to inject below bottom layer
// This performs a linear transform only.
// The class supports reduced-dimension mappings (block-diagonal structure)
// for use with neighbor-frame augmented input feature vectors:
//  - 'diagblocks' diagonal block matrices. All outside elements are 0.
//    This should be either 1 (no block-diag structure) or equal to the
//    number of neighbor frames used in the input feature vector.
//  - the diag blocks can be pooled or non-pooled ('poolblocks')
//    This is implemented in training, actual matrix contains copies.
// ===========================================================================

class linearnetwork : public rbmbase
{
    size_t diagblocks;          // e.g. 11 (must match neighbor expansion; 1 for full matrix)
    bool poolblocks;            // true -> blocks of neighbor frames are pooled
    void validatedims() const   // check if dimensions match
    {
        if (W.cols() != a.rows())
            malformed ("invalid model file--W matrix dimensions mismatch bias dimensions");
    }
public:
#if 0   // do we ever need this?
    linearnetwork (matrix && pW, vector && pa)
    {
        W = std::move (pW);
        a = std::move (pa);
        validatedims();
    }
#endif
    linearnetwork (size_t vdim, size_t hdim, size_t diagblocks, bool poolblocks)
        : diagblocks (diagblocks), poolblocks (poolblocks)
    {
        W.resize (vdim, hdim);
        a.resize (hdim, 1);
        validatedims();
        // initialize to identity (does nothing unless trained)
        initidentity();
        setbiaszero();
        fprintf (stderr, "linearnetwork: %d diagonal blocks for %d x %d network, %spooled\n", diagblocks, vdim, hdim, poolblocks ? "" : "not ");
        vdimnumroundup = 0;
        hdimnumroundup = 0;
        numstream = 1;
    }

    linearnetwork (FILE * f) : rbmbase (f)
    {
        validatedims();
        fcheckTag (f, "BFLR");
        diagblocks = fgetint (f);
        poolblocks = fgetint (f) != 0;
        fcheckTag (f, "EFLR");
        numstream = W.cols() / W.rows();
        if ( numstream == 1 )
            hdimnumroundup = 0;
        else
            hdimnumroundup = (W.cols() - W.rows() * numstream) / (numstream - 1);
    }
    //virtual rbmbase * clone() const { return new linearnetwork (*this); }
    virtual string type() const { return "linearnetwork"; }


    // overloaded so we can write extra information
    virtual void write (FILE * f) const
    {
        rbmbase::write (f);
        fputTag (f, "BFLR");
        fputint (f, (int) diagblocks);
        fputint (f, poolblocks ? 1 : 0);
        fputTag (f, "EFLR");
    }

    // forward propagation
    virtual void forwardprop (const rbmstatevectorsref & v, rbmstatevectorsref & Ph, const bool linearonly=false) const
    {
        vtoh (v, Ph);   // h = W' v + a
    }

    virtual void pretrainingstats (const rbmstatevectorsref & Ph, rbmstatevectorsref & v1, rbmstatevectorsref & h1, unsigned int randomseed) const
    {
        Ph; v1; h1; randomseed;
        throw std::logic_error ("pretrainingstats: linearnetwork does not support pre-training");
    }

    virtual void pretrainingmodelupdate (const rbmstatevectorsref & v, const rbmstatevectorsref & h, rbmstatevectorsref & v1, rbmstatevectorsref & h1,
                                    float learningratepersample, double momentumpersample, bool resetmomentum)
    {
        v; h; v1; h1; learningratepersample; momentumpersample; resetmomentum;
        throw std::logic_error ("pretrainingmodelupdate: linearnetwork does not support pre-training");
    }

    // linearnetwork needs special version of this to implement block-diagonal structure
    virtual void backpropagationmodelupdate (const rbmstatevectorsref & ehxs, const rbmstatevectorsref & v,
                                             float learningratepersample, double momentumpersample, bool resetmomentum, int deferupdate, modelupdateinfo & bpinfo)
    {
        // first do the default update...
        bpinfo.nochangeifaboveorbelow = 0;  /*set to 0 for linear layer*/
        rbmbase::backpropagationmodelupdate (ehxs, v, learningratepersample, momentumpersample, resetmomentum, deferupdate, bpinfo);

        // ...and then post-process to enforce block-diagonal structure of matrix
        if (deferupdate > 0)
            return;
        bool setidentity = false;
        if (diagblocks == vdim())            // [v-hansu] indicate topsecond layer
            setidentity = true;
        W.setblockdiagonal (diagblocks, poolblocks, numstream, hdimnumroundup,setidentity);
        a.setblockdiagonal (diagblocks, poolblocks, numstream, hdimnumroundup, false);
    }

    // backward error propagation
    virtual void backpropagationstats (rbmstatevectorsref & /*in/out*/ eh, const rbmstatevectorsref & h, rbmstatevectorsref & /*out*/ ev) const
    {
        // for a linear layer, there is nothing to do here
        h; eh; ev;  // not used here

        // return value 'ev' is error back-propagated through network, to pass to next lower layer
        // TODO: Check the math whether this is actually correct once we ever use this as an intermediate rather than bottom layer.
        if (!ev.empty())    // (bottom level does not need this--pass an empty matrix)
            ehtoev (eh, ev);  // ev = W eh
    }

    virtual const matrix & peekweightmatrix() const
    {
        return W.peek();
    }

    virtual const vector & peekbias() const 
    {
        return a.peek();
    }

    virtual void forwardpropwithoutnonlinearity (const matrixstripe & v, matrixstripe & u, size_t i) const 
    {
        throw std::logic_error ("forwardpropwithoutnonlinearity: not implemented");
    }

    virtual void blowup(const size_t blowupfactor)        // added by Hang Su adaptation
    {
        const size_t wrowsori = W.rows();
        const size_t wcolsori = W.cols();
        hdimnumroundup = W.colstride() - wrowsori;
        vdimnumroundup = 0;         //remember vdimroundup only record the vdim roundup, so it shall be set to 0;
        numstream = blowupfactor;
        const size_t arowsori = a.rows();
        const size_t acolsori = a.cols();

        rmbmodelmatrix Wbackup;
        Wbackup.resize (wrowsori , wcolsori);
        foreach_coord(i, j, W)  Wbackup(i,j) = W(i,j);
        W.resize( wrowsori , (wcolsori + hdimnumroundup) * blowupfactor - hdimnumroundup);
        foreach_coord( i, j, W)    W(i, j) = 0;

        a.resize( (arowsori + hdimnumroundup) * blowupfactor - hdimnumroundup, acolsori );
        foreach_coord( i, j, a )    a(i, j) = 0;

        for (size_t blockindex = 0; blockindex < blowupfactor; blockindex++)
        {
            for( size_t j = 0; j < wcolsori; j++)
                for( size_t i = 0; i < wrowsori; i++)
                    W(i, j + blockindex * (wcolsori + hdimnumroundup)) = Wbackup(i,j);
            for (size_t j = 0; j < acolsori; j++) 
                for (size_t i = 0; i < arowsori; i++)
                    a(i + blockindex * (arowsori + hdimnumroundup), j) = 0;
        }
        validatedims();
    }

    virtual void blowup(const size_t blowupfactor, const std::vector<size_t> & statemap)  //added by Hang Su adaptation
    {
        throw std::logic_error ("blowup: rbm layer shall not use this function with statemap");
    }

    virtual void setlinearlayerweight(const matrix & adaptmatrix)
    {
        fprintf(stderr,"setlinearlayerweight: initial adaptation matrix to GMM adaptation matrix");
        initidentity(adaptmatrix);
        setbiaszero();
    }
};


class leakyrootnetwork : public rbm
{
	size_t rootorder;       // e.g. 5 for 5-th root; or 1 for leaky relu
	float leakiness;        // e.g. 0.01
public:

	leakyrootnetwork(matrix && W, vector && a, vector && b) : rbm(std::move(W), std::move(a), std::move(b)) {}
	leakyrootnetwork(size_t vdim, size_t hdim, unsigned int randomseed) : rbm(vdim, hdim, randomseed) {}
	
	// constructors: from scratch and from file

	template<typename FILEHANDLETYPE> leakyrootnetwork(FILEHANDLETYPE f) : rbm(f)
	{
		if (nonlinearitykind != leakyrootkind)
			throw std::logic_error("leakyrootnetwork: unexpectedly failed to read nonlinearitykind from file");
		fcheckTag(f, "BLRN");
		rootorder = fgetint(f);
		leakiness = fgetfloat(f);
		fcheckTag(f, "ELRN");
	}

	// overridden so we can write extra information
	template<typename FILEHANDLETYPE>
	void dowrite(FILEHANDLETYPE f) const
	{
		rbmbase::write(f);
		fputTag(f, "BLRN");
		fputint(f, (int)rootorder);
		fputfloat(f, leakiness);
		fputTag(f, "ELRN");
	}
	virtual void write(FILE * f) const { dowrite(f); }
//	virtual void write(HANDLE f) const { dowrite(f); }


protected:

	virtual string type() const { return "leakyrootnetwork"; }

	virtual void forwardprop(const rbmstatevectorsref & v, rbmstatevectorsref & Eh, const bool linearonly = false) const
	{
		vtoh(v, Eh);   // z = W' v + a


		Eh.leakyroot(rootorder, leakiness);
	}

	// backward error propagation
	virtual void backpropagationstats(rbmstatevectorsref & /*in/out*/ eh, const rbmstatevectorsref & h, rbmstatevectorsref & /*out*/ ev) const
	{
		assert(eh.cols() == h.cols() && eh.rows() == h.rows());

		// compute 'dW' = [ dh/d(w(i,j)) ] and 'da' = [ dh/d(a[i]) ]
		// err = eh .* derivative of non-linearity computed from its output value
		// update 'eh' in place for later use in accumulation

		// multiply by derivative
		eh.mulbydleakyroot(h, rootorder, leakiness);

		if (!ev.empty())    // (bottom level does not need this--pass an empty matrix)
			ehtoev(eh, ev);  // ev = W eh  (eh is the updated one)
		//ev.dump("error signal");
	}

	virtual void pretrainingstats(const rbmstatevectorsref & Eh, rbmstatevectorsref & v1, rbmstatevectorsref & h1, unsigned int randomseed) const
	{
		Eh; v1; h1; randomseed;
		throw std::logic_error("pretrainingstats: not implemented for leakyrootnetworks for now");
	}

	virtual void pretrainingmodelupdate(const rbmstatevectorsref & v, const rbmstatevectorsref & h, rbmstatevectorsref & v1, rbmstatevectorsref & h1,
		float learningratepersample, double momentumpersample)
	{
		v; h; v1; h1; learningratepersample; momentumpersample;
		throw std::logic_error("pretrainingmodelupdate: not implemented for leakyrootnetworks for now");
	}

	// computation of deltavs for unseen state compensation [v-hansu]
	virtual float forwardpropdelta(rbmstatevectorsref & deltah, const rbmstatevectorsref & deltav, const rbmstatevectorsref & h,
		/*const*/ rbmstatevectorsref & v, /*const*/ rbmstatevectorsref & eh, rbmstatevectorsref & vnorms,
		const float learningrateperframe, const double momentumpersample) const
	{
		throw::logic_error("forwardpropdelta: not implemented for leakyrootnetworks");
	}

	virtual void blowup(const size_t blowupfactor, const std::vector<size_t> & statemap)  //added by Hang Su adaptation
	{
		throw std::logic_error("blowup: relunetwork layer shall not use this function with statemap");
	}

	virtual void blowup(const size_t blowupfactor)
	{
		throw::logic_error("blowup: not implemented for leakyrootnetworks");
	}

	virtual void forwardpropwithoutnonlinearity(const matrixstripe & v, matrixstripe & u, size_t i) const
	{
		throw std::logic_error("forwardpropwithoutnonlinearity: not implemented for leakyrootnetworks");
	}
};

class relunetwork : public rbm
{
public:

	relunetwork(matrix && W, vector && a, vector && b) : rbm(std::move(W), std::move(a), std::move(b)) {}
	relunetwork(size_t vdim, size_t hdim, unsigned int randomseed) : rbm(vdim, hdim, randomseed) {}

	// constructors: from scratch and from file

	template<typename FILEHANDLETYPE> relunetwork(FILEHANDLETYPE f) : rbm(f){}

	// overridden so we can write extra information
	//	virtual void write(HANDLE f) const { dowrite(f); }


protected:

	virtual string type() const { return "relunetwork"; }

	virtual void forwardprop(const rbmstatevectorsref & v, rbmstatevectorsref & Eh, const bool linearonly = false) const
	{
		vtoh(v, Eh);   // z = W' v + a


		Eh.setto0ifbelow(0.0f);
	}

	// backward error propagation
	virtual void backpropagationstats(rbmstatevectorsref & /*in/out*/ eh, const rbmstatevectorsref & h, rbmstatevectorsref & /*out*/ ev) const
	{
		assert(eh.cols() == h.cols() && eh.rows() == h.rows());

		// compute 'dW' = [ dh/d(w(i,j)) ] and 'da' = [ dh/d(a[i]) ]
		// err = eh .* derivative of non-linearity computed from its output value
		// update 'eh' in place for later use in accumulation

		// multiply by derivative
		eh.mulbydlru(h);

		if (!ev.empty())    // (bottom level does not need this--pass an empty matrix)
			ehtoev(eh, ev);  // ev = W eh  (eh is the updated one)
		//ev.dump("error signal");
	}

	virtual void pretrainingstats(const rbmstatevectorsref & Eh, rbmstatevectorsref & v1, rbmstatevectorsref & h1, unsigned int randomseed) const
	{
		Eh; v1; h1; randomseed;
		throw std::logic_error("pretrainingstats: not implemented for leakyrootnetworks for now");
	}

	virtual void pretrainingmodelupdate(const rbmstatevectorsref & v, const rbmstatevectorsref & h, rbmstatevectorsref & v1, rbmstatevectorsref & h1,
		float learningratepersample, double momentumpersample)
	{
		v; h; v1; h1; learningratepersample; momentumpersample;
		throw std::logic_error("pretrainingmodelupdate: not implemented for leakyrootnetworks for now");
	}

	// computation of deltavs for unseen state compensation [v-hansu]
	virtual float forwardpropdelta(rbmstatevectorsref & deltah, const rbmstatevectorsref & deltav, const rbmstatevectorsref & h,
		/*const*/ rbmstatevectorsref & v, /*const*/ rbmstatevectorsref & eh, rbmstatevectorsref & vnorms,
		const float learningrateperframe, const double momentumpersample) const
	{
		throw::logic_error("forwardpropdelta: not implemented for leakyrootnetworks");
	}

	virtual void blowup(const size_t blowupfactor, const std::vector<size_t> & statemap)  //added by Hang Su adaptation
	{
		throw std::logic_error("blowup: relunetwork layer shall not use this function with statemap");
	}

	virtual void blowup(const size_t blowupfactor)
	{
		throw::logic_error("blowup: not implemented for leakyrootnetworks");
	}

	virtual void forwardpropwithoutnonlinearity(const matrixstripe & v, matrixstripe & u, size_t i) const
	{
		throw std::logic_error("forwardpropwithoutnonlinearity: not implemented for leakyrootnetworks");
	}
};



};};
