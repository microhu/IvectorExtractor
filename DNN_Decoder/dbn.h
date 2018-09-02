// dbn.h -- implementation of Hinton's Deep Belief Network
//
// F. Seide, Nov 2010 based on code provided by Yu Dong, MSR Speech Research Group

#pragma once

#include "rbm.h"
#include "dtnn.h"
#include <regex>
#include <queue>
namespace msra { namespace dbn {

// ===========================================================================
// annlayerfactory -- factory class to help with construction from a given type string
// ===========================================================================
class annlayerfactory
{
public:
    // create and read a network, given a type string
    static inline Iannlayer * createfromfile (const string & type, FILE * f)
    {
        if (type == "rbmgaussbernoulli")
            return new rbmgaussbernoulli (f);
        else if (type == "rbmbernoullibernoulli")
            return new rbmbernoullibernoulli (f);
		else if (type == "rbmisalinearbernoulli")  // Jian added for svd
            return new rbmbernoullibernoulli (f,1);
		else if (type == "perceptron")
			return new perceptron(f);
		else if (type == "relunetwork")
			return new relunetwork(f);
        else if (type == "linearnetwork")
            return new linearnetwork (f);
        else if (type == "dtnn")
            return new dtnn (f);
        else
            throw runtime_error ("createfromfile: invalid model type: " + type);
    }

    static inline Iannlayer * create(const string &type, size_t vdim, std::vector<size_t> hdims, unsigned int randomseed)
    {
        if (type == "perceptron" || type == "softmax" || type == "sm" )  // add a perception layer 
            return new perceptron (vdim, hdims[0], randomseed);
        else if (type == "rbmbernoullibernoulli" || type == "bb")
            return new rbmbernoullibernoulli (vdim, hdims[0], randomseed);
        else if (type == "rbmgaussbernoulli" || type == "gb")
            return new rbmgaussbernoulli(vdim, hdims[0], randomseed);
        else if (type == "linearnetwork" || type == "ln")
            return new linearnetwork(vdim, hdims[0], 11/*or 1 for non-pooled*/, true); 
		else if (type == "dtnn" || type == "tn")
			return new dtnn(vdim, hdims[0], hdims[1], randomseed);
		else if (type == "relunetwork")
			return new relunetwork(vdim, hdims[0], randomseed);
        else
            throw runtime_error ("create: invalid model type: " + type);
    }
};

#if 1 // TODO: move it to a proper place.[v-xieche]
void copydata (msra::cuda::matrix &dst, msra::cuda::matrix &src, msra::math::ssematrix<msra::math::ssematrixbase> &bufmatrix, int copyFlags)
{
    dst.assign (src, &bufmatrix(0,0), bufmatrix.getcolstride(), false, copyFlags);
}
#endif

// state mapping for class based adaptation __added by Hang Su adaptation

// ===========================================================================
// a DBN model
// This is a stack of RBMs (it also holds mean/std for v norm and Pu for u norm).
// ===========================================================================
class model
{
    vector mean, std;                                 // input normalization
    std::vector<unique_ptr<Iannlayer>> layers;          // the layers
    vector Pu;                                        // prior probs over u (empty until we have a supervised layer)
    bool computing;                                   // entercomputation() called?

    model & operator= (const model &);
    model (const model &);
    static void malformed (string msg) { throw runtime_error ("dbnmodel: invalid model: " + msg); }

    void read (FILE * f)
    {
        fcheckTag (f, "DBN\n");
        char buf[10000];
        fgetstring (f, buf);
        fcheckTag (f, "BDBN");
        int version = fgetint (f);
        if (version != 0) malformed ("unsupported version number");
        int numlayers = fgetint (f);

        mean.read (f, "gmean");
        std.read (f, "gstddev");
        if (mean.size() != std.size()) malformed ("inconsistent size of mean and std vector");

        // read all layers
        layers.resize (numlayers);
        fcheckTag (f, "BNET");
        foreach_index (i, layers)
        {
            char buf[100];
            string type = fgetstring (f, buf);
            layers[i].reset (annlayerfactory::createfromfile (type, f));
        }
        fcheckTag (f, "ENET");
        if (!layers.empty() && layers[0]->vdim() % mean.size() != 0) malformed ("inconsistent first-level weight dim and mean/std");

        // read Pu (priors) if we have a top layer
        if (!layers.empty())
        {
            size_t toplayer = layers.size() -1;
            if (layers[toplayer]->type() == "perceptron")   // ... not nice; can't we generalize this somehow? TODO: add virtual bool isoutputlayer() & allow it only on top
            {
                Pu.read (f, "Pu");
                if (Pu.size() != layers[toplayer]->hdim()) malformed ("inconsistent size of top-level weight matrix and priors");
            }
        }
        fcheckTag (f, "EDBN");
    }

    void write (FILE * f, const string & comment)
    {
        fputTag (f, "DBN\n");
        fputstring (f, comment);
        fputTag (f, "BDBN");
        fputint (f, 0);                     // a version number
        fputint (f, (int) layers.size());   // number of layers
        mean.write (f, "gmean");
        std.write (f, "gstddev");
        fputTag (f, "BNET");
        foreach_index (i, layers)
        {
            fputstring (f, layers[i]->type());
            layers[i]->write (f);
        }
        fputTag (f, "ENET");
        if (!layers.empty() && layers[layers.size() -1]->type() == "perceptron")
            Pu.write (f, "Pu"); // only write if the model is complete
        fputTag (f, "EDBN");
    }

    // helper for reading Matlab files: enumerate all names, find the highest numerical value
    template<typename DUMMY> static int gettoplayer (const DUMMY & inmats)
    {
        int toplayer = 0;
        for (auto iter = inmats.begin(); iter != inmats.end(); iter++)
        {
            const string & matname = iter->first;
            int i = atoi (matname.c_str());
            if (i > toplayer)
                toplayer = i;
        }
        return toplayer;
    }

    // load a DBN from two Matlab-5 formatted files
    void loadmatfile (const wstring & path)
    {
     ;
    }

    void checknotcomputing() const { if (computing) throw std::logic_error ("function called while in 'computing' state, forbidden"); }
    void checkcomputing() const { if (!computing) throw std::logic_error ("function called while not in 'computing' state, forbidden"); }

public:

    // Note: The public methods on this class are to be called outside of computation.
    // Once we start computing (entercomputation()), do not call any method in this object directly.

    // constructor for a fresh model  --we need to pass feature mean/std
    model (const std::vector<float> & datamean, const std::vector<float> & datastd) : mean (datamean), std (datastd), computing (false) {}

    // constructor that loads from input file
    model (const wstring & path) : computing (false) { load (path); }

    void load (const wstring & path)
    {
        auto_file_ptr f = fopenOrDie (path, L"rbS");

        // check if Matlab 5 format --Dong's experiments stored models in this format
        string tag = fgetTag (f);
        if (tag == "MATL")
        {
            fclose (f);
            loadmatfile (path);
            return;
        }
        rewind (f);

        // read the model
        read (f);
    }

    // serializing a snapshot to disk
    void save (const wstring & path, const string & comment)
    {
        checknotcomputing();
        auto_file_ptr f = fopenOrDie (path, L"wbS");
        write (f, comment);
        fflushOrDie (f);
    }

    // dimensions
    size_t vdim() const throw() { return layers[0]->vdim(); }
    size_t udim() const throw() { return layers[layers.size()-1]->hdim(); }
    size_t hdim (size_t layer) const throw() { return layers[layer]->hdim(); }  // note: 0-based, e.g. first hidden layer dim = hdim[0]
    size_t numlayers() const throw() { return layers.size(); }  // number of networks

    // for special purposes, we allow to get internal access (it's a research project)
    const Iannlayer & peeklayer (size_t i) const throw() { return *layers[i].get(); }

    // check dimensions --used at load time during training to verify we have the config right
    void checkdimensions (const std::vector<size_t> & dims) const
    {
        if (dims.size() <= layers.size())
            malformed ("checkdimensions: too few layerdims expected compared to model");
        foreach_index (i, layers)
        {
            if (layers[i]->vdim() != dims[i] || layers[i]->hdim() != dims[i+1])
            {
                fprintf (stderr, "for layer i(%d): layers[i]->vdim(%d) != dims[i](%d) || layers[i]->hdim(%d) != dims[i+1](%d) ", i, layers[i]->vdim(), dims[i],layers[i]->hdim(), dims[i+1]);
                malformed ("checkdimensions: expected layerdims mismatching the actual model");
            }
        }
    }

    // create a layer
    // Called when pretraining/finetuning an additional layer.
    // Initializes W to (deterministic pseudo-) random numbers.
    void addlayer (bool top, size_t vdim, size_t hdim, wstring layertype)
    {
        checknotcomputing();
        size_t layer = numlayers();
        layers.resize (layer +1);
        // create the layer
        unsigned int randomseed = (unsigned int) layer;
        if (top)  //TODO: change to type based decision
        {
            layers[layer].reset (new perceptron (vdim, hdim, randomseed));
            // for top layer, we also create the prior probabilities
            Pu.resize (hdim, 1);
            foreach_index (i, Pu)
                Pu[i] = 1.0f / Pu.size();
        }
        else if (layertype == L"dtnn" ||layertype == L"tn"  )
        {
            size_t h2dim = (size_t)sqrt((double)hdim);
            if (h2dim * h2dim != hdim)
                malformed ("addlayer: for dtnn layer the layer dim must be a square of a number");

            layers[layer].reset (new dtnn (vdim, h2dim, h2dim, randomseed));
        }
        else if (layer > 0)  //TODO: change to type based decision
            layers[layer].reset (new rbmbernoullibernoulli (vdim, hdim, randomseed));
        else  //TODO: change to type based decision
            layers[layer].reset (new rbmgaussbernoulli (vdim, hdim, randomseed));
    }

    // enlarge the model to do classes based adaptation
    void blowup(const size_t blowupfactor, const std::vector<size_t> & layerdims, const std::vector<size_t> & statemapping)    //added by Hang Su adaptation
    {
        for (size_t i = 0; i < layers.size() -1; i++)
        {
            if (layerdims[i+1] / layers[i]->hdim() != blowupfactor)
            {
                throw std::runtime_error ("modelblowup: layerdims does not match with blowupfactor.");
            }
        }
        for (size_t i = 0; i < layers.size() -1; i++)
            layers[i]->blowup(blowupfactor);
        layers[layers.size() - 1]->blowup(blowupfactor,statemapping);
    }

    void setlinearlayer(const msra::dbn::matrix & adaptmatrixinitpath)
    {
        layers[0]->setlinearlayerweight(adaptmatrixinitpath);
    }
private:
    // add some type layer in a given layer
    // TODO: Really only used for injectlinearlayer for now. Tricky because of different init parameters.
    // TODO: move into annlayerfactory class
    void insertlayer(const string &type, size_t vdim, size_t hdim, size_t injectlocation)
    {
        checknotcomputing();
        unsigned int randomseed = (unsigned int)injectlocation;
        if (type == "perceptron")  // add a perception layer 
			layers.insert(layers.begin() + injectlocation, std::unique_ptr<Iannlayer>(new perceptron (vdim, hdim, randomseed)));
        else if (type == "rbmbernoullibernoulli")
            layers.insert(layers.begin() + injectlocation, std::unique_ptr<Iannlayer>(new rbmbernoullibernoulli (vdim, hdim, randomseed)));
        else if (type == "rbmgaussbernoulli")
            layers.insert(layers.begin() + injectlocation, std::unique_ptr<Iannlayer>((new rbmgaussbernoulli(vdim, hdim, randomseed))));
        else if(type == "linearnetwork")
        {
            // notice: linearnetwork don't initialize the W and a as a rand number between (0,1)
            // feature dim can be 52 or 39
            const size_t featuredimoriplp = 52;
            const size_t featuredimorihlda = 39;
            size_t expandfactor;                // expandfactor records the num of blocks in linear network
            if (hdim % featuredimoriplp == 0)
                expandfactor = hdim / featuredimoriplp;
            else if (hdim % featuredimorihlda == 0)
                expandfactor = hdim / featuredimorihlda;
            else
                expandfactor = hdim;
            layers.insert(layers.begin() + injectlocation, std::unique_ptr<Iannlayer>(new linearnetwork(vdim, hdim, expandfactor, true))); 
        }
        else
            malformed ("unknown model type string"); 
    }
public:
    const vector& getmeanref () { return mean; }
    const vector& getstdref ()  { return std;  }
    const vector& getPuref  ()  { return Pu; }
    const std::vector<unique_ptr<Iannlayer>>& getlayersref() { return layers;}

    // insert a lineartransform layer at the bottom
    void injectlinearlayer(size_t injectlocation)
    {
        checknotcomputing();
        const size_t linearlayerdim = layers[injectlocation]->vdim();
        insertlayer("linearnetwork", linearlayerdim, linearlayerdim, injectlocation);
    }

    void reinitializelayer(size_t layer)
    {
        checknotcomputing();
        unsigned int randomseed = (unsigned int)layer;
        layers[layer].reset( annlayerfactory::create (layers[layer]->type(), layers[layer]->vdim(), layers[layer]->hdims(),  randomseed) );
    }


    // reduce number of layers --used for testing purposes only
    void shedlayers (size_t n)
    {
        checknotcomputing();
        if (n > layers.size())
            malformed ("shedlayers: trying to shed layers that don't exist");
        if (n == layers.size())
            return;
        fprintf (stderr, "shedlayers: reducing number of layers to %d\n", n);
        layers.resize (n);
        Pu.resize (0, 0);   // no top layer anymore --no priors
    }

    // implant the prior probability
    void setprior (const msra::dbn::vector & newPu)
    {
        checknotcomputing();
        if (numlayers() == 0 || layers[numlayers()-1]->type() != "perceptron")
            malformed ("setprior: attempted to set prior when no top layer available");
        if (newPu.size() != udim())
            malformed ("setprior: new prior vector of wrong dimension");
        Pu = newPu;
    }

    // function for more hacky experiments to set an entire layer to some value
    template<class WTYPE, class ATYPE>
    void overridelayer (size_t n, const WTYPE & W, const ATYPE & a, const size_t weightsetindex)
    {
        checknotcomputing();

        //we need to do conversion since template function setweight is not virtual
        if (layers[n]->type() == "dtnn")
        {
            dtnn & layer = dynamic_cast<dtnn &>(*layers[n]);
            layer.setweights (W, a, weightsetindex);
        }
        else
        {
            rbmbase & layer = dynamic_cast<rbmbase &>(*layers[n]);
            layer.setweights (W, a, weightsetindex);
        }
    }

    // for diagnostics we allow to read this out
    const msra::dbn::vector & getprior() const { return Pu; }

    // split a layer
    void splitlayer (size_t layer)
    {
        checknotcomputing();
        layers[layer-1]->doublenodes (true);
        layers[layer  ]->doublenodes (false);
    }

    void print()    // const?
    {
        checknotcomputing();

        fprintf (stderr, "\n@@@@@@@@@@@@@@@ Dump model weights @@@@@@@@@@@@@@@\n");

        printmat(mean);
        printmat(std);
        printmat(Pu);

        std::vector<unique_ptr<Iannlayer>>::iterator itr;
        size_t layer;
        for ( itr = layers.begin(), layer=0; itr != layers.end(); ++itr, ++layer )
        {
            fprintf (stderr, "\n###### layer %d ######\n", layer);
            (*itr)->print();
        }
    }

    void print (FILE *f)        // added by Hang Su
    {
        checknotcomputing();

        fprintf (f, "\n@@@@@@@@@@@@@@@ Dump model weights @@@@@@@@@@@@@@@\n");

        std::vector<unique_ptr<Iannlayer>>::iterator itr;
        size_t layer;
        for ( itr = layers.begin(), layer=0; layer < 2 && itr != layers.end(); ++itr, ++layer )
        {
            fprintf (f, "\n###### layer %d ######\n", layer);
            (*itr)->print(f);
        }
        itr = layers.end();
        itr--;
        fprintf (f, "\n###### layer %d ######\n", layer);
        (*itr)->print(f);
    }

    // dump a specific layer matrix to stdout [v-xieche]
    void dumplayer(size_t layer)
    {
        checknotcomputing();
        fprintf(stderr, "\n@@@@@@@@@@@  Dump matrix of layer %d @@@@@@@@@@@\n", layer);
        layers[layer]->dumplayer();
    }

    // print model distribution
    // Returns a pair (total params, total non-null params) for printing overall statistics.
    pair<unsigned int,unsigned int> printvaluedistribution() // const
    {
        checknotcomputing();

        fprintf (stderr, "\n@@@@@@@@@@@@@@@ Dump model weight distribution information @@@@@@@@@@@@@@@\n");

        unsigned int totalparams = 0, totalnonnullparams = 0;

        foreach_index (layer, layers)
        {
            fprintf (stderr, "\n###### layer %d ######\n", layer);
            auto stats = layers[layer]->printvaluedistribution (msra::strfun::strprintf ("[%d]", layer));
            totalparams += stats.first;
            totalnonnullparams += stats.second;
        }
        return make_pair (totalparams, totalnonnullparams);
    }

    // do necessary preparations to start any computation with the model
    // This is expensive and intended to be for an entire epoch.
    // 'type'can be:
    //  -1 -> backpropagation
    //  +1 -> pretraining
    //   0 -> evaluation
    // With CUDA, this loads the model into the CUDA RAM.
    void entercomputation (int type)
    {
        checknotcomputing();
        computing = true;
        foreach_index (j, layers)
            layers[j]->entercomputation (type);
    }
	void exitcomputation()
    {
        checkcomputing();
        foreach_index (j, layers)
            layers[j]->exitcomputation();
        computing = false;

    }
    class evaluator
    {
        void operator=(evaluator&);
    protected:
        std::vector<rbmstatevectors> layerstate;    // all layer inputs/outputs as rbmstatevectors
        // these are just remembered from dbnmodel
        const vector & mean, & std;
        const vector & Pu;
        const std::vector<unique_ptr<Iannlayer>> & layers;   // the layers

        size_t vdim() const throw() { return layers[0]->vdim(); }
        size_t udim() const throw() { return layers[layers.size()-1]->hdim(); }
        size_t nfwd() const throw() { return layerstate[0].cols(); }

        void alloclayerstate (std::vector<rbmstatevectors> & layerstate, size_t nfwd)
        {
            layerstate.resize (layers.size()+1);
            layerstate[0].resize (layers[0]->vdim(), nfwd);
            foreach_index (i, layers)
            {
                // layerstate[i] is input of layer i
                if (layerstate[i].rows() != layers[i]->vdim())
                    model::malformed ("dimension mismatch between layers");
                // layerstate[i+1] is for output of layer i = input of layer i+1
                layerstate[i+1].resize (layers[i]->hdim(), nfwd);
            }

#if 0    // hack for getting activation histogram [v-xieche]
            nnode = layers[1]->vdim();   // the first layer's hidden layer number.
            ngrid = 100;
            hist.resize(nnode, ngrid);
            foreach_coord(i, j, hist)
                hist(i,j) = 0;

#endif
            assert (Pu.empty() || udim() == Pu.size());
        }

    public:
        // This creates vectors to store the intermediate state activations (incl. v and u).
        // Note: We allow to instantiate this inside and outside 'computing' mode.
        evaluator (const model & M, size_t nfwd)
            : layers (M.layers), mean (M.mean), std (M.std), Pu (M.Pu)
        {
            alloclayerstate (layerstate, nfwd);
            assert (nfwd == this->nfwd());
        }
        // wait for completion of offloaded computation (CUDA)
		/*
        void synchronize()
        {
            if (!layerstate.empty())
                layerstate[0].synchronize();
        }
		*/
    public:

        // forward propagation through network; result is returned as a reference
        // Only the sub-range [ts,te) of 'v' is used, and only that is updated in the layerstate and return value.
        // This function is called directly only by the training. For recognition, you want to call logPuv() below.
        // 'numlayerstoeval' and 'prenonlinearity' are special modes for bottleneck features and training experiments.
        // 'numlayerstoeval == 0' is allowed and means to only scale the input features to layerstate[0].
        template<class VMATRIX> const rbmstatevectors & forwardprop (const VMATRIX & v, size_t ts, size_t te, size_t numlayerstoeval, bool prenonlinearity)
        {
            bool updatetoplayer = false;  // judge whether updating toplayer now.
            assert (ts < te && te <= v.cols());
            assert (te <= layerstate[0].cols());
            assert (v.rows() == vdim());

            // apply mean/std normalization
            // Note that mean/std are for a single frame, while v is a concatenation of
            // multiple neighbor frames.
            {
                rbmstatevectorsrefwriting inlayer (layerstate[0], ts, te-ts);
                for (size_t t = ts; t < te; t++) foreach_row (i, v)
                {
                    size_t k = i % mean.size();
                    inlayer(i,t-ts) = (v(i,t) - mean[k]) / std[k];
                }
                // destructor of inlayer syncs back the stripe to CUDA
            }
            // determine top layer to compute --if passed SIZE_MAX then compute the full network (default)
            if (numlayerstoeval == SIZE_MAX)
                numlayerstoeval = layers.size();
            else if (numlayerstoeval == 100)   // for temporary test purpose for the binary function. [v-xieche]
            {
                numlayerstoeval = layers.size();
                updatetoplayer = true;
            }

            // apply network
            for (size_t i = 0; i < numlayerstoeval; i++)
            {
                // compute the output of layer i
                // We use a rbmstatevectorsref stripe to compute only the requested sub-range [ts,te).
                rbmstatevectorsref inlayer  (layerstate[i].stripe   (ts, te - ts));
                rbmstatevectorsref outlayer (layerstate[i+1].stripe (ts, te - ts));
                if (i == numlayerstoeval -1 && prenonlinearity)   // last layer: option to bypass non-linearity
                    layers[i]->forwardprop (inlayer, outlayer, true);
                else
                    layers[i]->forwardprop (inlayer, outlayer);
            }
            // result is now in top layer's layerstate
            const auto & toplayerstate = layerstate[numlayerstoeval];
            return toplayerstate;
        }

    protected:

        // transfer uids[] vector to CUDA-side 'float' vector and return the stripe in ready-to-use form
        mutable rbmstatevectors fuids;          // tmp to move uids reference to CUDA
        template<class UIDSVECTOR>
        rbmstatevectorsref uidsstripe (const UIDSVECTOR & uids, size_t ts, size_t te) const
        {
            fuids.resize (1, nfwd());
            {
                rbmstatevectorsrefwriting fuids_stripe (fuids, ts, te-ts);
                for (size_t t = ts; t < te; t++)
                    fuids_stripe (0,t-ts) = (float) uids[t];    // store in a 'float' row vector
                // destructor syncs back fuids
            }

            // return the stripe in desired format
            return fuids.stripe (ts, te-ts);
        }

    public:

        // -------------------------------------------------------------------
        // helpers for training
        // -------------------------------------------------------------------

        // compute posterior of reference, as one way of convergence tracking
        // Returns av. log posterior. Also prints av. posterior and training-batch frame accuracy.
        mutable rbmstatevectors sumlogpps, sumpps, sumfcors; // [i] vector buffers for posterior statistics


        template<class UIDSVECTOR>
        std::pair<double, double> posteriorstats (const UIDSVECTOR & uids, size_t ts, size_t te) const
        {

            // space for intermediate (column-wise) results (for CUDA use)
            sumlogpps.resize (1, nfwd());
            sumpps.resize    (1, nfwd());
            sumfcors.resize  (1, nfwd());
            // inputs
            const auto   fu = uidsstripe (uids, ts, te);                    // ground truth in CUDA-compatible format
            const rbmstatevectorsref Pu (layerstate[layers.size()].stripe (ts, te-ts)); // actual probabilities

            // results
            double avlogpp;     // log posterior
            double avpp;        // posterior
            double avfcor;      // rate of frames correctly detected

            // compute it
            rbmstatevectorsref sumlogpps_stripe = sumlogpps.stripe (ts, te-ts);
            rbmstatevectorsref sumpps_stripe    = sumpps.stripe (ts, te-ts);
            rbmstatevectorsref sumfcors_stripe  = sumfcors.stripe (ts, te-ts);
            fu.posteriorstats (Pu, sumlogpps_stripe, sumpps_stripe, sumfcors_stripe, avlogpp, avpp, avfcor);  

            // we ony log av log pp
            fprintf (stderr, "posteriorstats: avlogPP=%.2f  avPP=%.2f  frames correct=%.1f%%\n",
                avlogpp, avpp, 100.0 * avfcor);
            return std::make_pair (avlogpp, avfcor);
        }

        // keep a running sum over all P(u|v), for use as a prior
        // Note: This is imprecise, because the model is not final.
        mutable rbmmodelmatrix Pusums;      // CUDA temp for accumulating Pu
        mutable matrix Pusumstmp;           // CPU temp
        void accumulatepriors (std::vector<double> & Pusum, size_t & Pusumcount, size_t ts, size_t te) const
        {
            if (Pusums.empty()) // Pusums follows the model-parameter paradigm, so need to 'entercomputation'
            {
                Pusums.resize (udim(), 1);
                Pusums.entercomputation();
            }
            // sum up all columns in CUDA space
            const auto & u = layerstate[layers.size()];
            Pusums.scaleandaddallcols (0.0f, u.stripe (ts, te-ts), 1.0f, Pusumstmp);
            // now accumulate into CPU-side overall accumulator
            Pusums.accumulate (Pusum);
            Pusumcount += (te-ts);
        }
        // accumulate from features (perform forwardprop)
        template<class VMATRIX>
        void accumulatepriors (const VMATRIX & v, std::vector<double> & Pusum, size_t & Pusumcount, size_t ts, size_t te)
        {
            // perform forwardprop if needed (we do in fixpriors())
            forwardprop (v, ts, te, SIZE_MAX, false);
            // now accumulate
            accumulatepriors (Pusum, Pusumcount, ts, te);
        }

        // -------------------------------------------------------------------
        // LL evaluation
        // -------------------------------------------------------------------

        // get overall log likelihood / p(v)  based oh a previously done forwardprop()
        // p(v|u) = p(u|v) * p(v) / p(u)
        // This computes p(u|v) / p(u) (since p(v) is a constant per frame)
        // 'v' is supposed to be the final feature vector (with neighbor frames or whatever)

	 template<class UMATRIX> void logLL (UMATRIX & Pugv, size_t effectivecols,const bool divbyprior)
        {
            const rbmstatevectorsrefreading u (layerstate[layers.size()], 0, effectivecols);
            assert (u.rows() == udim() && u.cols() <= Pugv.cols());

			if (divbyprior)                            // divide by prior
            {
				
				foreach_column(j,u)
				{
					if(2*j+2<=Pugv.cols())
					{
						for(size_t i=0;i<u.rows();i++)
						{
							
							const float Pugvij = u(i,j) / Pu[i];
							Pugv(i,2*j) = Pugvij > 1e-30f ? logf (Pugvij) : -1e30f;
							
						}
						for(size_t i=0;i<u.rows();i++)
						{
					    	Pugv(i,2*j+1) = Pugv(i,2*j);
						}
					}
					else
					{
						for(size_t i=0;i<u.rows();i++)
						{
							const float Pugvij = u(i,j) / Pu[i];
							Pugv(i,2*j) = Pugvij > 1e-30f ? logf (Pugvij) : -1e30f;
						}
					}
				}
				
            }
            else
            {
             foreach_column(j,u)
			{
				if(2*j+2<=Pugv.cols())
				{
					for(size_t i=0;i<u.rows();i++)
					{
						
						Pugv(i,2*j) =  u(i,j) > 1e-30f ? logf (u(i,j)) : -1e30f;
						
					}
					for(size_t i=0;i<u.rows();i++)
					{
					    Pugv(i,2*j+1) = Pugv(i,2*j);
					}
				}
				else
				{
					for(size_t i=0;i<u.rows();i++)
					{
						
						Pugv(i,2*j) =  u(i,j) > 1e-30f ? logf (u(i,j)) : -1e30f;
					}
				}
			  }
			 
            }
        }

        template<class UMATRIX> void logLL (UMATRIX & Pugv, const bool divbyprior)
        {
            const rbmstatevectorsrefreading u (layerstate[layers.size()], 0, Pugv.cols());
            assert (u.rows() == udim() && u.cols() == Pugv.cols());
			
			//
            assert (Pugv.cols() == u.cols());
            if (divbyprior)  // divide by prior
            {
                foreach_coord (i, j, Pugv)
				{
					 const float Pugvij = u(i,j) / Pu[i];
					 Pugv(i,j) = Pugvij > 1e-30f ? logf (Pugvij) : -1e30f;

                   /*
					if(u(i,j) / Pu[i]<expf(-1e2f)) // low boundary incase pugv go to -inf by v-wenh
						Pugv(i,j)=-1e2f;
					else
						Pugv(i,j) = logf (u(i,j)/Pu[i]); 
					*/
				}
			}
            else
            {
                foreach_coord (i, j, Pugv)
				{
					Pugv(i,j) =  u(i,j) > 1e-30f ? logf (u(i,j)) : -1e30f;
					/*
					if(u(i,j)<expf(-1e2f)) // low boundary incase pugv go to -inf by v-wenh
						Pugv(i,j)=-1e2f;
					else
                       Pugv(i,j) = logf (u(i,j));
					   */
				}
			}

        }

        // evaluate overall log likelihood / p(v)
        // TODO: rename to a correct name (it is NOT P(u|v))
        template<class VMATRIX, class UMATRIX> void logPuv (const VMATRIX & v, UMATRIX & Pugv, const bool divbyprior)
        {
		
#ifdef AYNC_FRAME_DECODING
			assert (v.cols() <= Pugv.cols());
#else
			assert (v.cols() == Pugv.cols());
#endif
            assert (Pugv.rows() == udim());
            assert (Pu.size() == udim());

            // perform forward propagation through network -> u
		
            forwardprop (v, 0, v.cols(), SIZE_MAX, false);
			//test
		

            // convert to scaled likelihoods
#ifdef AYNC_FRAME_DECODING
            logLL (Pugv,v.cols(), divbyprior);
#else
			logLL (Pugv, divbyprior);
#endif
		

        }

        // logPuv from previous forwardprop() pass

        // evaluate a layer's activations
        // 'v' is supposed to be the final feature vector (with neighbor frames or whatever)
        // The top layer's output activation values are placed in Eh.
        // Differs from logPuv in that no priors are applied and no log is taken.
        // Currently used in experimental ML initialization (top layer) and bottleneck features (intermediate layer).
        // 'layer' = SIZE_MAX means top layer. 'prenonlinearity' allows to bypass the sigmoid (for bottleneck features).
        template<class VMATRIX, class UMATRIX> void evaluate (const VMATRIX & v, UMATRIX & Eh, size_t atlayer/*or SIZE_MAX*/, bool prenonlinearity)
        {
            assert (v.cols() == Eh.cols());
            if (atlayer == layers.size()) assert (Eh.rows() == udim()); else assert (Eh.rows() == layerstate[atlayer].rows());

            // perform forward propagation through network -> u
            const rbmstatevectorsrefreading u (forwardprop (v, 0, v.cols(), atlayer, prenonlinearity), 0, v.cols());
            // u is layer activations[atlayer], i.e. it can be an intermediate result

            // copy out the result
            foreach_coord (i, j, Eh)
                Eh(i,j) = u(i,j);
        }
    };

    // -----------------------------------------------------------------------
    // class accmulator --for thread-local accumulation step
    // This is no longer used and can be merged with 'trainer'
    // -----------------------------------------------------------------------

    class accumulator : public evaluator
    {
    protected:
        rbmstatevectors v1, h1;                     // pre-training: updated v and h after 1 step of CD (top layer only)
        size_t firstbplayer;                        // first layer that gets updated by backpropagation
        std::vector<rbmstatevectors> errorstate;    // [firstbplayer..numlayers-1] back-propagated error vectors; note that the paper's e^L=errorstate[L+1]!
        rbmstatevectors keepsampleflags;            // [t] temp for sub-sampling frames (1.0 = keep; 0.0 = remove)
        bool istoplayer;
    public:
        // This creates:
        //  - v and h for forward propagation
        //  - pt: vectors to store updated v and h
        //  - bp: vectors to store the shared intermediate error values
        // ... TODO: rename 'nfwd' to 'T' or something like that
        accumulator (const model & M, size_t nfwd, bool istoplayer, size_t finetunedlayers) : evaluator (M, nfwd), istoplayer (istoplayer), firstbplayer (0)
        {
            if (istoplayer)
            {
                if (finetunedlayers > M.numlayers()) finetunedlayers = M.numlayers();
                firstbplayer = M.numlayers() - finetunedlayers;
                if (firstbplayer > 0)
                    fprintf (stderr, "accumulator: backpropagation limited to layers %d..%d\n", (int) firstbplayer, M.numlayers() -1);
                alloclayerstate (errorstate, nfwd);
                errorstate[firstbplayer].resize (0, nfwd);    // we don't want lowest level ... TODO: do this nicer
            }
            else
            {
                size_t toplayer = layers.size() -1;
                v1.resize (layers[toplayer]->vdim(), nfwd);
                h1.resize (layers[toplayer]->hdim(), nfwd);
            }
            keepsampleflags.resize (1, nfwd);
        }

        // unsupervised pre-training accumulation
        //  - input = columns of feature vectors
        // This operates on the time stripe [ts,te)
        void pretrainingstats2 (const matrixbase & v_in, size_t ts, size_t te, unsigned int randomseedframebase)
        {
            size_t toplayer = layers.size() -1;
            const auto & Ph = layerstate[toplayer +1];

            rbmstatevectorsref Ph_stripe (Ph.stripe (ts, te - ts));
            rbmstatevectorsref v1_stripe (v1.stripe (ts, te - ts));
            rbmstatevectorsref h1_stripe (h1.stripe (ts, te - ts));

            layers[toplayer]->pretrainingstats (Ph_stripe, v1_stripe, h1_stripe, randomseedframebase);
        }

        // compute log LL for reconstruction, for tracking pre-training
        // To make the av log LL more readable, the LL is normalized by the null hypothesis
        // of perfect reconstruction. That normalization only depends on the input data,
        // and it takes out a large constant offset of the number which does not add
        // value and makes small changes so much harder to see.
        // This function expects pretrainingstats() to have been called before.
        // TODO: This function is no longer reentrant, so what's the point of passing ts, te ?
        mutable rbmstatevectors glogllsums, logllsums; // [i] vector buffers for likelihood values for statistics
        double llstats (size_t ts, size_t te) const
        {
            const size_t toplayer = layers.size() -1;

            logllsums.resize  (layers[toplayer]->vdim(), 1);
            glogllsums.resize (layers[toplayer]->vdim(), 1);

            assert (layers[toplayer]->type() == "rbmgaussbernoulli" || layers[toplayer]->type() == "rbmbernoullibernoulli");
            const bool gaussian = (layers[toplayer]->type() == "rbmgaussbernoulli");

            double glogllsum = 0.0; // Gaussian (also for binary units, for diagnostics)
            double logllsum = 0.0;
            rbmstatevectorsref glogllsums_stripe (glogllsums.stripe (0, 1));
            rbmstatevectorsref logllsums_stripe  (logllsums.stripe (0, 1));
            layerstate[toplayer].stripe (ts, te-ts).llstats (v1.stripe (ts, te-ts), glogllsums_stripe, logllsums_stripe, glogllsum, logllsum);

            fprintf (stderr, "llstats: avlogLL=%.5f, av Gaussian logLL=%.5f\n", logllsum / (te - ts), glogllsum / (te - ts));

#if 1
            if (gaussian)
                return glogllsum / (te - ts);
            else
                return logllsum / (te - ts);
#else
            //return logllsum / (te - ts);    // av log LL per frame
            return glogllsum / (te - ts);    // returning Gaussian distance for now (although it seems more noisy)
#endif
        }

#if 0   // merged with backpropagationorpretrainingstats1   --TODO: clean this up a little
        // unsupervised pre-training accumulation, phase 1 (forwardprop())
        // same as backpropagationstats1(); merge them
        void pretrainingstats1 (const matrixbase & v, size_t ts, size_t te)
        {
            forwardprop (v, ts, te, SIZE_MAX, false);
        }
#endif

        // first stage of BP accumulation, the forward propagation
        // ... TODO (?): rename these two functions. They are It's not just accumulating, it's processing--bpstats()?
        void backpropagationorpretrainingstats1 (const matrixbase & v, size_t ts, size_t te)
        {
            forwardprop (v, ts, te, SIZE_MAX, false);
        }
		
        // update the output posteriors externally
        // This is needed for MMI training, which we try to reflect in the name of the function.
        // Call this before backpropagationstats2(), which will be based on these gammas.
        template<class UMATRIX> void setdenominatorgammas (const UMATRIX & pp, float keepweight)
        {
#if 0
            keepweight;
            // we replace the output layer's activations
            rbmstatevectorsrefwriting u (layerstate[layers.size()], 0, pp.cols());
            assert (u.rows() == pp.rows());

            // copy over the posteriors
            foreach_coord (s, t, pp)
                u(s,t) = pp(s,t);
#else
            // we interpolate the output layer's activations
            rbmstatevectorsref u (layerstate[layers.size()].stripe (0, pp.cols()));
            u.lockforreadwrite();
            assert (u.rows() == pp.rows());

            // copy over the posteriors
            foreach_coord (s, t, pp)
                u(s,t) = u(s,t) * keepweight + pp(s,t) * (1.0f - keepweight);
            u.unlock();
#endif
        }

        // update the top-level error signal from MMI training
        // Call this before backpropagationstats3(), which will be based on these gammas.
        template<class UMATRIX> void backpropagationstatsmmi2 (const UMATRIX & numgammas, const UMATRIX & dengammas, size_t ts, size_t te, float keepweight)
        {
            // we interpolate the output layer's activations
            const rbmstatevectorsrefreading Pu (layerstate[layers.size()], ts, te - ts);

            // TODO: why not use rbmstatevectorsrefwriting here?
            rbmstatevectorsref err (errorstate[layers.size()].stripe (ts, te - ts));  // -> error goes here
            err.lockforwriting();

            // copy over the posteriors
            for (size_t t = ts; t < te; t++)
                foreach_row (s, err)
                err(s,t-ts) = numgammas(s,t-ts) - (dengammas(s,t-ts) * (1.0f - keepweight) + Pu(s,t-ts) * keepweight);

            err.unlock();   // sync it back
        }

        // supervised back-propagation accumulation  --set error signal
        //  - input = columns of feature vectors
        //  - target output (uids[ts..te-1]) = indices of supervised class ids corresponding to vectors
        // Compute the top error signal.
        // This function operates only on the column range [ts,te) and can be safely called from multiple threads for disjoint column ranges.
        // forwardprop() must be run before entering this function.
        // TODO: why is prefevaluator a unique_ptr and not just a pointer?
        template<class UIDSVECTOR>
        void backpropagationstats2 (const matrixbase & v, const UIDSVECTOR & uids, size_t ts, size_t te, msra::dbn::model::evaluator * prefevaluator, const float alpha)
        {
            const auto & u = layerstate[layers.size()];
            const rbmstatevectorsref fu  (uidsstripe (uids, ts, te));                       // reference
            const rbmstatevectorsref Pu  (u.stripe (ts, te - ts));                          // actual probabilities
            rbmstatevectorsref       err (errorstate[layers.size()].stripe (ts, te - ts));  // -> error goes here

            if (prefevaluator && alpha > 0)
            {   // TODO: make this compatible with MMI training
                const auto & refu = prefevaluator->forwardprop (v, ts, te, SIZE_MAX, false);
                const rbmstatevectorsref refPu  (refu.stripe (ts, te - ts));                          // actual probabilities
                err.seterrorsignalwithklreg (fu, Pu, refPu, alpha);
            }
            else
            {
                err.seterrorsignal (fu, Pu);  // compute the error signal
            }
        }

        // drop frames
        // All fields are set up ready for back propagation; that is top-level error and all forward-prop information (layerstate[]).
        size_t dropframes (const std::vector<bool> & framestodrop, size_t ts, size_t te)
        {
            size_t framesfortraining = te - ts;
            assert (framestodrop.size() == framesfortraining);
            foreach_index (i, framestodrop) if (framestodrop[i]) framesfortraining--;
            if (framesfortraining == te - ts)    // nothing to do
                return framesfortraining;

            {
                rbmstatevectorsrefwriting keepsamples (keepsampleflags, ts, te-ts);
                for (size_t t = ts; t < te; t++)
                    keepsamples(0,t) = framestodrop[t] ? 0.0f : 1.0f;
                // destructor syncs back keepsampleflags
            }

            rbmstatevectorsref keepsamples (keepsampleflags.stripe (ts, te - ts));
            foreach_index (i, layerstate)
            {
                rbmstatevectorsref h (layerstate[i].stripe (ts, te - ts));
                h.dropframes (keepsamples);
            }

            rbmstatevectorsref err (errorstate[layers.size()].stripe (ts, te - ts));
            err.dropframes (keepsamples);

            return framesfortraining;
        }

        // propagate the error signal of all layers and frames for later use in accumulation
        //  - input = columns of feature vectors
        //  - target output (uids[ts..te-1]) = indices of supervised class ids corresponding to vectors
        // This function operates only on the column range [ts,te) and can be safely called from multiple threads for disjoint column ranges.
        // forwardprop() must be run before entering this function, and error signal must have been set as well.
        // TODO: rename all these numbered functions to something proper that describes what they actually do.
        void backpropagationstats3 (size_t ts, size_t te)
        {
#if 0       // OLD VERSION  --need to move forwardprop() out of here if one wants to resurrect it
            {
                // forward prop through layers  --activation probs get stored in layerstate[][]
                const rbmstatevectorsrefreading u (forwardprop (v, ts, te), ts, te-ts);

                // initialize top-level error state
                // TODO: do this inside CUDA space
                rbmstatevectorsrefwriting toperrorstate (errorstate[layers.size()], ts, te-ts);
                for (size_t t = ts; t < te; t++)
                {
                    foreach_row (i, toperrorstate)
                    {
                        const float utarget_ij = (i == uids[t]) ? 1.0f : 0.0f;
                        toperrorstate(i,t-ts) = utarget_ij - u(i,t-ts);
                    }
                }
                checknan (toperrorstate);
                // destructor syncs back toperrorstate
            }
#endif

            // back-propagate through the layers and accumulate
            for (size_t i = layers.size(); i > firstbplayer; i--)
            {
                rbmstatevectorsref h  (layerstate[i].stripe (ts, te - ts));

                rbmstatevectorsref eh (errorstate[i].stripe (ts, te - ts));   
                //checknan (eh);
                //checknan (h);
                rbmstatevectorsref ev (errorstate[i-1].stripe (ts, te - ts));   // note: empty for bottom layer

#ifdef LOGINSIGMOID    // calculate the sigmoid value from the log(epison + s(z)) [v-xieche]
                h.getorisigmoid();
#endif
                layers[i-1]->backpropagationstats (eh, h, ev);
                //checknan (ev);
                // Note: eh has now been updated and is ready for accumulation.
                // ev has been output as an input for the next stage, where it will,
                // in turn, be updated for accumulation.
            }
        }
    };


    // -----------------------------------------------------------------------
    // class trainer --for global model update
    // TODO: no longer needed to separate this from 'accumulator'
    // -----------------------------------------------------------------------

    // The 'trainer' is created per epoch, while multiple 'accumulator' are kept NUMA-locally.
    class trainer : public accumulator
    {
    public:
        // This creates vectors to store the shared intermediate state activations (incl. v and u).
        trainer (const model & M, size_t mbsize, bool istoplayer, size_t finetunedlayers) : accumulator (M, mbsize, istoplayer, finetunedlayers) { }

        // update the model based on layerstate[][] and errorstate[][]
        void pretrainingmodelupdate (size_t ts, size_t te, float learningratepersample, double momentumpersample, bool resetmomentum)
        {
            size_t toplayer = layers.size() -1; // pre-training applies to the top layer only
            rbmstatevectorsref v (layerstate[toplayer].stripe (ts, te - ts));
            rbmstatevectorsref h (layerstate[toplayer+1].stripe (ts, te - ts));
            rbmstatevectorsref v1ref (v1.stripe (ts, te - ts));
            rbmstatevectorsref h1ref (h1.stripe (ts, te - ts));
            layers[toplayer]->pretrainingmodelupdate (v, h, v1ref, h1ref, learningratepersample, momentumpersample, resetmomentum);
        }

        // update the model based on layerstate[][] and errorstate[][]
        void backpropagationmodelupdate (size_t ts, size_t te, float learningratepersample, double momentumpersample, bool resetmomentum, int deferupdate,
            float sparsethreshold, size_t restricttosinglelayer/*or SIZE_MAX*/,
            const regularizationtype regtype, const float regparam, const unique_ptr<msra::dbn::model> & prefmodel)
        {
            size_t begin = firstbplayer;
            size_t end = layers.size();
            if (restricttosinglelayer != SIZE_MAX)
            {
                if (begin < restricttosinglelayer)
                    begin = restricttosinglelayer;
                if (end > restricttosinglelayer + 1)
                    end = restricttosinglelayer + 1;
                if (end <= begin)
                {
                    fprintf (stderr, "begin %d, end %d\n", begin, end);
                    throw std::runtime_error ("backpropagationmodelupdate: restricttosinglelayer conflicts with firstbplayer or is out of range");
                }
            }
            // apply to entire network top-down
            for (size_t i = begin; i < end; i++)
            {
                // update parameters of model i based on BP step
                rbmstatevectorsref err (errorstate[i+1].stripe (ts, te - ts));
                rbmstatevectorsref act (layerstate[i].stripe (ts, te - ts));
                modelupdateinfo bpinfo;
                bpinfo.regtype = regtype;
                bpinfo.sparsethreshold = sparsethreshold;

                if (regtype == regL2 && prefmodel)
                {
                    bpinfo.alpha = regparam;
                    bpinfo.preflayer = &(*(prefmodel->layers[i]));
                }
                else if (regtype == regCL)
                {
                    bpinfo.nochangeifaboveorbelow = -regparam;
                }
                else if (regtype == regCS)
                {
                    bpinfo.nochangeifaboveorbelow = regparam;
                }

                layers[i]->backpropagationmodelupdate (err, act, learningratepersample, momentumpersample, resetmomentum, deferupdate, bpinfo);
            }
        }
    };
};

// ===========================================================================
// interface to abstract trainer implementations (for now, only used with compact trainer and pipeline trainer)
// ===========================================================================
class itrainer 
{
public:             // define the interface functions for plain training and pipeline training.[v-xieche]
    virtual void entercomputation (msra::dbn::model& model, bool istoplayer) = 0;
    virtual void exitcomputation (msra::dbn::model &model) = 0;
    virtual void inittraining (size_t actualmbsize) = 0;        // initiate buffer will be used in the following training.
    virtual void getminibatch (std::vector<size_t> & uids, bool &flag_remove_pipeline, size_t & restrictupdatelayer) = 0;
    virtual void processminibatch (const msra::dbn::matrixstripe &v, const std::vector<size_t> & uids, size_t ts, size_t te, size_t startlayertoeval, size_t restrictupdatelayer,
        size_t numlayerstoeval, bool prenonlinearity /* false */, msra::dbn::model::evaluator * prefevaluator, const float alpha,  bool resetmomentum,
        float learningratepersample, double momentumpersample, 
        float sparsethreshold, size_t restricttosinglelayer/*or SIZE_MAX*/,
        const msra::dbn::regularizationtype regtype, const float regparam, unique_ptr<msra::dbn::model> & prefmodel, std::vector<double> &Pusum, size_t totalPucount) = 0;
    virtual void getmbstats (std::vector<double> &Pusum, const std::vector<size_t> &uids, size_t &totalPucount, size_t actualmbsize, double &logpsum, double &fcorsum, size_t &logpframes) = 0;
    virtual ~itrainer() { }
};

};};
