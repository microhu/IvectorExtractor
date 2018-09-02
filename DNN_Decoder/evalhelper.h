#pragma once
//#define AYNC_FRAME_DECODING
//#define MULTI_LOGISTIC_REGRESSION
//#define MCE_LINEAR
#include <io.h> // for _dup(), _fileno()
#include <iostream>
#include "dbn.h"
#include "htkfeatio.h"  
#include "unicode.h"

using namespace std;
int msra::numa::node_override = -1;     
size_t msra::parallel::ppl_cores = 1;

static std::map<wstring,float> StatePriorLoad(const wstring& szTiedList, const msra::dbn::model & model)
{
	auto_file_ptr fp = fopenOrDie(szTiedList, L"r");
	map<wstring,float> stateList;
	for(int i = 0; !feof(fp); i++)
	{
		WSTRING st = fgetlinew(fp);
		stateList[st] = model.getprior()[i];
		if(i > model.getprior().size())
		{
			std::cout << "tiedList length mismatch with the model prior length" <<std::endl;
			break;
		}
	}
	return stateList;
}

map<wstring,vector<wstring>> mlfFileParser(const wstring &mlffile)
{
	auto_file_ptr fp = fopenOrDie(mlffile, L"r");

	map<wstring,vector<wstring>> mlfcontainer;
	vector<wstring> wordList;
	wstring fileName;

	WSTRING st = fgetlinew(fp);
	if(st != L"#!MLF!#")
	{
		fwprintf(stderr, L"%s mlf file header missing", mlffile.c_str());
	}
	while(!feof(fp))
	{
		WSTRING st = fgetlinew(fp);
		if(st[0] == '\"')
		{
			fileName = regex_replace (st, wregex (L"\\.[^\\.\\\\/:]*$"), wstring());  // delete extension (or not if none)
			fileName = regex_replace (fileName, wregex (L"/"), wstring(L"\\")); 
			fileName = regex_replace (fileName, wregex (L".*\\\\"), wstring()); 
		}
		else if(st == L"\.")
		{
			mlfcontainer[fileName] = wordList;
			wordList.clear(); 
		}
		else if(st[0] < '0' || st[0] > '9')
		{
			wordList.push_back(st);
		}
	}
	return mlfcontainer;

};
template<class MATRIX> void msraMatrix2doublearray(MATRIX & feat,double** score)
{
        size_t featdim = feat.rows();
        size_t numframes = feat.cols();
		for (size_t i = 0; i < numframes; i++)
		{
			 for(size_t k=0;k<featdim;k++)
			 {
				score[i][k]= -feat(k,i);
			 }
		}

};
template<class MATRIX> void msraMatrix2floatarray(MATRIX & feat, float** score)
{
	size_t featdim = feat.rows();
	size_t numframes = feat.cols();
	for (size_t i = 0; i < numframes; i++)
	{
		for (size_t k = 0; k<featdim; k++)
		{
			score[i][k] = -feat(k, i);
		}
	}

};

static void usage (const string msg = "")
{
    if (!msg.empty())
        fprintf (stderr, "%s\n", msg.c_str());
    fprintf (stderr, "dbn COMMAND ARGS\n");
    exit (1);
}
class args
{
    std::deque<wstring> argarray;
public:
    args (int argc, wchar_t ** argv)
    {
        for (int i = 1; i < argc; i++)
            argarray.push_back (argv[i]);
    }
    bool empty() const throw() { return argarray.empty(); }
    const wstring & front() const throw() { return argarray.front(); }
    wstring next (string msg)
    {
        if (empty())
            usage (msg + " expected");
        wstring arg = argarray.front();
        argarray.pop_front();
        return arg;
    }
    int nextint (string msg)
    {
        wstring val = next(msg);
        return _wtoi (val.c_str()); // ...TODO: add error handling
    }
   float nextfloat (string msg)
    {
        wstring val = next(msg);
        return (float)_wtof (val.c_str()); // ...TODO: add error handling
    }
};
wstring makeoutpath (const wstring & outdir, wstring file, const wstring & outext)
{
    // replace directory
    if (!outdir.empty())
    {
        file = regex_replace (file, wregex (L".*[\\\\/:]"), wstring()); // delete path
        size_t nsl = 0, nbsl = 0;   // count whether the path uses / or \ convention, and stick with it
        foreach_index (i, outdir)
        {
            if (outdir[i] == '/') nsl++;
            else if (outdir[i] == '\\') nbsl++;
        }
        file = outdir + (nbsl > nsl ? L"\\" : L"/") + file;   // prepend new path
    }
    // replace output extension
    if (!outext.empty())
    {
        file = regex_replace (file, wregex (L"\\.[^\\.\\\\/:]*$"), wstring());  // delete extension (or not if none)
        file += L"." + outext;      // and add the new one
    }
    return file;
}

template<class INV, class OUTV> static void copytosubvector (const INV & inv, size_t subvecindex, OUTV & outv)
{
    size_t subdim = inv.size();
    assert (outv.size() % subdim == 0);
    size_t k0 = subvecindex * subdim;
    foreach_index (k, inv)
        outv[k + k0] = inv[k];
}
static size_t augmentationextent (size_t featdim/*augment from*/, size_t modeldim/*to*/)
{
    const size_t windowframes = modeldim / featdim;   // total number of frames to generate
    const size_t extent = windowframes / 2;           // extend each side by this

    if (modeldim % featdim != 0)
        throw runtime_error ("augmentationextent: model vector size not multiple of input features");
    if (windowframes % 2 == 0)
        throw runtime_error (msra::strfun::strprintf ("augmentationextent: neighbor expansion of input features to %d not symmetrical", windowframes));

    return extent;
}
template<class MATRIX, class VECTOR> static void augmentneighbors (const MATRIX & frames, const std::vector<char> & boundaryflags, size_t t,
                                                                   VECTOR & v)
{
    // how many frames are we adding on each side
    const size_t extent = augmentationextent (frames[t].size(), v.size());

    // copy the frame and its neighbors
    // Once we hit a boundaryflag in either direction, do not move index beyond.
    copytosubvector (frames[t], extent, v);     // frame[t] sits right in the middle
    size_t t1 = t;  // index for frames on to the left
    size_t t2 = t;  // and right
    for (size_t n = 1; n <= extent; n++)
    {

        if (boundaryflags.empty())  // boundary flags not given: 'frames' is full utterance
        {
            if (t1 > 0) t1--;                   // index does not move beyond boundary
            if (t2 + 1 < frames.size()) t2++;
        }
        else
        {
            if (boundaryflags[t1] != -1) t1--;  // index does not move beyond a set boundaryflag,
            if (boundaryflags[t2] != 1) t2++;   // because that's the start/end of the utterance
        }
        copytosubvector (frames[t1], extent - n, v);
        copytosubvector (frames[t2], extent + n, v);
    }
}

template<class MATRIX> static void augmentneighbors (const std::vector<std::vector<float>> & frames, const std::vector<char> & boundaryflags,
                                                     size_t ts, size_t te,  // range [ts,te)
                                                     MATRIX & v)
{
#ifdef AYNC_FRAME_DECODING
	//size_t effeictiveindex=0;
	 for (size_t t = ts; t < te; t++)
    {
		if((t-ts)%2==0) // if skip more than one frame, it is native to change 2 to 3--> may cause faliure
		{
			auto v_t = v.col((t-ts)/2); // the vector to fill in
			augmentneighbors (frames, boundaryflags, t,v_t);
		//    effeictiveindex++;
		}
		
	 }
#else

    for (size_t t = ts; t < te; t++)
    {
        auto v_t = v.col(t-ts); // the vector to fill in
        augmentneighbors (frames, boundaryflags, t, v_t);
    }
#endif
}


class onlineeval // for online mode forward propagation to get the state posterior or likelyhood
{
	const bool divbyprior;
	const msra::dbn::model &model;
	std::vector<std::vector<float>> frames;
	std::vector<char>boundaryflags;
	std::size_t numframes;
	std::wstring outputpath;
	unsigned int sampperiod;
	wstring featKind;
private:
	void clear()
	{
		frames.clear();
		boundaryflags.clear();
		outputpath.clear();
		numframes=0;

	}
public:
	onlineeval(const msra::dbn::model & model, bool divbyprior=false): model(model),divbyprior(divbyprior)
	{
		frames.reserve(2048); // reserve for some space
	}
	template<class MATRIX> void evaltomatrix(const MATRIX &feat, const wstring & featkind, unsigned int samperiod, MATRIX & outmatrix )
	{
		 if(feat.cols()<2)
		 {
			 throw std::runtime_error("evaltofile: utterances < 2 frames not supported");
		 }
		 foreach_column(t,feat)
		 {
		   std::vector<float> v (&feat(0,t), &feat(0,t) + feat.rows());
		   frames.push_back(v);
		   boundaryflags.push_back ((t == 0) ? -1 : (t == feat.cols() -1) ? +1 : 0);
		 }
		 numframes=feat.cols();
		 this->sampperiod = samperiod;
		 this->featKind = featkind;
		 // pass parameter to dnn model to begin forward propagation

	     size_t framesinblock = frames.size();
		
#ifdef AYNC_FRAME_DECODING
		framesinblock=frames.size()/2 + frames.size()-(frames.size()/2)*2; // get the effective argument input
#endif
        msra::dbn::matrix agufeat (model.vdim(), framesinblock); 
		size_t outdim = model.udim();
        msra::dbn::matrix pred (outdim, frames.size()); 
		
        msra::dbn::model::evaluator eval (model, framesinblock);

		
        augmentneighbors (frames, boundaryflags, 0, frames.size(), agufeat);

		

		eval.logPuv(agufeat,pred,this->divbyprior);
		
		outmatrix=pred;
		clear();
		
	    
	}
	template<class MATRIX> void evaltofile(const MATRIX &feat, const wstring & featkind, unsigned int samperiod, const std::wstring &outpath)
	{
		 if(feat.cols()<2)
		 {
			 throw std::runtime_error("evaltofile: utterances < 2 frames not supported");
		 }
		 foreach_column(t,feat)
		 {
		   std::vector<float> v (&feat(0,t), &feat(0,t) + feat.rows());
		   frames.push_back(v);
		   boundaryflags.push_back ((t == 0) ? -1 : (t == feat.cols() -1) ? +1 : 0);
		 }
		 numframes=feat.cols();
		 this->outputpath=outpath;
		 this->sampperiod=samperiod;
		 this->featKind=featkind;
		 // pass parameter to dnn model to begin forward propagation

	    const size_t framesinblock = frames.size();
        //const size_t targetnfwd = 8;   
        msra::dbn::matrix agufeat (model.vdim(), framesinblock); 
		size_t outdim = model.udim();
        msra::dbn::matrix pred (outdim, framesinblock); 
		
        msra::dbn::model::evaluator eval (model, framesinblock);
        augmentneighbors (frames, boundaryflags, 0, framesinblock, agufeat);

		eval.logPuv(agufeat,pred,this->divbyprior);
		msra::asr::htkfeatwriter::writeAsASCII(this->outputpath, this->featKind, this->sampperiod, pred);
	    clear();
	}

	template<class MATRIX> void evalOrderedFirstNToFile(const MATRIX &feat, unsigned int samperiod, const std::wstring &outpath, int FirstN)
	{
		if (feat.cols()<2)
		{
			throw std::runtime_error("evaltofile: utterances < 2 frames not supported");
		}
		
		foreach_column(t, feat)
		{
			std::vector<float> v(&feat(0, t), &feat(0, t) + feat.rows());
			frames.push_back(v);
			boundaryflags.push_back((t == 0) ? -1 : (t == feat.cols() - 1) ? +1 : 0);
		}
		numframes = feat.cols();
		this->outputpath = outpath;
		this->sampperiod = samperiod;
		// pass parameter to dnn model to begin forward propagation

		const size_t framesinblock = frames.size();
		//const size_t targetnfwd = 8;   
		msra::dbn::matrix agufeat(model.vdim(), framesinblock);
		size_t outdim = model.udim();
		msra::dbn::matrix pred(outdim, framesinblock);

		msra::dbn::model::evaluator eval(model, framesinblock);
		augmentneighbors(frames, boundaryflags, 0, framesinblock, agufeat);
		eval.evaluate(agufeat, pred, model.numlayers(), false/*prenonlinearity*/);
	//	eval.logPuv(agufeat, pred, this->divbyprior);
		msra::asr::htkfeatwriter::writeOrderedFrames(this->outputpath, L"USER", sampperiod, pred, FirstN);
		clear();
	}
	template<class MATRIX> void forwardPropagationGetEMStatistics(const MATRIX &feat, const wstring sidFeatPath, map<size_t, size_t> selectedIndMap, int M, const std::wstring &outpath)
	{
		if (feat.cols()<2)
		{
			throw std::runtime_error("evaltofile: utterances < 2 frames not supported");
		}

		foreach_column(t, feat)
		{
			std::vector<float> v(&feat(0, t), &feat(0, t) + feat.rows());
			frames.push_back(v);
			boundaryflags.push_back((t == 0) ? -1 : (t == feat.cols() - 1) ? +1 : 0);
		}
		numframes = feat.cols();
		this->outputpath = outpath;

		// pass parameter to dnn model to begin forward propagation

		const size_t framesinblock = frames.size();
		//const size_t targetnfwd = 8;   
		msra::dbn::matrix agufeat(model.vdim(), framesinblock);
		size_t outdim = model.udim();
		msra::dbn::matrix pred(outdim, framesinblock);

		msra::dbn::model::evaluator eval(model, framesinblock);
		augmentneighbors(frames, boundaryflags, 0, framesinblock, agufeat);
		eval.evaluate(agufeat, pred, model.numlayers(), false/*prenonlinearity*/);
		//	eval.logPuv(agufeat, pred, this->divbyprior);
	
		msra::asr::htkfeatwriter::writeEMStatistics(this->outputpath, sidFeatPath, pred, selectedIndMap, M);
		clear();
	}

	template<class MATRIX> void forwardPropagationGetClusteredStatePosterior(const MATRIX &feat, unsigned int sampleperiod, map<size_t, size_t> &senone2stateMap, size_t totalStateNumber, const std::wstring &outpath)
	{
		if (feat.cols()<2)
		{
			throw std::runtime_error("evaltofile: utterances < 2 frames not supported");
		}

		foreach_column(t, feat)
		{
			std::vector<float> v(&feat(0, t), &feat(0, t) + feat.rows());
			frames.push_back(v);
			boundaryflags.push_back((t == 0) ? -1 : (t == feat.cols() - 1) ? +1 : 0);
		}
		numframes = feat.cols();
		this->outputpath = outpath;

		// pass parameter to dnn model to begin forward propagation

		const size_t framesinblock = frames.size();
		//const size_t targetnfwd = 8;   
		msra::dbn::matrix agufeat(model.vdim(), framesinblock);
		size_t outdim = model.udim();
		msra::dbn::matrix pred(outdim, framesinblock);

		msra::dbn::model::evaluator eval(model, framesinblock);
		augmentneighbors(frames, boundaryflags, 0, framesinblock, agufeat);
		eval.evaluate(agufeat, pred, model.numlayers(), false/*prenonlinearity*/);
		//	eval.logPuv(agufeat, pred, this->divbyprior);

		msra::dbn::matrix clusteredPos(totalStateNumber, framesinblock);
		clusteredPos.setzero();
		foreach_column(t, pred)
		{
			for (size_t k = 0; k < pred.rows(); k++)
				clusteredPos(senone2stateMap[k], t) += pred(k, t);
		}
		//your code here
		msra::asr::htkfeatwriter::write(this->outputpath, L"USER", sampleperiod, clusteredPos);
		clear();
	}
};
static void evaluate (const wstring & modelpath, const vector<wstring> & infiles, const wstring & outdir, const wstring & outext,
                      const wstring & scriptoutpath, bool makemode,  bool divbyprior)
{
    // which mode do we run in?
    const char * operation = "lleval";
    // load the model
    fprintf (stderr, "%s: loading model '%S'\n", operation, modelpath.c_str());
    msra::dbn::model model (modelpath);
    // feature reader
    msra::asr::htkfeatreader reader;
    model.entercomputation (0);
	onlineeval onlineeval(model,divbyprior); // get the posterior
    // process
    size_t numuptodate = 0;
    vector<wstring> outfiles; 
	outfiles.reserve (infiles.size());
    foreach_index (i, infiles)
    {
        const auto path = reader.parse (infiles[i]);    // parse a=b[s,e] syntax if present
        wstring outfile = makeoutpath (outdir, path, outext);
        outfiles.push_back (outfile);
        // read file
        msra::dbn::matrix feat;
        wstring featkind;
        unsigned int sampperiod;
        reader.read (path, featkind, sampperiod, feat);   // whole file read as columns of feature vectors
        fprintf (stderr, "evaluate: reading %d frames of %S\n", feat.cols(), ((wstring)path).c_str());
		onlineeval.evaltofile(feat, featkind, sampperiod, outfile);
    }

}

