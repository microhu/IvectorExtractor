#pragma once

#include <vector>
#include <string>
using namespace std;


#include "Common.h"
#include "CMatrix.h"
#include "mmintrin.h"

/// min gamma
const double MIN_GAMMA = 1e-8;

/// accumulator buffer size 
const int ACC_BUF_SIZE = 4;  // this value (4) is got by experiments with R=400
const int ACC_BUF_MIN_SIZE = 2;
const int ACC_BUF_MAX_SIZE = 16;
const float L2CACHE_USE_RATIO = 0.80f;

/// update flag
#define UPDATE_TRANS        1
#define UPDATE_VAR          2
#define UPDATE_IVEC         4

/// file name
#define IVEC_EXTENSION	          "ivec"
#define OUT_T_MATRIX_FILE_NAME    "Xform"
#define OUT_MODEL_FILE_NAME       "SM"
#define STAT_EXTENSION            "stat"

#ifdef OUTPUT_IGHF
#define IGMM_EXTENSION            "gmm"
#define IHMM_EXTENSION            "hmm"
#define IFLG_EXTENSION            "flg"
#endif

const ULONG WriterThreadScale = 8;

enum IVectorProcessType {IVec_UpdateParameters, IVec_EstimateIVector };

struct SIVectorInParameters
{
	/// args
	int               traceLevel;
	int               numThread;
	int               numWriterPerThread;
	int               numAccCacheLine;      // acc cache size
	int               dim;					// i-vector dimension
	int               numIter;				// iteration number
	float             TScale;               // scale for T random initialized values
	float             pruneTh;              // prune th. used in GMM calculating
	bool              fUpdateTrans;			// update T
	bool              fUpdateVar;			// update variance
	bool              fUpdateIVec;          // update ivector
	bool              fBinaryFormat;		// input and output file format
	bool              fSkip;				// skip iteration
	bool              fAcousticFeature;     // input data is acoustic feature
	bool              fNaturalReadOrder;    // nature read order
	bool              fUseDoubleAcc;		// accumulators are double format
	bool              fUseDoubleStat;       // input stat is double
    bool              fSaveiStatOnly;       // save stat (gamma) for each speech segment
#ifdef OUTPUT_IGHF
    bool              fSaveiGMMOnly;        // save gmm only for each speech segment
    bool              fSaveiHMMOnly;        // save hmm only for each speech segment
    bool              fOutFrameLabelofGmm;  // output label of gaussian in GMM for each frame
    bool              fOutFrameLabelofUBM;  // output label of gaussian in UBM for each frame
#endif
	const char        * pszJobID;			// job id
	const char        * pszInModelFile;		// input model file name
	const char        * pszInUBMFile;       // input UBM file
	const char        * pszInTransFile;		// input transform (T) file name
	const char        * pszWorkDir;			// working directory
	const char        * pszOutIVecDir;      // directory of output ivector
	const char        * pszHldaTransFn;     // HLDA transform file
};

struct SIVectorHeader
{
    ULONG K;  // feature vector dimension
    ULONG R;  // i-vector dimension
    ULONG C;  // Count of mixture components
};

struct SIVectorAccHeader
{
	float m_cTask; // Total number of i-vector Tasks accumulated
    float m_totData;   // Total frames of data
	float m_totOcc; // Total occupancy
	float m_totLike; // Total likelihood
	float m_totFunc; // Total objective function
};

struct SIVectorRemoteWriterStat
{
	HTKhdr                   hdr;
	BYTE *                   buf;
	int                      nOffset; // for suport multi-utterance-stat in one stat file
};

struct SIVectorAccBuffer 
{	// used to speed up calculating
	vector<CMatrix> Matrix; // buffer for L and Eww
	// vector<CSymmetricMatrix_Double> Matrix_Double;
	vector<vector<double>>   Gamma;
	vector<vector<double>>   GammaX;
	vector<vector<double>>   GammaXX;
	vector<string>           filename;
	int                      nUsed;
	int                      nMaxSize;

	// accumulators used in thread
	SIVectorAccHeader        accHeader;
	CMatrix                  accGammaXEw;
	vector<double>           accGamma; 
	vector<double>           accGammaXX;

	// remote writer stat
	vector<SIVectorRemoteWriterStat>    remoteWriter;

    // input feature if have
    vector<vector<float>>    feat;
};

class CIVector : protected SIVectorHeader
{
public:
	CIVector(const int numThread = 0);

	~CIVector(void);
	// Initialize members in CIVector class
	void Allocate(const SIVectorInParameters &paras);
	
	// Pre-calculate and broadcast
	void PrepareNewIteration(IVectorProcessType type);
	
	// Initialize accumulator buffer
	void InitializeBufferInThread(SIVectorAccBuffer &AccBuf);

	// Get file from platform, and fill into bufer
	//bool LoadRecordStatIntoBuffer(SPlatform::SPFile &file, SIVectorRemoteWriterStat &stat, SIVectorAccBuffer &AccBuf);

	// Calculate L and accumulate Gamma and GammaXX in local for all records in buffer
	void AccumulateRecordStatBuffer(SIVectorAccBuffer &AccBuf);

	// Estimate i-vector, cal likelihood, etc for one record 
	void AccumulateRecordStat(SIVectorAccBuffer &AccBuf, ULONG curBufIdx);

	// Accumulate GammaEww in buffer
	void AccumulateGammaEww(SIVectorAccBuffer &AccBuf);

	// Accumulate header, GammaXEw, Gamma and GammaXX
	void Accumulate(SIVectorAccBuffer &AccBuf);

	// WriteAccumulator + reduce
	//void Reduce(SPlatform::DoubleAccumulatorVector &acc);

	// Check complete
	bool IterationComplete(bool fSkip, const char * pszSpecialExtensionName);
	
	ULONG getAccSize();
	
	// Update T and Var
	//void Update(SPlatform::DoubleAccumulatorVector &acc);
	
	void SaveUpdatedResults(const char * pszSpecialExtensionName);

protected:
	CMatrix iVar;  // matrix of 1/variance, C-by-K
	vector<float> iVarFloor;
	vector<float> gConstMix; // C-by-1, log(ivar) - K * log(TPI) for each mixture
	CMatrix Tt;    // transpose of T matrix, R-by-C*K
	CMatrix Ts;           // Tt*ivar
	vector<CMatrix> TsT; // Tt*ivar*T, block matrix: R(R+1)/2 (total num is C)

protected:
	void ClearStatistics();
	HRESULT LoadOneRecordStat(char* pFile, vector<double> &Gamma, vector<double> &GammaX, vector<double> &GammaXX, vector<float> &featBuf);
	HRESULT ReadHeaderOfRecord(char* pFile, ULONG &nRecord);
	int Serialize(BYTE *buf);
	int Deserialize(BYTE *buf);
	void CalculateTs();
	void CalculateTsT();
	//void WriteAccumulator(SPlatform::DoubleAccumulatorVector & acc);
	//void ReadAccumulator(SPlatform::DoubleAccumulatorVector & acc);
	void EstimateTMatrix();
	void EstimateVar();
	
	inline size_t CSIdx(size_t mixIdx) const
    {
        return (mixIdx % ulCS);
    }
	void LockList(ULONG idx);
	void unlockList(ULONG idx);
	int tryLockList(ULONG idx);
	bool LoadModel(const char *fname);
	bool SaveModel(const char *fname);
	bool InitTMatrix(const char *fname, const float TScaleMin = 0.0f, const float TScaleMax = 0.0f);

protected:
	
	bool m_fBinaryFormat;
	bool m_fAcousticFeature;
	IVectorProcessType m_type;
	int m_updateFlag;
	string m_szWorkDir;
	string m_szOutIVecDir;
	string m_szOutTransFile;
	string m_szOutModelFile;
	string m_szJobID;

	/// model
	CGaussianMixture m_GMM;
	CGaussianMixture m_UBM;

	/// Critical sections used to protect the accumulator space
	static const ULONG ulCS = 1024;
	CRITICAL_SECTION m_csList[ulCS];  
	CRITICAL_SECTION m_csHeader;
	CRITICAL_SECTION m_csGxEw;
	CRITICAL_SECTION m_csG;
	CRITICAL_SECTION m_csGxx;

	/// statistic
	SIVectorAccHeader m_accHeader;     // size = 5
	CMatrix m_accGammaXEw;	         // size = [C*K, R]
	// only used in update T matrix
	vector<CMatrix> m_accGammaEww; // size = C*[R, R]
	// vector<CSymmetricMatrix_Double> m_accGammaEww_Double;
	// only used in update variance
	vector<double> m_accGamma;          // size = C
	vector<double> m_accGammaXX;        // size = C*K
	bool m_fUseDoubleAcc;
	bool m_fUseDoubleStat;
    bool m_fSaveiStatOnly;
#ifdef OUTPUT_IGHF
    bool m_fOutGMMFamily;
    bool m_fSaveiGMMOnly;
    bool m_fSaveiHMMOnly;
    bool m_fOutFrameLabelofGmm;  // output label of gaussian in GMM for each frame
    bool m_fOutFrameLabelofUBM;
#endif
	/// accumulator cache
	int m_numCacheLine;
	int m_numThread;

	/// Remote Writer
	int m_numWriterThread;
	//MSHpcRemoteWriter m_IVecWriter;
};

