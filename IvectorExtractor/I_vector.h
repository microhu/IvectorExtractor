#pragma once
#include <vector>
#include <string>
using namespace std;

#include "Common.h"
#include "CMatrix.h"
#include "mmintrin.h"
#include "Gmm.h"
const double MIN_GAMMA = 1e-8;
struct SIVectorHeader
{
	ULONG K;  // feature vector dimension
	ULONG R;  // i-vector dimension
	ULONG C;  // Count of mixture components
};
class I_vector: SIVectorHeader  // changed it to public
{
public:
	I_vector(bool byteOrder);
	void Initialize(const char* gmmModelPath, const char* ivecModelPath);
	void IvectorEstimation(char* fname, char* fIvecFilename, bool inputIsStatisticFlag);
	~I_vector();
protected:
	bool natureReadOrder;
	CMatrix iVar;  // matrix of 1/variance, C-by-K
	vector<float> iVarFloor;
	vector<float> gConstMix; // C-by-1, log(ivar) - K * log(TPI) for each mixture
	CMatrix Tt;    // transpose of T matrix, R-by-C*K
	CMatrix Ts;           // Tt*ivar
	vector<CMatrix> TsT; // Tt*ivar*T, block matrix: R(R+1)/2 (total num is C)
	CGaussianMixture m_GMM;
	//SIVectorHeader GDim;

	void CalculateTs();
	void CalculateTsT();
	static const ULONG ulCS = 1024;
	bool LoadModel(const char *fname);
	bool LoadTmatrix(const char *fname);
	void writeIvectorHTKFormat(char * fIvecFilename, vector<float> ivecData, HTKhdr header);
	inline size_t CSIdx(size_t mixIdx) const
	{
		return (mixIdx % ulCS);
	}
};

