#pragma once

#include <vector>
#include <Windows.h>
using namespace std;

#include "Common.h"
#include "CMatrix.h"

//#include "..\MSHpcClusteringLib\MSHpcClustering.h"

#ifdef SSE_OPTIMIZED
#include "mmintrin.h"
#endif


class CGaussianComponent
{
protected:
	vector<float> mean;
	vector<float> var;
	//vector<float> invvar;
	float gConst;

	double gamma;
	vector<double> gammaX;
	vector<double> gammaXX;
	CRITICAL_SECTION *cs;

public:
	friend class CGaussianMixture;

	//CGaussComponent(vector<float> &mean_in,vector<float> &var_in ,double gamma);
	//CGaussComponent(vector<float> &mean_in,vector<float> &var_in,float gconst_in);
	//CGaussComponent(int dim_in = -1);
	

	CGaussianComponent();
	CGaussianComponent(const CGaussianComponent &gc_in);
	~CGaussianComponent();

	float MOutP(const vector<float> &x);
	void Accumulate(const vector<float> &x, const double p);
	void Update();

protected:
	//float GetMahalanobisDistance(const vector<float> &X);
	//float GetProbDensity(const vector<float> &X);
	void FixGConst();
	void ApplyVarFloor(const vector<float> &varFloor);
	//void ClearStatistics();
	//void WriteAccumulatorVector(SPlatform::DoubleAccumulatorVector &vec);
	//void UpdateParameters(SPlatform::DoubleAccumulatorVector &vec,vector<float> &varFloor);
	//void Accumulate(vector<float> &X,double gammaZnk);
};

class CGaussianMixture
{
//public:
	//SPlatform::DoubleAccumulatorVector likelihood;

protected:
	int dim;
	int nMix;
	bool naturalReadOrder;
	double lr;

	vector<float> wt;
	vector<CGaussianComponent> mixture;

	vector<float> varFloor;
	CMatrix transMat;

	// HTK macro name
	string featType;
	string hmmName;
	vector<string> tailTransP;

private:
	float pruneTh;
	
public:
	CGaussianMixture(bool naturalReadOrder_in=false);
	//CGaussMixture(char *LBGFile1,char *LBGFile2);
	//CGaussMixture(int gnum,int dim_in);
	//CGaussMixture();
	//	~CGaussMixture();

	//void GetMixProbDensity(const vector<float> X,vector<float> &P);
	//void AccumulateHTKFeatureFile(SPFile *data);
//#ifdef LOCALREADING
//	void AccumulateHTKFeatureFile(FILE *data);
//#endif
	//void CalcLikelihood();
	//void Update(SPlatform::DoubleAccumulatorMatrix &mat);

	//void WriteASCIIResults(char *resultPath);
	void GetVarFloor(vector<float> &vec) const
	{
		vec = varFloor;
	}
	void setReadOrder(bool order)
	{
		naturalReadOrder = order;
	}
	void GetMixtureVar(vector<float> &vec, const size_t iComponent) const;
	void LoadGMMFile(const char *gmmFn);
	void SaveGMMFile(const char *gmmFn);
	void LoadLBGFile(const char *lbgFn_glb, const float vfRatio, const char *lbgFn);
	void LoadHldaTransFile(const char *fnHldaTrans);
	void Initialize();
	void AccumulateHTKFeatureFile(char * fname);
	void AccumulateHTKFeatureFile(char* fname, vector<float> &Gamma, vector<float> &GammaX);
	void AccumulateStatisticFile(char* fname, vector<float> &Gamma, vector<float>&GammaX);
	void WriteAccumulatorVector(char * fname, char *mode = "wb");
	int GetNumMixture() const
	{
		return(nMix);
	}

	int GetDim() const
	{
		return(dim);
	}

	void SetPruningTh(const float pruneTh_in)
	{
		pruneTh = pruneTh_in;
	}
	//void WriteAccumulatorMatrix(SPlatform::DoubleAccumulatorMatrix &mat);
	//void ClearStatistics(); 

	//int GetNumMixtures();
	//int GetDimension();
};
