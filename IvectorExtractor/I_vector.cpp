#include "I_vector.h"
#include <assert.h>
#include <stdlib.h>
#include <direct.h>

I_vector::I_vector(bool byteOrder)
{
	natureReadOrder = byteOrder;
	m_GMM.setReadOrder(byteOrder);
}
I_vector::~I_vector()
{
}


bool I_vector::LoadModel(const char * fname)
{
	m_GMM.LoadGMMFile(fname);
	C = m_GMM.GetNumMixture();
	K = m_GMM.GetDim();
	/// 1/varFloor
	m_GMM.GetVarFloor(iVarFloor);
	for (size_t i = 0; i < K; ++i)
	{
		iVarFloor[i] = 1 / iVarFloor[i];
	}

	/// 1/var
	iVar.setSize(C, K);
	vector<float> vec;
	for (size_t i = 0; i < C; ++i)
	{
		m_GMM.GetMixtureVar(vec, i);

		for (size_t j = 0; j < K; ++j)
		{
			vec[j] = 1 / vec[j];
			if (vec[j] > iVarFloor[j])
			{
				//TraceHR(E_UNEXPECTED, "CIVector::LoadModel: fail loading model: iVar[%d][%d] > iVarFloor[%d]", i, j, i);
			}
		}
		iVar.setRow(vec,i);
	}
	return true;
}
bool I_vector::LoadTmatrix(const char* fname)
{
	Tt.load(fname);
	Tt.transpose();
	R = Tt.nRow;
	CalculateTs();
	CalculateTsT();
	return true;
}
void I_vector::Initialize(const char* gmmModelPath, const char* ivecModelPath)
{
	LoadModel(gmmModelPath);
	LoadTmatrix(ivecModelPath);
	
	gConstMix.assign(C, 0);
	const float *pivar = (const float *)iVar.getRawPtr_const();
	for (size_t mixIdx = 0; mixIdx < C; ++mixIdx)
	{
		for (size_t i = 0; i < K; ++i, ++pivar)
		{
			gConstMix[mixIdx] += log(*pivar);
		}
		gConstMix[mixIdx] -= float(K * log(TPI));
	}

}

void I_vector::CalculateTs()
{
	if (iVar.empty())
	{
		;
	}

	Ts.assign(Tt);
	float * pTs = (float *)Ts.getRawPtr();
	const float * pivar;
	size_t N = Ts.nCol; // nrow or ncol
	__m128 _A, _B;
	size_t j;
	for (size_t i = 0; i < R; ++i)
	{
		for (j = 4, pivar = (const float *)iVar.getRawPtr(); j < N; j += 4, pivar += 4, pTs += 4)
		{
			_A = _mm_loadu_ps(pivar);
			_B = _mm_loadu_ps(pTs);
			_B = _mm_mul_ps(_A, _B);
			_mm_storeu_ps(pTs, _B);
		}
		for (j -= 4; j < N; ++j, ++pivar, ++pTs)
		{
			*pTs *= *pivar;
		}
	}
}

void I_vector::CalculateTsT()
{
	if (Ts.empty())
	{
		CalculateTs();
	}
	TsT.assign(C, CMatrix(R, R, 0.0));

	float * pTs = (float *)Ts.getRawPtr();
	const float * pTt;
	__m128 _A, _B, _C;
	for (size_t i = 0; i < R; ++i)
	{
		for (size_t mixIdx = 0; mixIdx < C; ++mixIdx, pTs += K)
		{
			for (size_t j = i; j < R; ++j)
			{
				const float * p = pTs;
				pTt = (float *)Tt[j] + mixIdx * K;
				_C = _mm_set1_ps(0.0);
				size_t k;
				for (k = 4; k < K; k += 4, p += 4, pTt += 4)
				{
					_A = _mm_loadu_ps(p);
					_B = _mm_loadu_ps(pTt);
					_B = _mm_mul_ps(_A, _B);
					_C = _mm_add_ps(_C, _B);
				}
				_C = _mm_hadd_ps(_C, _C);
				_C = _mm_hadd_ps(_C, _C);
				TsT[mixIdx][i][j] = _C.m128_f32[0];
				for (k -= 4; k < K; ++k, ++p, ++pTt)
				{
					TsT[mixIdx][i][j] += *pTt * (*p);
				}
				TsT[mixIdx][j][i] = TsT[mixIdx][i][j];
			}
		}
	}
}
void I_vector::writeIvectorHTKFormat(char * fIvecFilename, vector<float> ivecData, HTKhdr header)
{
	FILE *fp;
	int number = 0;
	fp = fopen(fIvecFilename, "wb");
	number=fwrite(&header, sizeof(header), 1, fp);
	for (int i = 0; i < ivecData.size(); i++)
		fwrite(&ivecData[i], sizeof(ivecData[0]), 1, fp);
	fclose(fp);
	printf("ivector extraction done for utterance: %s\n", fIvecFilename);
	
}
void I_vector::IvectorEstimation(char* fname, char* fIvecFilename, bool inputIsStatistic)
{
	size_t size = C*K;
	vector<float> Gamma_f;
	vector<float> GammaX_f; // input
	CMatrix Matrix_f; // L-I
	Gamma_f.resize(C);
	GammaX_f.resize(size); 
	Matrix_f.setSize(R, R);
	if (inputIsStatistic)
	{
		m_GMM.AccumulateStatisticFile(fname, Gamma_f, GammaX_f);
	}
	else
	{
		m_GMM.AccumulateHTKFeatureFile(fname, Gamma_f, GammaX_f);
	}


	// calculate L-I
	for (size_t mixIdx = 0; mixIdx < C; ++mixIdx)
	{
		if (Gamma_f[mixIdx] > MIN_GAMMA)
		{
			Matrix_f.WeightedAdd(TsT[mixIdx], Gamma_f[mixIdx]); // L - I
		}
	}
	
	
	CMatrix  &symMatrix_1 = Matrix_f; // symMatrix_1 is L right now, at last it will be Eww
	CMatrix  symMatrix_2;

	//symMatrix_1[0][0] += 1;
	for (size_t i = 0; i < R; ++i)
	{
		symMatrix_1[i][i] += 1;
	}

	symMatrix_1.invChol(symMatrix_2); // sysMatrix_2 is the inverse L right now

	/// Calculate i-vector (E[w(s)])
	vector<float> SG_f, Ew_f; //GammaX_f;// GammaXX_f;



	SG_f = Ts * GammaX_f;
	Ew_f = symMatrix_2 * SG_f;

	vector<float> swapvec;
	swapvec.resize(R);
	for (size_t i = 0; i < R; ++i)
	{
		swapvec[i] = Ew_f[i];
		Swap32((int *)&swapvec[i]);
	}
	HTKhdr ihdr;
	ihdr.nSamples = 1;
	ihdr.sampKind = 9;
	ihdr.sampPeriod = 100000;
	ihdr.sampSize = 4*R; //4*sample size
	Swap32(&ihdr.nSamples);
	Swap32(&ihdr.sampPeriod);
	Swap16(&ihdr.sampSize);
	Swap16(&ihdr.sampKind);

	// writer i-vector 
	writeIvectorHTKFormat(fIvecFilename, swapvec, ihdr);

}
