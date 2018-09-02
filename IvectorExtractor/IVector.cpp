#include "IVector.h"
#include <assert.h>
#include <stdlib.h>
#include <direct.h>


CIVector::CIVector(const int numThread)
{
	m_fBinaryFormat = false;
	m_fAcousticFeature = false;
	m_fUseDoubleAcc = false;
    m_fSaveiStatOnly = false;
#ifdef OUTPUT_IGHF
    m_fSaveiGMMOnly = false;
    m_fSaveiHMMOnly = false;
    m_fOutFrameLabelofGmm = false;
    m_fOutFrameLabelofUBM = false;
    m_fOutGMMFamily = false;
#endif
	m_type = IVec_UpdateParameters;
	m_updateFlag = 0;
	K = R = C = 0;
	m_numCacheLine = 0;
	m_numThread = numThread;
	m_numWriterThread = 0;
	InitializeCriticalSection(&m_csHeader);
	InitializeCriticalSection(&m_csGxEw);
	InitializeCriticalSection(&m_csG);
	InitializeCriticalSection(&m_csGxx);
	for (ULONG i = 0; i < ulCS; ++i)
	{
		InitializeCriticalSection(&m_csList[i]);
	}
}

CIVector::~CIVector(void)
{
	DeleteCriticalSection(&m_csHeader);
	DeleteCriticalSection(&m_csGxEw);
	DeleteCriticalSection(&m_csG);
	DeleteCriticalSection(&m_csGxx);
	for (ULONG i = 0; i < ulCS; ++i)
	{
		DeleteCriticalSection(&m_csList[i]);
	}
}

void CIVector::Allocate(const SIVectorInParameters &paras)
{

	R = paras.dim;

		m_szOutTransFile = ConcatenateFileFullPath(m_szWorkDir.c_str(), OUT_T_MATRIX_FILE_NAME, IVEC_EXTENSION);
		m_szOutModelFile = ConcatenateFileFullPath(m_szWorkDir.c_str(), OUT_MODEL_FILE_NAME, NULL);

		/// main node, load model and T matrix (store transpose of T)
		// load model
		LoadModel(paras.pszInModelFile);

		// load Tt matrix
		InitTMatrix(paras.pszInTransFile, -paras.TScale, paras.TScale);
		SaveUpdatedResults("init");

		// load UBM / HLDA if data type is acoustic feature

        if (m_fAcousticFeature)
		{
			
				m_UBM.LoadGMMFile(paras.pszInUBMFile);
				m_UBM.SetReadOrder(paras.fNaturalReadOrder);
				m_UBM.SetPruningTh(paras.pruneTh);
			
		}

	    if (m_fAcousticFeature)
	    {
		    m_UBM.Initialize();
	    }
    
    
}

void CIVector::AccumulateRecordStat(SIVectorAccBuffer &AccBuf, ULONG curBufIdx)
{
	HRESULT hr = S_OK;
	CMatrix  &symMatrix_1 = AccBuf.Matrix[curBufIdx]; // symMatrix_1 is L right now, at last it will be Eww
	CMatrix  symMatrix_2;
	vector<double>     &Gamma   = AccBuf.Gamma[curBufIdx];
	vector<double>     &GammaX  = AccBuf.GammaX[curBufIdx];
	vector<double>     &GammaXX = AccBuf.GammaXX[curBufIdx];
    vector<float>      &featBuf = AccBuf.feat[curBufIdx];
	SIVectorAccHeader  &accHeader   = AccBuf.accHeader;
	CMatrix            &accGammaXEw = AccBuf.accGammaXEw;
    

	string &fStatFileName = AccBuf.filename[curBufIdx];
	string fIVecFileName;
    string fIStatFileName;
#ifdef OUTPUT_IGHF
    string fIGMMFileName;
    string fIHMMFileName;
    string fIFLGFileName;
#endif
	SIVectorRemoteWriterStat &remoteWriter = AccBuf.remoteWriter[curBufIdx];

	FILE *f = NULL;

	/// set ivector file name
	if (m_type == IVec_EstimateIVector)
	{
		string::size_type pos1 = fStatFileName.find_last_of('\\');
		string::size_type pos2 = fStatFileName.find_last_of('.');
		string szIvecExt = m_szJobID.empty() ? IVEC_EXTENSION : (m_szJobID + "." + IVEC_EXTENSION);
        string szIStatExt = m_szJobID.empty() ? STAT_EXTENSION : (m_szJobID + "." + STAT_EXTENSION);
#ifdef OUTPUT_IGHF
        string szIgmmExt = m_szJobID.empty() ? IGMM_EXTENSION : (m_szJobID + "." + IGMM_EXTENSION);
        string szIhmmExt = m_szJobID.empty() ? IHMM_EXTENSION : (m_szJobID + "." + IHMM_EXTENSION);
        string szIflgExt = m_szJobID.empty() ? IHMM_EXTENSION : (m_szJobID + "." + IFLG_EXTENSION);
#endif
		if (m_szOutIVecDir.empty())
		{
			// ivector filename = stat file name + ".ivec"
			string pathname = fStatFileName.substr(0, pos2 + 1);
			fIVecFileName = pathname + szIvecExt;
            fIStatFileName = pathname + szIStatExt;
#ifdef OUTPUT_IGHF
            fIGMMFileName = pathname + szIgmmExt;
            fIHMMFileName = pathname + szIhmmExt;
            fIFLGFileName = pathname + szIflgExt;
#endif
		}
		else
		{
			string name = fStatFileName.substr(pos1 + 1, pos2 - pos1 - 1);
			fIVecFileName = ConcatenateFileFullPath(m_szOutIVecDir.c_str(), name.c_str(), szIvecExt.c_str());
            fIStatFileName = ConcatenateFileFullPath(m_szOutIVecDir.c_str(), name.c_str(), szIStatExt.c_str());
#ifdef OUTPUT_IGHF
            fIGMMFileName = ConcatenateFileFullPath(m_szOutIVecDir.c_str(), name.c_str(), szIgmmExt.c_str());
            fIHMMFileName = ConcatenateFileFullPath(m_szOutIVecDir.c_str(), name.c_str(), szIhmmExt.c_str());
            fIFLGFileName = ConcatenateFileFullPath(m_szOutIVecDir.c_str(), name.c_str(), szIflgExt.c_str());
#endif
		}
	}
	
	/// Calculate invert of l(s) 
	// + I
	if (m_fUseDoubleAcc)
	{
		for (size_t i = 0; i < R; ++i)
		{
			symMatrix_1.doubleElement(i, i) += 1;
		}
	}
	else
	{
		for (size_t i = 0; i < R; ++i)
		{
			symMatrix_1.floatElement(i, i) += 1;
		}
	}

	hr = symMatrix_1.cholesky(symMatrix_2); 
	TraceHR(hr, "CIVector::AccumulateRecordStat: fail to inverse L.\n");

	symMatrix_2.invAfterChol(symMatrix_1); // symMatrix_1 is inverse L right now
	_DEBUG_TRACE((TRACE_DEBUGINFO, "Debug info: L.invChol() completed.\n"));
	float logdet = (float)symMatrix_2.logdetAfterChol();
		
	/// Calculate i-vector (E[w(s)])
	vector<float> SG_f, Ew_f, GammaX_f, GammaXX_f;
	vector<double> SG_d, Ew_d;
	if (m_fUseDoubleAcc)
	{
		SG_d = Ts * GammaX;
		Ew_d = symMatrix_1 * SG_d;
	}
	else
	{
		size_t size = C*K;
		GammaX_f.resize(size);
		GammaXX_f.resize(size);
		for (size_t i = 0; i < size; ++i)
		{
			GammaX_f[i] = (float)GammaX[i];
			GammaXX_f[i] = (float)GammaXX[i];
		}
		SG_f = Ts * GammaX_f;
		Ew_f = symMatrix_1 * SG_f;
	}
	_DEBUG_TRACE((TRACE_DEBUGINFO, "Debug info: computing Ew completed.\n"));
	if (m_type == IVec_UpdateParameters)
	{
		/// Calculate 
		float likelihood = 0.0, accEwT = 0.0;
		double gConst = 0.0;
		if (m_fUseDoubleAcc)
		{
			symMatrix_1.Add(Ew_d);       // symMatrix_1 is Eww right now, Eww = invL + Ew * Ew;
			_DEBUG_TRACE((TRACE_DEBUGINFO, "Debug info: computing Eww completed.\n"));
			
			/// Calculate likelihood
			accEwT = likelihood = (float)CMatrix::DotProduct(Ew_d, SG_d);
			for (size_t mixIdx = 0; mixIdx < C; ++mixIdx)
			{
				if (Gamma[mixIdx] > MIN_GAMMA)
				{
					gConst += gConstMix[mixIdx] * Gamma[mixIdx] - CMatrix::DotProduct((const double *)iVar[mixIdx], &GammaXX[mixIdx * K], K);
				}
			}
			likelihood += (float)gConst;
			likelihood -= logdet;
			_DEBUG_TRACE((TRACE_DEBUGINFO, "Debug info: computing likelihood completed.\n"));

			// GammaX * Ew
			accGammaXEw.Add(GammaX, Ew_d);
			_DEBUG_TRACE((TRACE_DEBUGINFO, "Debug info: acc GammaXEw completed.\n"));
		}
		else
		{
			symMatrix_1.Add(Ew_f);       // symMatrix_1 is Eww right now, Eww = invL + Ew * Ew;
			_DEBUG_TRACE((TRACE_DEBUGINFO, "Debug info: computing Eww completed.\n"));
			
			/// Calculate likelihood
			accEwT = likelihood = CMatrix::DotProduct(Ew_f, SG_f);
			for (size_t mixIdx = 0; mixIdx < C; ++mixIdx)
			{
				if (Gamma[mixIdx] > MIN_GAMMA)
				{
					gConst += gConstMix[mixIdx] * Gamma[mixIdx] - CMatrix::DotProduct((const float *)iVar[mixIdx], &GammaXX_f[mixIdx * K], K);
				}
			}
			likelihood += (float)gConst;
			likelihood -= logdet;
			_DEBUG_TRACE((TRACE_DEBUGINFO, "Debug info: computing likelihood completed.\n"));

			// GammaX * Ew
			accGammaXEw.Add(GammaX_f, Ew_f);
			_DEBUG_TRACE((TRACE_DEBUGINFO, "Debug info: acc GammaXEw completed.\n"));
		}
		/// Accumulate statistic
		// header
		accHeader.m_totData += 1;
		accHeader.m_totLike += likelihood;
		accHeader.m_totFunc += accEwT;		// just used for check
		accHeader.m_totOcc  += logdet;		// just used for check
		_DEBUG_TRACE((TRACE_DEBUGINFO, "Debug info: acc likelihood completed, likelihood = %f.\n", accHeader.m_totLike));
	}
	else if (m_type == IVec_EstimateIVector)
	{
#ifdef OUTPUT_IGHF
        if (m_fOutGMMFamily)
        {
            // save i-gmm only
            CMatrix temT;
            Tt.transpose(temT);
            if (m_fUseDoubleAcc)
		    {
			    Ew_f.resize(R);
			    for (size_t i = 0; i < R; ++i)
			    {
				    Ew_f[i] = (float)Ew_d[i];
			    }
		    }
            vector<float> mean = temT * Ew_f;

            CGaussianMixture gmm(m_UBM);
            gmm.AddToMixtureMean(mean);
            if (m_fSaveiGMMOnly)
            {
                gmm.SaveGMMFile(fIGMMFileName.c_str(), false);
            }
            else if(m_fSaveiHMMOnly)
            {
                gmm.SaveHMMFile(fIHMMFileName.c_str(), false);
            }

            if (m_fOutFrameLabelofGmm)
            {
                ULONG nSamples = featBuf.size() / K;
                if (nSamples * K != featBuf.size())
                {
                    TraceHR(E_UNEXPECTED, "CIVector::AccumulateRecordStat: feat dimension mismatch.\n");
                }
                int maxTry = 10;
                FILE *fp;
                while (fopen_s(&fp, fIFLGFileName.c_str(), "wt") != 0 && maxTry > 0)
                {
                    Sleep(5000);
                    maxTry--;
                }
                if(maxTry <= 0)
	            {
		            TraceHR(E_UNEXPECTED, "AccumulateRecordStat: Cannot create %s", fIFLGFileName.c_str());
	            }
                fprintf(fp, "%d %d\n", gmm.GetNumMixture(), nSamples);
                vector<float> temp(K);
                for (size_t i = 0; i < nSamples; ++i)
                {
                    temp.assign(featBuf.begin() + i*K, featBuf.begin() + (i+1)*K);
                    size_t label = m_fOutFrameLabelofUBM ? m_UBM.GetMaxLikelihoodComponentIdx(temp) : gmm.GetMaxLikelihoodComponentIdx(temp);
                    fprintf(fp, "%d ", label);
                }
                fprintf(fp, "\n");
                fclose(fp);
            }
        }
        else
#endif
        {
		    // Store i-vector in buf
		    if (remoteWriter.buf == NULL || remoteWriter.nOffset < 0 || remoteWriter.nOffset >= remoteWriter.hdr.nSamples)
		    {
			    TraceHR(E_UNEXPECTED, "CIVector::AccumulateRecordStat: unexpected remoteWriter.\n");
		    }
		    BYTE *p = remoteWriter.buf + (remoteWriter.hdr.sampSize * remoteWriter.nOffset) + sizeof(HTKhdr);
		    if (m_fUseDoubleAcc)
		    {
			    Ew_f.resize(R);
			    for (size_t i = 0; i < R; ++i)
			    {
				    Ew_f[i] = (float)Ew_d[i];
			    }
		    }
            vector<float> swapvec;
            string fnOut;
            if (m_fSaveiStatOnly)
            {
                fnOut = fIStatFileName;
                swapvec.resize(C);
                for (size_t i = 0; i < C; ++i)
                {
                    swapvec[i] = (float)Gamma[i];
                    Swap32((int *)&swapvec[i]);
                }
            }
            else
            {
                fnOut = fIVecFileName;
                swapvec.resize(R);
                for (size_t i = 0; i < R; ++i)
                {
                    swapvec[i] = Ew_f[i];
                    Swap32((int *)&swapvec[i]);
                }
            }
		    CopyMemory(p, &swapvec[0], remoteWriter.hdr.sampSize);
		    _DEBUG_TRACE((TRACE_DEBUGINFO, "Writed i-vector in file %s.\n", fnOut.c_str()));
		    if (remoteWriter.nOffset == remoteWriter.hdr.nSamples - 1)
		    {
			    // remote write i-vector
                int len = sizeof(HTKhdr) + remoteWriter.hdr.nSamples * remoteWriter.hdr.sampSize;
			    m_IVecWriter.PushFile(fnOut, (PVOID) remoteWriter.buf, len);
			    delete remoteWriter.buf;
			    _DEBUG_TRACE((TRACE_DEBUGINFO, "Saved %d i-vectors in: %s \n", remoteWriter.hdr.nSamples, fnOut.c_str()));
		    }
        }
	}
}

HRESULT CIVector::LoadOneRecordStat(SPFile *pFile, vector<double> &Gamma, vector<double> &GammaX, vector<double> &GammaXX, vector<float> &featBuf)
{
	if (Gamma.size() != C) Gamma.assign(C, 0.0);
	if (GammaX.size() != C*K) GammaX.assign(C*K, 0.0);
	if (GammaXX.size() != C*K) GammaXX.assign(C*K, 0.0);

	if (m_fAcousticFeature)
	{
#ifdef OUTPUT_IGHF
        if (m_fOutFrameLabelofGmm)
            m_UBM.AccumulateHTKFeatureFile(*pFile, Gamma, GammaX, GammaXX, featBuf);
        else
#endif
		    m_UBM.AccumulateHTKFeatureFile(*pFile, Gamma, GammaX, GammaXX);
	}
	else
	{
		if (m_fUseDoubleStat)
		{
			for (size_t i = 0; i < C; ++i)
			{
				if(pFile->fread_s(&Gamma[i], sizeof(double), sizeof(double), 1) != 1)
				{
					return E_UNEXPECTED;
				}
				if(pFile->fread_s(&GammaX[i * K], sizeof(double) * K, sizeof(double), K) != K)
				{
					return E_UNEXPECTED;
				}
				if(pFile->fread_s(&GammaXX[i * K], sizeof(double) * K, sizeof(double), K) != K)
				{
					return E_UNEXPECTED;
				}
			}
		}
		else
		{
			float buf;
			size_t idx = 0;
			for (size_t i = 0; i < C; ++i)
			{
				if(pFile->fread_s(&buf, sizeof(float), sizeof(float), 1) != 1)
				{
					return E_UNEXPECTED;
				}
				Gamma[i] = (double)buf;
				idx = i * K;
				for (size_t j = 0; j < K; ++j)
				{
					if(pFile->fread_s(&buf, sizeof(float), sizeof(float), 1) != 1)
					{
						return E_UNEXPECTED;
					}
					GammaX[idx +j] = (double)buf;
				}
				for (size_t j = 0; j < K; ++j)
				{
					if(pFile->fread_s(&buf, sizeof(float), sizeof(float), 1) != 1)
					{
						return E_UNEXPECTED;
					}
					GammaXX[idx + j] = (double)buf;
				}
			}
		}
	}
	/// debug
	/*FILE *fp = NULL;
	string fname = pFile->GetFilePath() + ".stat";
	if(fopen_s(&fp, fname.c_str(), "wt"))
	{
		TraceHR(E_UNEXPECTED, "LoadGMMFile: Cannot create %s", fname.c_str());
	}
	fprintf(fp, "%s\n", fname.c_str());
	fprintf(fp, "Gamma: ");
	for (size_t i = 0; i < C; ++i)
	{
		fprintf(fp, "%e ", Gamma[i]);
	}
	fprintf(fp, "\nGammaX: ");
	for (size_t i = 0; i < GammaX.size(); ++i)
	{
		fprintf(fp, "%e ", GammaX[i]);
	}
	fprintf(fp, "\n");
	fclose(fp);*/
	return S_OK;
}

HRESULT CIVector::ReadHeaderOfRecord(SPFile *pFile, ULONG &nRecord)
{
	if (m_fAcousticFeature)
	{
		// load record from acoustic feature file
		// skip reading header
		nRecord = 1;
	}
	else
	{
		// load record from statistic file
		ULONG header[2];
		if(pFile->fread_s(&header, sizeof(ULONG)*2, sizeof(ULONG), 2) != 2)
		{
			return E_UNEXPECTED;
		}
		if (header[0] == 0 || header[1] != C * (1 + 2 * K))
		{
			return E_INVALIDARG;
		}
		nRecord = header[0];
	}
	return S_OK;
}

void CIVector::ClearStatistics()
{
	m_accHeader.m_cTask = 0;
	m_accHeader.m_totData = 0;
	m_accHeader.m_totFunc = 0;
	m_accHeader.m_totLike = 0;
	m_accHeader.m_totOcc = 0;

	string strFormat = (m_fUseDoubleAcc) ? "double" : "float";

	m_accGammaEww.resize(C);
	for (size_t i = 0; i < C; ++i)
	{
		m_accGammaEww[i].assign(R, R, 0.0, strFormat.c_str(), "symmetric");
	}
	m_accGammaXEw.assign(C*K, R, 0.0, strFormat.c_str());
	m_accGamma.assign(C, 0.0);
	m_accGammaXX.assign(C * K, 0.0);
}

int CIVector::Serialize(BYTE *buf)
{
	BYTE *p = buf;

	/// header C, K
	if(buf) memcpy(p, &C, sizeof(ULONG));
	p += sizeof(ULONG);
	if(buf) memcpy(p, &K, sizeof(ULONG));
	p += sizeof(ULONG);

	/// iVar matrix
	if(buf) memcpy(p, iVar.getRawPtr_const(), iVar.getRawSizeInByte());
	p += iVar.getRawSizeInByte();

	/// Tt matrix
	// Tt matrix
	if(buf) memcpy(p, Tt.getRawPtr_const(), Tt.getRawSizeInByte());
	p += Tt.getRawSizeInByte();

	/// Ts matrix
	// Ts matrix
	if(buf) memcpy(p, Ts.getRawPtr_const(), Ts.getRawSizeInByte());
	p += Ts.getRawSizeInByte();

	/// TsT
	for (ULONG i = 0; i < C; ++i)
	{
		// TsT[i]
		if(buf) memcpy(p, TsT[i].getRawPtr_const(), TsT[i].getRawSizeInByte());
		p += TsT[i].getRawSizeInByte();
	}

	/// gConstMix
	if(buf) memcpy(p, &gConstMix[0], sizeof(float) * C);
	p += sizeof(float) * C;

	return (int)(p-buf);
}

int CIVector::Deserialize(BYTE *buf)
{
	BYTE *p = buf;

	/// header C, K
	if(buf) memcpy(&C, p, sizeof(ULONG));
	p += sizeof(ULONG);
	if(buf) memcpy(&K, p, sizeof(ULONG));
	p += sizeof(ULONG);
	
	string strFormat = (m_fUseDoubleAcc) ? "double" : "float";
	/// iVar matrix
	iVar.assign(C, K, 0.0, strFormat.c_str());
	if(buf) memcpy(iVar.getRawPtr(), p, iVar.getRawSizeInByte());
	p += iVar.getRawSizeInByte();

	/// Tt matrix
	Tt.assign(R, C*K, 0.0, strFormat.c_str());
	// Tt matrix
	if(buf) memcpy(Tt.getRawPtr(), p, Tt.getRawSizeInByte());
	p += Tt.getRawSizeInByte();

	/// Ts matrix
	Ts.assign(R, C*K, 0.0, strFormat.c_str());
	// Ts matrix
	if(buf) memcpy(Ts.getRawPtr(), p, Ts.getRawSizeInByte());
	p += Ts.getRawSizeInByte();

	/// TsT
	TsT.resize(C);
	for (ULONG i = 0; i < C; ++i)
	{
		// TsT[i]
		TsT[i].assign(R, R, 0.0, strFormat.c_str(), "symmetric");
		if(buf) memcpy(TsT[i].getRawPtr(), p, TsT[i].getRawSizeInByte());
		p += TsT[i].getRawSizeInByte();
	}

	/// gConstMix
	gConstMix.resize(C);
	if(buf) memcpy(&gConstMix[0], p, sizeof(float) * C);
	p += sizeof(float) * C;

	return (int)(p-buf);
}

// Calculate Ts, TsT and broadcast C, K, Tt, Ts and TsT
void CIVector::PrepareNewIteration(IVectorProcessType type)
{
	BYTE *buf;
	int len, recv;

	m_type = type;

		/// main node, calculate Ts and TsT
		CalculateTs();
		CalculateTsT();

		/// calculate gConst of each mixture (used in calculating likelihood)
		// gConst = log(ivar) - K * log(TPI)
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

		/// conver iVar, Tt, Ts and TsT to double format if using double acc
		if (m_fUseDoubleAcc)
		{
			iVar.ConvertTo(Double);
			Tt.ConvertTo(Double);
			Ts.ConvertTo(Double);
			for (size_t i = 0; i < C; ++i)
			{
				TsT[i].ConvertTo(Double);
			}
		}

		/// and broadcast C, K, Tt, Ts and TsT
		len = Serialize(NULL);
		buf = new BYTE[len];
		Serialize(buf);
		delete [] buf;
	
	/// initialize statistics
	ClearStatistics();
}

void CIVector::CalculateTs()
{
	if (iVar.empty())
	{
		//TraceHR(E_UNEXPECTED, "CIVector::CalculateTs: no ivar");
	}

	Ts.assign(Tt);
	float * pTs = (float * )Ts.getRawPtr();
	const float * pivar;
	size_t N = Ts.getCol();
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

void CIVector::CalculateTsT()
{
	if (Ts.empty())
	{
		CalculateTs();
	}
	TsT.assign(C, CMatrix(R,R,0.0,Symmetric, Float));

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
				TsT[mixIdx].floatElement(i,j) = _C.m128_f32[0];
				for (k -= 4; k < K; ++k, ++p, ++pTt)
				{
					TsT[mixIdx].floatElement(i,j) += *pTt * (*p);
				}
			}
		}
	}
}

void CIVector::Accumulate(SIVectorAccBuffer &AccBuf)
{
	if (m_type == IVec_EstimateIVector)
	{
		return;
	}

	/// acc header
	EnterCriticalSection(&m_csHeader);
	try
	{
		m_accHeader.m_totData += AccBuf.accHeader.m_totData;
		m_accHeader.m_totLike += AccBuf.accHeader.m_totLike;
		m_accHeader.m_cTask += AccBuf.accHeader.m_cTask;
		m_accHeader.m_totFunc += AccBuf.accHeader.m_totFunc;
		m_accHeader.m_totOcc += AccBuf.accHeader.m_totOcc;
	}
	catch (...)
	{
		TraceHR(E_UNEXPECTED, "CIVector::Accumulate: something error when acc header.\n");
	}
	LeaveCriticalSection(&m_csHeader);
	Trace(TRACE_DEBUGINFO, "Debug info: finally acc header completed.\n");

	/// GammaX * Ew;
	EnterCriticalSection(&m_csGxEw);
	try
	{
		m_accGammaXEw += AccBuf.accGammaXEw;
	}
	catch (...)
	{
		TraceHR(E_UNEXPECTED, "CIVector::Accumulate: something error when acc accGammaXEw.\n");
	}
	LeaveCriticalSection(&m_csGxEw);
	Trace(TRACE_DEBUGINFO, "Debug info: finally acc GammaX * Ew completed.\n");

	/// Gamma and GammaXX;
	if (m_updateFlag & UPDATE_VAR)
	{
		// Gamma
		EnterCriticalSection(&m_csG);
		try
		{
			CMatrix::VectorAddTo(m_accGamma, AccBuf.accGamma);
		}
		catch (...)
		{
			TraceHR(E_UNEXPECTED, "CIVector::AccumulateRecordStat: something error when acc accGamma.\n");
		}
		LeaveCriticalSection(&m_csG);
		Trace(TRACE_DEBUGINFO, "Debug info: finally acc Gamma completed.\n");

		// GammaXX
		EnterCriticalSection(&m_csGxx);
		try
		{
			CMatrix::VectorAddTo(m_accGammaXX, AccBuf.accGammaXX);
		}
		catch (...)
		{
			TraceHR(E_UNEXPECTED, "CIVector::AccumulateRecordStat: something error when acc accGammaXX.\n");
		}
		LeaveCriticalSection(&m_csGxx);
		Trace(TRACE_DEBUGINFO, "Debug info: finally acc GammaXX completed.\n");
	}
}

void CIVector::AccumulateGammaEww(SIVectorAccBuffer &AccBuf)
{

	if (m_type == IVec_UpdateParameters && (m_updateFlag & UPDATE_TRANS))
	{
		vector<ULONG> flagList(C);
		ULONG count = 0;
		ULONG mixIdx, idx;
		for (ULONG i = 0; i < C; ++i)
		{
			flagList[i] = i;
		}
		while (count < C)
		{
			idx = count;
			while (idx < C)
			{
				mixIdx = flagList[idx];
				if (tryLockList(mixIdx) != 0 )
				{
					for (int i = 0; i < AccBuf.nUsed; ++i)
					{
						if (AccBuf.Gamma[i][mixIdx] > MIN_GAMMA)
						{
							try
							{
								m_accGammaEww[mixIdx].WeightedAdd(AccBuf.Matrix[i], AccBuf.Gamma[i][mixIdx]);
							}
							catch (...)
							{
								TraceHR(E_UNEXPECTED, "CIVector::AccumulateGammaEww: something error when acc accGammaEww.\n");
							}
						}
					}
					unlockList(mixIdx);
					flagList[idx] = flagList[count];
					++count;
					++idx;
				}
				else
				{
					++idx;
				}
			}
		}
	}
	_DEBUG_TRACE((TRACE_DEBUGINFO, "Debug info: acc GammaEww completed.\n"));
}

void CIVector::WriteAccumulator(SPlatform::DoubleAccumulatorVector &acc)
{
	
	size_t size = acc.GetLength();
	size_t idx = 0;
	if (size != getAccSize())
	{
		TraceHR(E_UNEXPECTED, "CIVector::WriteAccumulator: acc size mismatch, acc.GetLength() = %d.\n", size);
	}

	Trace(TRACE_DETAILS, "start writting accumulators, size = %d.\n",size);

	/// header
	acc.Set(m_accHeader.m_cTask, idx); ++idx;
	acc.Set(m_accHeader.m_totData, idx); ++idx;
	acc.Set(m_accHeader.m_totOcc, idx); ++idx;
	acc.Set(m_accHeader.m_totLike, idx); ++idx;
	acc.Set(m_accHeader.m_totFunc, idx); ++idx;

	Trace(TRACE_DETAILS, "accHeader.m_cTask = %f\n", m_accHeader.m_cTask);
	Trace(TRACE_DETAILS, "accHeader.m_totData = %f\n", m_accHeader.m_totData);
	Trace(TRACE_DETAILS, "accHeader.m_totOcc = %f\n", m_accHeader.m_totOcc);
	Trace(TRACE_DETAILS, "accHeader.m_totLike = %f\n", m_accHeader.m_totLike);
	Trace(TRACE_DETAILS, "accHeader.m_totFunc = %f\n", m_accHeader.m_totFunc);

	if (m_fUseDoubleAcc)
	{
		/// accGammaXEw
		const double *p = (const double *)m_accGammaXEw.getRawPtr_const();
		size = m_accGammaXEw.getRawSize();
		for (size_t i = 0; i < size; ++i)
		{
			acc.Set(*p, idx);
			++p;
			++idx;
		}
		Trace(TRACE_DETAILS, "writed accGammaXEw, size = [%d, %d].\n", m_accGammaXEw.getRow(), m_accGammaXEw.getCol());
	
		/// accGammaEww
		for (size_t k = 0; k < C; ++k)
		{
			p = (const double *)m_accGammaEww[k].getRawPtr_const();
			size = m_accGammaEww[k].getRawSize();
			for (size_t i = 0; i < size; ++i)
			{
				acc.Set(*p, idx);
				++p;
				++idx;
			}
		}
		Trace(TRACE_DETAILS, "writed accGammaEww, size = [%d, %d].\n",m_accGammaEww.size(), size);
	}
	else
	{
		/// accGammaXEw
		const float *p = (const float *)m_accGammaXEw.getRawPtr_const();
		size = m_accGammaXEw.getRawSize();
		for (size_t i = 0; i < size; ++i)
		{
			acc.Set(*p, idx);
			++p;
			++idx;
		}
		Trace(TRACE_DETAILS, "writed accGammaXEw, size = [%d, %d].\n", m_accGammaXEw.getRow(), m_accGammaXEw.getCol());
	
		/// accGammaEww
		for (size_t k = 0; k < C; ++k)
		{
			p = (const float *)m_accGammaEww[k].getRawPtr_const();
			size = m_accGammaEww[k].getRawSize();
			for (size_t i = 0; i < size; ++i)
			{
				acc.Set(*p, idx);
				++p;
				++idx;
			}
		}
		Trace(TRACE_DETAILS, "writed accGammaEww, size = [%d, %d].\n",m_accGammaEww.size(), size);
	
	}
	/// accGamma
	for (size_t i = 0; i < m_accGamma.size(); ++i)
	{
		acc.Set(m_accGamma[i], idx); ++idx;
	}
	Trace(TRACE_DETAILS, "writed accGamma, size = %d.\n",m_accGamma.size());

	/// accGammaXX
	for (size_t i = 0; i < m_accGammaXX.size(); ++i)
	{
		acc.Set(m_accGammaXX[i], idx); ++idx;
	}
	Trace(TRACE_DETAILS, "writed accGammaXX, size = %d.\n",m_accGammaXX.size());

	Trace(TRACE_DETAILS, "finished writting accumulators, size = %d.\n", idx);
}

void CIVector::Reduce(SPlatform::DoubleAccumulatorVector &acc)
{
	Trace(TRACE_DETAILS, "IVector Task: reduce statistics.\n");
	WriteAccumulator(acc);
	acc.Reduce();
}

void CIVector::ReadAccumulator(SPlatform::DoubleAccumulatorVector &acc)
{
	size_t size = acc.GetLength();
	size_t idx = 0;
	if (size != getAccSize())
	{
		TraceHR(E_UNEXPECTED, "CIVector::ReadAccumulator: acc size mismatch, acc.GetLength() = %d.\n", size);
	}
	Trace(TRACE_DETAILS, "start reading accumulators, size = %d.\n",size);

	/// header
	m_accHeader.m_cTask = (float)acc.Get(idx); ++idx;
	m_accHeader.m_totData = (float)acc.Get(idx); ++idx;
	m_accHeader.m_totOcc = (float)acc.Get(idx); ++idx;
	m_accHeader.m_totLike = (float)acc.Get(idx); ++idx;
	m_accHeader.m_totFunc = (float)acc.Get(idx); ++idx;

	Trace(TRACE_DETAILS, "accHeader.m_cTask = %f\n", m_accHeader.m_cTask);
	Trace(TRACE_DETAILS, "accHeader.m_totData = %f\n", m_accHeader.m_totData);
	Trace(TRACE_DETAILS, "accHeader.m_totOcc = %f\n", m_accHeader.m_totOcc);
	Trace(TRACE_DETAILS, "accHeader.m_totLike = %f\n", m_accHeader.m_totLike);
	Trace(TRACE_DETAILS, "accHeader.m_totFunc = %f\n", m_accHeader.m_totFunc);

	/// accGammaXEw
	if (m_fUseDoubleAcc)
	{
		double *p = (double *)m_accGammaXEw.getRawPtr();
		size = m_accGammaXEw.getRawSize();
		for (size_t i = 0; i < size; ++i)
		{
			*p = acc.Get(idx);
			++p;
			++idx;
		}
		Trace(TRACE_DETAILS, "read accGammaXEw, size = [%d, %d].\n", m_accGammaXEw.getRow(), m_accGammaXEw.getCol());
	
		/// accGammaEww
		for (size_t k = 0; k < C; ++k)
		{
			p = (double *)m_accGammaEww[k].getRawPtr();
			size = m_accGammaEww[k].getRawSize();
			for (size_t i = 0; i < size; ++i)
			{
				*p = acc.Get(idx);
				++p;
				++idx;
			}
		}
		Trace(TRACE_DETAILS, "read accGammaEww, size = [%d, %d].\n",m_accGammaEww.size(), size);
	}
	else
	{
		float *p = (float *)m_accGammaXEw.getRawPtr();
		size = m_accGammaXEw.getRawSize();
		for (size_t i = 0; i < size; ++i)
		{
			*p = (float)acc.Get(idx);
			++p;
			++idx;
		}
		Trace(TRACE_DETAILS, "read accGammaXEw, size = [%d, %d].\n", m_accGammaXEw.getRow(), m_accGammaXEw.getCol());
	
		/// accGammaEww
		for (size_t k = 0; k < C; ++k)
		{
			p = (float *)m_accGammaEww[k].getRawPtr();
			size = m_accGammaEww[k].getRawSize();
			for (size_t i = 0; i < size; ++i)
			{
				*p = (float)acc.Get(idx);
				++p;
				++idx;
			}
		}
		Trace(TRACE_DETAILS, "read accGammaEww, size = [%d, %d].\n",m_accGammaEww.size(), size);

	}
	/// accGamma
	for (size_t i = 0; i < m_accGamma.size(); ++i)
	{
		m_accGamma[i] = acc.Get(idx); ++idx;
		if (m_accGamma[i] < 0)
		{
			TraceHR(E_UNEXPECTED, "CIVector::ReadAccumulator: accGamma[%d] < 0.\n", i);
		}
	}
	Trace(TRACE_DETAILS, "read accGamma, size = %d.\n", m_accGamma.size());

	/// accGammaXX
	for (size_t i = 0; i < m_accGammaXX.size(); ++i)
	{
		m_accGammaXX[i] = acc.Get(idx); ++idx;
		if (m_accGammaXX[i] < 0)
		{
			TraceHR(E_UNEXPECTED, "CIVector::ReadAccumulator: accGammaXX[%d] < 0.\n", i);
		}
	}
	Trace(TRACE_DETAILS, "read accGammaXX, size = %d.\n", m_accGammaXX.size());
	Trace(TRACE_DETAILS, "finished reading accumulators, size = %d.\n", idx);
}

void CIVector::Update(SPlatform::DoubleAccumulatorVector &acc)
{
	if(platform.GetRank() == 0)
	{
		/// main node, update T (and iVar)
		Trace(TRACE_LOG, "IVector Task: updating parameters.\n");
		ReadAccumulator(acc);

		Trace(TRACE_LOG, "IVector Task: total record number: %d.\n", (ULONG)m_accHeader.m_totData);
		Trace(TRACE_LOG, "IVector Task: likelihood: %e.\n", m_accHeader.m_totLike);
		if (!(m_updateFlag & UPDATE_VAR))
		{
			Trace(TRACE_LOG, "IVector Task: likelihood (constants removed): %e.\n", m_accHeader.m_totFunc - m_accHeader.m_totOcc);
		}
		Trace(TRACE_DETAILS, "IVector Task: likelihood detail: Ew*Ts*GammX=%e, log|L|=%e, gConst=%e.\n", m_accHeader.m_totFunc, m_accHeader.m_totOcc, m_accHeader.m_totLike + m_accHeader.m_totOcc - m_accHeader.m_totFunc);
		
		// update T matrix
		if (m_updateFlag & UPDATE_TRANS)
		{
			EstimateTMatrix();
		}

		// update iVar
		if (m_updateFlag & UPDATE_VAR)
		{
			EstimateVar();
		}

		if (m_fUseDoubleAcc)
		{
			iVar.ConvertTo(Float);
			Tt.ConvertTo(Float);
		}

	}
}

void CIVector::EstimateTMatrix()
{
	HRESULT hr = S_OK;
	size_t idx = 0;
	string strFormat = (m_fUseDoubleAcc) ? "double" : "float";
	CMatrix tempGammaEww;
	for (size_t mixIdx = 0; mixIdx < C; ++mixIdx)
	{
		// cholesky decompose with A matrix (A = accGammaEww[mixIdx])
		hr = m_accGammaEww[mixIdx].cholesky(tempGammaEww);
		if (FAILED(hr))
		{
			Trace(TRACE_ERRORS, "Warning: CIVector::EstimateTMatrix: fail to decompose matrix when mixIdx=%d.\n", mixIdx);
			if (m_accGammaEww[mixIdx].isZeroMatrix())
			{
				Trace(TRACE_ERRORS, "Warning: CIVector::EstimateTMatrix: matrix is zero.\n");
			}
			else
			{
				TraceHR(E_UNEXPECTED, "CIVector::EstimateTMatrix: unknown reason.\n");
			}
		}
		else
		{
			// Linear solve A*T = b
			idx = mixIdx * K;
			if (m_fUseDoubleAcc)
			{
				vector<double> b, x;
				for (size_t j = 0; j < K; ++j)
				{
					m_accGammaXEw.getRow(idx, b);
					// chol method
					CMatrix::cholSub(tempGammaEww, b, x);
					// set new T
					Tt.setCol(idx, x);
					++idx;
				}
			}
			else
			{
				vector<float> b, x;
				for (size_t j = 0; j < K; ++j)
				{
					m_accGammaXEw.getRow(idx, b);
					// chol method
					CMatrix::cholSub(tempGammaEww, b, x);
					// set new T
					Tt.setCol(idx, x);
					++idx;
				}
			}
		}
	}
}

void CIVector::EstimateVar()
{
	size_t idx = 0;
	if (m_fUseDoubleAcc)
	{
		for (size_t mixIdx = 0; mixIdx < C; ++mixIdx)
		{
			idx = mixIdx * K;
			for (size_t k = 0; k < K; ++k)
			{
				double Mck = 0;
				for (size_t r = 0; r < R; ++r)
				{
					Mck += m_accGammaXEw.doubleElement(idx + k, r) * Tt.doubleElement(r, idx + k);
				}
				double sigma = m_accGammaXX[idx + k] - Mck;
				if (sigma <= 0)
				{
					Trace(TRACE_ERRORS, "Warning: CIVector::EstimateVar: Sigma=%e <= 0, accGammaXX=%e, accGamma=%e, skip update variance of [mixture %d dim %d].\n", sigma, m_accGammaXX[idx + k], m_accGamma[mixIdx], mixIdx, k);
				}
				else
				{
					iVar.doubleElement(mixIdx, k) = min(m_accGamma[mixIdx]/sigma, iVarFloor[k]);
				}
			}
		}
	}
	else
	{
		for (size_t mixIdx = 0; mixIdx < C; ++mixIdx)
		{
			idx = mixIdx * K;
			for (size_t k = 0; k < K; ++k)
			{
				float Mck = 0;
				for (size_t r = 0; r < R; ++r)
				{
					Mck += m_accGammaXEw.floatElement(idx + k, r) * Tt.floatElement(r, idx + k);
				}
				double sigma = m_accGammaXX[idx + k] - Mck;
				if (sigma <= 0)
				{
					Trace(TRACE_ERRORS, "Warning: CIVector::EstimateVar: Sigma=%e <= 0, accGammaXX=%e, accGamma=%e, skip update variance of [mixture %d dim %d].\n", sigma, m_accGammaXX[idx + k], m_accGamma[mixIdx], mixIdx, k);
				}
				else
				{
					iVar.floatElement(mixIdx, k) = (float)min(m_accGamma[mixIdx]/sigma, iVarFloor[k]);
				}
			}
		}
	}
}


ULONG CIVector::getAccSize()
{
	size_t accSize = 0;
	size_t size = 0;

	// header size
	if (sizeof(SIVectorAccHeader) % sizeof(float) != 0)
	{
		TraceHR(E_UNEXPECTED, "CIVector::getAccSize: unexpect accHeader size");
	}
	accSize += sizeof(SIVectorAccHeader) / sizeof(float);

	// accGammaXEw size
	size = m_accGammaXEw.getRawSize();
	if (size != C * K * R)
	{
		TraceHR(E_UNEXPECTED, "CIVector::getAccSize: unexpect accGammaXEw size");
	}
	accSize += size;

	// accGammaEww size
	if (m_accGammaEww.size() != 0)
	{
		size = m_accGammaEww.size() * m_accGammaEww[0].getRawSize();
	}
	else
	{
		size = 0;
	}
	if (size != C * R * (R + 1) / 2)
	{
		TraceHR(E_UNEXPECTED, "CIVector::getAccSize: unexpect accGammaEww size");
	}
	accSize += size;

	// accGamma size
	size = m_accGamma.size();
	if (size != C)
	{
		TraceHR(E_UNEXPECTED, "CIVector::getAccSize: unexpect accGamma size");
	}
	accSize += size;

	// accGammaXX size
	size = m_accGammaXX.size();
	if (size != C * K)
	{
		TraceHR(E_UNEXPECTED, "CIVector::getAccSize: unexpect accGammaXX size");
	}
	accSize += size;

	return (ULONG)accSize;
}

void CIVector::SaveUpdatedResults(const char * pszSpecialExtensionName)
{
	if(platform.GetRank() == 0)
	{
		if (m_updateFlag & UPDATE_TRANS)
		{
			string szTransFileName = ConcatenateFileFullPath(NULL, m_szOutTransFile.c_str(), pszSpecialExtensionName);
			CMatrix T;
			Tt.transpose(T);
			const char * mode = (m_fBinaryFormat) ? "wb" : "wt";
			T.save(szTransFileName.c_str(), mode);
			Trace(TRACE_PROGRESS, "Saved T matrix in %s\n", szTransFileName.c_str());
		}
		if (m_updateFlag & UPDATE_VAR)
		{
			string szModelFileName = ConcatenateFileFullPath(NULL, m_szOutModelFile.c_str(), pszSpecialExtensionName);
			SaveModel(szModelFileName.c_str());
			Trace(TRACE_PROGRESS, "Saved model in %s\n", szModelFileName.c_str());
		}
	}
}

bool CIVector::IterationComplete(bool fSkip, const char * pszSpecialExtensionName)
{
	int flag = 0; // flag == 1 => complete
	int recv = 0;
	if(platform.GetRank() == 0)
	{
		const char *mode = (m_fBinaryFormat) ? "rb" : "rt";
		if (fSkip == false)
		{
			flag = 0;
		}
		else
		{
			FILE * f = NULL;
			if (m_updateFlag & UPDATE_TRANS)
			{
				string szTransFileName = ConcatenateFileFullPath(NULL, m_szOutTransFile.c_str(), pszSpecialExtensionName);
				if (fopen_s(&f, szTransFileName.c_str(), mode) == 0)
				{
					fclose(f);
					CMatrix bufT;
					bufT.load(szTransFileName.c_str(), mode);
					if (bufT.getRow() != C * K || bufT.getCol() != R)
					{
						flag = 0;
					}
					else
					{
						if (m_updateFlag & UPDATE_VAR)
						{
							string szModelFileName = ConcatenateFileFullPath(NULL, m_szOutModelFile.c_str(), pszSpecialExtensionName);
							if (fopen_s(&f, szModelFileName.c_str(), "r") == 0)
							{
								fclose(f);
								LoadModel(szModelFileName.c_str());
								flag = 1;
							}
							else
							{
								flag = 0;
							}
						}
						else
						{		
							flag = 1;
						}
					}
					if (flag == 1)
					{
						bufT.transpose(Tt);
					}
				}
			}
		}
		// broadcast flag
		platform.BroadcastSendMem(&flag, sizeof(int));
	}
	else
	{
		/// other nodes receive results
		platform.BroadcastRecvMem(&flag, sizeof(int), &recv);
	}
	return (flag == 0) ? false : true;
}

void CIVector::LockList(ULONG idx)
{
	EnterCriticalSection(&m_csList[CSIdx(idx)]);
}

void CIVector::unlockList(ULONG idx)
{
	LeaveCriticalSection(&m_csList[CSIdx(idx)]);
}

int CIVector::tryLockList(ULONG idx)
{
	return TryEnterCriticalSection(&m_csList[CSIdx(idx)]);
}

void CIVector::InitializeBufferInThread(SIVectorAccBuffer &AccBuf)
{
	AccBuf.nUsed = 0;
	AccBuf.nMaxSize = m_numCacheLine;
	AccBuf.Matrix.resize(m_numCacheLine);
	AccBuf.Gamma.resize(m_numCacheLine);
	AccBuf.GammaX.resize(m_numCacheLine);
	AccBuf.GammaXX.resize(m_numCacheLine);
	AccBuf.filename.resize(m_numCacheLine);
	AccBuf.remoteWriter.resize(m_numCacheLine);

	string strFormat = (m_fUseDoubleAcc) ? "double" : "float";

	for (int i = 0; i < m_numCacheLine; ++i)
	{
		AccBuf.Matrix[i].assign(R, R, 0.0, strFormat.c_str(), "symmetric");
		AccBuf.Gamma[i].resize(C);
		AccBuf.GammaX[i].resize(C*K);
		AccBuf.GammaXX[i].resize(C*K);
	}

	if (m_type == IVec_UpdateParameters)
	{
		SIVectorAccHeader acc = {0, 0, 0, 0, 0};
		AccBuf.accHeader = acc;
		AccBuf.accGammaXEw.assign(C*K, R, 0.0, strFormat.c_str());
		AccBuf.accGamma.assign(C, 0.0);
		AccBuf.accGammaXX.assign(C*K, 0.0);
	}

	/// remote writer
	HTKhdr hdr = {0, 10000 , 0, 9};
	for (int i = 0; i < m_numCacheLine; ++i)
	{
		AccBuf.remoteWriter[i].hdr = hdr;
		AccBuf.remoteWriter[i].buf = NULL;
		AccBuf.remoteWriter[i].nOffset = -1;
	}

#ifdef OUTPUT_IGHF
    if (m_fOutFrameLabelofGmm)
        AccBuf.feat.resize(m_numCacheLine);
#endif
}

bool CIVector::LoadRecordStatIntoBuffer(SPFile &file, SIVectorRemoteWriterStat &stat, SIVectorAccBuffer &AccBuf)
{
	HRESULT hr = S_OK;

	while (AccBuf.nUsed < m_numCacheLine)
	{
		// buffer is not full
		while(stat.nOffset < stat.hdr.nSamples && AccBuf.nUsed < m_numCacheLine)
		{
			// load one file form platform
            hr = LoadOneRecordStat(&file, AccBuf.Gamma[AccBuf.nUsed], AccBuf.GammaX[AccBuf.nUsed], AccBuf.GammaXX[AccBuf.nUsed], AccBuf.feat[AccBuf.nUsed]);
			TraceHR(hr, "CIVector::LoadRecordStatIntoBuffer: fail to load record stat, file %s", file.GetFilePath().c_str());
			
			AccBuf.filename[AccBuf.nUsed] = ReplaceSubstr(file.GetFilePath(), "/", "\\");
			AccBuf.remoteWriter[AccBuf.nUsed] = stat;
			
			++stat.nOffset;
			++AccBuf.nUsed;
		}
		if (stat.nOffset >= stat.hdr.nSamples)
		{
			// buffer is not full, but no data in current file. So get next file from platform
			if (platform.GetNext(file))
			{
				ULONG nRecord = 0;
				hr = ReadHeaderOfRecord(&file, nRecord);
				TraceHR(hr, "CIVector::AccumulateRecordStat: unexpected file header in file %s, nRecord=%d", file.GetFilePath().c_str(), nRecord);
				stat.hdr.nSamples = (int)nRecord;
				stat.nOffset = 0;
				// remote writer
				if (m_type == IVec_EstimateIVector)
				{
                    int ndim = (m_fSaveiStatOnly) ? C : R;
					int len = sizeof(HTKhdr) + sizeof(float) * nRecord * ndim;
					stat.buf = new BYTE[len];
					stat.hdr.sampSize = (short) ndim * sizeof(float);

                    // swap
                    HTKhdr header = stat.hdr;
                    Swap32(&header.nSamples);
		            Swap32(&header.sampPeriod);
		            Swap16(&header.sampSize);
		            Swap16(&header.sampKind);

					CopyMemory(stat.buf, &header, sizeof(HTKhdr));
				}
			}
			else
			{
				// buffer is empty and no more files, then can be finish return false
				return (AccBuf.nUsed == 0) ? false : true;
			}
		}
	}
	return true;
}

// calculate L and accumulate Gamma and GammaXX in local for all records in buffer
void CIVector::AccumulateRecordStatBuffer(SIVectorAccBuffer &AccBuf)
{
	_DEBUG_TRACE((TRACE_DEBUGINFO, "Debug info: Accumulate record stat in buffer, bufUsed=%d.\n", AccBuf.nUsed));
	/// calculate L
	for (int i = 0; i < AccBuf.nUsed; ++i)
	{		
		AccBuf.Matrix[i].assign_zero(R);	
	}
	for (size_t mixIdx = 0; mixIdx < C; ++mixIdx)
	{
		for (int i = 0; i < AccBuf.nUsed; ++i)
		{
			if (AccBuf.Gamma[i][mixIdx] > MIN_GAMMA)
			{
				AccBuf.Matrix[i].WeightedAdd(TsT[mixIdx], AccBuf.Gamma[i][mixIdx]); // L - I
			}
		}
	}
	_DEBUG_TRACE((TRACE_DEBUGINFO, "Debug info: L.WeightedAdd() completed.\n"));

	/// Accumulate statistic into local acc buffer
	if (m_type == IVec_UpdateParameters && (m_updateFlag & UPDATE_VAR))
	{
		for (int i = 0; i < AccBuf.nUsed; ++i)
		{
			// Gamma
			CMatrix::VectorAddTo(AccBuf.accGamma, AccBuf.Gamma[i]);
			_DEBUG_TRACE((TRACE_DEBUGINFO, "Debug info: acc Gamma completed.\n"));
			// GammaXX
			CMatrix::VectorAddTo(AccBuf.accGammaXX, AccBuf.GammaXX[i]);
			_DEBUG_TRACE((TRACE_DEBUGINFO, "Debug info: acc GammaXX completed.\n"));
		}
	}
}

bool CIVector::LoadModel(const char * fname)
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
				TraceHR(E_UNEXPECTED, "CIVector::LoadModel: fail loading model: iVar[%d][%d] > iVarFloor[%d]", i, j, i);
			}
		}
		iVar.setRow(i, vec);
	}
	return true;
}

bool CIVector::SaveModel(const char * fname)
{
	/// 1/var
	vector<float> vec;
	for (ULONG i = 0; i < C; ++i)
	{
		iVar.getRow(i, vec);
		for (ULONG j = 0; j < K; ++j)
		{
			vec[j] = 1 / vec[j];
		}
		m_GMM.SetMixtureVar(vec, i);
	}
	m_GMM.SaveGMMFile(fname);
	return true;
}

bool CIVector::InitTMatrix(const char *fname, const float TScaleMin, const float TScaleMax)
{
	if (fname == NULL)
	{
		if (R == 0 || C == 0 || K == 0)
		{
			TraceHR(E_UNEXPECTED, "CIVector::InitTMatrix: fail initialize T matrix: R = %d, C = %d, K = %d", R, C, K);
		}
		// initialize Tt matrix
		Tt.assign(Normal, Float);
		if (TScaleMin < TScaleMax)
		{
			// initialize T according GMM variance and TScaleMin and TScaleMax
			vector<float> vec;
			vector<float> vecRow(R);
			float minVal, offset, randval;
			Tt.assign_zero(C*K, R);
			for (ULONG i = 0; i < C; ++i)
			{
				m_GMM.GetMixtureVar(vec, i);
				for (ULONG j = 0; j < K; ++j)
				{
					vec[j] = sqrt(vec[j]);
					minVal = vec[j] * TScaleMin;
					offset = vec[j] * TScaleMax - minVal;
					for (ULONG k = 0; k < R; ++k)
					{
						 randval = (float)(rand() % 101) / 100;
						 vecRow[k] = minVal + offset * randval;
					}
					Tt.setRow(i*K+j, vecRow);
				}
			}
			// store transpose of T matrix
			Tt.transpose();
		}
		else
		{
			TraceHR(E_UNEXPECTED, "CIVector::InitTMatrix: fail initialize T matrix: tscale min = %e, max = %e \n", TScaleMin, TScaleMax);
		}
	}
	else
	{
		// load T from file
		Tt.load(fname);
		if (Tt.getRow() == 0 || Tt.getCol() == 0)
		{
			TraceHR(E_UNEXPECTED, "CIVector::InitTMatrix: fail loading T matrix: row = %d, col = %d", Tt.getRow(), Tt.getCol());
		}
		if (Tt.getRow() != C*K || Tt.getCol() != R)
		{
			TraceHR(E_UNEXPECTED, "CIVector::InitTMatrix: unexpect T matrix");
		}

		// store transpose of T matrix
		Tt.transpose();
	}
	return true;
}