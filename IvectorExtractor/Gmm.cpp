#include "Gmm.h"




// member functions of CGaussianComponent class

CGaussianComponent::CGaussianComponent()
{
	cs = new CRITICAL_SECTION;
	InitializeCriticalSection(cs);
}

CGaussianComponent::CGaussianComponent(const CGaussianComponent &gc_in)
{
	*this = gc_in;

	// be careful to the CS opeartions
	cs = new CRITICAL_SECTION;
	InitializeCriticalSection(cs);
}

CGaussianComponent::~CGaussianComponent()
{
	DeleteCriticalSection(cs);
	delete cs;
}

void CGaussianComponent::FixGConst()
{
	/*double detSigma = 1.0;
	for(int i=0; i<(int)var.size(); i++)
	{
	detSigma *= var[i];
	}
	gconst = (float)(-0.5 * (var.size() * log(2 * PI) + log(detSigma)));*/

	float z;
	float sum = float(var.size()*log(TPI));
	for(int i=0; i<(int)var.size(); i++)
	{
		z = float(var[i]<=MINLARG ? LZERO : log(var[i]));
		sum += z;
	}

	gConst = sum;
}

void CGaussianComponent::ApplyVarFloor(const vector<float> &varFloor)
{
	int nFloored = 0;

	for(int i=0; i<(int)var.size(); i++)
	{
		if(var[i] < varFloor[i])
		{
			var[i] = varFloor[i];
			nFloored++;
		}
	}

	if(nFloored != 0)
	{
		printf("ApplyVarFloor: %d out of %d var floored\n", nFloored, var.size());
	}
}

float CGaussianComponent::MOutP(const vector<float> &x)
{
#ifdef SSE_OPTIMIZED
	int i;
	int dim = x.size();
	const float *p1, *p2, *p3;
	__m128 m1, m2, s1;

	s1 = _mm_setzero_ps();
	for(i=0, p1=&mean[0], p2=&x[0], p3=&var[0]; i+4<=dim; i+=4)
	{
		m1 = _mm_loadu_ps(p1+i);
		m2 = _mm_loadu_ps(p2+i);
		m1 = _mm_sub_ps(m1, m2);
		m1 = _mm_mul_ps(m1, m1);
		m2 = _mm_loadu_ps(p3+i);
		m1 = _mm_div_ps(m1, m2);
		s1 = _mm_add_ps(s1, m1);
	}
	for(; i<dim; i++)
	{
		m1 = _mm_load_ss(p1+i);
		m2 = _mm_load_ss(p2+i);
		m1 = _mm_sub_ss(m1, m2);
		m1 = _mm_mul_ss(m1, m1);
		m2 = _mm_load_ss(p3+i);
		m1 = _mm_div_ss(m1, m2);
		s1 = _mm_add_ss(s1, m1);
	}

	// add together
	s1 = _mm_hadd_ps(s1, s1);
	s1 = _mm_hadd_ps(s1, s1);

	return -0.5f*(s1.m128_f32[0]+gConst);

#else
	float sum, xmm;

	sum = gConst;
	for(int i=0; i<(int)x.size(); i++)
	{
		xmm = x[i]-mean[i];
		sum += xmm*xmm/var[i];
	}
	return -0.5f*sum;
#endif
}


// member functions of CGaussianMixture class
CGaussianMixture::CGaussianMixture(bool naturalReadOrder_in) : naturalReadOrder(naturalReadOrder_in)
{
	featType = "USER";
	hmmName = "UBM";
}

void CGaussianMixture::LoadGMMFile(const char *gmmFn)
{
	char buff[LINEMAX], mask[LINEMAX];

	// only the main rank should load the files
	

	// CAUTION: a very naive reading method of HTK format
	FILE *fp;
	if(fopen_s(&fp, gmmFn, "r"))
	{
		printf("LoadGMMFile: Cannot open %s", gmmFn);
	}

	// ~o and dimension
	if(fscanf_s(fp, "~o\n<STREAMINFO> 1 %d\n", &dim) != 1)
	{
		printf("LoadGMMFile: Error loading <STREAMINFO>");
		//platform.Abort();
	}

	//  feature type
	sprintf_s(mask, "<VECSIZE> %d<NULLD><%%s\n", dim);
	if(fscanf_s(fp, mask, buff, LINEMAX) != 1)
	{
		printf("LoadGMMFile: Error loading feature type");
		//platform.Abort();
	}
	if(strcmp(strrchr(buff, '<'), "<DIAGC>") != 0)
	{
		printf("LoadGMMFile: Only support DIAGC");
		//platform.Abort();
	}
	*strchr(buff, '>') = '\0';
	featType = buff;

	// macros
	while (fscanf_s(fp, "~%c ", buff, LINEMAX) == 1)
	{
		switch(buff[0])
		{
		case 'v':
			// skip "varFloor1 and <VARIANCE>"
			fgets(buff, LINEMAX, fp);
			fgets(buff, LINEMAX, fp);
			
			// read varFloor
			varFloor.resize(dim);
			for(int i=0; i<dim; i++)
			{
				if(fscanf_s(fp, " %f", &varFloor[0]+i) != 1)
				{
					printf("LoadGMMFile: Error reading varFloor, dim %d", i);
					//platform.Abort();
				}
			}

			// skip \n
			fgets(buff, LINEMAX, fp);
			break;
		case 'h':
			// hmm name
			fgets(buff, LINEMAX, fp);
			*strrchr(buff, '\"') = '\0';
			hmmName = strchr(buff, '\"')+1;

			// <BEGINHMM> and <NUMSTATES>
			int nStates;
			if (fscanf_s(fp, "<BEGINHMM>\n<NUMSTATES> %d\n", &nStates, LINEMAX) != 1)
			{
				printf("LoadGMMFile: Error reading <BEGINHMM> or <NUMSTATES>");
				//platform.Abort();
			}

			// check
			if(nStates != 3)
			{
				printf("LoadGMMFile: Only single-state GMM is supported");
				//platform.Abort();
			}

			// skip <STATE> 2
			fgets(buff, LINEMAX, fp);

			// try to access <NUMMIXES>
			if (fscanf_s(fp, "<NUMMIXES> %d\n", &nMix, LINEMAX) == 1)
			{
				// multi-mixture case
				mixture.resize(nMix);
				wt.resize(nMix);
			}
			else
			{
				// single-mixture case
				nMix = 1;
				mixture.resize(1);
				wt.assign(1, 1.0f);
			}

			// mixture components
			for(int i=0; i<nMix; i++)
			{
				if(nMix > 1)
				{
					sprintf_s(mask, "<MIXTURE> %d %%f\n", i+1);
					if (fscanf_s(fp, mask, &wt[0] + i, LINEMAX) != 1)
					{
						printf("LoadGMMFile: Error reading weight for mixture [%d]", i+1);
						//platform.Abort();
					}
				}

				// skip <MEAN>
				fgets(buff, LINEMAX, fp);
				mixture[i].mean.resize(dim);
				for(int j=0; j<dim; j++)
				{
					if (fscanf_s(fp, " %f", &mixture[i].mean[0] + j, LINEMAX) != 1)
					{
						printf("LoadGMMFile: Error reading mean, dim %d", i);
						//platform.Abort();
					}
				}
				fgets(buff, LINEMAX, fp);

				// skip <VARIANCE>
				fgets(buff, LINEMAX, fp);
				mixture[i].var.resize(dim);
				for(int j=0; j<dim; j++)
				{
					if (fscanf_s(fp, " %f", &mixture[i].var[0] + j, LINEMAX) != 1)
					{
						printf("LoadGMMFile: Error reading mean, dim %d", i);
						//platform.Abort();
					}
				}
				fgets(buff, LINEMAX, fp);

				// possible GConst
				fscanf_s(fp, "<GCONST> %f\n", &mixture[i].gConst, LINEMAX);
			}

			// remaining parts
			while(fgets(buff, LINEMAX, fp))
			{
				tailTransP.push_back(buff);
			}

			break;
		default:
			printf("LoadGMMFile: Unknow macro %c", buff[0]);
			//platform.Abort();
		}
	}

	fclose(fp);

	// finalize
	for(int i=0; i<nMix; i++)
	{
		mixture[i].ApplyVarFloor(varFloor);
		mixture[i].FixGConst();
	}
}

void CGaussianMixture::SaveGMMFile(const char *gmmFn)
{
	// only the main rank should save the file


	FILE *fp;
	if(fopen_s(&fp, gmmFn, "w"))
	{
		printf("LoadGMMFile: Cannot create %s", gmmFn);
	}

	// header
	fprintf(fp, "~o\n<STREAMINFO> 1 %d\n<VECSIZE> %d<NULLD><%s><DIAGC>\n", dim, dim, featType.c_str());

	// varFloor
	fprintf(fp, "~v \"varFloor1\"\n<VARIANCE> %d\n", dim);
	for(int i=0; i<dim; i++)
	{
		fprintf(fp, " %e", varFloor[i]);
	}
	fprintf(fp, "\n");

	// ~h + others
	fprintf(fp, "~h \"%s\"\n<BEGINHMM>\n<NUMSTATES> 3\n<STATE> 2\n", hmmName.c_str());

	// <NUMMIXES>
	if(nMix > 1)
	{
		fprintf(fp, "<NUMMIXES> %d\n", nMix);
	}

	for(int i=0; i<nMix; i++)
	{
		// weight
		if(nMix > 1)
		{
			fprintf(fp, "<MIXTURE> %d %e\n", i+1, wt[i]);
		}

		// mean
		fprintf(fp, "<MEAN> %d\n", dim);
		for(int j=0; j<dim; j++)
		{
			fprintf(fp, " %e", mixture[i].mean[j]);
		}
		fprintf(fp, "\n");

		// var
		fprintf(fp, "<VARIANCE> %d\n", dim);
		for(int j=0; j<dim; j++)
		{
			fprintf(fp, " %e", mixture[i].var[j]);
		}
		fprintf(fp, "\n");

		// gconst
		fprintf(fp, "<GCONST> %e\n", mixture[i].gConst);
	}

	if(tailTransP.empty())
	{
		fprintf(fp, "<TRANSP> 3\n 0.000000e+000 1.000000e+000 0.000000e+000\n 0.000000e+000 9.000000e-001 1.000000e-001\n 0.000000e+000 0.000000e+000 0.000000e+000\n<ENDHMM>\n");
	}
	else
	{
		for(int i=0; i<(int)tailTransP.size(); i++)
		{
			fputs(tailTransP[i].c_str(), fp);
		}
	}

	fclose(fp);
}

void CGaussianMixture::LoadHldaTransFile(const char *fnHldaTrans)
{

	transMat.load(fnHldaTrans, "rtHLDA");

	if((int)transMat.nRow < dim)
	{
		printf("LoadHldaTransFile: Dimension error [%d < %d]\n", transMat.nRow, dim);
		return;
	}
	else if(transMat.nRow != dim)
	{
		printf("LoadHldaTransFile: Dimension reduced from %d to %d\n", transMat.nRow, dim);
		transMat.resizeRow(dim);
	}
}

void CGaussianMixture::Initialize()
{
	//int recv;
	//BYTE *buf;

	// clear statistics
	for(int i=0; i<nMix; i++)
	{
		mixture[i].gamma = 0;
		mixture[i].gammaX.assign(dim, 0);
		mixture[i].gammaXX.assign(dim, 0);
	}

	// set log-likelihood
	lr = 0;
}
void CGaussianMixture::AccumulateStatisticFile(char* fname, vector<float> &Gamma, vector<float>&GammaX)
{
	int featDim, mixNum;
	vector<float> tempVec;
	vector<INT32> tempVal;
	tempVal.resize(1);
	mixNum = Gamma.size(); // number of mixtures: 3501
	featDim = GammaX.size() / mixNum; // feature dimension for each frame: 60
	tempVec.resize(featDim);

	FILE *statFile;// statstics of zero order and centralized first order
	if (fopen_s(&statFile, fname, "rb"))
	{
		printf("StatExt: can not open file %s\n", fname);
		return;
	}
	if (fread_s(&tempVal[0], sizeof(INT32), sizeof(INT32), 1, statFile) != 1) // 1 sample
	{
		printf("read statistic file error: %s", fname);
		return;
	}
	if (fread_s(&tempVal[0], sizeof(INT32), sizeof(INT32), 1, statFile) != 1) // number of total parameters 
	{
		printf("read statistic file error: %s", fname);
		return;
	}
	if (tempVal[0] != mixNum*(2 * featDim + 1)) // zero, centralized first order and second order
	{
		printf("dimension mismatch! number of parameters is %d, while model mixture is %d and featdim is %d", tempVal[0], mixNum, featDim);
		return;
	}
	for (int i = 0; i < mixNum; i++)
	{
		if (fread_s(&tempVec[0], sizeof(float), sizeof(float), 1, statFile) != 1) // number of total parameters 
		{
			printf("read statistic file error: %s", fname);
			return;
		}
		Gamma[i] = tempVec[0];
		if (fread_s(&tempVec[0], sizeof(float)*featDim, sizeof(float), featDim, statFile) != featDim) // number of total parameters 
		{
			printf("read statistic file error: %s", fname);
			return;
		}
		for (int j = 0; j < featDim; j++)
		{
			GammaX[i*featDim + j] = tempVec[j];
		}
		if (fread_s(&tempVec[0], sizeof(float)*featDim, sizeof(float), featDim, statFile) != featDim) // number of total parameters 
		{
			printf("read statistic file error: %s", fname);
			return;
		}
		// only the zero and first order statistics are used, so the second statistic is not stored 
	}
	fclose(statFile);
}

void CGaussianMixture::AccumulateHTKFeatureFile(char* fname, vector<float> &Gamma, vector<float> &GammaX)
{
	vector<float> data, feat, pp;
	double sum;
	double p;
	HTKhdr header;
	int featDim;
	//CMatrix dataMat;

	// resize posterior probabilty vector
	pp.resize(nMix);

	FILE * featFile;
	if (fopen_s(&featFile, fname, "rb"))
	{
		printf("StatExt: can not open file %s\n", fname);
		return;
	}
	fread_s(&header, sizeof(HTKhdr), sizeof(HTKhdr), 1, featFile);
	if (!naturalReadOrder)
	{
		Swap32(&header.nSamples);
		Swap32(&header.sampPeriod);
		Swap16(&header.sampSize);
		Swap16(&header.sampKind);
	}

	featDim = header.sampSize / sizeof(float);

	// dimension check
	if (transMat.empty())
	{
		if (dim != featDim)
		{
			printf("AccumulateHTKFeatureFile: Dimension mismatch [%d vs %d]\n", dim, featDim);
			return;
		}
		feat.resize(dim);
	}
	else
	{
		if (transMat.nCol != featDim)
		{
			printf("AccumulateHTKFeatureFile: Dimension mismatch [%d vs %d]\n", transMat.nCol, featDim);
			return;
		}
		feat.resize(featDim);
	}

	// read frame-by-frame
	//printf("%d\n",header.nSamples);
	for (int i = 0; i<header.nSamples; i++)
	{
		if (fread_s(&feat[0], header.sampSize, sizeof(float), featDim, featFile) != featDim)
		{
			printf("AccumulateHTKFeatureFile: Error loading feature frame\n");
			return;
		}

		if (!naturalReadOrder)
		{
			int *tmp = (int *)(&feat[0]);
			for (int j = 0; j<featDim; j++)
			{
				Swap32(tmp + j);
			}
		}

		if (!transMat.empty())
		{
			// apply linear transformation
			data = transMat * feat;
		}
		else
		{
			data = feat;
		}

		// calculate MOutP
		sum = LZERO;
		for (int k = 0; k<nMix; k++)
		{
			pp[k] = mixture[k].MOutP(data) + log(wt[k]);
			sum = LAdd(sum, pp[k]);
		}

		// accumulate statistics
	//	for (int i = 0; i<nMix; i++)
//		{
			// check if it is worthwile
			//if(pp[i]-sum > pruneTh)
	//		{
	//			mixture[i].Accumulate(data, exp(pp[i] - sum));
//			}
	//	}

		for (int k = 0; k < nMix; k++)
		{
			p = exp(pp[k] - sum);
			Gamma[k] += p;

			for (int j = 0; j<(int)featDim; j++)
			{

				GammaX[k*featDim + j] += p * (data[j] - mixture[k].mean[j]);
				
			//	GammaXX[k][j] += p * (data[j] - mixture[k].mean[j])*(data[j] - mixture[k].mean[j]);
			}
		}

		lr += sum;
	}

	fclose(featFile);
	data.clear();
	feat.clear();
	pp.clear();

}
void CGaussianMixture::AccumulateHTKFeatureFile(char * fname)
{
	vector<float> data, feat, pp;
	double sum;
	HTKhdr header;
	int featDim;
	CMatrix dataMat;

	// resize posterior probabilty vector
	pp.resize(nMix);

	FILE * featFile;
	if(fopen_s(&featFile,fname,"rb"))
	{
		printf("StatExt: can not open file %s\n",fname);
		return;
	}
	fread_s(&header,sizeof(HTKhdr),sizeof(HTKhdr),1,featFile);
	if(!naturalReadOrder)
	{
		Swap32(&header.nSamples);
		Swap32(&header.sampPeriod);
		Swap16(&header.sampSize);
		Swap16(&header.sampKind);
	}

	featDim = header.sampSize / sizeof(float);

		// dimension check
	if(transMat.empty())
	{
		if(dim != featDim)
		{
			printf("AccumulateHTKFeatureFile: Dimension mismatch [%d vs %d]\n", dim, featDim);
			return;
		}
		feat.resize(dim);
	}
	else
	{
		if(transMat.nCol != featDim)
		{
			printf("AccumulateHTKFeatureFile: Dimension mismatch [%d vs %d]\n", transMat.nCol, featDim);
			return;
		}
		feat.resize(featDim);
	}

	// read frame-by-frame
	//printf("%d\n",header.nSamples);
	for(int i=0; i<header.nSamples; i++)
	{
		if(fread_s(&feat[0], header.sampSize,sizeof(float), featDim, featFile) != featDim)
		{
			printf("AccumulateHTKFeatureFile: Error loading feature frame\n");
			return;
		}

		if(!naturalReadOrder)
		{
			int *tmp = (int *)(&feat[0]);
			for(int j=0; j<featDim; j++)
			{
				Swap32(tmp+j);
			}
		}

		if(!transMat.empty())
		{
			// apply linear transformation
			data = transMat * feat;
		}
		else
		{
			data = feat;
		}

		// calculate MOutP
		sum = LZERO;
		for(int i=0; i<nMix; i++)
		{
			pp[i] = mixture[i].MOutP(data) + log(wt[i]);
			sum = LAdd(sum, pp[i]);
		}

		// accumulate statistics
		for(int i=0; i<nMix; i++)
		{
			// check if it is worthwile
			//if(pp[i]-sum > pruneTh)
			{
				mixture[i].Accumulate(data, exp(pp[i]-sum));
			}
		}
		lr += sum;
	}

	fclose(featFile);
}

void CGaussianMixture::WriteAccumulatorVector(char * fname, char *mode)
{
	// log likelihood
	// gammas
	CMatrix uttStat;
	uttStat.nRow = 1;
	uttStat.nCol = nMix*(1+dim+dim);
#ifdef USE_STL_VECTOR

	uttStat.m_Matrix.assign(nMix*(1+dim+dim),0);

#else

	uttStat.m_Matrix = new float[nMix*(1+dim+dim)];

#endif
	
	int idx = 0;
	//float gamma = 0;
	for(int i=0; i<nMix; i++)
	{
		uttStat.m_Matrix[idx++] = (float)mixture[i].gamma;
		//gamma=gamma+mixture[i].gamma;
		
		for(int j=0; j<dim; j++)
		{
			uttStat.m_Matrix[idx++] = (float)mixture[i].gammaX[j];
		}
		for(int j=0; j<dim; j++)
		{
			uttStat.m_Matrix[idx++] = (float)mixture[i].gammaXX[j];
		}
	}
	//printf("%f\n",gamma);
	uttStat.save(fname, mode);
}

void CGaussianComponent::Accumulate(const vector<float> &x, const double p)
{
	EnterCriticalSection(cs);

	__try
	{
		gamma += p;

		for(int i=0; i<(int)x.size(); i++)
		{
			gammaX[i] += p * (x[i]-mean[i]);
			gammaXX[i] += p * (x[i]-mean[i])*(x[i]-mean[i]);
		}
	}
	__finally
	{
		LeaveCriticalSection(cs);
	}
}
void CGaussianMixture::GetMixtureVar(vector<float> &vec, const size_t iComponent) const
{
	
	if (iComponent >= nMix || iComponent < 0)
	{
		//TraceHR(E_UNEXPECTED, "CGaussianMixture::GetMixtureVar: Component index need from 0 to %d, input = %d", nMix, iComponent);
	}
	vec = mixture[iComponent].var;
}