#include "CMatrix.h"
#include <assert.h>

CMatrix::CMatrix(void)
{
	nRow = nCol = 0;
#ifndef USE_STL_VECTOR
	m_Matrix = NULL;
#else
#endif
}


CMatrix::~CMatrix(void)
{
#ifndef USE_STL_VECTOR
	clear();
#endif
}

CMatrix::CMatrix(const ULONG row, const ULONG col, const float val)
{
#ifdef USE_STL_VECTOR
	nRow = row;
	nCol = col;
	m_Matrix.assign(row * col, val);
#else
	m_Matrix = NULL;
	assign(row, col, val);
#endif
	
}

CMatrix::CMatrix(const vector<float> &vec1, const vector<float> &vec2)
{
#ifndef USE_STL_VECTOR
	m_Matrix = NULL;
#endif
	assign(vec1, vec2);
}

CMatrix::CMatrix(const CMatrix &mat)
{
#ifndef USE_STL_VECTOR
	m_Matrix = NULL;
#endif
	assign(mat);
}

void CMatrix::assign(const vector<float> &vec1, const vector<float> &vec2)
{
	assert(vec1.size() != 0 && vec2.size() != 0);

	setSize(vec1.size(), vec2.size());

#ifdef SSE_OPTIMIZED

#ifdef USE_STL_VECTOR
	ULONG i = 0;
	ULONG j = 0;
	const float *p1, *p2;
	float *p = getRawPtr();
	__m128 _A, _B;
	for (i = 0, p1 = &vec1[0]; i < nRow; ++i, ++p1)
	{
		_B = _mm_set1_ps(*(p1));
		for (j = 4, p2 = &vec2[0]; j <= nCol; j += 4, p += 4, p2 += 4)
		{
			_A = _mm_loadu_ps(p2);
			_A = _mm_mul_ps(_A, _B);
			_mm_storeu_ps(p, _A);
		}
		for (j -= 4; j < nCol; ++j, ++p, ++p2)
		{
			*p = *p1 * *p2;
		}
	}
#else
	ULONG i = 0;
	ULONG j = 0;
	const float *p1, *p2;
	float *p = getRawPtr();
	__m128 _A, _B;
	if (vec2.size() % 4 != 0)
	{	// _mm_storeu_ps
		for (i = 0, p1 = &vec1[0]; i < nRow; ++i, ++p1)
		{
			_B = _mm_set1_ps(*(p1));
			for (j = 4, p2 = &vec2[0]; j <= nCol; j += 4, p += 4, p2 += 4)
			{
				_A = _mm_loadu_ps(p2);
				_A = _mm_mul_ps(_A, _B);
				_mm_storeu_ps(p, _A);
			}
			for (j -= 4; j < nCol; ++j, ++p, ++p2)
			{
				*p = *p1 * *p2;
			}
		}
	}
	else
	{	// _mm_store_ps
		for (i = 0, p1 = &vec1[0]; i < nRow; ++i, ++p1)
		{
			_B = _mm_set1_ps(*(p1));
			for (j = 4, p2 = &vec2[0]; j <= nCol; j += 4, p += 4, p2 += 4)
			{
				_A = _mm_loadu_ps(p2);
				_A = _mm_mul_ps(_A, _B);
				_mm_store_ps(p, _A);
			}
		}
	}
#endif


#else

	float *p = getRawPtr();
	for (ULONG i = 0; i < nRow; ++i)
	{
		for (ULONG j = 0; j < nCol; ++j, ++p)
		{
			*p = vec1[i] * vec2[j];
		}
	}

#endif
}

void CMatrix::assign(const ULONG row, const ULONG col, const float *pmat)
{
	assert(row != 0 || col != 0 || pmat != NULL);

	setSize(row, col);

#ifdef SSE_OPTIMIZED
	__m128 _A;
	ULONG N = nRow * nCol;
	float *p = getRawPtr();
	const float *p2 = pmat;
	ULONG i = 4;
	for (; i <= N; i += 4, p += 4, p2 += 4)
	{
		_A = _mm_loadu_ps(p2);
		_MM_STORE_PS(p, _A);
	}
	for (i -= 4; i < N; ++i, ++p, ++p2)
	{
		*p = *p2;
	}
#else
	ULONG N = nRow * nCol;
	float *p = getRawPtr();
	const float *p2 = pmat;
	for (ULONG i = 0; i < N; ++i, ++p, ++p2)
	{
		*p = *p2;
	}
#endif

}

void CMatrix::assign(const ULONG row, const ULONG col, const float val)
{
	assert(row != 0 && col != 0);
#ifdef USE_STL_VECTOR
	nRow = row;
	nCol = col;
	m_Matrix.assign(nRow * nCol, val);
#else

	setSize(row, col);

#ifdef SSE_OPTIMIZED
	__m128 _A;
	ULONG N = nRow * nCol;
	_A = _mm_set1_ps(val);
	float *p = getRawPtr();
	ULONG i = 4;
	for (; i <= N; i += 4, p += 4)
	{
		_mm_store_ps(p, _A);
	}
	for (i -= 4; i < N; ++i, ++p)
	{
		*p = val;
	}
#else
	ULONG N = nRow * nCol;
	float *p = getRawPtr();
	for (ULONG i = 0; i < N; ++i, ++p)
	{
		*p = val;
	}
#endif

#endif
}

void CMatrix::assign(const CMatrix &mat)
{
	
#ifdef USE_STL_VECTOR	
	*this = mat;
#else
	setSize(mat.nRow, mat.nCol);
	CopyMemory(m_Matrix, mat.getRawPtr_const(), nRow * nCol * sizeof(float));
#endif
}

void CMatrix::assign_rand(const ULONG row, const ULONG col, const float minVal, const float maxVal)
{
	assert(row != 0 || col != 0 || minVal <= maxVal);

	setSize(row, col);

	ULONG N = row * col;
	float randval;
	for (ULONG i = 0; i < N; ++i)
	{
		 randval = (float)(rand() % 101) / 100;
		 //m_pTrans[i] = 2 * (0.5 - randval);
		 m_Matrix[i] = minVal + (maxVal - minVal) * randval;
	}
}

void CMatrix::setSize(const ULONG row, const ULONG col)
{
#ifdef USE_STL_VECTOR
	nRow = row;
	nCol = col;
	m_Matrix.resize(nRow * nCol);
	ZeroMemory(getRawPtr(), getRawSizeInByte());
#else
	clear();
	nRow = row;
	nCol = col;
	m_Matrix = (float*)_aligned_malloc(sizeof(float) * (nRow * nCol), SSE_ALIGN);
	if (m_Matrix == NULL || (ULONG)m_Matrix % SSE_ALIGN != 0)
	{
		clear();
	}
	ZeroMemory(getRawPtr(), getRawSizeInByte());
#endif
}

void CMatrix::resizeRow(const ULONG row)
{
#ifdef USE_STL_VECTOR
	m_Matrix.resize(row * nCol);
	nRow = row;
#else
	if (row < nRow)
	{
		nRow = row;
	}
	else if (row > nRow)
	{
		float * buf = (float*)_aligned_malloc(sizeof(float) * (row * nCol), SSE_ALIGN);
		if (buf == NULL || (ULONG)buf % SSE_ALIGN != 0)
		{
			clear();
		}
		CopyMemory(buf, m_Matrix, nRow * nCol);
		_aligned_free(m_Matrix);
		m_Matrix = buf;
		nRow = row;
	}
#endif
}

// return row i (i = 0, 1, ..., nRow-1) as a vector<float>
vector<float> CMatrix::getRow(ULONG i) const
{
	assert(i < nRow);
#ifdef USE_STL_VECTOR
	vector<float> v;
	vector<float>::const_iterator st = m_Matrix.begin() + nCol * i;
	vector<float>::const_iterator ed = m_Matrix.begin() + nCol * (i + 1);
	v.insert(v.end(), st, ed);
	return v;
#else
	vector<float> v(nCol);
	CopyMemory(&v[0], &m_Matrix[nCol * i], nCol * sizeof(float));
	return v;
#endif
}

void CMatrix::getRow(ULONG i, vector<float> &vec)
{
	assert(i < nRow);
	
#ifdef USE_STL_VECTOR
	vec.clear();
	vector<float>::const_iterator st = m_Matrix.begin() + nCol * i;
	vector<float>::const_iterator ed = m_Matrix.begin() + nCol * (i + 1);
	vec.insert(vec.end(), st, ed);
#else
	vec.resize(nCol);
	CopyMemory(&vec[0], &m_Matrix[nCol * i], nCol * sizeof(float));
#endif
}

// return col j (j = 0, 1, ..., nCol-1) as a vector<float>
vector<float> CMatrix::getCol(ULONG j) const
{
	assert(j < nCol);
	vector<float> v;
	for (ULONG i = 0; i < nRow; i++)
	{
		v.push_back(m_Matrix[i * nCol + j]);
	}
	return v;
}

void CMatrix::getCol(ULONG j, vector<float> &vec)
{
	assert(j < nCol);
	vec.clear();
	for (ULONG i = 0; i < nRow; i++)
	{
		vec.push_back(m_Matrix[i * nCol + j]);
	}
}

// set row i (i = 0, 1, ..., nRow-1) as a vector<float>
void CMatrix::setRow(const vector<float> & vec, ULONG i)
{
	assert(vec.size() == nCol && i < nRow);

	CopyMemory(&m_Matrix[nCol * i], &vec[0], nCol * sizeof(float));
}
// set col j (j = 0, 1, ..., nCol-1) as a vector<float>
void CMatrix::setCol(const vector<float> & vec, ULONG j)
{
	assert(vec.size() == nRow && j < nCol);

	for (ULONG i = 0; i < nRow; ++i)
	{
		m_Matrix[i * nCol + j] = vec[i];
	}
}

void CMatrix::popRow(vector<float> &vec)
{
	assert(nRow >= 1);
#ifdef USE_STL_VECTOR
	vec.clear();
	vector<float>::const_iterator st = m_Matrix.end() - nCol;
	vector<float>::const_iterator ed = m_Matrix.end();
	vec.insert(vec.end(), st, ed);
	m_Matrix.erase(st, ed);
	--nRow;
#else
	vec.resize(nCol);
	CopyMemory(&vec[0], &m_Matrix[nCol * (nRow - 1)], nCol * sizeof(float));
	--nRow;
#endif
}

void CMatrix::pushRow(vector<float> &vec)
{
	assert(vec.size() == nCol);
#ifdef USE_STL_VECTOR
	m_Matrix.insert(m_Matrix.end(), vec.begin(), vec.end());
	++nRow;
#else
	float * buf = (float*)_aligned_malloc(sizeof(float) * ((nRow + 1) * nCol), SSE_ALIGN);
	if (buf == NULL || (ULONG)buf % SSE_ALIGN != 0)
	{
		clear();
	}
	CopyMemory(buf, m_Matrix, nRow * nCol * sizeof(float));
	_aligned_free(m_Matrix);
	m_Matrix = buf;
	CopyMemory(&m_Matrix[nCol * nRow], &vec[0], nCol * sizeof(float));
	++nRow;
#endif
}

void CMatrix::save(const char *fname, const char *mode) const
{
	FILE *f = NULL;
	if (empty())
	{
		fclose(f);
		return;
	}
	const float *p = &m_Matrix[0];
	if (strcmp(mode, "wt") == 0)
	{
		if (fopen_s(&f, fname, "wt") != 0)
		{
			printf("CMatrix::save: can't create file: %s", fname);
		}
		fprintf(f, "%u\t%u\n", nRow, nCol);
		for(ULONG i = 0; i < nRow; ++i)
		{
			for (ULONG j = 0; j < nCol; ++j)
			{
				fprintf(f, "%e\t", *p);
				++p;
			}
			fprintf(f, "\n");
		}
	}
	else
	{
		if (fopen_s(&f, fname, "wb") != 0)
		{
			printf("CMatrix::save: can't create file: %s", fname);
		}
		fwrite(&nRow, sizeof(ULONG), 1, f);
		fwrite(&nCol, sizeof(ULONG), 1, f);
		for(ULONG i = 0; i < nRow; ++i)
		{
			fwrite(p, sizeof(float), nCol, f);
			p += nCol;
		}
	}
    fclose(f);
}

void CMatrix::load(const char *fname, const char *mode)
{
	CMatrixHeader sth = { 0, 0 };
	FILE * f = NULL;

	if (strcmp(mode, "rt") == 0 || strcmp(mode, "rtHLDA") == 0)
	{
		if (fopen_s(&f, fname, "rt") != 0)
		{
			printf("CMatrix::load: can't open file: %s", fname);
		}

		if(strcmp(mode, "rt") == 0)
		{
			if (fscanf_s(f, "%u %u",&sth.nRow, &sth.nCol) != 2)
			{
				fclose(f);
				printf( "CMatrix::load: can't read the header");
			}
		}
		else if(strcmp(mode, "rtHLDA") == 0)
		{
			ULONG placeHolder;
			if (fscanf_s(f, "%u %u %u", &placeHolder, &sth.nRow, &sth.nCol) != 3)
			{
				fclose(f);
				printf( "CMatrix::load: can't read the header");
			}
		}

		clear();
		if (sth.nRow > 0 && sth.nCol > 0)
		{
			assign(sth.nRow, sth.nCol, 0.0);
			float *p = &m_Matrix[0];
			float buf;
			for (ULONG i = 0; i < nRow; ++i)
			{
				for (ULONG j = 0; j < nCol; ++j)
				{
					if (fscanf_s(f, "%f",&buf) != 1)
					{
						fclose(f);
						printf("CMatrix::load: matrix and header mismatch");
					}
					else
					{
						*p = buf;
						++p;
					}
				}
			}
		}
	}
	else
	{
		if (fopen_s(&f, fname, "rb") != 0)
		{
			printf("CMatrix::load: can't open file: %s", fname);
		}
		if (fread(&sth, sizeof(CMatrixHeader), 1, f) != 1)
		{
			fclose(f);
			printf("CMatrix::load: can't read the header");
		}

		clear();
		if (sth.nRow > 0 && sth.nCol > 0)
		{
			assign(sth.nRow, sth.nCol, 0.0);
			if (fread(&m_Matrix[0], sizeof(float), nCol * nRow, f) != nCol * nRow)
			{
				fclose(f);
				printf("CMatrix::load: matrix and header mismatch");
			}
		}
	}
	fclose(f);
}

void CMatrix::transpose()
{
	CMatrix mat;
	transpose(mat);
	assign(mat);
}

void CMatrix::transpose(CMatrix &mat)
{
	mat.assign(nCol, nRow);
	float* p = mat.getRawPtr();
	for (ULONG i = 0; i < nRow; ++i)
	{
        for (ULONG j = 0; j < nCol; ++j)
		{
			p[j * nRow + i] = m_Matrix[i * nCol + j];
		}
	}
}

float CMatrix::logdet()
{
	vector<ULONG> perm(nRow);
	int sign;
	CMatrix mat(*this);
	LUDecompose(mat, perm, sign);

	// |mat|
    float d = 0;
    for (ULONG i = 0; i < nRow; i ++)
    {
        d += log (fabs(mat[i][i]));
    }
	return d;
}

vector<float> CMatrix::diag() const
{
	assert(nRow == nCol);

    vector<float> vec;
    for (ULONG i = 0; i < nRow; ++i)
		vec.push_back(m_Matrix[i * nCol + i]);
    return vec;
}

CMatrix CMatrix::invChol()
{
	assert(nCol > 0 && nRow > 0 && nRow == nCol);

	CMatrix inv(nRow, nCol, 0.0);
	
	invChol(inv);

	return inv;
}

void CMatrix::invChol(CMatrix &inv)
{
	assert(nCol > 0 && nRow > 0 && nRow == nCol);
	inv.setSize(nRow, nRow);

	// cholesky decomposition
	CMatrix mat(*this);
	cholesky(mat);

	// Back substitutions
	vector<float> b, x; 
	b.assign(nRow, 0.0);
	for (ULONG j = 0; j < nCol; ++j)
	{
		b[j] = 1.0;
		cholSub(mat, b, x);
		for (ULONG i = 0; i < nRow; ++i) 
			inv[i][j] = x[i];
		b[j] = 0.0;
	}
}

void CMatrix::cholesky(CMatrix &mat)
{

	assert(mat.nCol > 0 && mat.nRow > 0 && mat.nRow == mat.nCol);

#ifdef SSE_OPTIMIZED

	LONG Nr4;
	float * pj, * pi;
	float buf;
	__m128 _A, _B, _C;
	for (ULONG j = 0; j < mat.nRow; j++)
	{
		Nr4 = j - (j % 4);
		for (ULONG i = j; i < mat.nRow; i++)
		{
			pj = mat[j];
			pi = mat[i];
			buf = 0;
			_C = _mm_set1_ps(0.0);
			for (LONG k = 0; k < Nr4; k += 4)
			{
				_A = _mm_loadu_ps(pj);
				_B = _mm_loadu_ps(pi);
				_B = _mm_mul_ps(_A, _B);
				_C = _mm_add_ps(_C, _B);
				pi += 4;
				pj += 4;
			}
			_B = _mm_movehl_ps(_C, _C);
			_C = _mm_add_ps(_C, _B);
			_B = _mm_shuffle_ps(_C, _C, 0xB1);
			_C = _mm_add_ps(_C, _B);
			_mm_store_ss(&buf, _C);
			for (ULONG k = Nr4; k < j; k++)
			{
				buf += *pj * (*pi);
				pi++;pj++;
			}
			mat[j][i] -= buf;
		}
		// diag components
		if (mat[j][j] <= 0.0)
		{
			printf("CMatrix::cholesky: the matrix is not positive\n");
		}
		mat[j][j] = sqrt(mat[j][j]);
		// non-diag components
		for (ULONG i = j + 1; i < mat.nRow; i++)
		{
			mat[i][j] = mat[j][i] / mat[j][j];
			mat[j][i] = mat[i][j];
		}
	}

#else

	// i loop
	for (ULONG i = 0; i < mat.nRow; ++i) 
	{
		// dkag components
		for (ULONG k = 0; k < i; ++k)
		{
			mat[i][i] -= mat[i][k] * mat[i][k];
		}
		if (mat[i][i] <= 0.0)
		{
			printf("CMatrix::cholesky: the matrix is not positive");
		}

		mat[i][i] = sqrt(mat[i][i]);
			
		// non-dkag components
		for (ULONG j = i + 1; j < mat.nRow; ++j) 
		{
			for (ULONG k = 0; k < i; ++k) 
				mat[i][j] -= mat[i][k] * mat[j][k];
			mat[j][i] = mat[i][j] / mat[i][i];
		}
	}
	
#endif
}

void CMatrix::linearSolveChol(CMatrix &A, vector<float> & b, vector<float> & x, bool oneStep)
{
	cholesky(A);
	cholSub(A, b, x, oneStep);
}

void CMatrix::cholSub(CMatrix &A, vector<float> &b, vector<float> &x, bool oneStep)
{

	assert(A.nCol > 0 && A.nRow > 0 && A.nRow == A.nCol && A.nCol == b.size());

#ifdef SSE_OPTIMIZED

	float sum, buf;
	float * pi, * px;
	ULONG Nr4;
	x.assign(A.nRow, 0);
	__m128 _A, _B, _C;
	for (ULONG i = 0; i < A.nRow; i++)
	{
		sum = b[i];
		pi = A[i];
		px = &x[0];
		Nr4 = i - (i % 4);
		_C = _mm_set1_ps(0.0);
		for (ULONG k = 0; k < Nr4; k += 4)
		{
			_A = _mm_loadu_ps(pi);
			_B = _mm_loadu_ps(px);
			_B = _mm_mul_ps(_A, _B);
			_C = _mm_add_ps(_C, _B);
			pi += 4;
			px += 4;
		}
		_B = _mm_movehl_ps(_C, _C);
		_C = _mm_add_ps(_C, _B);
		_B = _mm_shuffle_ps(_C, _C, 0xB1);
		_C = _mm_add_ps(_C, _B);
		_mm_store_ss(&buf, _C);
		for (ULONG k = Nr4; k < i; k++)
		{
			buf += *px * (*pi);
			pi++;px++;
		}
		sum -= buf;
		x[i] = sum / A[i][i];
	}

	if (oneStep == true)
	{
		return;
	}

	float * pj;
	x[A.nCol - 1] /= A[A.nCol - 1][A.nCol - 1];
	for (LONG j = A.nCol - 2; j >= 0; j--)
	{
		sum = x[j];
		Nr4 = A.nRow - ((A.nRow - j - 1) % 4);
		pj = &A[j][j + 1];
		px = &x[j + 1];
		_C = _mm_set1_ps(0.0);
		for (ULONG k = j + 1; k < Nr4; k += 4)
		{
			_A = _mm_loadu_ps(pj);
			_B = _mm_loadu_ps(px);
			_B = _mm_mul_ps(_A, _B);
			_C = _mm_add_ps(_C, _B);
			pj += 4;
			px += 4;
		}
		_B = _mm_movehl_ps(_C, _C);
		_C = _mm_add_ps(_C, _B);
		_B = _mm_shuffle_ps(_C, _C, 0xB1);
		_C = _mm_add_ps(_C, _B);
		_mm_store_ss(&buf, _C);
		for (ULONG k = Nr4; k < A.nRow; k++)
		{
			buf += *px * (*pj);
			pj++;px++;
		}
		sum -= buf;
		x[j] = sum / A[j][j];
	}

#else

	// Ly = b, y save in x
	float sum;
	x.assign(A.nRow, 0);
	for (ULONG i = 0; i < A.nRow; ++i)
	{
		sum = b[i];
		for (ULONG j = 0; j < i; ++j)
			sum -= A[i][j] * x[j];
		x[i] = sum / A[i][i];
	}

	if (oneStep == true)
		return;

	// L'x = y
	for (int j = A.nRow - 1; j >= 0; --j)
	{
		sum = x[j];
		for (int i = j + 1; i < (int)A.nRow; ++i)
			sum -= A[i][j] * x[i];
		x[j] = sum / A[j][j];
	}

#endif
}

CMatrix & CMatrix::operator += (const CMatrix & mat)
{
	assert(nRow == 0 || (nRow == mat.nRow && nCol == mat.nCol));

	if (empty())
	{
		*this = mat;
		return *this;
	}

#ifdef SSE_OPTIMIZED

	ULONG N = nRow * nCol;
	__m128 _A, _B, _C;
	const float * pmat = mat.getRawPtr_const();
	float * p = getRawPtr();
	ULONG i = 4;
	for (; i <= N; i += 4, p += 4, pmat += 4)
	{
		_A = _MM_LOAD_PS(p);
		_B = _MM_LOAD_PS(pmat);
		_C = _mm_add_ps(_A, _B);
		_MM_STORE_PS(p, _C); 
	}
	for (i -= 4; i < N; ++i, ++p, ++pmat)
	{
		*p += *pmat;
	}
	return *this;

#else

	ULONG N = nRow * nCol;
	const float * pmat = mat.getRawPtr_const();
	float * p = getRawPtr();
	for (ULONG i = 0; i < N; ++i, ++p, ++pmat)
	{
		*p += *pmat;
	}
	return *this;

#endif
}

CMatrix & CMatrix::operator -= (const CMatrix & mat)
{
	assert(nRow == mat.nRow && nCol == mat.nCol);

#ifdef SSE_OPTIMIZED

	ULONG N = nRow * nCol;
	__m128 _A, _B, _C;
	const float * pmat = mat.getRawPtr_const();
	float * p = getRawPtr();
	ULONG i = 4;
	for (; i <= N; i += 4, p += 4, pmat += 4)
	{
		_A = _MM_LOAD_PS(p);
		_B = _MM_LOAD_PS(pmat);
		_C = _mm_sub_ps(_A, _B);
		_MM_STORE_PS(p, _C); 
	}
	for (i -= 4; i < N; ++i, ++p, ++pmat)
	{
		*p -= *pmat;
	}
	return *this;

#else

	ULONG N = nRow * nCol;
	const float * pmat = mat.getRawPtr_const();
	float * p = getRawPtr();
	for (ULONG i = 0; i < N; ++i, ++p, ++pmat)
	{
		*p -= *pmat;
	}
	return *this;

#endif
}

CMatrix & CMatrix::operator *= (float x)
{

#ifdef SSE_OPTIMIZED

	ULONG N = nRow * nCol;
	__m128 _A, _B, _C;
	_B = _mm_set1_ps(x);
	float *p = getRawPtr();
	ULONG i = 4;
	for (; i <= N; i += 4, p += 4)
	{
		_A = _MM_LOAD_PS(p);
		_C = _mm_mul_ps(_A, _B);
		_MM_STORE_PS(p, _C); 
	}
	for (i -= 4; i < N; ++i, ++p)
	{
		*p *= x;
	}
	return *this;

#else

	ULONG N = nRow * nCol;
	float * p = getRawPtr();
	for (ULONG i = 0; i < N; ++i, ++p)
	{
		*p *= x;
	}
	return *this;

#endif
}

#ifndef USE_STL_VECTOR	
CMatrix & CMatrix::operator = (const CMatrix & mat)
{
	assign(mat);
	return *this;
}
#endif
CMatrix CMatrix::operator + (const CMatrix & mat) const
{
	return CMatrix(*this) += mat;
}

CMatrix CMatrix::operator - (const CMatrix & mat) const
{
    return CMatrix(*this) -= mat;
}

CMatrix CMatrix::operator * (float x) const
{
    return CMatrix(*this) *= x;
}

vector<float> CMatrix::operator * (const vector<float> & vec) const
{
	assert(nCol == vec.size());

#ifdef SSE_OPTIMIZED

#ifdef USE_STL_VECTOR
    vector<float> v;
	v.assign(nRow, 0);
	__m128 _A, _B, _C;
	const float *pvec;
	const float *p = getRawPtr_const();
    for (ULONG i = 0; i < nRow; ++i)
	{
		_C = _mm_set1_ps(0.0);
		pvec = &vec[0];
		ULONG j = 4;
		for (; j <= nCol; j += 4, p += 4, pvec += 4)
		{
			_A = _mm_loadu_ps(p);
			_B = _mm_loadu_ps(pvec);
			_B = _mm_mul_ps(_A, _B);
			_C = _mm_add_ps(_C, _B);
		}
		_B = _mm_movehl_ps(_C, _C);
        _C = _mm_add_ps(_C, _B);
        _B = _mm_shuffle_ps(_C, _C, 0xB1);
        _C = _mm_add_ps(_C, _B);
		_mm_store_ss(&v[i], _C);
		for (j -= 4; j < nCol; ++j, ++p, ++pvec)
		{
			v[i] += *p * *pvec;
		} 
	}
    return v;
#else

	vector<float> v;
	v.assign(nRow, 0);
	__m128 _A, _B, _C;
	const float *pvec;
	const float *p = getRawPtr_const();
	if (nCol % 4 != 0)
	{	// _mm_loadu_ps
		for (ULONG i = 0; i < nRow; ++i)
		{
			_C = _mm_set1_ps(0.0);
			pvec = &vec[0];
			ULONG j;
			for (j = 4; j <= nCol; j += 4, p += 4, pvec += 4)
			{
				_A = _mm_loadu_ps(p);
				_B = _mm_loadu_ps(pvec);
				_B = _mm_mul_ps(_A, _B);
				_C = _mm_add_ps(_C, _B);
			}
			_B = _mm_movehl_ps(_C, _C);
			_C = _mm_add_ps(_C, _B);
			_B = _mm_shuffle_ps(_C, _C, 0xB1);
			_C = _mm_add_ps(_C, _B);
			_mm_store_ss(&v[i], _C);
			for (j -= 4; j < nCol; ++j, ++p, ++pvec)
			{
				v[i] += *p * *pvec;
			} 
		}
	}
	else
	{	// _mm_load_ps
		for (ULONG i = 0; i < nRow; ++i)
		{
			_C = _mm_set1_ps(0.0);
			pvec = &vec[0];
			ULONG j = 4;
			for (; j <= nCol; j += 4, p += 4, pvec += 4)
			{
				_A = _mm_load_ps(p);
				_B = _mm_loadu_ps(pvec);
				_B = _mm_mul_ps(_A, _B);
				_C = _mm_add_ps(_C, _B);
			}
			_B = _mm_movehl_ps(_C, _C);
			_C = _mm_add_ps(_C, _B);
			_B = _mm_shuffle_ps(_C, _C, 0xB1);
			_C = _mm_add_ps(_C, _B);
			_mm_store_ss(&v[i], _C);
		}
	}

    return v;

#endif

#else

	vector<float> v;
	v.assign(nRow, 0);
	for (ULONG i = 0; i < nRow; ++i)
	{
		for (ULONG j = 0; j < nCol; ++j)
		{
			v[i] += m_Matrix[i * nCol + j] * vec[j];
		}
	}
	return v;

#endif

}

CMatrix CMatrix::operator * (const CMatrix & mat) const
{
	assert(nCol == mat.nRow);

	CMatrix m(nRow, mat.nCol, 0);
	for (ULONG j = 0; j < mat.nCol; ++j)
	{
		m.setCol((*this) * mat.getCol(j), j);
	}
    return m;
}

CMatrix CMatrix::MulSelfTranspose() // A * A*
{
	assert(nRow > 0 && nCol > 0);
	vector<float> w(nCol, 1);
	return MulSelfTranspose(w);
}

CMatrix CMatrix::MulSelfTranspose(vector<float> &w) // A * diag(w) * A*
{
	assert (nRow > 0 && nCol > 0 && w.size() == nCol);

#ifdef SSE_OPTIMIZED
	
	CMatrix mat(nRow, nRow, 0);
	__m128 _A, _B, _C;
	const float *pi, *pj, *ptemp;
	ULONG Ns4 = nCol - 4;
	for (ULONG i = 0; i < nRow; ++i)
	{
		ptemp = pj = &m_Matrix[i * nCol];
		for (ULONG j = i; j < nRow; ++j)
		{
			pi = ptemp;
			_C = _mm_set1_ps(0.0);
			ULONG k = 0;
			if (nCol >= 4)
			{
				for (k = 0; k <= Ns4; k += 4, pi += 4, pj += 4)
				{
					_A = _mm_loadu_ps(pi);
					_B = _mm_loadu_ps(pj);
					_B = _mm_mul_ps(_A, _B);
					_A = _mm_loadu_ps(&w[k]);
					_B = _mm_mul_ps(_A, _B);
					_C = _mm_add_ps(_C, _B);
				}
			}
			_B = _mm_movehl_ps(_C, _C);
			_C = _mm_add_ps(_C, _B);
			_B = _mm_shuffle_ps(_C, _C, 0xB1);
			_C = _mm_add_ps(_C, _B);
			_mm_store_ss(&mat[i][j], _C);
			for (; k < nCol; ++k, ++pi, ++pj)
			{
				mat[i][j] += *pi * w[k] * *pj;
			} 
		}
	}
	for (ULONG i = 0; i < nRow; ++i)
	{
		for (ULONG j = 0; j < i; ++j)
		{
			mat[i][j] = mat[j][i];
		}
	}

	return mat;
#else

	CMatrix mat(nRow, nRow, 0);
	const float *pi, *pj, *ptemp;
	for (ULONG i = 0; i < nRow; ++i)
	{
		ptemp = pj = &m_Matrix[i * nCol];
		for (ULONG j = i; j < nRow; ++j)
		{
			pi = ptemp;
			for (ULONG k = 0; k < nCol; ++k, ++pi, ++pj)
			{
				mat[i][j] += *pi * w[k] * *pj;
			}
		}
	}
	for (ULONG i = 0; i < nRow; ++i)
	{
		for (ULONG j = 0; j < i; ++j)
		{
			mat[i][j] = mat[j][i];
		}
	}
	return mat;
#endif
}

void CMatrix::WeightedAdd(CMatrix &mat, float w) // A += mat*w
{
	assert(nRow == mat.nRow && nCol == mat.nCol);

#ifdef SSE_OPTIMIZED

	ULONG N = nRow * nCol;
	__m128 _A, _B, _C;
	const float * pmat = mat.getRawPtr_const();
	float * p = getRawPtr();
	_C = _mm_set1_ps(w);
	ULONG i;
	for (i = 4; i <= N; i += 4, p += 4, pmat += 4)
	{
		_A = _MM_LOAD_PS(pmat);
		_A = _mm_mul_ps(_A, _C);
		_B = _MM_LOAD_PS(p);
		_B = _mm_add_ps(_A, _B);
		_MM_STORE_PS(p, _B);
	}
	for (i -= 4; i < N; ++i, ++p, ++pmat)
	{
		*p += *pmat * w;
	}

#else

	ULONG N = nRow * nCol;
	const float * pmat = mat.getRawPtr();
	float * p = getRawPtr();
	for (ULONG i = 0; i < N; ++i, ++p, ++pmat)
	{
		*p += *pmat * w;
	}

#endif
}

void CMatrix::Add(const vector<float> &vec1, const vector<float> &vec2) // A += vec1 * vec2^T
{

	assert(vec1.size() == nRow && vec2.size() == nCol);

#ifdef SSE_OPTIMIZED

#ifdef USE_STL_VECTOR
	ULONG i = 0, j = 0;
	const float *p1, *p2;
	float *p = getRawPtr();
	__m128 _A, _B, _C;
	for (i = 0, p1 = &vec1[0]; i < nRow; ++i, ++p1)
	{
		_C = _mm_set1_ps(*(p1));
		for (j = 4, p2 = &vec2[0]; j <= nCol; j += 4, p += 4, p2 += 4)
		{
			_A = _mm_loadu_ps(p2);
			_B = _mm_mul_ps(_A, _C);
			_A = _mm_loadu_ps(p);
			_B = _mm_add_ps(_A, _B);
			_mm_storeu_ps(p, _B);
		}
		for (j -= 4; j < nCol; ++j, ++p, ++p2)
		{
			*p += *p1 * *p2;
		}
	}
#else
	ULONG i = 0, j = 0;
	const float *p1, *p2;
	float *p = getRawPtr();
	__m128 _A, _B, _C;
	if (nCol % 4 != 0)
	{	// _mm_loadu_ps
		for (i = 0, p1 = &vec1[0]; i < nRow; ++i, ++p1)
		{
			_C = _mm_set1_ps(*(p1));
			for (j = 4, p2 = &vec2[0]; j <= nCol; j += 4, p += 4, p2 += 4)
			{
				_A = _mm_loadu_ps(p2);
				_B = _mm_mul_ps(_A, _C);
				_A = _mm_loadu_ps(p);
				_B = _mm_add_ps(_A, _B);
				_mm_storeu_ps(p, _B);
			}
			for (j -= 4; j < nCol; ++j, ++p, ++p2)
			{
				*p += *p1 * *p2;
			}
		}
	}
	else
	{	// _mm_load_ps
		for (i = 0, p1 = &vec1[0]; i < nRow; ++i, ++p1)
		{
			_C = _mm_set1_ps(*(p1));
			for (j = 4, p2 = &vec2[0]; j <= nCol; j += 4, p += 4, p2 += 4)
			{
				_A = _mm_loadu_ps(p2);
				_B = _mm_mul_ps(_A, _C);
				_A = _mm_load_ps(p);
				_B = _mm_add_ps(_A, _B);
				_mm_store_ps(p, _B);
			}
		}
	}
#endif
#else

	float * p = getRawPtr();
	for (ULONG i = 0; i < nRow; ++i)
	{
		for (ULONG j = 0; j < nCol; ++j, ++p)
		{
			*p += vec1[i] * vec2[j];
		}
	}
#endif
}

// LU decomposition on Matrix a.
// The permutation of rows is returned in perm and 
// sign is returned as +/-1 depending on whether 
// there was an even/odd number of row interchanges
void CMatrix::LUDecompose(CMatrix &A, vector<ULONG> &perm, int &sign)
{
	ULONG N = A.nRow;
    ULONG imax = 0;
    vector<float> vv(N);
    sign = 1;
    for (ULONG i = 0; i < N; i ++)
    {
        float scale = 0.0;
        for (ULONG j = 0; j < N; j ++)
        {
            float xx = fabs(A[i][j]);
            if (xx > scale)
                scale = xx;
        }

        if (scale == 0.0)
        {
			printf("CMatrix::LUDecompose: Matrix is Singular");
        }
        else
        {
            vv[i] = 1.0f / scale;
        }
    }

    for (ULONG j = 0; j < N; j ++)
    {
        for (ULONG i = 0; i < j; i ++)
        {
            float sum = A[i][j];
            for (ULONG k = 0; k < i; k ++)
            {
                sum -= A[i][k] * A[k][j];
            }
            A[i][j] = sum;
        }

        float scale=0.0;
        for (ULONG i = j; i < N; i ++)
        {
            float sum = A[i][j];
            for (ULONG k = 0; k < j; k ++)
            {
                sum -= A[i][k] * A[k][j];
            }
            A[i][j] = sum;

            float yy = vv[i]*fabs(sum);
            if (yy >=scale)
            {
                scale = yy;
                imax = i;
            }
        }

        if (j != imax)
        {
			vector<float> tempRow = A.getRow(imax);
			A.setRow(A.getRow(j), imax);
			A.setRow(tempRow, j);
            sign = -sign;
            vv[imax] = vv[j];
        }

        perm[j] = imax;
        if (A[j][j] == 0.0)
        {
			printf("CMatrix::LUDecompose: Matrix is Singular");
        }
        else if (j != N-1)
        {
            float yy = 1.0f / A[j][j];
            for (ULONG i = j+1; i < N; i ++)
            {
                A[i][j] *= yy;
            }
        }
    }
}

// vecAugend = vecAugend + vecAddend
void CMatrix::VectorAddTo(vector<float> &vecAugend, vector<float> &vecAddend)
{
	assert(vecAugend.size() == vecAddend.size() && vecAddend.size() != 0);

	if (vecAugend.empty())
	{
		vecAugend = vecAddend;
		return;
	}

#ifdef SSE_OPTIMIZED

	ULONG N = vecAugend.size();
	__m128 _A, _B, _C;
	const float * padd = &vecAddend[0];
	float * p = &vecAugend[0];
	ULONG i;
	for (i = 4; i <= N; i += 4, p += 4, padd += 4)
	{
		_A = _mm_loadu_ps(p);
		_B = _mm_loadu_ps(padd);
		_C = _mm_add_ps(_A, _B);
		_mm_storeu_ps(p, _C); 
	}
	for (i -= 4; i < N; ++i, ++p, ++padd)
	{
		*p += *padd;
	}

#else

	ULONG N = vecAugend.size();
	const float * padd = &vecAddend[0];
	float * p = &vecAugend[0];
	for (ULONG i = 0; i < N; ++i, ++p, ++padd)
	{
		*p += *padd;
	}

#endif
}

float CMatrix::DotProduct(vector<float> &vec1, vector<float> &vec2)
{
	assert(vec1.size() != 0 && vec1.size() == vec2.size());
	float * p1 = &vec1[0];
	float * p2 = &vec2[0];
	return DotProduct(p1, p2, vec1.size());
}

float CMatrix::DotProduct(float *pvec1, float *pvec2, ULONG size)
{

#ifdef SSE_OPTIMIZED

	float val = 0;
	__m128 _A, _B, _C;
	ULONG i;
	_C = _mm_set1_ps(0.0);
	for (i = 4; i <= size; i += 4, pvec1 += 4, pvec2 += 4)
	{
		_A = _mm_loadu_ps(pvec1);
		_B = _mm_loadu_ps(pvec2);
		_B = _mm_mul_ps(_A, _B);
		_C = _mm_add_ps(_C, _B); 
	}
	_B = _mm_movehl_ps(_C, _C);
	_C = _mm_add_ps(_C, _B);
	_B = _mm_shuffle_ps(_C, _C, 0xB1);
	_C = _mm_add_ps(_C, _B);
	_mm_store_ss(&val, _C);
	for (i -= 4; i < size; ++i, ++pvec1, ++pvec2)
	{
		val += *pvec1 * (*pvec2);
	}
	return val;

#else

	float val = 0;
	for (ULONG i = 0; i < size; ++i, ++pvec1, ++pvec2)
	{
		val += *pvec1 * (*pvec2);
	}
	return val;

#endif
}



/*************************************************/
/*        Symmetric Matrix Class                 */
/*************************************************/

CSymmetricMatrix::CSymmetricMatrix(void)
{
	nRow = 0;
#ifndef USE_STL_VECTOR
	m_Matrix = NULL;
#endif
}


CSymmetricMatrix::~CSymmetricMatrix(void)
{
#ifndef USE_STL_VECTOR
	clear();
#endif
}

CSymmetricMatrix::CSymmetricMatrix(const ULONG row, const float val)
{
#ifdef USE_STL_VECTOR
	nRow = row;
	m_Matrix.assign(getRawSize(), val);
#else
	m_Matrix = NULL;
	assign(row, val);
#endif
	
}

CSymmetricMatrix::CSymmetricMatrix(const vector<float> &vec)
{
#ifndef USE_STL_VECTOR
	m_Matrix = NULL;
#endif
	assign(vec);
}

CSymmetricMatrix::CSymmetricMatrix(const CSymmetricMatrix &mat)
{
#ifndef USE_STL_VECTOR
	m_Matrix = NULL;
#endif
	assign(mat);
}

void CSymmetricMatrix::assign(const vector<float> &vec)
{
	assert(vec.size() != 0);

	setSize(vec.size());

#ifdef SSE_OPTIMIZED
	ULONG i = 0, j = 0;
	const float *p1, *p2;
	float *p = getRawPtr();
	__m128 _A, _B;
	for (i = 0, p1 = &vec[0]; i < nRow; ++i, ++p1)
	{
		_B = _mm_set1_ps(*(p1));
		for (j = 3, p2 = &vec[0]; j <= i; j += 4, p += 4, p2 += 4)
		{
			_A = _mm_loadu_ps(p2);
			_A = _mm_mul_ps(_A, _B);
			_mm_storeu_ps(p, _A);
		}
		for (j -= 3; j <= i; ++j, ++p, ++p2)
		{
			*p = *p1 * *p2;
		}
	}
#else

	float *p = getRawPtr();
	for (ULONG i = 0; i < nRow; ++i)
	{
		for (ULONG j = 0; j <= i; ++j, ++p)
		{
			*p = vec[i] * vec[j];
		}
	}

#endif
}

void CSymmetricMatrix::assign(const ULONG row, const float val)
{
	assert(row != 0);
#ifdef USE_STL_VECTOR
	nRow = row;
	m_Matrix.assign(getRawSize(), val);
#else

	setSize(row);

#ifdef SSE_OPTIMIZED
	__m128 _A;
	ULONG N = getRawSize();
	_A = _mm_set1_ps(val);
	float *p = getRawPtr();
	ULONG i;
	for (i = 4; i <= N; i += 4, p += 4)
	{
		_mm_store_ps(p, _A);
	}
	for (i -= 4; i < N; ++i, ++p)
	{
		*p = val;
	}
#else
	ULONG N = getRawSize();
	float *p = getRawPtr();
	for (ULONG i = 0; i < N; ++i, ++p)
	{
		*p = val;
	}
#endif

#endif
}

void CSymmetricMatrix::assign(const CSymmetricMatrix &mat)
{
	
#ifdef USE_STL_VECTOR	
	*this = mat;
#else
	setSize(mat.nRow);
	CopyMemory(m_Matrix, mat.getRawPtr_const(), getRawSize() * sizeof(float));
#endif
}

void CSymmetricMatrix::setSize(const ULONG row)
{
#ifdef USE_STL_VECTOR
	nRow = row;
	m_Matrix.resize(getRawSize());
#else
	clear();
	nRow = row;
	m_Matrix = (float*)_aligned_malloc(sizeof(float) * (getRawSize()), SSE_ALIGN);
	if (m_Matrix == NULL || (ULONG)m_Matrix % SSE_ALIGN != 0)
	{
		clear();
	}
#endif
}

void CSymmetricMatrix::ConvertTo(CMatrix & tarMat)
{
	tarMat.setSize(nRow, nRow);
	float *ptar = tarMat.getRawPtr();
	const float * p = getRawPtr_const();
	for (ULONG i = 0; i < nRow; ++i)
	{
		CopyMemory(ptar, p, (i + 1) * sizeof(float));
		p += i + 1;
		ptar += nRow;
	}
	for (ULONG i = 0; i < nRow; ++i)
	{
		for (ULONG j = i + 1; j < nRow; ++j)
		{
			tarMat[i][j] = tarMat[j][i];
		}
	}
}

void CSymmetricMatrix::ConvertFrom(const CMatrix & srcMat)
{
	if (srcMat.nRow != srcMat.nCol)
	{
		printf("CSymmetricMatrix::ConvertFrom: nRow[=%d] != nCol[=%d].\n", srcMat.nRow, srcMat.nCol);
	}
	setSize(srcMat.nRow);
	const float *psrc = srcMat.getRawPtr_const();
	float * p = getRawPtr();
	for (ULONG i = 0; i < nRow; ++i)
	{
		CopyMemory(p, psrc, (i + 1) * sizeof(float));
		p += i + 1;
		psrc += nRow;
	}
}

float & CSymmetricMatrix::element(ULONG i, ULONG j)
{
	if (i >= nRow || j >= nRow)
	{
		printf("CSymmetricMatrix::element: out of range, nRow=%d, i=%d, j=%d.\n", nRow, i, j);
	}
	if (i < j)
	{
		swap(i, j);
	}
	return m_Matrix[(i+1)*i/2+j];
}

CSymmetricMatrix & CSymmetricMatrix::operator += (const CSymmetricMatrix & mat)
{
	assert(nRow == 0 || nRow == mat.nRow);

	if (empty())
	{
		*this = mat;
		return *this;
	}

#ifdef SSE_OPTIMIZED

	ULONG N = getRawSize();
	__m128 _A, _B, _C;
	const float * pmat = mat.getRawPtr_const();
	float * p = getRawPtr();
	ULONG i;
	for (i = 4; i <= N; i += 4, p += 4, pmat += 4)
	{
		_A = _MM_LOAD_PS(p);
		_B = _MM_LOAD_PS(pmat);
		_C = _mm_add_ps(_A, _B);
		_MM_STORE_PS(p, _C); 
	}
	for (i -= 4; i < N; ++i, ++p, ++pmat)
	{
		*p += *pmat;
	}
	return *this;

#else

	ULONG N = getRawSize();
	const float * pmat = mat.getRawPtr_const();
	float * p = getRawPtr();
	for (ULONG i = 0; i < N; ++i, ++p, ++pmat)
	{
		*p += *pmat;
	}
	return *this;

#endif
}

CSymmetricMatrix & CSymmetricMatrix::operator -= (const CSymmetricMatrix & mat)
{
	assert(nRow == mat.nRow);

#ifdef SSE_OPTIMIZED

	ULONG N = getRawSize();
	__m128 _A, _B, _C;
	const float * pmat = mat.getRawPtr_const();
	float * p = getRawPtr();
	ULONG i;
	for (i = 4; i <= N; i += 4, p += 4, pmat += 4)
	{
		_A = _MM_LOAD_PS(p);
		_B = _MM_LOAD_PS(pmat);
		_C = _mm_sub_ps(_A, _B);
		_MM_STORE_PS(p, _C); 
	}
	for (i -= 4; i < N; ++i, ++p, ++pmat)
	{
		*p -= *pmat;
	}
	return *this;

#else

	ULONG N = getRawSize();
	const float * pmat = mat.getRawPtr_const();
	float * p = getRawPtr();
	for (ULONG i = 0; i < N; ++i, ++p, ++pmat)
	{
		*p -= *pmat;
	}
	return *this;

#endif
}

CSymmetricMatrix & CSymmetricMatrix::operator *= (float x)
{

#ifdef SSE_OPTIMIZED

	ULONG N = getRawSize();
	__m128 _A, _B, _C;
	_B = _mm_set1_ps(x);
	float *p = getRawPtr();
	ULONG i;
	for (i = 4; i <= N; i += 4, p += 4)
	{
		_A = _MM_LOAD_PS(p);
		_C = _mm_mul_ps(_A, _B);
		_MM_STORE_PS(p, _C); 
	}
	for (i -= 4; i < N; ++i, ++p)
	{
		*p *= x;
	}
	return *this;

#else

	ULONG N = getRawSize();
	float * p = getRawPtr();
	for (ULONG i = 0; i < N; ++i, ++p)
	{
		*p *= x;
	}
	return *this;

#endif
}

CSymmetricMatrix CSymmetricMatrix::operator + (const CSymmetricMatrix & mat) const
{
	return CSymmetricMatrix(*this) += mat;
}

CSymmetricMatrix CSymmetricMatrix::operator - (const CSymmetricMatrix & mat) const
{
    return CSymmetricMatrix(*this) -= mat;
}

CSymmetricMatrix CSymmetricMatrix::operator * (float x) const
{
    return CSymmetricMatrix(*this) *= x;
}

void CSymmetricMatrix::WeightedAdd(CSymmetricMatrix &mat, float w) // A += mat*w
{
	assert(nRow == mat.nRow);

#ifdef SSE_OPTIMIZED

	ULONG N = getRawSize();
	__m128 _A, _B, _C;
	const float * pmat = mat.getRawPtr_const();
	float * p = getRawPtr();
	_C = _mm_set1_ps(w);
	ULONG i = 4;
	for (; i <= N; i += 4, p += 4, pmat += 4)
	{
		_A = _MM_LOAD_PS(pmat);
		_A = _mm_mul_ps(_A, _C);
		_B = _MM_LOAD_PS(p);
		_B = _mm_add_ps(_A, _B);
		_MM_STORE_PS(p, _B);
	}
	for (i -= 4; i < N; ++i, ++p, ++pmat)
	{
		*p += *pmat * w;
	}

#else

	ULONG N = getRawSize();
	const float * pmat = mat.getRawPtr();
	float * p = getRawPtr();
	for (ULONG i = 0; i < N; ++i, ++p, ++pmat)
	{
		*p += *pmat * w;
	}

#endif
}

void CSymmetricMatrix::Add(const vector<float> &vec) // A += vec * vec^T
{

	assert(vec.size() == nRow);

#ifdef SSE_OPTIMIZED

	ULONG i = 0, j = 0;
	const float *p1, *p2;
	float *p = getRawPtr();
	__m128 _A, _B, _C;
	for (i = 0, p1 = &vec[0]; i < nRow; ++i, ++p1)
	{
		_C = _mm_set1_ps(*(p1));
		for (j = 3, p2 = &vec[0]; j <= i; j += 4, p += 4, p2 += 4)
		{
			_A = _mm_loadu_ps(p2);
			_B = _mm_mul_ps(_A, _C);
			_A = _mm_loadu_ps(p);
			_B = _mm_add_ps(_A, _B);
			_mm_storeu_ps(p, _B);
		}
		for (j -= 3; j <= i; ++j, ++p, ++p2)
		{
			*p += *p1 * *p2;
		}
	}
#else

	float * p = getRawPtr();
	for (ULONG i = 0; i < nRow; ++i)
	{
		for (ULONG j = 0; j <= i; ++j, ++p)
		{
			*p += vec[i] * vec[j];
		}
	}
#endif
}

void CSymmetricMatrix::invChol(CSymmetricMatrix &inv)
{
	assert(nRow > 0);

	inv.setSize(nRow);
	// cholesky decomposition
	CSymmetricMatrix mat;
	cholesky(mat);

	// Back substitutions
	vector<float> b, x; 
	b.assign(nRow, 0.0);
	for (ULONG j = 0; j < nRow; ++j)
	{
		b[j] = 1.0;
		cholSub(mat, b, x);
		for (ULONG i = j; i < nRow; ++i) 
			inv[i][j] = x[i];
		b[j] = 0.0;
	}
}

void CSymmetricMatrix::cholesky(CSymmetricMatrix &mat)
{

#ifdef SSE_OPTIMIZED

	mat = *this;

	LONG Nr4;
	float * pj, * pi;
	float buf;
	__m128 _A, _B, _C;
	for (ULONG j = 0; j < nRow; j++)
	{
		Nr4 = j - (j % 4);
		for (ULONG i = j; i < nRow; i++)
		{
			pj = mat[j];
			pi = mat[i];
			buf = 0;
			if (Nr4 > 0)
			{
				_C = _mm_set1_ps(0.0);
				for (LONG k = 0; k < Nr4; k += 4)
				{
					_A = _mm_loadu_ps(pj);
					_B = _mm_loadu_ps(pi);
					_B = _mm_mul_ps(_A, _B);
					_C = _mm_add_ps(_C, _B);
					pi += 4;
					pj += 4;
				}
				_B = _mm_movehl_ps(_C, _C);
				_C = _mm_add_ps(_C, _B);
				_B = _mm_shuffle_ps(_C, _C, 0xB1);
				_C = _mm_add_ps(_C, _B);
				_mm_store_ss(&buf, _C);
			}
			for (ULONG k = Nr4; k < j; k++)
			{
				buf += *pj * (*pi);
				pi++;pj++;
			}
			mat[i][j] = (*this)[i][j] - buf;
		}
		// diag components
		if (mat[j][j] <= 0.0)
		{
			printf("CSymmetricMatrix::cholesky: the matrix is not positive\n");
		}
		mat[j][j] = sqrt(mat[j][j]);
		// non-diag components
		for (ULONG i = j + 1; i < nRow; i++)
		{
			mat[i][j] = mat[i][j] / mat[j][j];
		}
	}

#else
	mat.setSize(nRow);
	// i loop
	for (ULONG i = 0; i < nRow; ++i) 
	{
		// dkag components
		mat.element(i,i) = element(i,i);
		for (ULONG k = 0; k < i; ++k)
		{
			mat.element(i,i) -= element(i,k) * element(i,k);
		}
		if (mat.element(i,i) <= 0.0)
		{
			printf("CSymmetricMatrix::cholesky: the matrix is not positive");
		}

		mat.element(i,i) = sqrt(mat.element(i,i));
			
		// non-dkag components
		for (ULONG j = i + 1; j < mat.nRow; ++j) 
		{
			mat.element(i,j) = element(i,j);
			for (ULONG k = 0; k < i; ++k) 
				mat.element(i,j) -= element(i,k) * element(j,k);
			mat.element(i,j) = mat.element(i,j) / mat.element(i,i);
		}
	}
#endif
}

void CSymmetricMatrix::cholSub(CSymmetricMatrix &A, vector<float> &b, vector<float> &x, bool oneStep)
{
	assert(A.nRow > 0 && A.nRow == b.size());

#ifdef SSE_OPTIMIZED

	float sum, buf;
	float * pi, * px;
	ULONG Nr4;
	x.assign(A.nRow, 0);
	__m128 _A, _B, _C;
	for (ULONG i = 0; i < A.nRow; i++)
	{
		sum = b[i];
		pi = A[i];
		px = &x[0];
		Nr4 = i - (i % 4);
		_C = _mm_set1_ps(0.0);
		for (ULONG k = 0; k < Nr4; k += 4)
		{
			_A = _mm_loadu_ps(pi);
			_B = _mm_loadu_ps(px);
			_B = _mm_mul_ps(_A, _B);
			_C = _mm_add_ps(_C, _B);
			pi += 4;
			px += 4;
		}
		_B = _mm_movehl_ps(_C, _C);
		_C = _mm_add_ps(_C, _B);
		_B = _mm_shuffle_ps(_C, _C, 0xB1);
		_C = _mm_add_ps(_C, _B);
		_mm_store_ss(&buf, _C);
		for (ULONG k = Nr4; k < i; k++)
		{
			buf += *px * (*pi);
			pi++;px++;
		}
		sum -= buf;
		x[i] = sum / A[i][i];
	}

	if (oneStep == true)
	{
		return;
	}

	//float * pj;
	//x[A.nRow - 1] /= A[A.nRow - 1][A.nRow - 1];
	//for (LONG j = A.nRow - 2; j >= 0; j--)
	//{
	//	sum = x[j];
	//	Nr4 = A.nRow - ((A.nRow - j - 1) % 4);
	//	pj = &A[j][j + 1];
	//	px = &x[j + 1];
	//	_C = _mm_set1_ps(0.0);
	//	for (LONG k = j + 1; k < Nr4; k += 4)
	//	{
	//		_A = _mm_loadu_ps(pj);
	//		_B = _mm_loadu_ps(px);
	//		_B = _mm_mul_ps(_A, _B);
	//		_C = _mm_add_ps(_C, _B);
	//		pj += 4;
	//		px += 4;
	//	}
	//	_B = _mm_movehl_ps(_C, _C);
	//	_C = _mm_add_ps(_C, _B);
	//	_B = _mm_shuffle_ps(_C, _C, 0xB1);
	//	_C = _mm_add_ps(_C, _B);
	//	_mm_store_ss(&buf, _C);
	//	for (ULONG k = Nr4; k < A.nRow; k++)
	//	{
	//		buf += *px * (*pj);
	//		pj++;px++;
	//	}
	//	sum -= buf;
	//	x[j] = sum / A[j][j];
	//}

#else

	// Ly = b, y save in x
	float sum;
	x.assign(A.nRow, 0);
	for (ULONG i = 0; i < A.nRow; ++i)
	{
		sum = b[i];
		for (ULONG j = 0; j < i; ++j)
			sum -= A[i][j] * x[j];
		x[i] = sum / A[i][i];
	}

	if (oneStep == true)
		return;
#endif
	// L'x = y
	for (int j = A.nRow - 1; j >= 0; --j)
	{
		sum = x[j];
		for (int i = j + 1; i < (int)A.nRow; ++i)
			sum -= A[i][j] * x[i];
		x[j] = sum / A[j][j];
	}
}