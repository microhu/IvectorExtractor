#pragma once
#include <vector>
#include <stdio.h>
#include <Windows.h>
using namespace std;


#ifdef SSE_OPTIMIZED
#include "mmintrin.h"
#endif

#define SSE_ALIGN 16

#ifndef USE_STL_VECTOR
//#define USE_STL_VECTOR
#endif
#ifndef SSE_OPTIMIZED
#define SSE_OPTIMIZED
#endif

#ifdef USE_STL_VECTOR
#define _MM_LOAD_PS(a)  _mm_loadu_ps((a))
#define _MM_STORE_PS(a,b)  _mm_storeu_ps((a), (b))
#else
#define _MM_LOAD_PS(a)  _mm_load_ps((a))
#define _MM_STORE_PS(a,b)  _mm_store_ps((a), (b)) 
#endif 

struct CMatrixHeader
{
    ULONG nRow;
    ULONG nCol;
};

class CMatrix : public CMatrixHeader
{
	friend class CGaussianMixture;
public:
	CMatrix(void);
	~CMatrix(void);
    CMatrix(const ULONG row, const ULONG col, const float val = 0);
	CMatrix(const vector<float> &vec1, const vector<float> &vec2); // A = vec1 * vec2^T
	CMatrix(const CMatrix &mat);

	void assign(const vector<float> &vec1, const vector<float> &vec2);
	void assign(const ULONG row, const ULONG col, const float *pmat);
	void assign(const ULONG row, const ULONG col, float val = 0.0);
	void assign(const CMatrix &mat);
	void assign_rand(const ULONG row, const ULONG col, const float minVal, const float maxVal);
	void setSize(const ULONG row, const ULONG col);

	inline float * operator [] (ULONG i)
	{
		return &m_Matrix[i * nCol];
	}
	inline float * getRawPtr()
	{
		return (empty() ? NULL : &m_Matrix[0]);
	}
	inline const float * getRawPtr_const() const
	{
		return (empty() ? NULL : &m_Matrix[0]);
	}
	inline ULONG getRawSize()
	{
#ifdef USE_STL_VECTOR
		return m_Matrix.size();
#else
		return nRow * nCol;
#endif
	}
	inline ULONG getRawSizeInByte()
	{
#ifdef USE_STL_VECTOR
		return m_Matrix.size()*sizeof(float);
#else
		return nRow * nCol*sizeof(float);
#endif
	}
	inline bool empty() const
	{
#ifdef USE_STL_VECTOR
		return m_Matrix.empty();
#else
		return ((nRow * nCol == 0) ? true : false);
#endif
	}

	inline void clear()
	{
#ifdef USE_STL_VECTOR
		if (m_Matrix.empty() == false)
		{
			m_Matrix.clear();
		}
#else
		if (m_Matrix != NULL)
		{
			_aligned_free(m_Matrix);
		}
		m_Matrix = NULL;
#endif
		nRow = 0;
		nCol = 0;
	}
	
	void resizeRow(const ULONG row);
	vector<float> getRow(ULONG i) const;// return row i (i = 0, 1, ..., nRow-1) as a vector<float>
	vector<float> getCol(ULONG j) const; // return col j (j = 0, 1, ..., nCol-1) as a vector<float>
	void getRow(ULONG i, vector<float> &vec);
	void getCol(ULONG j, vector<float> &vec);
	void setRow(const vector<float> & vec, ULONG i); // set row i (i = 0, 1, ..., nRow-1) as a vector<float>
	void setCol(const vector<float> & vec, ULONG j); // set col j (j = 0, 1, ..., nCol-1) as a vector<float>
	void popRow(vector<float> &vec);
	void pushRow(vector<float> &vec);
	void save(const char *fname, const char *mode = "wt") const;
	void load(const char *fname, const char *mode = "rt");

	void transpose();
	void transpose(CMatrix &mat);
	float logdet();
	vector<float> CMatrix::diag() const;
	CMatrix invChol();
	void invChol(CMatrix &inv);
	static void cholesky(CMatrix &mat);
	static void linearSolveChol(CMatrix &A, vector<float> &b, vector<float> &x, bool oneStep = false);
	static void cholSub(CMatrix &A, vector<float> &b, vector<float> &x, bool oneStep = false);
	static void LUDecompose(CMatrix &A, vector<ULONG> &perm, int &sign);

#ifndef USE_STL_VECTOR	
	CMatrix & operator = (const CMatrix & mat);
#endif
	CMatrix & operator += (const CMatrix & mat);
    CMatrix & operator -= (const CMatrix & mat);
    CMatrix & operator *= (float x);
	CMatrix operator + (const CMatrix & mat) const;
    CMatrix operator - (const CMatrix & mat) const;
    CMatrix operator * (float x) const;
    vector<float> operator * (const vector<float> & vec) const;
    CMatrix operator * (const CMatrix & mat) const;

	CMatrix MulSelfTranspose(); // A * A*
	CMatrix MulSelfTranspose(vector<float> &w); // A * diag(w) * A*
	void WeightedAdd(CMatrix &mat, float w); // A += mat*w
	void Add(const vector<float> &vec1, const vector<float> &vec2); // A += vec1 * vec2^T

	/// vector opt
	// vecAugend = vecAugend + vecAddend
	static void VectorAddTo(vector<float> &vecAugend, vector<float> &vecAddend); 
	static float DotProduct(vector<float> &vec1, vector<float> &vec2);
	static float DotProduct(float *pvec1, float *pvec2, ULONG size);
private:

#ifdef USE_STL_VECTOR
	inline vector<float>::iterator begin()
	{
		return m_Matrix.begin();
	}
	inline vector<float>::iterator end()
	{
		return m_Matrix.end();
	}
#endif

private:
#ifdef USE_STL_VECTOR

	vector<float> m_Matrix;

#else

	float * m_Matrix;

#endif

};

/*************************************************/
/*        Symmetric Matrix Class                 */
/*************************************************/
/*
m_Matrix = [a11 a21 a22 a31 a32 a33]
Matrix:
	|a11        |
	|a21 a22    |
	|a31 a32 a33|
*/
struct CSymmetricMatrixHeader
{
    ULONG nRow;
};

// enum SymmetricMatrixStoreType {Lower, Upper };

class CSymmetricMatrix : public CSymmetricMatrixHeader
{
public:
	CSymmetricMatrix(void);
	~CSymmetricMatrix(void);
    CSymmetricMatrix(const ULONG row, const float val = 0);
	CSymmetricMatrix(const vector<float> &vec); // A = vec * vec^T
	CSymmetricMatrix(const CSymmetricMatrix &mat);

	void assign(const vector<float> &vec);
	void assign(const ULONG row, float val = 0.0);
	void assign(const CSymmetricMatrix &mat);
	void setSize(const ULONG row);

	void ConvertTo(CMatrix & tarMat);
	void ConvertFrom(const CMatrix & srcMat);
	float & element(ULONG i, ULONG j);

	inline float * getRawPtr()
	{
		return (empty() ? NULL : &m_Matrix[0]);
	}
	inline const float * getRawPtr_const() const
	{
		return (empty() ? NULL : &m_Matrix[0]);
	}
	inline ULONG getRawSize()
	{
		return nRow * (nRow + 1) / 2 ;
	}

	inline bool empty() const
	{
		return ((nRow == 0) ? true : false);
	}

	inline void clear()
	{
#ifdef USE_STL_VECTOR
		m_Matrix.clear();
#else
		if (m_Matrix != NULL)
		{
			_aligned_free(m_Matrix);
		}
		m_Matrix = NULL;
#endif
		nRow = 0;
	}

#ifndef USE_STL_VECTOR	
	CSymmetricMatrix & operator = (const CSymmetricMatrix & mat)
	{
		assign(mat);
		return *this;
	}
#endif
	CSymmetricMatrix & operator += (const CSymmetricMatrix & mat);
    CSymmetricMatrix & operator -= (const CSymmetricMatrix & mat);
    CSymmetricMatrix & operator *= (float x);
	CSymmetricMatrix operator + (const CSymmetricMatrix & mat) const;
    CSymmetricMatrix operator - (const CSymmetricMatrix & mat) const;
    CSymmetricMatrix operator * (float x) const;
    
	void invChol(CSymmetricMatrix &inv);
	void cholesky(CSymmetricMatrix &mat);
	static void cholSub(CSymmetricMatrix &A, vector<float> &b, vector<float> &x, bool oneStep = false);

	void WeightedAdd(CSymmetricMatrix &mat, float w); // A += mat*w
	void Add(const vector<float> &vec); // A += vec * vec^T

private:
	inline float * operator [] (ULONG i)
	{
		return &m_Matrix[i * (i + 1) / 2];
	}

private:

#ifdef USE_STL_VECTOR

	vector<float> m_Matrix;

#else

	float * m_Matrix;

#endif

	// SymmetricMatrixStoreType m_type;
};