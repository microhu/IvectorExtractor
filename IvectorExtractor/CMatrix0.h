#pragma once
#include <vector>
#include <stdio.h>
#include "Common.h"
#include "windows.h"
using namespace std;



#define ULONG unsigned long
#include "mmintrin.h"
#define SSE_ALIGN 16

/*************************************************/
/*              Matrix Class                     */
/*************************************************/
/* Normal matrix
// elements stored row-by-row
m_Matrix = [a11 a12 a13 a21 a22 a23 a31 a32 a33]
Matrix:
	|a11 a12 a13|
	|a21 a22 a23|
	|a31 a32 a33|
*/
/* Symmetric matrix
// elements stored in lower-triangle
m_Matrix = [a11 a21 a22 a31 a32 a33]
Matrix:
	|a11 a21 a31|
	|a21 a22 a32|
	|a31 a32 a33|
*/
enum MatrixType {Normal, Symmetric};
enum MatrixDataFormat {Float, Double};

struct CMatrixHeader
{
    size_t nRow;
    size_t nCol;
	MatrixType type;
	MatrixDataFormat format;
};

class CMatrix : public CMatrixHeader // changed to public
{
public:
	CMatrix(const char* property_1 = NULL, const char* property_2 = NULL);
	CMatrix(const MatrixType type_in, const MatrixDataFormat format_in);
	CMatrix(const size_t row, const size_t col, const double val = 0.0, const char* property_1 = NULL, const char* property_2 = NULL);
	CMatrix(const size_t row, const size_t col, const double val, const MatrixType type_in, const MatrixDataFormat format_in);
	CMatrix(const vector<float> &vec1, const vector<float> &vec2); // A = vec1 * vec2^T
	CMatrix(const vector<double> &vec1, const vector<double> &vec2);
	CMatrix(const CMatrix &mat);
	~CMatrix(void);
	
	/// assign
	void assign(const MatrixType type_in, const MatrixDataFormat format_in);
	void assign(const size_t row = 0, const size_t col = 0, const double val = 0.0, const char* property_1 = NULL, const char* property_2 = NULL);
	void assign(const size_t row, const size_t col, const double val, const MatrixType type_in, const MatrixDataFormat format_in);
	void assign(const size_t row, const size_t col, const float *p, const MatrixType type_in = Normal, const MatrixDataFormat format_in = Float);
	void assign(const size_t row, const size_t col, const double *p, const MatrixType type_in = Normal, const MatrixDataFormat format_in = Double);
	void assign(const vector<float> &vec1, const vector<float> &vec2);
	void assign(const vector<double> &vec1, const vector<double> &vec2);
	void assign(const CMatrix &mat);
	void assign_rand(const size_t row, const size_t col, const double minVal, const double maxVal);
	void assign_zero(const size_t row, const size_t col = 0);
	void setSize(const size_t row, const size_t col = 0);
	void clear(void);

	/// get stat
	bool isZeroMatrix(void);
	size_t getRawSize(void) const;
	size_t getRawSizeInByte(void) const;
	size_t getRawSizeNeeded(const size_t row, const size_t col = 0) const;
	size_t getRawSizeNeededInByte(const size_t row, const size_t col = 0) const;
	inline size_t getRow(void) const
	{
		return nRow;
	}
	inline size_t getCol(void) const
	{
		return (type == Normal) ? nCol : nRow;
	}
	inline bool empty(void) const
	{
		return ((nRow == 0) ? true : false);
	}

	/// get address
	void * getRawPtr(void);
	const void * getRawPtr_const(void) const;
	size_t getElementRawIdx(const size_t i, const size_t j) const;
	void * operator [] (const size_t i);
	const void * operator [] (const size_t i) const;

	/// I/O
	void resizeRow(const size_t row);
	void getRow(const size_t i, vector<float> &vec);
	void getRow(const size_t i, vector<double> &vec);
	void getCol(const size_t j, vector<float> &vec);
	void setRow(const size_t i, const vector<float> &vec); // set row i (i = 0, 1, ..., nRow-1) as a vector<float>
	void setRow(const size_t i, const vector<double> &vec);
	void setCol(const size_t j, const vector<float> &vec); // set col j (j = 0, 1, ..., nCol-1) as a vector<float>
	void setCol(const size_t j, const vector<double> &vec);
	void popRow(vector<float> &vec);
	void pushRow(const vector<float> &vec);
	template <typename T> void setElement(const T &val, const size_t i, const size_t j);
	float & floatElement(const size_t i, const size_t j);
	double & doubleElement(const size_t i, const size_t j);
	void save(const char *fname, const char *mode = "wt") const;
	void load(const char *fname, const char *mode = "rt");
	void ConvertTo(CMatrix &mat) const;
	void ConvertTo(const MatrixDataFormat new_format);
	
    /// operations
	void transpose(void);
	void transpose(CMatrix &mat) const;

	CMatrix & operator = (const CMatrix & mat);
	CMatrix & operator += (const CMatrix & mat);
    CMatrix & operator -= (const CMatrix & mat);
	CMatrix operator + (const CMatrix & mat) const;
    CMatrix operator - (const CMatrix & mat) const;
    vector<float> operator * (const vector<float> & vec) const;
	vector<double> operator * (const vector<double> & vec) const;
	void WeightedAdd(CMatrix &mat, const double w); // A += mat*w
	void Add(const vector<float> &vec1, const vector<float> &vec2); // A += vec1 * vec2^T
	void Add(const vector<double> &vec1, const vector<double> &vec2);
	void Add(const vector<float> &vec);
	void Add(const vector<double> &vec);

	void invAfterChol(CMatrix &inv); // this matrix is a lower triangular matrix got by Cholesky decomposition 
	void cholesky(CMatrix &mat);
	static void cholSub(const CMatrix &A, vector<float> &b, vector<float> &x, bool oneStep = false);
	static void cholSub(const CMatrix &A, vector<double> &b, vector<double> &x, bool oneStep = false);
	double logdetAfterChol();

	/// vector opt
	// vecAugend = vecAugend + vecAddend
	static void VectorAddTo(vector<float> &vecAugend, const vector<float> &vecAddend); 
	static void VectorAddTo(vector<double> &vecAugend, const vector<double> &vecAddend); 
	static float DotProduct(vector<float> &vec1, vector<float> &vec2);
	static float DotProduct(const float *pvec1, const float *pvec2, size_t size);
	static double DotProduct(vector<double> &vec1, vector<double> &vec2);
	static double DotProduct(const double *pvec1, const double *pvec2, size_t size);

private:
	void setMatrixFormat(const char* property_1 = NULL, const char* property_2 = NULL);

private:
	float * m_FloatMatrix;
	vector<double> m_DoubleMatrix;
};