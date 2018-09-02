#include "CMatrix.h"
#include <assert.h>

//using namespace SPlatform;


CMatrix::CMatrix(const char* property_1, const char* property_2)
{
	m_FloatMatrix = NULL;
	m_DoubleMatrix.clear();
	type = Normal;
	format = Float;
	nRow = nCol = 0;
	assign(0, 0, 0.0, property_1, property_2);
}

CMatrix::CMatrix(const MatrixType type_in, const MatrixDataFormat format_in)
{
	m_FloatMatrix = NULL;
	m_DoubleMatrix.clear();
	type = type_in;
	format = format_in;
	nRow = nCol = 0;
}

CMatrix::CMatrix(const size_t row, const size_t col, const double val, const char* property_1, const char* property_2)
{
	m_FloatMatrix = NULL;
	m_DoubleMatrix.clear();
	type = Normal;
	format = Float;
	nRow = nCol = 0;
	assign(row, col, val, property_1, property_2);
}

CMatrix::CMatrix(const size_t row, const size_t col, const double val, const MatrixType type_in, const MatrixDataFormat format_in)
{
	m_FloatMatrix = NULL;
	m_DoubleMatrix.clear();
	assign(row, col, val, type_in, format_in);
}

CMatrix::CMatrix(const vector<float> &vec1, const vector<float> &vec2)
{
	m_FloatMatrix = NULL;
	m_DoubleMatrix.clear();
	type = Normal;
	format = Float; // default float
	nRow = nCol = 0;
	assign(vec1, vec2);
}

CMatrix::CMatrix(const vector<double> &vec1, const vector<double> &vec2)
{
	m_FloatMatrix = NULL;
	m_DoubleMatrix.clear();
	type = Normal;
	format = Double; // default double
	nRow = nCol = 0;
	assign(vec1, vec2);
}

CMatrix::CMatrix(const CMatrix &mat)
{
	m_FloatMatrix = NULL;
	m_DoubleMatrix.clear();
	nRow = nCol = 0;
	assign(mat);
}

CMatrix::~CMatrix(void)
{
	clear();
}

void CMatrix::assign(const MatrixType type_in, const MatrixDataFormat format_in)
{
	type = type_in;
	format = format_in;
	clear();
}

void CMatrix::assign(const size_t row, const size_t col, const double val, const char* property_1, const char* property_2)
{
	// set matrix format (type & data format)
	setMatrixFormat(property_1, property_2);

	// set matrix size
	setSize(row, col);

	// assign
	size_t size = getRawSize();
	if (format == Float)
	{
		if (val < ZERO && val > -ZERO)
		{
			assign_zero(nRow, nCol);
		}
		else
		{
			__m128 _A;
			_A = _mm_set1_ps((float)val);
			float *p = m_FloatMatrix;
			size_t i;
			for (i = 4; i <= size; i += 4, p += 4)
			{
				_mm_store_ps(p, _A);
			}
			for (i -= 4; i < size; ++i)
			{
				*p++ = (float)val;
			}
		}
	}
	else // if (format == Double)
	{
		m_DoubleMatrix.assign(size, val);
	}
}

void CMatrix::assign(const size_t row, const size_t col, const double val, const MatrixType type_in, const MatrixDataFormat format_in)
{
	type = type_in;
	format = format_in;

	// set matrix size
	setSize(row, col);

	// assign
	size_t size = getRawSize();
	if (format == Float)
	{
		if (val < ZERO && val > -ZERO)
		{
			assign_zero(nRow, nCol);
		}
		else
		{
			__m128 _A;
			_A = _mm_set1_ps((float)val);
			float *p = m_FloatMatrix;
			size_t i;
			for (i = 4; i <= size; i += 4, p += 4)
			{
				_mm_store_ps(p, _A);
			}
			for (i -= 4; i < size; ++i)
			{
				*p++ = (float)val;
			}
		}
	}
	else // if (format == Double)
	{
		m_DoubleMatrix.assign(size, val);
	}
}

void CMatrix::assign(const size_t row, const size_t col, const float *p, const MatrixType type_in, const MatrixDataFormat format_in)
{
	type = type_in;
	format = format_in;

	// set matrix size
	setSize(row, col);
	size_t size = getRawSize();
	if (format == Float)
	{
		CopyMemory(getRawPtr(), p, getRawSizeInByte());
	}
	else // if (format == Double)
	{
		double *ptr = (double *)getRawPtr();
		for (size_t i = 0; i < size; ++i, ++ptr, ++p)
		{
			*ptr = (float)*p;
		}
	}
}

void CMatrix::assign(const size_t row, const size_t col, const double *p, const MatrixType type_in, const MatrixDataFormat format_in)
{
	type = type_in;
	format = format_in;

	// set matrix size
	setSize(row, col);
	size_t size = getRawSize();
	if (format == Double)
	{
		CopyMemory(getRawPtr(), p, getRawSizeInByte());
	}
	else // if (format == Float)
	{
		float *ptr = (float *)getRawPtr();
		for (size_t i = 0; i < size; ++i, ++ptr, ++p)
		{
			*ptr = (float)*p;
		}
	}
}

void CMatrix::assign(const vector<float> &vec1, const vector<float> &vec2)
{
	if (vec1.size() == 0 || vec2.size() == 0)
	{
	//	CMSHPCTrace::TraceHR(E_UNEXPECTED, "CMatrix::assign(const &vector<float>, const &vector<float>): input vector can't be empty.");
	}

	if (format != Float || type != Normal)
	{
	//	CMSHPCTrace::TraceHR(E_UNEXPECTED, "CMatrix::assign(const &vector<float>, const &vector<float>): only float & normal matrix supports this operate.");
	}
	else
	{
		setSize(vec1.size(), vec2.size());

		size_t i, j;
		const float *p1, *p2;
		float *p = m_FloatMatrix;
		__m128 _A, _B;
		if (vec2.size() % 4 != 0)
		{
			for (i = 0, p1 = &vec1[0]; i < nRow; ++i, ++p1)
			{
				_B = _mm_set1_ps(*(p1));
				for (j = 4, p2 = &vec2[0]; j <= nCol; j += 4, p += 4, p2 += 4)
				{
					_A = _mm_loadu_ps(p2);
					_A = _mm_mul_ps(_A, _B);
					_mm_storeu_ps(p, _A);
				}
				for (j -= 4; j < nCol; ++j)
				{
					*p++ = *p1 * *p2++;
				}
			}
		}
		else
		{
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
	}
}

void CMatrix::assign(const vector<double> &vec1, const vector<double> &vec2)
{
	if (vec1.size() == 0 || vec2.size() == 0)
	{
	//	CMSHPCTrace::TraceHR(E_UNEXPECTED, "CMatrix::assign(const &vector<double>, const &vector<double>): input vector can't be empty.");
	}

	if (format != Double || type != Normal)
	{
	//	CMSHPCTrace::TraceHR(E_UNEXPECTED, "CMatrix::assign(const &vector<double>, const &vector<double>): only double & normal matrix supports this operate.");
	}
	else
	{
		setSize(vec1.size(), vec2.size());

		size_t i, j;
		const double *p1, *p2;
		double *p = &m_DoubleMatrix[0];
		
		for (i = 0, p1 = &vec1[0]; i < nRow; ++i, ++p1)
		{
			for (j = 0, p2 = &vec2[0]; j < nCol; ++j)
			{
				*p++ = *p1 * *p2++;
			}
		}
	}
}

void CMatrix::assign(const CMatrix &mat)
{
	type = mat.type;
	format = mat.format;
	setSize(mat.nRow, mat.nCol);
	CopyMemory(getRawPtr(), mat.getRawPtr_const(), getRawSizeInByte());
}

void CMatrix::assign_rand(const size_t row, const size_t col, const double minVal, const double maxVal)
{
	if (row == 0 || col == 0 || minVal > maxVal)
	{
//		CMSHPCTrace::TraceHR(E_UNEXPECTED, "CMatrix::assign_rand(const CMatrix &mat): matrix type or raw format mismatch.");
	}

	setSize(row, col);

	size_t size = getRawSize();
	double randval;
	double offset = maxVal - minVal;
	if (format == Float)
	{
		float *p = m_FloatMatrix;
		for (size_t i = 0; i < size; ++i)
		{
			 randval = (rand() % 101) / 100;
			 *p++ = (float)(minVal + offset * randval);
		}
	}
	else // if (format == Double)
	{
		double *p = &m_DoubleMatrix[0];
		for (size_t i = 0; i < size; ++i)
		{
			 randval = (rand() % 101) / 100;
			 *p++ = minVal + offset * randval;
		}
	}
}

void CMatrix::assign_zero(const size_t row, const size_t col)
{
	if (row == 0)
	{
		return;
	}

	setSize(row, col);

	ZeroMemory(getRawPtr(), getRawSizeInByte());
}

void CMatrix::setSize(const size_t row, const size_t col)
{
	size_t size = getRawSize();
	size_t sizeNeeded = getRawSizeNeeded(row, col);
	nRow = row;
	nCol = (col == 0) ? row : col;
	if (type == Symmetric && nRow != nCol)
	{
	//	CMSHPCTrace::TraceHR(E_UNEXPECTED, "CMatrix::setSize: symmetric matrix must be given the same row and col size (%d vs. %d).\n", nRow, nCol);
	}

	if (sizeNeeded <= size)
	{
		if ((format == Float && m_FloatMatrix != NULL) || (format == Double && !m_DoubleMatrix.empty()))
		{
			return;
		}
	}

	if (format == Float)
	{
		m_DoubleMatrix.clear();
		if (m_FloatMatrix != NULL)
		{
			_aligned_free(m_FloatMatrix);
		}
		m_FloatMatrix = NULL;
		m_FloatMatrix = (float*)_aligned_malloc(sizeof(float) * sizeNeeded, SSE_ALIGN);
		if (m_FloatMatrix == NULL || (unsigned long)m_FloatMatrix % SSE_ALIGN != 0)
		{
			clear();
		//	CMSHPCTrace::TraceHR(E_OUTOFMEMORY, "CMatrix: out of memory.\n");
		}
	}
	else // if (format == Double)
	{
		if (m_FloatMatrix != NULL)
		{
			_aligned_free(m_FloatMatrix);
		}
		m_FloatMatrix = NULL;
		m_DoubleMatrix.resize(sizeNeeded);
	}
}

void CMatrix::clear(void)
{
	nRow = 0;
	nCol = 0;
	if (m_FloatMatrix != NULL)
	{
		_aligned_free(m_FloatMatrix);
	}
	m_FloatMatrix = NULL;
	m_DoubleMatrix.clear();
}

bool CMatrix::isZeroMatrix()
{
	size_t size = getRawSize();
	if (format == Float)
	{
		for (size_t i = 0; i < size; ++i)
		{
			if (m_FloatMatrix[i] > ZERO)
			{
				return false;
			}
		}
	}
	else // if (format == Double)
	{
		for (size_t i = 0; i < size; ++i)
		{
			if (m_DoubleMatrix[i] > ZERO)
			{
				return false;
			}
		}
	}
	return true;
}

size_t CMatrix::getRawSize(void) const
{
	if (type == Normal)
	{
		return (nCol == 0) ? nRow * nRow : nRow * nCol;
	}
	else // if (type == Symmetric)
	{
		return nRow * (nRow + 1) / 2;
	}
}

size_t CMatrix::getRawSizeInByte(void) const
{
	size_t size = getRawSize();
	if (format == Float)
	{
		return size * sizeof(float);
	}
	else // if (format == Double)
	{
		return size * sizeof(double);
	}
}

size_t CMatrix::getRawSizeNeeded(const size_t row, const size_t col) const
{
	if (type == Normal)
	{
		return (col == 0) ? row * row : row * col;
	}
	else // if (type == Symmetric)
	{
		size_t maxval = max(row, col);
		if (col != 0 && row != col)
		{
	//		CMSHPCTrace::Trace(0, TRACE_WARNINGS, "Warning: CMatrix::getRawSizeNeeded: symmetric matrix must be given the same row and col size (%d vs. %d), use row as %d.", row, col, max(row, col));
		}
		return maxval * (maxval + 1) / 2;
	}
}

size_t CMatrix::getRawSizeNeededInByte(const size_t row, const size_t col) const
{
	size_t size = getRawSizeNeeded(row, col);
	if (format == Float)
	{
		return size * sizeof(float);
	}
	else // if (format == Double)
	{
		return size * sizeof(double);
	}
}

void * CMatrix::getRawPtr(void)
{
	if (format == Float)
	{
		return (void *)m_FloatMatrix;
	}
	else // if (format == Double)
	{
		return (m_DoubleMatrix.empty()) ? NULL : (void *)&m_DoubleMatrix[0];
	}
}

const void * CMatrix::getRawPtr_const(void) const
{
	if (format == Float)
	{
		return (const void *)m_FloatMatrix;
	}
	else // if (format == Double)
	{
		return (m_DoubleMatrix.empty()) ? NULL : (const void *)&m_DoubleMatrix[0];
	}
}

size_t CMatrix::getElementRawIdx(const size_t i, const size_t j) const
{
	if (type == Symmetric)
	{
		return i > j ? (i+1)*i/2+j : (j+1)*j/2+i;
	}
	else // if (type == Normal)
	{
		return i*nCol+j;
	}
}

void * CMatrix::operator [] (const size_t i)
{
	if (i >= nRow)
	{
//		CMSHPCTrace::TraceHR(E_UNEXPECTED, "CMatrix::operator []: row index = %d out of [0, %d].", i, nRow-1);
	}
	
	if (type == Normal)
	{
		return (format == Float ? (void *)&m_FloatMatrix[i * nCol] : (void *)&m_DoubleMatrix[i * nCol]);
	}
	else // if (type == Symmetric)
	{
		return (format == Float ? (void *)&m_FloatMatrix[i * (i + 1) / 2] : (void *)&m_DoubleMatrix[i * (i + 1) / 2]);
	}
}

const void * CMatrix::operator [] (const size_t i) const
{
	if (i >= nRow)
	{
	//	CMSHPCTrace::TraceHR(E_UNEXPECTED, "CMatrix::operator []: row index = %d out of [0, %d].", i, nRow-1);
	}
	
	if (type == Normal)
	{
		return (format == Float ? (const void *)&m_FloatMatrix[i * nCol] : (const void *)&m_DoubleMatrix[i * nCol]);
	}
	else // if (type == Symmetric)
	{
		return (format == Float ? (const void *)&m_FloatMatrix[i * (i + 1) / 2] : (const void *)&m_DoubleMatrix[i * (i + 1) / 2]);
	}
}

void CMatrix::resizeRow(const size_t row)
{
	if (row < nRow)
	{
		nRow = row;
	}
	else if (row > nRow)
	{
		if (format == Float)
		{
			float * buf = (float*)_aligned_malloc(getRawSizeNeededInByte(row, nCol), SSE_ALIGN);
			if (buf == NULL || (ULONG)buf % SSE_ALIGN != 0)
			{
				clear();
	//			CMSHPCTrace::TraceHR(E_OUTOFMEMORY, "CMatrix: out of memory.\n");
			}
			CopyMemory(buf, m_FloatMatrix, getRawSizeInByte());
			_aligned_free(m_FloatMatrix);
			m_FloatMatrix = buf;
			nRow = row;
		}
		else // if (format == Double)
		{
			m_DoubleMatrix.resize(getRawSizeNeeded(row, nCol));
			nRow = row;
		}
	}
}

// return row i (i = 0, 1, ..., nRow-1) as a vector<float>
void CMatrix::getRow(const size_t i, vector<float> &vec)
{
	if (i >= nRow)
	{
	//	CMSHPCTrace::TraceHR(E_UNEXPECTED, "CMatrix::getRow: row index = %d out of [0, %d].", i, nRow-1);
	}
	if (format != Float)
	{
	//	CMSHPCTrace::TraceHR(E_UNEXPECTED, "CMatrix::getRow(const size_t i, vector<float> &vec): only support float format matrix.");
	}

	vec.resize(nCol);
	if (type == Normal)
	{
		CopyMemory(&vec[0], (*this)[i], nCol * sizeof(float));
	}
	else // if (format == Symmetric)
	{
		CopyMemory(&vec[0], (*this)[i], (i+1) * sizeof(float));
		float *p;
		for (size_t j = i+1; j < nRow; ++j)
		{
			p = (float *)(*this)[j];
			vec[j] = p[i];
		}
	}
}

void CMatrix::getRow(const size_t i, vector<double> &vec)
{
	if (i >= nRow)
	{
	//	CMSHPCTrace::TraceHR(E_UNEXPECTED, "CMatrix::getRow: row index = %d out of [0, %d].", i, nRow-1);
	}
	if (format != Double)
	{
	//	CMSHPCTrace::TraceHR(E_UNEXPECTED, "CMatrix::getRow(const size_t i, vector<double> &vec): only support double format matrix.");
	}

	vec.resize(nCol);
	if (type == Normal)
	{
		CopyMemory(&vec[0], (*this)[i], nCol * sizeof(double));
	}
	else // if (format == Symmetric)
	{
		CopyMemory(&vec[0], (*this)[i], (i+1) * sizeof(double));
		double *p;
		for (size_t j = i+1; j < nRow; ++j)
		{
			p = (double *)(*this)[j];
			vec[j] = p[i];
		}
	}
}

// return col j (j = 0, 1, ..., nCol-1) as a vector<float>
void CMatrix::getCol(const size_t j, vector<float> &vec)
{
	if (j >= nCol)
	{
	//	CMSHPCTrace::TraceHR(E_UNEXPECTED, "CMatrix::getCol: col index = %d out of [0, %d].", j, nCol-1);
	}
	if (format != Float)
	{
	//	CMSHPCTrace::TraceHR(E_UNEXPECTED, "CMatrix::getCol(const size_t i, vector<float> &vec): only support float format matrix.");
	}
	vec.resize(nRow);
	float *p;
	if (type == Normal)
	{
		for (size_t i = 0; i < nRow; ++i)
		{
			p = (float *)(*this)[i];
			vec[i] = p[j];
		}
	}
	else // if (format == Symmetric)
	{
		getRow(j, vec);
	}
}

// set row i (i = 0, 1, ..., nRow-1) as a vector<float>
void CMatrix::setRow(const size_t i, const vector<float> &vec)
{
	if (i >= nRow || vec.size() != nCol)
	{
	//	CMSHPCTrace::TraceHR(E_UNEXPECTED, "CMatrix::setRow: row index = %d, vec size = %d, matrix size is [%d, %d].", i, vec.size(), nRow, nCol);
	}
	if (format != Float)
	{
		//CMSHPCTrace::TraceHR(E_UNEXPECTED, "CMatrix::setRow(const size_t i, vector<float> &vec): only support float format matrix.");
	}

	if (type == Normal)
	{
		CopyMemory((*this)[i], &vec[0], nCol * sizeof(float));
	}
	else // if (format == Symmetric)
	{
		CopyMemory((*this)[i], &vec[0], (i+1) * sizeof(float));
		float *p;
		for (size_t j = i+1; j < nRow; ++j)
		{
			p = (float *)(*this)[j];
			p[i] = vec[j];
		}
	}
}

void CMatrix::setRow(const size_t i, const vector<double> &vec)
{
	if (i >= nRow || vec.size() != nCol)
	{
	//	CMSHPCTrace::TraceHR(E_UNEXPECTED, "CMatrix::setRow: row index = %d, vec size = %d, matrix size is [%d, %d].", i, vec.size(), nRow, nCol);
	}
	if (format != Double)
	{
	//	CMSHPCTrace::TraceHR(E_UNEXPECTED, "CMatrix::setRow(const size_t i, vector<double> &vec): only support double format matrix.");
	}

	if (type == Normal)
	{
		CopyMemory((*this)[i], &vec[0], nCol * sizeof(double));
	}
	else // if (format == Symmetric)
	{
		CopyMemory((*this)[i], &vec[0], (i+1) * sizeof(double));
		double *p;
		for (size_t j = i+1; j < nRow; ++j)
		{
			p = (double *)(*this)[j];
			p[i] = vec[j];
		}
	}
}

// set col j (j = 0, 1, ..., nCol-1) as a vector<float>
void CMatrix::setCol(const size_t j, const vector<float> &vec)
{
	if (j >= nCol || vec.size() != nRow)
	{
	//	CMSHPCTrace::TraceHR(E_UNEXPECTED, "CMatrix::setCol: col index = %d, vec size = %d, matrix size is [%d, %d].", j, vec.size(), nRow, nCol);
	}
	if (format != Float)
	{
	//	CMSHPCTrace::TraceHR(E_UNEXPECTED, "CMatrix::setCol(const size_t i, vector<float> &vec): only support float format matrix.");
	}

	float *p;
	if (type == Normal)
	{
		for (size_t i = 0; i < nRow; ++i)
		{
			p = (float *)(*this)[i];
			p[j] = vec[i];
		}
	}
	else // if (format == Symmetric)
	{
		setRow(j, vec);
	}
}

void CMatrix::setCol(const size_t j, const vector<double> &vec)
{
	if (j >= nCol || vec.size() != nRow)
	{
	//	CMSHPCTrace::TraceHR(E_UNEXPECTED, "CMatrix::setCol: col index = %d, vec size = %d, matrix size is [%d, %d].", j, vec.size(), nRow, nCol);
	}
	if (format != Double)
	{
	//	CMSHPCTrace::TraceHR(E_UNEXPECTED, "CMatrix::setCol(const size_t i, vector<double> &vec): only support double format matrix.");
	}

	double *p;
	if (type == Normal)
	{
		for (size_t i = 0; i < nRow; ++i)
		{
			p = (double *)(*this)[i];
			p[j] = vec[i];
		}
	}
	else // if (format == Symmetric)
	{
		setRow(j, vec);
	}
}

void CMatrix::popRow(vector<float> &vec)
{
	if (type != Normal || format != Float)
	{
	//	CMSHPCTrace::TraceHR(E_UNEXPECTED, "CMatrix::popRow(vector<float> &vec): only support float format normal matrix.");
	}
	if (empty())
	{
		vec.clear();
	}
	else
	{
		vec.resize(nCol);
		CopyMemory(&vec[0], (*this)[nRow - 1], nCol * sizeof(float));
		--nRow;
	}
}

void CMatrix::pushRow(const vector<float> &vec)
{
	if (type != Normal || format != Float)
	{
	//	CMSHPCTrace::TraceHR(E_UNEXPECTED, "CMatrix::pushRow(vector<float> &vec): only support float format normal matrix.");
	}
	if (vec.size() != nCol)
	{
	//	CMSHPCTrace::TraceHR(E_UNEXPECTED, "CMatrix::pushRow: vec size = %d, matrix size is [%d, %d].", vec.size(), nRow, nCol);
	}

	float * buf = (float*)_aligned_malloc(sizeof(float) * ((nRow + 1) * nCol), SSE_ALIGN);
	if (buf == NULL || (ULONG)buf % SSE_ALIGN != 0)
	{
		clear();
	//	CMSHPCTrace::TraceHR(E_OUTOFMEMORY, "CMatrix: out of memory.\n");
	}
	CopyMemory(buf, &m_FloatMatrix[0], nRow * nCol * sizeof(float));
	_aligned_free(m_FloatMatrix);
	m_FloatMatrix = buf;
	CopyMemory(&m_FloatMatrix[nCol * nRow], &vec[0], nCol * sizeof(float));
	++nRow;
}

template <typename T> void CMatrix::setElement(const T &val, const size_t i, const size_t j)
{
	if (i >= nRow || j >= nCol)
	{
		CMSHPCTrace::TraceHR(E_UNEXPECTED, "CMatrix::setElement: out of range, nRow=%d, nCol=%d, i=%d, j=%d.\n", nRow, nCol, i, j);
	}
	size_t idx = 0;
	if (type == Normal)
	{
		idx = i * nRow + j;
	}
	else // if (type == Symmetric)
	{
		idx = i > j ? (i+1)*i/2+j : (j+1)*j/2+i;
	}
	if (format == Float)
	{
		m_FloatMatrix[idx] = (float)val;
	}
	else // if (format == Double)
	{
		m_DoubleMatrix[idx] = (double)val;
	}
}

float & CMatrix::floatElement(const size_t i, const size_t j)
{
	if (format != Float)
	{
	//	CMSHPCTrace::TraceHR(E_UNEXPECTED, "CMatrix::floatElement: only support float format matrix.");
	}
	if (i >= nRow || j >= nCol)
	{
	//	CMSHPCTrace::TraceHR(E_UNEXPECTED, "CMatrix::floatElement::element: out of range, matrix size = [%d, %d], i=%d, j=%d.\n", nRow, nCol, i, j);
	}

	if (type == Symmetric)
	{
		return (i > j ? m_FloatMatrix[(i+1)*i/2+j] : m_FloatMatrix[(j+1)*j/2+i]);
	}
	else // if (type == Normal)
	{
		return m_FloatMatrix[i*nCol+j];
	}
}

double & CMatrix::doubleElement(const size_t i, const size_t j)
{
	if (format != Double)
	{
	//	CMSHPCTrace::TraceHR(E_UNEXPECTED, "CMatrix::doubleElement: only support double format matrix.");
	}
	if (i >= nRow || j >= nCol)
	{
	//	CMSHPCTrace::TraceHR(E_UNEXPECTED, "CMatrix::doubleElement::element: out of range, matrix size = [%d, %d], i=%d, j=%d.\n", nRow, nCol, i, j);
	}

	if (type == Symmetric)
	{
		return (i > j ? m_DoubleMatrix[(i+1)*i/2+j] : m_DoubleMatrix[(j+1)*j/2+i]);
	}
	else // if (type == Normal)
	{
		return m_DoubleMatrix[i*nCol+j];
	}
}

void CMatrix::save(const char *fname, const char *mode) const
{
	if (type != Normal || format != Float)
	{
	//	CMSHPCTrace::TraceHR(E_UNEXPECTED, "CMatrix::save: only support float format normal matrix.");
	}
	FILE *f = NULL;
	if (empty())
	{
		fclose(f);
		return;
	}
	const float *p = &m_FloatMatrix[0];
	if (strcmp(mode, "wt") == 0)
	{
		if (fopen_s(&f, fname, "wt") != 0)
		{
		//	CMSHPCTrace::TraceHR(E_UNEXPECTED, "CMatrix::save: can't create file: %s", fname);
		}
		fprintf(f, "%u\t%u\n", nRow, nCol);
		for(size_t i = 0; i < nRow; ++i)
		{
			for (size_t j = 0; j < nCol; ++j)
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
		//	CMSHPCTrace::TraceHR(E_UNEXPECTED, "CMatrix::save: can't create file: %s", fname);
		}
		unsigned long row = (unsigned long)nRow, col = (ULONG)nCol;
		fwrite(&row, sizeof(ULONG), 1, f);
		fwrite(&col, sizeof(ULONG), 1, f);
		for(size_t i = 0; i < nRow; ++i)
		{
			fwrite(p, sizeof(float), nCol, f);
			p += nCol;
		}
	}
    fclose(f);
}

void CMatrix::load(const char *fname, const char *mode)
{
	if (type != Normal || format != Float)
	{
	//	CMSHPCTrace::TraceHR(E_UNEXPECTED, "CMatrix::load: only support float format normal matrix.");
	}

	ULONG row, col;
	FILE * f = NULL;

	if (strcmp(mode, "rt") == 0 || strcmp(mode, "rtHLDA") == 0)
	{
		if (fopen_s(&f, fname, "rt") != 0)
		{
		//	CMSHPCTrace::TraceHR(E_UNEXPECTED, "CMatrix::load: can't open file: %s", fname);
		}

		if(strcmp(mode, "rt") == 0)
		{
			if (fscanf_s(f, "%u %u",&row, &col) != 2)
			{
				fclose(f);
			//	CMSHPCTrace::TraceHR(E_UNEXPECTED, "CMatrix::load: can't read the header");
			}
		}
		else if(strcmp(mode, "rtHLDA") == 0)
		{
			ULONG placeHolder;
			if (fscanf_s(f, "%u %u %u", &placeHolder, &row, &col) != 3)
			{
				fclose(f);
		//		CMSHPCTrace::TraceHR(E_UNEXPECTED, "CMatrix::load: can't read the header");
			}
		}

		clear();
		if (row > 0 && col > 0)
		{
			assign(row, col, 0.0);
			float *p = &m_FloatMatrix[0];
			float buf;
			for (size_t i = 0; i < nRow; ++i)
			{
				for (size_t j = 0; j < nCol; ++j)
				{
					if (fscanf_s(f, "%f",&buf) != 1)
					{
						fclose(f);
				//		CMSHPCTrace::TraceHR(E_UNEXPECTED, "CMatrix::load: matrix and header mismatch");
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
		//	CMSHPCTrace::TraceHR(E_UNEXPECTED, "CMatrix::load: can't open file: %s", fname);
		}
		if (fread(&row, sizeof(ULONG), 1, f) != 1)
		{
			fclose(f);
	//		CMSHPCTrace::TraceHR(E_UNEXPECTED, "CMatrix::load: can't read the header");
		}
		if (fread(&col, sizeof(ULONG), 1, f) != 1)
		{
			fclose(f);
		//	CMSHPCTrace::TraceHR(E_UNEXPECTED, "CMatrix::load: can't read the header");
		}

		clear();
		if (row > 0 && col > 0)
		{
			assign(row, col, 0.0);
			if (fread(&m_FloatMatrix[0], sizeof(float), nCol * nRow, f) != nCol * nRow)
			{
				fclose(f);
		//		CMSHPCTrace::TraceHR(E_UNEXPECTED, "CMatrix::load: matrix and header mismatch");
			}
		}
	}
	fclose(f);
}

void CMatrix::ConvertTo(CMatrix &mat) const
{
	if ((type == mat.type && format == mat.format) || empty())
	{
		mat.assign(*this);
		return;
	}

	if (type == Normal && mat.type == Symmetric)
	{
	//	CMSHPCTrace::TraceHR(E_UNEXPECTED, "CMatrix::ConvertTo: don't support converting noraml matrix to symmetric matrix.");
	}

	mat.setSize(nRow, nCol);
	// Symmetric matrix convert to normal matrix
	if (type == Symmetric && mat.type == Normal)
	{
		if (mat.format == Float)
		{
			float *ptar = (float *)mat.getRawPtr();
			if (format == Float)
			{
				const float * p = (const float * )getRawPtr_const();
				for (size_t i = 0; i < nRow; ++i)
				{
					CopyMemory(ptar, p, (i + 1) * sizeof(float));
					p += i + 1;
					ptar += nRow;
				}
			}
			else // if (float == Double)
			{
				const double * p = (const double * )getRawPtr_const();
				for (size_t i = 0; i < nRow; ++i)
				{
					for (size_t j = 0; j <= i; ++j)
					{
						ptar[j] = (float)p[j];
					}
					p += i + 1;
					ptar += nRow;
				}
			}

			for (size_t i = 0; i < nRow; ++i)
			{
				for (size_t j = i + 1; j < nRow; ++j)
				{
					mat.floatElement(i, j) = mat.floatElement(j, i);
				}
			}
		}
		else // if (mat.format == Double)
		{
			double *ptar = (double *)mat.getRawPtr();
			if (format == Float)
			{
				const float * p = (const float * )getRawPtr_const();
				for (size_t i = 0; i < nRow; ++i)
				{
					for (size_t j = 0; j <= i; ++j)
					{
						ptar[j] = (double)p[j];
					}
					p += i + 1;
					ptar += nRow;
				}
			}
			else // if (float == Double)
			{
				const double * p = (const double * )getRawPtr_const();
				for (size_t i = 0; i < nRow; ++i)
				{
					CopyMemory(ptar, p, (i + 1) * sizeof(double));
					p += i + 1;
					ptar += nRow;
				}
			}
			
			for (size_t i = 0; i < nRow; ++i)
			{
				for (size_t j = i + 1; j < nRow; ++j)
				{
					mat.doubleElement(i, j) = mat.doubleElement(j, i);
				}
			}
		}
	}
	else if (type == mat.type)
	{
		if (mat.format == Float)
		{
			float *ptar = (float *)mat.getRawPtr();
			const double * p = (const double *)getRawPtr_const();
			size_t size = getRawSize();
			for (size_t i = 0; i < size; ++i, ++ptar, ++p)
			{
				*ptar = (float)(*p);
			}
		}
		else // if (mat.format == Double)
		{
			double *ptar = (double *)mat.getRawPtr();
			const float * p = (const float *)getRawPtr_const();
			size_t size = getRawSize();
			for (size_t i = 0; i < size; ++i, ++ptar, ++p)
			{
				*ptar = (double)(*p);
			}
		}
	}
}

void CMatrix::ConvertTo(const MatrixDataFormat new_format)
{
	if (format == new_format)
	{
		return;
	}
	
	size_t size = getRawSize();
	if (new_format == Float)
	{
		if (m_FloatMatrix != NULL)
		{
			_aligned_free(m_FloatMatrix);
			m_FloatMatrix = NULL;
		}
		m_FloatMatrix = (float*)_aligned_malloc(sizeof(float) * size, SSE_ALIGN);
		if (m_FloatMatrix == NULL || (ULONG)m_FloatMatrix % SSE_ALIGN != 0)
		{
			clear();
		//	CMSHPCTrace::TraceHR(E_OUTOFMEMORY, "CMatrix: out of memory.\n");
		}
		for (size_t i = 0; i < size; ++i)
		{
			m_FloatMatrix[i] = (float)m_DoubleMatrix[i];
		}
		m_DoubleMatrix.clear();
		format = Float;
	}
	else // if (new_format == Double)
	{
		m_DoubleMatrix.resize(size);
		for (size_t i = 0; i < size; ++i)
		{
			m_DoubleMatrix[i] = (double)m_FloatMatrix[i];
		}
		_aligned_free(m_FloatMatrix);
		m_FloatMatrix = NULL;
		format = Double;
	}
}

void CMatrix::transpose(void)
{
	CMatrix mat;
	transpose(mat);
	assign(mat);
}

void CMatrix::transpose(CMatrix &mat) const
{
	if (type == Symmetric)
	{
		mat.assign(*this);
	}
	else // if (type == Normal)
	{
		mat.assign(nCol, nRow, 0.0, type, format);
		if (format == Float)
		{
			float *pmat, *p;
			for (size_t i = 0; i < nRow; ++i)
			{
				p = (float *)(*this)[i];
				for (size_t j = 0; j < nCol; ++j)
				{
					pmat = (float *)mat[j];
					pmat[i] = p[j];
				}
			}
		}
		else // if (format == Double)
		{
			double *pmat, *p;
			for (size_t i = 0; i < nRow; ++i)
			{
				p = (double *)(*this)[i];
				for (size_t j = 0; j < nCol; ++j)
				{
					pmat = (double *)mat[j];
					pmat[i] = p[j];
				}
			}
		}
	}
	
}

CMatrix & CMatrix::operator = (const CMatrix & mat)
{
	assign(mat);
	return *this;
}

CMatrix & CMatrix::operator += (const CMatrix & mat)
{
	if (empty())
	{
		*this = mat;
		return *this;
	}

	if (type != mat.type || format != mat.format || nRow != mat.nRow || nCol != mat.nCol)
	{
	//	CMSHPCTrace::TraceHR(E_UNEXPECTED, "CMatrix::+=:two matrix mismatch.\n");
	}

	size_t size = getRawSize();
	if (format == Float)
	{
		__m128 _A, _B, _C;
		const float * pmat = (const float *)mat.getRawPtr_const();
		float * p = (float *)getRawPtr();
		size_t i;
		for (i = 4; i <= size; i += 4, p += 4, pmat += 4)
		{
			_A = _mm_load_ps(p);
			_B = _mm_load_ps(pmat);
			_C = _mm_add_ps(_A, _B);
			_mm_store_ps(p, _C); 
		}
		for (i -= 4; i < size; ++i)
		{
			*p++ += *pmat++;
		}
	}
	else // if (format == Double)
	{
		const double * pmat = (const double *)mat.getRawPtr_const();
		double * p = (double *)getRawPtr();
		for (size_t i = 0; i < size; ++i)
		{
			*p++ += *pmat++;
		}
	}
	return *this;
}

CMatrix & CMatrix::operator -= (const CMatrix & mat)
{
	if (empty())
	{
		*this = mat;
		return *this;
	}

	if (type != mat.type || format != mat.format || nRow != mat.nRow || nCol != mat.nCol)
	{
	//	CMSHPCTrace::TraceHR(E_UNEXPECTED, "CMatrix::+=:two matrix mismatch.\n");
	}

	size_t size = getRawSize();
	if (format == Float)
	{
		__m128 _A, _B, _C;
		const float * pmat = (const float *)mat.getRawPtr_const();
		float * p = (float *)getRawPtr();
		size_t i;
		for (i = 4; i <= size; i += 4, p += 4, pmat += 4)
		{
			_A = _mm_load_ps(p);
			_B = _mm_load_ps(pmat);
			_C = _mm_sub_ps(_A, _B);
			_mm_store_ps(p, _C); 
		}
		for (i -= 4; i < size; ++i)
		{
			*p++ -= *pmat++;
		}
	}
	else // if (format == Double)
	{
		const double * pmat = (const double *)mat.getRawPtr_const();
		double * p = (double *)getRawPtr();
		for (size_t i = 0; i < size; ++i)
		{
			*p++ -= *pmat++;
		}
	}
	return *this;
}

CMatrix CMatrix::operator + (const CMatrix & mat) const
{
	return CMatrix(*this) += mat;
}

CMatrix CMatrix::operator - (const CMatrix & mat) const
{
    return CMatrix(*this) -= mat;
}

vector<float> CMatrix::operator * (const vector<float> & vec) const
{
	if (format != Float)
	{
	//	CMSHPCTrace::TraceHR(E_UNEXPECTED, "CMatrix::operator * (const vector<float> & vec):only support float format normal matrix.\n");
	}

	if (vec.size() != nCol)
	{
//		CMSHPCTrace::TraceHR(E_UNEXPECTED, "CMatrix::operator * (const vector<float> & vec):mismatch, nCol = %d, vec size = %d.\n", nCol, vec.size());
	}

	vector<float> v(nRow, 0);
	__m128 _A, _B, _C;
	const float *pvec;
	const float *p = (const float *)getRawPtr_const();
	if (type == Normal)
	{
		if (nCol % 4 != 0)
		{	// _mm_loadu_ps
			for (size_t i = 0; i < nRow; ++i)
			{
				_C = _mm_setzero_ps();
				pvec = &vec[0];
				size_t j;
				for (j = 4; j <= nCol; j += 4, p += 4, pvec += 4)
				{
					_A = _mm_loadu_ps(p);
					_B = _mm_loadu_ps(pvec);
					_B = _mm_mul_ps(_A, _B);
					_C = _mm_add_ps(_C, _B);
				}
				_C = _mm_hadd_ps(_C, _C);
				_C = _mm_hadd_ps(_C, _C);
				_mm_store_ss(&v[i], _C);
				for (j -= 4; j < nCol; ++j)
				{
					v[i] += *p++ * *pvec++;
				} 
			}
		}
		else
		{	// _mm_load_ps
			for (size_t i = 0; i < nRow; ++i)
			{
				_C = _mm_setzero_ps();
				pvec = &vec[0];
				size_t j;
				for (j = 4; j <= nCol; j += 4, p += 4, pvec += 4)
				{
					_A = _mm_load_ps(p);
					_B = _mm_loadu_ps(pvec);
					_B = _mm_mul_ps(_A, _B);
					_C = _mm_add_ps(_C, _B);
				}
				_C = _mm_hadd_ps(_C, _C);
				_C = _mm_hadd_ps(_C, _C);
				_mm_store_ss(&v[i], _C);
			}
		}
	}
	else
	{
		for (size_t i = 0; i < nRow; ++i)
		{
			_C = _mm_setzero_ps();
			pvec = &vec[0];
			p = (const float *)(*this)[i];
			size_t j;
			for (j = 3; j <= i; j += 4, p += 4, pvec += 4)
			{
				_A = _mm_loadu_ps(p);
				_B = _mm_loadu_ps(pvec);
				_B = _mm_mul_ps(_A, _B);
				_C = _mm_add_ps(_C, _B);
			}
			for (j -= 3; j <= i; ++j, ++p, ++pvec)
			{
				_A = _mm_load_ss(p);
				_B = _mm_load_ss(pvec);
				_B = _mm_mul_ss(_A, _B);
				_C = _mm_add_ps(_C, _B);
			}
			_C = _mm_hadd_ps(_C, _C);
			_C = _mm_hadd_ps(_C, _C);
			v[i] = _C.m128_f32[0];

			for (j = i + 1; j < nRow; ++j, ++pvec)
			{
				v[i] += m_FloatMatrix[(j+1)*j/2+i] * *pvec;
			} 
		}
	}
    return v;
}

vector<double> CMatrix::operator * (const vector<double> & vec) const
{
	if (format != Double)
	{
	//	CMSHPCTrace::TraceHR(E_UNEXPECTED, "CMatrix::operator * (const vector<double> & vec):only support float format normal matrix.\n");
	}

	if (vec.size() != nCol)
	{
	//	CMSHPCTrace::TraceHR(E_UNEXPECTED, "CMatrix::operator * (const vector<double> & vec):mismatch, nCol = %d, vec size = %d.\n", nCol, vec.size());
	}

	vector<double> v(nRow, 0);
	const double *pvec;
	const double *p = (const double *)getRawPtr_const();
	if (type == Normal)
	{
		for (size_t i = 0; i < nRow; ++i)
		{
			pvec = &vec[0];
			for (size_t j = 0; j < nCol; ++j)
			{
				v[i] += *p++ * *pvec++;
			} 
		}
	}
	else
	{
		for (size_t i = 0; i < nRow; ++i)
		{
			pvec = &vec[0];
			p = (const double *)(*this)[i];
			size_t j;
			for (j = 0; j <= i; ++j, ++p, ++pvec)
			{
				v[i] += *p * *pvec;
			}
			for (j = i + 1; j < nRow; ++j, ++pvec)
			{
				v[i] += m_DoubleMatrix[(j+1)*j/2+i] * *pvec;
			} 
		}
	}
    return v;
}

void CMatrix::WeightedAdd(CMatrix &mat, const double w) // A += mat*w
{
	if (mat.nRow != nRow || mat.nCol != nCol)
	{
	//	CMSHPCTrace::TraceHR(E_UNEXPECTED, "CMatrix::WeightedAdd: matrix mismatch.\n");
	}

	size_t size = getRawSize();
	size_t i;
	if (format == Float && mat.format == Float)
	{
		__m128 _A, _B, _C;
		const float * pmat = (const float * )mat.getRawPtr_const();
		float * p = m_FloatMatrix;
		_C = _mm_set1_ps((float)w);
		for (i = 4; i <= size; i += 4, p += 4, pmat += 4)
		{
			_A = _mm_load_ps(pmat);
			_A = _mm_mul_ps(_A, _C);
			_B = _mm_load_ps(p);
			_B = _mm_add_ps(_A, _B);
			_mm_store_ps(p, _B);
		}
		for (i -= 4; i < size; ++i)
		{
			*p++ += *pmat++ * (float)w;
		}
	}
	else if (format == Double && mat.format == Double)
	{
		const double *pmat = (const double * )mat.getRawPtr_const();
		double *p = &m_DoubleMatrix[0];
		for (i = 0; i < size; ++i)
		{
			*p++ += *pmat++ * w;
		}
	}
	else if (format == Double && mat.format == Float)
	{
		const float *pmat = (const float * )mat.getRawPtr_const();
		double *p = &m_DoubleMatrix[0];
		for (i = 0; i < size; ++i, ++p, ++pmat)
		{
			*p += w * (double)(*pmat);
		}
	}
	else
	{
	//	CMSHPCTrace::TraceHR(E_UNEXPECTED, "CMatrix::WeightedAdd: didn't support form double to float.\n");
	}
}

void CMatrix::Add(const vector<float> &vec1, const vector<float> &vec2) // A += vec1 * vec2^T
{

	if (type != Normal || format != Float)
	{
	//	CMSHPCTrace::TraceHR(E_UNEXPECTED, "CMatrix::Add(<float>, <float>): only support float format normal matrix.\n");
	}

	if (nRow != vec1.size() || nCol != vec2.size())
	{
	//	CMSHPCTrace::TraceHR(E_UNEXPECTED, "CMatrix::Add(<float>, <float>): matrix mismatch, matrix size = [%dX%d], vec1 size = %d, vec2 size = %d.\n", nRow, nCol, vec1.size(), vec2.size());
	}

	size_t i = 0, j = 0;
	const float *p1, *p2;
	float *p = m_FloatMatrix;
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
			for (j -= 4; j < nCol; ++j)
			{
				*p++ += *p1 * *p2++;
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
}

void CMatrix::Add(const vector<double> &vec1, const vector<double> &vec2) // A += vec1 * vec2^T
{
	if (type != Normal || format != Double)
	{
	//	CMSHPCTrace::TraceHR(E_UNEXPECTED, "CMatrix::Add(<double>, <double>): only support double format normal matrix.\n");
	}

	if (nRow != vec1.size() || nCol != vec2.size())
	{
//		CMSHPCTrace::TraceHR(E_UNEXPECTED, "CMatrix::Add(<double>, <double>): matrix mismatch, matrix size = [%dX%d], vec1 size = %d, vec2 size = %d.\n", nRow, nCol, vec1.size(), vec2.size());
	}

	size_t i = 0, j = 0;
	const double *p1, *p2;
	double *p = &m_DoubleMatrix[0];
	for (i = 0, p1 = &vec1[0]; i < nRow; ++i, ++p1)
	{
		for (j = 0, p2 = &vec2[0]; j < nCol; ++j)
		{
			*p++ += *p1 * *p2++;
		}
	}
}

void CMatrix::Add(const vector<float> &vec)
{
	if (type != Symmetric || format != Float)
	{
	//	CMSHPCTrace::TraceHR(E_UNEXPECTED, "CMatrix::Add(<float>): only support float format symmetric matrix.\n");
	}

	if (nRow != vec.size())
	{
	//	CMSHPCTrace::TraceHR(E_UNEXPECTED, "CMatrix::Add(<float>): matrix mismatch, matrix size = [%dX%d], vec size = %d.\n", nRow, nCol, vec.size());
	}

	size_t i = 0, j = 0;
	const float *p1, *p2;
	float *p = (float *)getRawPtr();
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
}

void CMatrix::Add(const vector<double> &vec)
{
	if (type != Symmetric || format != Double)
	{
	//	CMSHPCTrace::TraceHR(E_UNEXPECTED, "CMatrix::Add(<double>): only support double format symmetric matrix.\n");
	}

	if (nRow != vec.size())
	{
	//	CMSHPCTrace::TraceHR(E_UNEXPECTED, "CMatrix::Add(<double>): matrix mismatch, matrix size = [%dX%d], vec size = %d.\n", nRow, nCol, vec.size());
	}

	size_t i = 0, j = 0;
	const double *p1, *p2;
	double *p = (double *)getRawPtr();
	for (i = 0, p1 = &vec[0]; i < nRow; ++i, ++p1)
	{
		for (j = 0, p2 = &vec[0]; j <= i; ++j, ++p, ++p2)
		{
			*p += *p1 * *p2;
		}
	}
}

void CMatrix::invAfterChol(CMatrix &inv)
{
	if (type != Symmetric)
	{
	//	CMSHPCTrace::TraceHR(E_UNEXPECTED, "CMatrix::invAfterChol: only support Symmetric matrix.\n");
	}

	inv.assign(*this);
	
	if (format == Float)
	{
		// L^{-1}
		vector<float> b(nRow, 0.0);
		CMatrix mat(Normal, Float);
		mat.assign_zero(nRow, nRow);
		const float *pi, *pj, *px;
		float * x;
		float sum;	
		size_t k;
		__m128 _A, _B, _C;
		for (size_t j = 0; j < nRow; ++j)
		{
			b[j] = 1.0;
			x = (float * )mat[j];
			for (size_t i = j; i < nRow; ++i)
			{
				sum = b[i];
				pi = (const float *)(*this)[i];
				px = x;
				_C = _mm_setzero_ps();
				for (k = 4; k <= i; k += 4, pi += 4, px += 4)
				{
					_A = _mm_loadu_ps(pi);
					_B = _mm_loadu_ps(px);
					_B = _mm_mul_ps(_A, _B);
					_C = _mm_add_ps(_C, _B);
				}
				for (k -= 4; k < i; ++k, ++pi, ++px)
				{
					_A = _mm_load_ss(pi);
					_B = _mm_load_ss(px);
					_B = _mm_mul_ss(_A, _B);
					_C = _mm_add_ss(_C, _B);
				}
				_C = _mm_hadd_ps(_C, _C);
				_C = _mm_hadd_ps(_C, _C);
				sum -= _C.m128_f32[0];
				x[i] = sum / floatElement(i, i);
			}
			b[j] = 0.0;
		}

		// L^{T}^{-1} * L^{-1}
		for (size_t i = 0; i < nRow; ++i)
		{
			for (size_t j = i; j < nRow; ++j)
			{
				pi = (const float *)mat[i] + j;
				pj = (const float *)mat[j] + j;
				_C = _mm_setzero_ps();
				for (k = 4 + j; k <= nRow; k += 4, pi += 4, pj += 4)
				{
					_A = _mm_loadu_ps(pj);
					_B = _mm_loadu_ps(pi);
					_B = _mm_mul_ps(_A, _B);
					_C = _mm_add_ps(_C, _B);
				}
				for (k -= 4; k < nRow; ++k, ++pi, ++pj)
				{
					_A = _mm_load_ss(pj);
					_B = _mm_load_ss(pi);
					_B = _mm_mul_ss(_A, _B);
					_C = _mm_add_ss(_C, _B);
				}
				_C = _mm_hadd_ps(_C, _C);
				_C = _mm_hadd_ps(_C, _C);
				inv.floatElement(i,j) = _C.m128_f32[0];
			}
		}
	}
	else // if (format == Double)
	{
		// L^{-1}
		vector<double> b(nRow, 0.0);
		CMatrix mat(Normal, Double);
		mat.assign_zero(nRow, nRow);
		const double *pi, *pj, *px;
		double * x;
		double sum;	
		size_t k;
		for (size_t j = 0; j < nRow; ++j)
		{
			b[j] = 1.0;
			x = (double * )mat[j];
			for (size_t i = j; i < nRow; ++i)
			{
				sum = b[i];
				pi = (const double *)(*this)[i];
				px = x;
				for (k = 0; k < i; ++k, ++pi, ++px)
				{
					sum -= *pi * *px;
				}
				x[i] = sum / doubleElement(i, i);
			}
			b[j] = 0.0;
		}

		// L^{T}^{-1} * L^{-1}
		for (size_t i = 0; i < nRow; ++i)
		{
			for (size_t j = i; j < nRow; ++j)
			{
				pi = (const double *)mat[i] + j;
				pj = (const double *)mat[j] + j;
				sum = 0;
				for (k = j; k < nRow; ++k, ++pi, ++pj)
				{
					sum += *pi * *pj;
				}
				inv.doubleElement(i,j) = sum;
			}
		}
	}
}

void CMatrix::cholesky(CMatrix &mat)
{
	if (type != Symmetric)
	{
		//CMSHPCTrace::TraceHR(E_UNEXPECTED, "CMatrix::cholesky: only support Symmetric matrix.\n");
	}

	mat.assign(*this);
	if (format == Float)
	{
		const float *pj, *pi;
		float buf, matjj;
		__m128 _A, _B, _C;
		for (size_t j = 0; j < nRow; ++j)
		{
			matjj = 0;
			for (size_t i = j; i < nRow; ++i)
			{
				pj = (const float * )mat[j];
				pi = (const float * )mat[i];
				buf = 0;
				_C = _mm_set1_ps(0.0);
				size_t k;
				for (k = 4; k <= j; k += 4, pi += 4, pj += 4)
				{
					_A = _mm_loadu_ps(pj);
					_B = _mm_loadu_ps(pi);
					_B = _mm_mul_ps(_A, _B);
					_C = _mm_add_ps(_C, _B);
				}
				_C = _mm_hadd_ps(_C, _C);
				_C = _mm_hadd_ps(_C, _C);
				_mm_store_ss(&buf, _C);
				for (k -= 4; k < j; ++k)
				{
					buf += *pj++ * (*pi++);
				}

				if (i == j) // diag components
				{
					matjj = floatElement(j, j) - buf;
					if (matjj <= 0.0)
					{
						//CMSHPCTrace::Trace(0, TRACE_ERRORS, "Warning: CSymmetricMatrix::cholesky: the matrix is not positive.\n");
						//return E_UNEXPECTED;
					}
					matjj = mat.floatElement(j, j) = sqrt(matjj);
				}
				else // non-diag components
				{
					mat.floatElement(i, j) = (floatElement(i, j) - buf) / matjj;
				}
			}
		}
	}
	else // if (format == Double)
	{
		const double * pj, * pi;
		double buf, matjj;
		for (size_t j = 0; j < nRow; ++j)
		{
			pj = (const double * )mat[j];
			matjj = 0;
			for (size_t i = j; i < nRow; ++i)
			{
				pi = (const double * )mat[i];
				buf = 0;
				for (size_t k = 0; k < j; ++k)
				{
					buf += pj[k] * pi[k];
				}

				if (i == j) // diag components
				{
					matjj = doubleElement(j, j) - buf;
					if (matjj <= 0.0)
					{
					//	CMSHPCTrace::Trace(0, TRACE_ERRORS, "Warning: CSymmetricMatrix::cholesky: the matrix is not positive.\n");
					//	return E_UNEXPECTED;
					}
					matjj = mat.doubleElement(j, j) = sqrt(matjj);
				}
				else // non-diag components
				{
					mat.doubleElement(i, j) = (doubleElement(i, j) - buf) / matjj;
				}
			}
		}
	}
//	return S_OK;
}

void CMatrix::cholSub(const CMatrix &A, vector<float> &b, vector<float> &x, bool oneStep)
{
	assert(A.nCol > 0 && A.nRow > 0 && A.nRow == A.nCol && A.nCol == b.size());
	
	if (A.type != Symmetric || A.format != Float)
	{
	//	CMSHPCTrace::TraceHR(E_UNEXPECTED, "CMatrix::cholSub: A must be a float format lower triangular matrix which is the product of Cholesky decomposition.\n");
	}

	if (A.nRow != b.size())
	{
	//	CMSHPCTrace::TraceHR(E_UNEXPECTED, "CMatrix::cholSub: mismatch, A size = [%d, %d], b size = %d.\n", A.nRow, A.nRow, b.size());
	}
	CMatrix mat(Normal, A.format);
	A.ConvertTo(mat);
	float sum = 0, buf = 0;
	const float *pi, *px;
	x.assign(A.nRow, 0);
	__m128 _A, _B, _C;
	size_t k = 0;
	for (size_t i = 0; i < mat.nRow; ++i)
	{
		sum = b[i];
		pi = (const float *)mat[i];
		px = &x[0];
		_C = _mm_set1_ps(0.0);
		for (k = 4; k <= i; k += 4, pi += 4, px += 4)
		{
			_A = _mm_loadu_ps(pi);
			_B = _mm_loadu_ps(px);
			_B = _mm_mul_ps(_A, _B);
			_C = _mm_add_ps(_C, _B);
		}
		_C = _mm_hadd_ps(_C, _C);
		_C = _mm_hadd_ps(_C, _C);
		_mm_store_ss(&buf, _C);
		for (k -= 4; k < i; ++k, ++pi, ++px)
		{
			buf += *px * (*pi);	
		}
		sum -= buf;
		x[i] = sum / mat.floatElement(i,i);
	}

	if (oneStep == true)
	{
		return;
	}

	const float * pj;
	x[mat.nCol - 1] /= mat.floatElement(mat.nCol - 1, mat.nCol - 1);
	for (int j = (int)mat.nCol - 2; j >= 0; j--)
	{
		sum = x[j];
		pj = (const float *)mat[j];
		pj += j + 1;
		px = &x[j + 1];
		_C = _mm_set1_ps(0.0);
		for (k = j + 5; k < mat.nRow; k += 4, pj += 4, px += 4)
		{
			_A = _mm_loadu_ps(pj);
			_B = _mm_loadu_ps(px);
			_B = _mm_mul_ps(_A, _B);
			_C = _mm_add_ps(_C, _B);
		}
		_C = _mm_hadd_ps(_C, _C);
		_C = _mm_hadd_ps(_C, _C);
		_mm_store_ss(&buf, _C);
		for (k -= 4; k < mat.nRow; ++k)
		{
			buf += *px * (*pj);
			pj++;px++;
		}
		sum -= buf;
		x[j] = sum / mat.floatElement(j,j);
	}
}

void CMatrix::cholSub(const CMatrix &A, vector<double> &b, vector<double> &x, bool oneStep)
{
	assert(A.nCol > 0 && A.nRow > 0 && A.nRow == A.nCol && A.nCol == b.size());
	
	if (A.type != Symmetric || A.format != Double)
	{
		//CMSHPCTrace::TraceHR(E_UNEXPECTED, "CMatrix::cholSub: A must be a double format lower triangular matrix which is the product of Cholesky decomposition.\n");
	}

	if (A.nRow != b.size())
	{
		//CMSHPCTrace::TraceHR(E_UNEXPECTED, "CMatrix::cholSub: mismatch, A size = [%d, %d], b size = %d.\n", A.nRow, A.nRow, b.size());
	}
	CMatrix mat(Normal, A.format);
	A.ConvertTo(mat);
	double sum = 0, buf = 0;
	const double *pi, *px;
	x.assign(A.nRow, 0);
	size_t k = 0;
	for (size_t i = 0; i < mat.nRow; ++i)
	{
		sum = b[i];
		pi = (const double *)mat[i];
		px = &x[0];
		buf = 0;
		for (k = 0; k < i; ++k, ++pi, ++px)
		{
			buf += *px * (*pi);	
		}
		sum -= buf;
		x[i] = sum / mat.doubleElement(i,i);
	}

	if (oneStep == true)
	{
		return;
	}

	const double * pj;
	x[mat.nCol - 1] /= mat.doubleElement(mat.nCol - 1, mat.nCol - 1);
	for (int j = (int)mat.nCol - 2; j >= 0; j--)
	{
		sum = x[j];
		pj = (const double *)mat[j];
		pj += j + 1;
		px = &x[j + 1];
		buf = 0;
		for (k = j+1; k < mat.nRow; ++k)
		{
			buf += *px * (*pj);
			pj++;px++;
		}
		sum -= buf;
		x[j] = sum / mat.doubleElement(j,j);
	}
}

double CMatrix::logdetAfterChol()
{
	if (type != Symmetric)
	{
		//CMSHPCTrace::TraceHR(E_UNEXPECTED, "CMatrix::logdetAfterChol: only support symmetric matrix.\n");
	}

	double det = 0;
	if (format == Float)
	{
		for (size_t i = 0; i < nRow; ++i)
		{
			det += log(m_FloatMatrix[i*(i+3)/2]);
		}
	}
	else // if (format == Double)
	{
		for (size_t i = 0; i < nRow; ++i)
		{
			det += log(m_DoubleMatrix[i*(i+3)/2]);
		}
	}
	return (2*det);
}

// vecAugend = vecAugend + vecAddend
void CMatrix::VectorAddTo(vector<float> &vecAugend, const vector<float> &vecAddend)
{
	assert(vecAugend.size() == vecAddend.size() && vecAddend.size() != 0);

	if (vecAugend.empty())
	{
		vecAugend = vecAddend;
		return;
	}

	size_t size = vecAugend.size();
	__m128 _A, _B, _C;
	const float * padd = &vecAddend[0];
	float * p = &vecAugend[0];
	size_t i;
	for (i = 4; i <= size; i += 4, p += 4, padd += 4)
	{
		_A = _mm_loadu_ps(p);
		_B = _mm_loadu_ps(padd);
		_C = _mm_add_ps(_A, _B);
		_mm_storeu_ps(p, _C); 
	}
	for (i -= 4; i < size; ++i)
	{
		*p++ += *padd++;
	}
}

void CMatrix::VectorAddTo(vector<double> &vecAugend, const vector<double> &vecAddend)
{
	assert(vecAugend.size() == vecAddend.size() && vecAddend.size() != 0);

	if (vecAugend.empty())
	{
		vecAugend = vecAddend;
		return;
	}

	size_t size = vecAugend.size();
	const double *padd = &vecAddend[0];
	double *p = &vecAugend[0];
	for (size_t i = 0; i < size; ++i)
	{
		*p++ += *padd++;
	}
}

float CMatrix::DotProduct(vector<float> &vec1, vector<float> &vec2)
{
	assert(vec1.size() != 0 && vec1.size() == vec2.size());
	const float * p1 = &vec1[0];
	const float * p2 = &vec2[0];
	return DotProduct(p1, p2, vec1.size());
}

float CMatrix::DotProduct(const float *pvec1, const float *pvec2, size_t size)
{
	float val = 0;
	__m128 _A, _B, _C;
	size_t i;
	_C = _mm_set1_ps(0.0);
	for (i = 4; i <= size; i += 4, pvec1 += 4, pvec2 += 4)
	{
		_A = _mm_loadu_ps(pvec1);
		_B = _mm_loadu_ps(pvec2);
		_B = _mm_mul_ps(_A, _B);
		_C = _mm_add_ps(_C, _B); 
	}
	_C = _mm_hadd_ps(_C, _C);
	_C = _mm_hadd_ps(_C, _C);
	_mm_store_ss(&val, _C);
	for (i -= 4; i < size; ++i, ++pvec1, ++pvec2)
	{
		val += *pvec1 * (*pvec2);
	}
	return val;
}

double CMatrix::DotProduct(vector<double> &vec1, vector<double> &vec2)
{
	assert(vec1.size() != 0 && vec1.size() == vec2.size());
	const double * p1 = &vec1[0];
	const double * p2 = &vec2[0];
	return DotProduct(p1, p2, vec1.size());
}

double CMatrix::DotProduct(const double *pvec1, const double *pvec2, size_t size)
{
	double val = 0;
	size_t i;
	for (i = 0; i < size; ++i, ++pvec1, ++pvec2)
	{
		val += *pvec1 * (*pvec2);
	}
	return val;
}

/// private functions
void CMatrix::setMatrixFormat(const char* property_1, const char* property_2)
{
	const char *str[2] = {property_1, property_2};
	for (size_t i = 0; i < 2; ++i)
	{
		if (str[i] == NULL)
		{
			return;
		}
		else
		{
			if (strcmp(str[i], "normal") == 0)
			{
				type = Normal;
			}
			else if (strcmp(str[i], "symmetric") == 0)
			{
				type = Symmetric;
			}
			else if (strcmp(str[i], "float") == 0)
			{
				format = Float;
			}
			else if (strcmp(str[i], "double") == 0)
			{
				format = Double;
			}
			else
			{
			//	CMSHPCTrace::TraceHR(E_INVALIDARG, "CMatrix::SetMatrixFormat: unknown matrix format: %s", str);
			}
		}
	}
	
}
