#pragma once
#include <math.h>
#include <string>

const int LINEMAX = 2048;
const double TPI = 6.28318530717959;     /* PI*2 */
const double LZERO = (-1.0E10);   /* ~log(0) */
const double ZERO = (1.0E-10);   
const double LSMALL = (-0.5E10);   /* log values < LSMALL are set to LZERO */
const double minLogExp = -log(-LZERO);
const double MINLARG = 2.45E-308;  /* lowest log() arg  = exp(MINEARG) */

// struct to pass info to threads



/* SwapInt32: swap byte order of int32 data value *p */
void Swap32(int *p);

/* SwapShort: swap byte order of short data value *p */
void Swap16(short *p);

/* EXPORT->LAdd: Return sum x + y on log scale, 
                sum < LSMALL is floored to LZERO */
double LAdd(double x, double y);

/* itoa: convert int to string*/
std::string itoa(const int i);

/* ReplaceSubstr: relpace srcSubstr in srcStr with dstSubstr.*/
std::string ReplaceSubstr(const std::string srcStr, const std::string srcSubstr, const std::string dstSubstr);

/*ConcatenateFileFullPath: file full path <= pszPath + "\\" + pszName + "." + pszExt*/
std::string ConcatenateFileFullPath(const char * pszPath, const char * pszName, const char * pszExt);

