#include "Common.h"

/* SwapInt32: swap byte order of int32 data value *p */
void Swap32(int *p)
{
   char temp,*q;
   
   q = (char*) p;
   temp = *q; *q = *(q+3); *(q+3) = temp;
   temp = *(q+1); *(q+1) = *(q+2); *(q+2) = temp;
}

/* SwapShort: swap byte order of short data value *p */
void Swap16(short *p)
{
   char temp,*q;
   
   q = (char*) p;
   temp = *q; *q = *(q+1); *(q+1) = temp;
}

/* EXPORT->LAdd: Return sum x + y on log scale, 
                sum < LSMALL is floored to LZERO */
double LAdd(double x, double y)
{
   double temp,diff,z;
   
   if (x<y) {
      temp = x; x = y; y = temp;
   }
   diff = y-x;
   if (diff<minLogExp) 
      return  (x<LSMALL)?LZERO:x;
   else {
      z = exp(diff);
      return x+log(1.0+z);
   }
}

/* itoa: convert int to string*/
std::string itoa(const int i)
{
	char buff[65];
	_itoa_s(i,buff,65,10);
	std::string str(buff, strlen(buff));
	return str;
}

/* ReplaceSubstr: relpace srcSubstr in srcStr with dstSubstr.*/
std::string ReplaceSubstr(const std::string srcStr, const std::string srcSubstr, const std::string dstSubstr)
{
	std::string::size_type pos, nextpos, oldlen, newlen;
	std::string dstStr = srcStr;
	nextpos = 0;
	oldlen = srcSubstr.size();
	newlen = dstSubstr.size();
	while((pos = dstStr.find(srcSubstr, nextpos)) != std::string::npos)
	{
		dstStr.replace(pos, oldlen, dstSubstr);
		nextpos = pos + newlen;
	}
	return dstStr;
}

/*ConcatenateFileFullPath: file full path <= path + filename + extension*/
std::string ConcatenateFileFullPath(const char * pszPath, const char * pszName, const char * pszExt)
{
	std::string file;
	file.clear();
	if (pszPath != NULL)
	{
		std::string path(pszPath);
		path = ReplaceSubstr(path, "/", "\\");
		if (pszName != NULL && path[path.size() - 1] != '\\')
		{
			path += '\\';
		}
		file += path;
	}
	if (pszName != NULL)
	{
		std::string name(pszName);
		file += name;
	}
	else if (pszExt != NULL)
	{
		return file;
	}
	if (pszExt != NULL)
	{
		std::string ext(pszExt);
		file += "." + ext;
	}
	return file;
}