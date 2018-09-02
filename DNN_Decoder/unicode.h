
static std::string wchar2string(const wchar_t* pwch) 
{ 
	size_t origsize = wcslen(pwch) + 1; 
	size_t convertedChars = 0; 
	const size_t newsize = origsize*2; 
	char *nstring = new char[newsize]; 
	wcstombs_s(&convertedChars, nstring, newsize, pwch, _TRUNCATE); 
	return string(nstring); 
} 

static std::wstring char2wstring(const char* pch)
{
	size_t origSize = strlen(pch) + 1;
	size_t convertedWChars = 0;
	const size_t newSize = origSize * 2;
	wchar_t *nstring = new wchar_t[newSize];
	mbstowcs_s(&convertedWChars, nstring, newSize, pch, _TRUNCATE);
	return wstring(nstring);
}