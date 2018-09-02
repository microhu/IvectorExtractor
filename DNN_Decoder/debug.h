#pragma once

#include <time.h>

// #define _FULL_DEBUG_

#ifdef _FULL_DEBUG_
#define _BASIC_DEBUG_
#endif


#ifdef _BASIC_DEBUG_
//#define LOG0(...) do { time_t rawTime; struct tm* timeInfo; rawTime= time (0); timeInfo=localtime(&rawTime); fwprintf (outfile, L"[%ls] : " , _wasctime(timeInfo) ); fwprintf (outfile,  __VA_ARGS__);} while(0)
#define LOG0(...) do { clock_t time; time = clock(); fwprintf (outfile, L"[%f] : " , (float)time ); fwprintf (outfile,  __VA_ARGS__);fflush(outfile);} while(0)
#else
#undef _FULL_DEBUG_
#define LOG0(...)  ;
#endif

#ifdef _FULL_DEBUG_
#define LOG(...) do { time_t rawTime; struct tm* timeInfo; rawTime= time (0); timeInfo=localtime(&rawTime); fwprintf (outfile, L"[%ls] : ", _wasctime(timeInfo) ); fwprintf (outfile,  __VA_ARGS__);fflush(outfile);} while(0)
#else
#define LOG(...)  ;
#endif

