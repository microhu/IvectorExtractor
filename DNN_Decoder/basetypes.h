// 
// basetypes.h - basic types that C++ lacks
// 
// Copyright (c) 2002, Microsoft Corporation. All rights reserved.
// 
// $Log: /Speech_To_Speech_Translation/dbn/dbn/basetypes.h $
// 
// 228   4/06/12 11:29 Fseide
// (edited a comment)
// 
// 227   11/01/11 10:55 Fseide
// added a todouble(const char *) to avoid conversion to std::string
// 
// 226   6/10/11 16:16 Fseide
// now uses stod() only in VS 2010+
// 
// 225   5/23/11 11:03a Fseide
// resurrected the call to stod()
// 
// 224   5/13/11 1:30p Kit
// fixed compiler warning
// 
// 223   5/13/11 1:00p Kit
// seems stod doesn't exist in kit's vs2010 - changed #ifdef - needs frank
// to review
// 
// 222   5/13/11 12:56p Kit
// updated todouble implementation for vs2005
// 
// 221   5/13/11 12:51p Kit
// fixed stod compatibility with vs2005
// 
// 220   5/13/11 12:38p Kit
// stod => rolled back to previous version and added std::
// 
// 219   5/12/11 2:50p V-weiwuz
// Add complie wrap to the function stod(), in order to make it can be
// complied in VS 2005.
// 
// 218   4/25/11 11:05a F-gli
// add todouble function: from string to double
// 
// 217   4/11/11 4:47p Fseide
// added matrix::size()  --not so nice!
// 
// 216   1/07/11 22:16 Fseide
// added a comment to document a potential perf bug
// 
// 215   11/30/10 4:06p Fseide
// fixed an error message in auto_file_ptr
// 
// 214   11/23/10 15:28 Fseide
// added another variant of toint()--really need to clean that part up
// 
// 213   11/19/10 8:24 Fseide
// (one more int vs. size_t fix...)
// 
// 212   11/18/10 10:03a Fseide
// (added overloads to deal with ideosyncrasies fixed_vector considering
// indices as 'int' rather than 'size_t'
// 
// 211   11/17/10 16:06 Fseide
// two new methods in msra::strfun, toint() and todouble()
// 
// 210   11/12/10 11:36 Fseide
// moved matrix loop macros out again as to not pollute this file
// 
// 209   11/12/10 9:06 Fseide
// added foreach_ macros for matrix type
// 
// 208   11/09/10 8:55 Fseide
// split out new function bytereverse() from byteswap()
// 
// 207   11/08/10 15:09 Fseide
// new util function byteswap() to swap an entire array
// 
// 206   11/08/10 12:38 Fseide
// auto_file_ptr etc. moved more towards the end;
// new class textreader for reading a text file line by line
// 
// 205   11/08/10 10:57 Fseide
// added auto_file_ptr::swap()
// 
// 204   11/04/10 9:22p Fseide
// added matrix::elemtype
// 
// 203   9/21/10 3:16p V-jimji
// template<class T,class FR = void> class auto_clean. add class FR = void
// 
// 202   8/26/10 20:40 Fseide
// replaced some outdated (void) argument-list syntax by ()
// 
// 201   8/26/10 20:39 Fseide
// ||throw_hr() now returns the original HRESULT (suggested in Win32Prg by
// Stewart Tootill);
// throw_hr() now has a default (NULL);
// added mappings for runtime_error and logic_error to catch_hr_return
// 
// 200   8/25/10 12:47 Fseide
// added functions CoTaskMemString() and ZeroStruct()
// 
// 199   8/25/10 11:16 Fseide
// added class noncopyable, catch_hr_return macro, and class auto_co_ptr
// 
// 198   8/18/10 6:03p V-jimji
// handling for compiling under /clr:pure
// 
// 197   8/18/10 5:52p V-jimji
// undid remove of stdcalls - seems we actually need this
// 
// 195   7/27/10 7:10 Fseide
// (added throw_hr in addition to bad_hr, is slightly more efficient)
// 
// 194   7/02/10 10:40a Kit
// fixed for wince platform
// 
// 193   6/11/10 12:03p Kit
// reverted tokenized strtok loop to explicitly using a context variable
// to avoid using TLS (perf hit)
// 
// 192   6/11/10 11:29a Kit
// tidied up comments regarding intrin.h
// 
// 191   6/11/10 11:16a Kit
// temporarily excluded frank's changes for fixing strlen intrinsicts
// using compiler version #ifdef - TODO, frank will find the right way to
// do this
// refactored wince code duplication in tls class
// 
// 190   6/11/10 10:22 Fseide
// 
// 189   11/30/09 1:28p Kit
// updated to compile under wince 5.0
// 
// 188   7/17/09 17:26 Fseide
// added a missing 'const' in class matrix
// 
// 187   6/16/09 11:17 Fseide
// redone the auto_timer
// 
// 186   6/16/09 8:41 Fseide
// (changed a comment about auto_timer)
// 
// 185   6/16/09 8:29 Fseide
// new class auto_timer to provide a super-simple runtime (wallclock)
// measurement facility
// 
// 184   6/14/09 21:18 Fseide
// fixed_vector: added a (duplicate?) assignment operator, because the
// templated was did not always chosen for unknown reasons
// 
// 183   6/10/09 17:53 Fseide
// bug fix: avoid div/0 error in matrix::cols()
// 
// 182   6/10/09 9:16 Fseide
// added a 'const' version of fixed_vector::begin()
// 
// 181   6/08/09 21:40 Fseide
// added a templated copy constructor to fixed_vector
// 
// 180   6/08/09 7:30 Fseide
// new member matrix::empty
// 
// 179   6/05/09 15:28 Fseide
// moved class matrix here
// 
// 178   6/04/09 10:09 Fseide
// (compacted fixed_vector a bit)
// 
// 177   6/03/09 15:44 Fseide
// new member fixed_vector::indexof()
// 
// 176   6/02/09 17:59 Fseide
// added missing parentheses to foreach_index() macro
// 
// 175   6/02/09 16:59 Fseide
// fixed_vector: operator= now takes anything with [] and size(), and
// constructor is now robust to size 0 (will skip the 'new')
// 
// 174   5/31/09 16:19 Fseide
// added a missing namespace in the tokenizer class
// 
// 173   5/31/09 16:03 Fseide
// moved class tokenizer here
// 
// 172   5/22/09 7:24 Fseide
// (CCritSec: removed an unnecessary 'protected')
// 
// 171   5/18/09 15:47 Fseide
// disabled an incorrect code-analysis warning for SetCurrentThreadName()
// 
// 170   5/14/09 18:18 Fseide
// moved auto_co_initialize to basetypes.h
// 
// 169   5/11/09 11:56a Kit
// fixed bug in auto_clean comments
// 
// 168   5/11/09 11:56a Kit
// fixed compile bug in auto_clean
// 
// 167   5/11/09 11:44a Kit
// renamed auto_dll_ptr to auto_clean
// 
// 166   09/05/11 10:46 Gocheng
// Missing files in previous check-in: Client changes for recording only
// mode
// 
// 165   09-05-06 11:21 Llu
// extended auto_handle into a template class to specify the handle type
// 
// 164   4/14/09 10:37 Fseide
// (added two typecasts to avoid a compiler error with a stricter
// compiler)
// 
// 163   19/01/09 6:21p Kit
// undid SECURE_SCL=0 for release as it seems it causes crashes
// 
// 162   19/01/09 5:19p Kit
// removed accidentally included #define for SECURE_SCL
// 
// 161   19/01/09 5:14p Kit
// updated SECURE_SCL check
// 
// 160   19/01/09 4:15p Kit
// fixed unbalanced #if #endif
// 
// 159   19/01/09 4:11p Kit
// added check for release mode to validate that _SECURE_SCL==0
// 
// 158   19/01/09 3:06p Kit
// disabled SECURE_SCL=0 for release mode.  Instead, its now added to the
// project settings for each project.
// 
// 157   19/01/09 12:43p Kit
// undoing accidental checkin #156 - now same as checkin #155
// 
// 156   19/01/09 12:06p Kit
// 
// 155   1/12/09 18:27 Fseide
// disabled _SECURE_SCL in Release mode to avoid array-bounds checks
// 
// 154   1/09/09 7:41 Fseide
// (fixed a warning)
// 
// 153   1/08/09 16:46 Fseide
// oops, last fix was incorrect, a Debugging artifact...
// 
// 152   1/08/09 16:38 Fseide
// bug fix in command_line constructor: now correctly skips the first
// _two_ arguments (path and command name)
// 
// 151   1/08/09 16:13 Fseide
// auto_file_ptr now handles stdin/out/err correctly (does not close them)
// 
// 150   1/08/09 15:33 Fseide
// added auto_handle and FormatWin32Error()
// 
// 149   1/08/09 11:07 Fseide
// added a simple helper for processing the command line arguments
// 
// 148   1/08/09 10:59 Fseide
// finally added split() and join() to msra::strfun namespace
// 
// 147   1/08/09 9:24 Fseide
// (changed a comment)
// 
// 146   1/07/09 8:52 Fseide
// new function cacpy() to replace strncpy() when applied to fixed-size
// character arrays that are not supposed to be 0-terminated C strings
// 
// 145   12/29/08 8:50 Fseide
// cleaned up the "unsafe" overloads section
// 
// 144   12/26/08 11:00 Fseide
// (added a typecast)
// 
// 143   12/26/08 8:48 Fseide
// new function SetCurrentThreadName()
// 
// 142   12/11/08 7:37p Qiluo
// refine wrappers of _vsnprintf & _vsnwprintf
// 
// 141   12/11/08 3:31p Qiluo
// refine some wrapper functions
// 
// 140   12/11/08 2:55p Qiluo
// (fix a comment, and reformat code)
// 
// 139   12/10/08 4:12p Qiluo
// updated comments for tricky templates for strlen and wcslen
// 
// 138   12/10/08 3:26p Qiluo
// refine wcsncpy wrapper
// 
// 137   12/10/08 12:32p Qiluo
// refine strerror wrapper with lock
// 
// 136   12/09/08 7:00p Qiluo
// reverted stringerror to strerror_ but now use static map
// 
// 135   12/09/08 6:36p Qiluo
// fixed some compiler bug
// 
// 134   12/09/08 6:31p Qiluo
// changed __declspec(thread) to win32 tls functions because the first is
// broken on pre-vista OSs for DLLs.  Also removed strerror_ function,
// replaced by stringerror
// 
// 133   12/02/08 10:44a Qiluo
// add swscanf wrapper
// 
// 132   12/01/08 5:56p Qiluo
// add wcslen wrappers
// 
// 131   12/01/08 2:32p Qiluo
// add fscanf wrapper
// 
// 130   11/28/08 6:00p Qiluo
// refine strlen wrapper as in $/Audio_Indexing/api/lib/hapivitelib/hapi.h
// (v49 v50)
// 
// 129   11/28/08 4:08p Qiluo
// remove unused macro
// 
// 128   11/28/08 11:56a Qiluo
// rewrite strlen wrapper
// 
// 127   11/25/08 1:19p Yushi
// avoided warning C4706
// 
// 126   11/19/08 5:44p Qiluo
// temporary solution to make this header self-contrain
// 
// 125   11/19/08 2:08p Qiluo
// rewrite strlen wrapper
// 
// 124   11/18/08 4:13p Qiluo
// remove wcsncpy wrapper, use our own function
// 
// 123   11/18/08 11:03 Fseide
// fixed overload of fopen() to map to _fsopen() instead of fopen_s() to
// recreate more similar semantics (read-only shared access);
// included string.h to ensure no conflict with our mapping tricks
// 
// 122   11/17/08 4:12p Qiluo
// remove unused markers
// 
// 121   11/14/08 7:43p Qiluo
// add strlen wrapper
// 
// 120   11/14/08 18:37 Fseide
// added variants of sscanf() that map to sscanf_s() (without adding
// value)
// 
// 119   11/14/08 5:33p Qiluo
// replace sscanf wrapper with a simple one
// 
// 118   11/14/08 16:51 Fseide
// (fixed some string-function templates)
// 
// 117   11/14/08 3:26p Qiluo
// add wrapper for _vsnwprintf
// 
// 116   11/14/08 2:57p Qiluo
// add wrapper for wcstok, _vsnprintf, wcsncpy
// 
// 115   11/14/08 12:05p Qiluo
// (add comment)
// 
// 114   11/14/08 11:47a Qiluo
// add unlink marker
// 
// 113   11/14/08 11:24a Qiluo
// add sscanf wrapper
// 
// 112   11/11/08 18:57 Fseide
// (fixed a comment)
// 
// 111   11/11/08 18:48 Fseide
// fixed fopen_s() usage
// 
// 110   11/11/08 18:35 Fseide
// changed definition of "overloads" to avoid compile errors if users use
// the same function names
// 
// 109   11/11/08 18:28 Fseide
// added backwards-compatible "overloads" for fopen(), _wfopen(), and
// strerror(), which are not really "unsafe"
// 
// 108   11/11/08 17:56 Fseide
// removed strbxxx() calls, to be replaced by "safe" fixed-buffer
// overloads
// 
// 107   11/11/08 17:50 Fseide
// (a comment added)
// 
// 106   11/11/08 17:36 Fseide
// added an #undef for _CRT_SECURE_CPP_OVERLOAD_STANDARD_NAMES to allow
// multiple defines
// 
// 105   11/11/08 17:32 Fseide
// set the correct "safe" flag (_CRT_SECURE_CPP_OVERLOAD_STANDARD_NAMES)
// and got rid of vsprintf()
// 
// 104   11/11/08 17:26 Fseide
// added vsprintf() overload but somehow not recognized if here
// 
// 103   11/10/08 19:43 Fseide
// removed 'const' from strtok() as some code modifies the result (but
// found a few missing 'const' in code this way)
// 
// 102   11/10/08 19:32 Fseide
// trying 'const' return value from strtok()
// 
// 101   11/10/08 19:11 Fseide
// changed [w]strprintf() to use "safe" vprintf() version
// 
// 100   11/10/08 18:52 Fseide
// added "safe" overloads for strlen() and strtok() (the latter by means
// of a macro)
// 
// 99    11/10/08 17:59 Fseide
// (partial new code for strprintf(), not active)
// 
// 98    18/07/08 10:33a Kit
// rolledback to version 96
// 
// 96    6/18/08 18:34 Fseide
// new method fill()
// 
// 95    6/18/08 11:41 Fseide
// added #pragma once
// 
// 94    6/18/08 10:52 Fseide
// moved functional-programming style macros (foreach_index, map_array,
// reduce_array) to basetypes.h
// 
// 93    7/12/07 9:18 Fseide
// added size constructor to ARRAY, now can do ARRAY<t> x (size)
// 
// 92    5/18/07 6:53p Rogeryu
// bug fix for utf8
// 
// 91    4/02/07 15:06 Fseide
// new method wstrprintf()
// 
// 90    3/29/07 10:24a Rogeryu
// bug fix
// 
// 89    3/28/07 17:11 Fseide
// bad_hr now based on std::exception
// 
// 88    3/28/07 13:09 Fseide
// (cosmetic change)
// 
// 87    3/27/07 19:52 Fseide
// (fixed an error message)
// 
// 86    3/27/07 19:31 Fseide
// wcstombs() and mbstowcs() changed to inline to get rid of 'unused
// static function removed' warning
// 
// 85    3/27/07 17:58 Fseide
// CUnicodeHelper replaced by functions defined in namespace msra::strfun
// 
// 84    2/14/07 15:38 Fseide
// (fixed compiler warnings when compiling managed)
// 
// 82    2/14/07 14:04 Fseide
// #defined _CRT_SECURE_NO_DEPRECATE, as this is Microsoft nonsense
// 
// 81    1/22/07 15:01 Fseide
// new method strprintf();
// renamed hr_exception to bad_hr (should also base it on
// runtime_exception)
// 
// 80    12/06/06 21:54 Fseide
// added operator->() to auto_file_ptr such that the macros like feof()
// also work
// 
// 79    11/28/06 20:18 Fseide
// new class bad_hr
// 
// 78    11/28/06 12:22 Fseide
// moved the disabling of 4701 (local var may be used w/o init) back to
// cpp file as this is too serious a warning to globally disable, as
// pointed out by Kit
// 
// 77    11/27/06 13:55 Fseide
// disabled another warning (caused by a compiler bug in VS.NET 2003) when
// compiling under Managed Extensions
// 
// 76    11/27/06 11:41 Fseide
// added constructor to auto_file_ptr to allow auto_file_ptr f = fopen()
// 
// 75    10/28/06 16:47 Fseide
// 
// 74    10/28/06 16:40 Fseide
// changed global fclose(auto_file_ptr) to a static to avoid link problems
// 
// 73    10/28/06 16:31 Fseide
// (a comment added)
// 
// 72    10/28/06 16:24 Fseide
// new class auto_file_ptr to mimic FILE* but with automatic fclose() at
// destruction time
// 
// 71    8/27/06 0:01 Fseide
// implemented proper swap() functions for ARRAY and fixed_vector (the
// default uses assignments and therefore does not guarantee nothrow and
// is also not efficient)
// 
// 70    5/14/06 17:33 Fseide
// added fixed_vector::operator=
// 
// 69    5/14/06 12:27 Fseide
// new method fixed_vector<>::clear()
// 
// 68    4/29/06 6:49p Yushli
// separated the #ifdef check for OACR_ASSUME as it does not work
// otherwise with oacr.h (seems to be inconsistent)
// 
// 67    4/26/06 10:29 Fseide
// fixed_vector<> can now be sized after instantiation
// 
// 66    4/20/06 7:49 Fseide
// new dummy macro OACR_ASSUME if not compiling using OACR
// 
// 65    4/13/06 2:17p Fseide
// new simple class fixed_vector
// 
// 64    3/24/06 5:27p Rogeryu
// disable 2 oacr warnings in a newly modified code piece
// 
// 63    3/24/06 2:59p Rogeryu
// changed ASSERT macro in _CHECKED configuration to something different
// to avoid an internal compiler error in VS 2003 (ugh!!)
// 
// 62    3/24/06 13:01 Fseide
// changed SAFE_DELETE() into something that has no OACR warning
// 
// 61    06-03-24 12:13 Llu
// fixed a bug in the wide-string version of the macros
// 
// 60    3/24/06 11:58 Fseide
// (a comment changed)
// 
// 59    3/24/06 11:49 Fseide
// strbcpy etc macros replaced by C++ template functions to guarantee one
// cannot pass in a char ptr as target
// 
// 58    3/22/06 9:14p Rogeryu
// review and suppress one oacr warning
// 
// 57    3/22/06 8:00p Fseide
// (a comment added)
// 
// 56    3/22/06 7:32p Fseide
// fixed a dummy SAL
// 
// 55    3/22/06 6:39p Fseide
// moved back-off #defines for SAL annotations and __override behind the
// #includes, such that we pick up sal support if present;
// changed condition for #define for some SAL annotations that seem not to
// be part of VS 2005
// 
// 54    3/22/06 5:45p Rogeryu
// eliminated strsafe's deprecation of string functions, this is managed
// through a different process
// 
// 53    3/22/06 5:32p Rogeryu
// added safe string macros
// 
// 52    3/22/06 4:57p Rogeryu
// refine comments
// 
// 51    3/21/06 6:46p Rogeryu
// review and fix oacr level2_security warnings
// 
// 50    3/21/06 5:21p Rogeryu
// review and fix level2_security OACR warnings
// 
// 49    3/21/06 11:11a Rogeryu
// work with OACR warnings
// 
// 48    3/20/06 22:00 Fseide
// 
// 47    06-03-17 17:03 Yushli
// comment out global #pragma warning (disable : 4996) since we are
// suppress this warning per function
// 
// 46    06-03-14 16:53 Yushli
// 
// 45    06-02-28 16:21 Kjchen
// 
// 44    06-02-24 20:11 Kjchen
// add defines for OACR
// 
// 43    2/24/06 8:03p Kjchen
// depress oacr warnings
// 
// 42    2/22/06 11:54a Yushli
// disable warning 4702 also in checked version
// 
// 41    11/15/05 9:15p Fseide
// disabled 'XXX() deprecated' warning, as it is impossible for us to fix
// all at this stage, and we know what we're doing
// 
// 40    10/14/05 10:22 Fseide
// removed two ASSERTions because it is hard to make it compile in Checked
// mode due to dependency on message.h (yak...)
// 
// 39    10/13/05 9:49p Rogeryu
// temporarily comment out the line that complains
// 
// 38    10/13/05 9:40p Rogeryu
// remove a compiling error by include "message.h"
// 
// 37    10/13/05 18:41 Fseide
// UTF8 and UTF16 finally fixed to really use the UTF-8 code page
// 
// 36    8/25/05 10:16a Kjchen
// fix comments
// 
// 35    8/25/05 10:10a Kjchen
// merge changes in OneNote
// 
// 34    5/16/05 12:39 Fseide
// disabled warning 4996 about use of deprecated functions -- we use the
// std C lib and we know what we are doing
// 
// 33    5/10/05 14:57 Fseide
// disabling of cast accuracy-loss warning removed
// 
// 32    5/10/05 12:15 Fseide
// (disabled unreachable-code warning in Release mode, as these are caused
// by STL)
// 
// 31    5/10/05 11:45 Fseide
// removed disabling of '<' signed/unsigned warning
// 
// 30    5/10/05 11:34 Fseide
// added an overload of size() to ARRAY that returns int instead of
// unsigned int, to avoid hundreds of meaningless typecasts
// 
// 29    5/09/05 18:36 Fseide
// CAutoNewHandler removed
// 
// 28    5/09/05 18:34 Fseide
// disabled one warning again as it broke another lib
// 
// 27    5/09/05 18:32 Fseide
// two #pragma disable warning removed
// 
// 26    5/09/05 15:21 Fseide
// CAutoLock: added default hidden constructor and assignment operator
// 
// 25    4/04/05 12:34 Fseide
// now understands new _CHECKED #define, which enables all ASSERTions
// while compiling with optimization
// 
// 24    3/17/05 4:38 Fseide
// (removed some #ifdef'ed-out garbage)
// 
// 23    3/17/05 4:34 Fseide
// CUnicodeHelper class: changed STRING back to std::string, WSTRING
// accordingly (because, what's the point?)
// 
// 22    2/20/05 17:27 Fseide
// uncommented code to make copy constructor and assignment operator
// inaccessible, because now I understand what this is for...
// 
// 21    11/13/04 18:33 Fseide
// CCritSec and CAutoLock moved to basetypes.h, critsec.h is now empty and
// obsolete
// 
// 20    8/08/03 14:40 Fseide
// (another warning disabled)
// 
// 19    7/30/03 5:10p Fseide
// new methods in CUnicodeHelper for UTF conversion (currently fake...)
// 
// 18    7/30/03 1:21p Fseide
// 0-terminator handling in CUnicodeHelper fixed according to spec of
// conversion functions
// 
// 17    7/30/03 1:10p Fseide
// (forgot 'public')
// 
// 16    7/30/03 1:05p Fseide
// added class CUnicodeHelper
// 
// 15    03-07-25 1:11 I-rogery
// first version of latgen: simple recognition program
// 
// 14    7/23/03 1:01p Fseide
// CAutoNewHandler changed to use ANSI C++ standard set_new_handler()
// mechanism
// 
// 13    7/23/03 12:49p Fseide
// new class CAutoNewHandler
// 
// 12    7/06/03 10:32p Fseide
// #define ASSERT surrounded by #ifndef ASSERT
// 
// 11    3/04/03 2:10p Fseide
// disabled another warning which seems to be wrong
// 
// 10    3/04/03 1:57p Fseide
// two warnings disabled (comparing signed/unsigned and conversion
// double->int)
// 
// 9     6/12/02 10:29p Jlzhou
// corrected definition of operator[] (now always reference)
// 
// 8     6/12/02 9:03p Jlzhou
// in DEBUG build, ARRAY<x> now implements array-bounds checking
// 
// 7     6/07/02 7:23p Fseide
// defined STRING
// 
// 6     4/03/02 3:57p Fseide
// VSS keyword and copyright added
// 
// F. Seide 5 Mar 2002
// 

#pragma once
#ifndef _BASETYPES_
#define _BASETYPES_

// [kit]: seems SECURE_SCL=0 doesn't work - causes crashes in release mode
// there are some complaints along this line on the web
// so disabled for now
//
//// we have agreed that _SECURE_SCL is disabled for release builds
//// it would be super dangerous to mix projects where this is inconsistent
//// this is one way to detect possible mismatches
//#ifdef NDEBUG
//#if !defined(_CHECKED) && _SECURE_SCL != 0 
//#error "_SECURE_SCL should be disabled for release builds"
//#endif
//#endif

#ifndef UNDER_CE    // fixed-buffer overloads not available for wince
#ifdef _CRT_SECURE_CPP_OVERLOAD_STANDARD_NAMES  // fixed-buffer overloads for strcpy() etc.
#undef _CRT_SECURE_CPP_OVERLOAD_STANDARD_NAMES
#endif
#define _CRT_SECURE_CPP_OVERLOAD_STANDARD_NAMES 1
#endif

#pragma warning (push)
#pragma warning (disable: 4793)    // caused by varargs

// disable certain parts of basetypes for wince compilation
#ifdef UNDER_CE
#define BASETYPES_NO_UNSAFECRTOVERLOAD // disable unsafe CRT overloads (safe functions don't exist in wince)
#define BASETYPES_NO_STRPRINTF         // dependent functions here are not defined for wince
#endif

#ifndef OACR    // dummies when we are not compiling under Office
#define OACR_WARNING_SUPPRESS(x, y)
#define OACR_WARNING_DISABLE(x, y)
#define OACR_WARNING_PUSH
#define OACR_WARNING_POP
#endif
#ifndef OACR_ASSUME	// this seems to be a different one
#define OACR_ASSUME(x)
#endif

// following oacr warnings are not level1 or level2-security
// in currect stage we want to ignore those warnings
// if necessay this can be fixed at later stage

// not a bug
OACR_WARNING_DISABLE(EXC_NOT_CAUGHT_BY_REFERENCE, "Not indicating a bug or security threat.");
OACR_WARNING_DISABLE(LOCALDECLHIDESLOCAL, "Not indicating a bug or security threat.");

// not reviewed
OACR_WARNING_DISABLE(MISSING_OVERRIDE, "Not level1 or level2_security.");
OACR_WARNING_DISABLE(EMPTY_DTOR, "Not level1 or level2_security.");
OACR_WARNING_DISABLE(DEREF_NULL_PTR, "Not level1 or level2_security.");
OACR_WARNING_DISABLE(INVALID_PARAM_VALUE_1, "Not level1 or level2_security.");
OACR_WARNING_DISABLE(VIRTUAL_CALL_IN_CTOR, "Not level1 or level2_security.");
OACR_WARNING_DISABLE(POTENTIAL_ARGUMENT_TYPE_MISMATCH, "Not level1 or level2_security.");

// determine WIN32 api calling convention
// it seems this is normally stdcall?? but when compiling as /clr:pure or /clr:Safe
// this is not supported, so in this case, we need to use the 'default' calling convention
// TODO: can we reuse the #define of WINAPI??
#ifdef _M_CEE_SAFE 
#define WINAPI_CC __clrcall
#elif _M_CEE
#define WINAPI_CC __clrcall
#else
#define WINAPI_CC __stdcall
#endif

// fix some warnings in STL
#if !defined(_DEBUG) || defined(_CHECKED) || defined(_MANAGED)
#pragma warning(disable : 4702) // unreachable code
#endif

#include <stdio.h>
#include <string.h>     // include here because we redefine some names later
#include <string>
#include <vector>
#include <cmath>        // for HUGE_VAL
#include <tchar.h>
#include <assert.h>
#include <map>
#include <windows.h>    // for CRITICAL_SECTION
#pragma push_macro("STRSAFE_NO_DEPRECATE")
#define STRSAFE_NO_DEPRECATE    // deprecation managed elsewhere, not by strsafe
#include <strsafe.h>    // for strbcpy() etc templates
#pragma pop_macro("STRSAFE_NO_DEPRECATE")

// CRT error handling seems to not be included in wince headers
// so we define our own imports
#ifdef UNDER_CE

// TODO: is this true - is GetLastError == errno?? - also this adds a dependency on windows.h
#define errno GetLastError() 

// strerror(x) - x here is normally errno - TODO: make this return errno as a string
#define strerror(x) "strerror error but can't report error number sorry!"
#endif

#ifndef __in // dummies for sal annotations if compiler does not support it
#define __in
#define __inout_z
#define __in_count(x)
#define __inout_cap(x)
#define __inout_cap_c(x)
#endif
#ifndef __out_z_cap	// non-VS2005 annotations
#define __out_cap(x)
#define __out_z_cap(x)
#define __out_cap_c(x)
#endif

#ifndef __override      // and some more non-std extensions required by Office
#define __override virtual
#endif

// disable warnings for which fixing would make code less readable
#pragma warning(disable : 4290) // throw() declaration ignored
#pragma warning(disable : 4244) // conversion from typeA to typeB, possible loss of data

// ----------------------------------------------------------------------------
// basic macros
// ----------------------------------------------------------------------------

#define SAFE_DELETE(p)  { if(p) { delete (p); (p)=NULL; } }
#define SAFE_RELEASE(p) { if(p) { (p)->Release(); (p)=NULL; } }     // nasty! use CComPtr<>
#ifndef ASSERT
#ifdef _CHECKED // basetypes.h expects this function to be defined (it is in message.h)
extern void _CHECKED_ASSERT_error(const char * file, int line, const char * exp);
#define ASSERT(exp) ((exp)||(_CHECKED_ASSERT_error(__FILE__,__LINE__,#exp),0))
#else
#define ASSERT assert
#endif
#endif

// ----------------------------------------------------------------------------
// basic data types
// ----------------------------------------------------------------------------

namespace msra { namespace basetypes {

// class ARRAY -- std::vector with array-bounds checking
// VS 2008 and above do this, so there is no longer a need for this.

template<class _ElemType>
class ARRAY : public std::vector<_ElemType>
{
#if defined (_DEBUG) || defined (_CHECKED)	// debug version with range checking
    static void throwOutOfBounds()
    {   // (moved to separate function hoping to keep inlined code smaller
        OACR_WARNING_PUSH;
        OACR_WARNING_DISABLE(IGNOREDBYCOMMA, "Reviewd OK. Special trick below to show a message when assertion fails"
            "[rogeryu 2006/03/24]");
        OACR_WARNING_DISABLE(BOGUS_EXPRESSION_LIST, "This is intentional. [rogeryu 2006/03/24]");
        ASSERT (("ARRAY::operator[] out of bounds", false));
        OACR_WARNING_POP;
    }
#endif

public:

    ARRAY() : std::vector<_ElemType> () { }
    ARRAY (int size) : std::vector<_ElemType> (size) { }

#if defined (_DEBUG) || defined (_CHECKED)	// debug version with range checking
    // ------------------------------------------------------------------------
    // operator[]: with array-bounds checking
    // ------------------------------------------------------------------------

    inline _ElemType & operator[] (int index)		    // writing
    {
        if (index < 0 || index >= size()) throwOutOfBounds();
        return (*(std::vector<_ElemType>*) this)[index];
    }

    // ------------------------------------------------------------------------

    inline const _ElemType & operator[] (int index) const	// reading
    {
        if (index < 0 || index >= size()) throwOutOfBounds();
        return (*(std::vector<_ElemType>*) this)[index];
    }
#endif

    // ------------------------------------------------------------------------
    // size(): same as base class, but returning an 'int' instead of 'size_t'
    // to allow for better readable code
    // ------------------------------------------------------------------------

    inline int size() const
    {
        size_t siz = ((std::vector<_ElemType>*) this)->size();
        return (int) siz;
    }
};
// overload swap(), otherwise we'd fallback to 3-way assignment & possibly throw
template<class _T> inline void swap (ARRAY<_T> & L, ARRAY<_T> & R) throw()
{ swap ((std::vector<_T> &) L, (std::vector<_T> &) R); }

// class fixed_vector - non-resizable vector

template<class _T> class fixed_vector
{
    _T * p;                 // pointer array
    size_t n;               // number of elements
    void check (int index) const { index; ASSERT (index >= 0 && (size_t) index < n); }
    void check (size_t index) const { index; ASSERT (index < n); }
    // ... TODO: when I make this public, LinearTransform.h acts totally up but I cannot see where it comes from.
    //fixed_vector (const fixed_vector & other) : n (0), p (NULL) { *this = other; }
public:
    fixed_vector() : n (0), p (NULL) { }
    void resize (int size) { clear(); if (size > 0) { p = new _T[size]; n = size; } }
    void resize (size_t size) { clear(); if (size > 0) { p = new _T[size]; n = size; } }
    fixed_vector (int size) : n (size), p (size > 0 ? new _T[size] : NULL) { }
    fixed_vector (size_t size) : n ((int) size), p (size > 0 ? new _T[size] : NULL) { }
    ~fixed_vector() { delete[] p; }
    inline int size() const { return (int) n; }
    inline int capacity() const { return (int) n; }
    inline bool empty() const { return n == 0; }
    void clear() { delete[] p; p = NULL; n = 0; }
    _T *       begin()       { return p; }
    const _T * begin() const { return p; }
    _T * end()   { return p + n; } // note: n == 0 so result is NULL
    inline       _T & operator[] (int index)          { check (index); return p[index]; }  // writing
    inline const _T & operator[] (int index) const    { check (index); return p[index]; }  // reading
    inline       _T & operator[] (size_t index)       { check (index); return p[index]; }  // writing
    inline const _T & operator[] (size_t index) const { check (index); return p[index]; }  // reading
    inline int indexof (const _T & elem) const { ASSERT (&elem >= p && &elem < p + n); return &elem - p; }
    inline void swap (fixed_vector & other) throw() { std::swap (other.p, p); std::swap (other.n, n); }
    template<class VECTOR> fixed_vector & operator= (const VECTOR & other)
    {
        int other_n = (int) other.size();
        fixed_vector tmp (other_n);
        for (int k = 0; k < other_n; k++) tmp[k] = other[k];
        swap (tmp);
        return *this;
    }
    fixed_vector & operator= (const fixed_vector & other)
    {
        int other_n = (int) other.size();
        fixed_vector tmp (other_n);
        for (int k = 0; k < other_n; k++) tmp[k] = other[k];
        swap (tmp);
        return *this;
    }
    template<class VECTOR> fixed_vector (const VECTOR & other) : n (0), p (NULL) { *this = other; }
};
template<class _T> inline void swap (fixed_vector<_T> & L, fixed_vector<_T> & R) throw() { L.swap (R); }

// class matrix - simple fixed-size 2-dimensional array, access elements as m(i,j)
// stored as concatenation of rows

template<class T> class matrix : fixed_vector<T>
{
    size_t numcols;
    size_t locate (size_t i, size_t j) const { ASSERT (i < rows() && j < cols()); return i * cols() + j; }
public:
    typedef T elemtype;
    matrix() : numcols (0) {}
    matrix (size_t n, size_t m) { resize (n, m); }
    void resize (size_t n, size_t m) { numcols = m; fixed_vector::resize (n * m); }
    size_t cols() const { return numcols; }
    size_t rows() const { return empty() ? 0 : size() / cols(); }
    size_t size() const { return fixed_vector::size(); }    // use this for reading and writing... not nice!
    bool empty() const { return fixed_vector::empty(); }
    T &       operator() (size_t i, size_t j)       { return (*this)[locate(i,j)]; }
    const T & operator() (size_t i, size_t j) const { return (*this)[locate(i,j)]; }
    void swap (matrix & other) throw() { std::swap (numcols, other.numcols); fixed_vector::swap (other); }
};
template<class _T> inline void swap (matrix<_T> & L, matrix<_T> & R) throw() { L.swap (R); }

// TODO: get rid of these
typedef std::string STRING;
typedef std::wstring WSTRING;
typedef std::basic_string<TCHAR> TSTRING;	// wide/narrow character string

// derive from this for noncopyable classes (will get you private unimplemented copy constructors)
// ... TODO: change all of basetypes classes/structs to use this
class noncopyable
{
    noncopyable & operator= (const noncopyable &);
    noncopyable (const noncopyable &);
public:
    noncopyable(){}
};

// class CCritSec and CAutoLock -- simple critical section handling
class CCritSec
{
    CCritSec (const CCritSec &); CCritSec & operator= (const CCritSec &);
    CRITICAL_SECTION m_CritSec;
public:
    CCritSec() { InitializeCriticalSection(&m_CritSec); };
    ~CCritSec() { DeleteCriticalSection(&m_CritSec); };
    void Lock() { EnterCriticalSection(&m_CritSec); };
    void Unlock() { LeaveCriticalSection(&m_CritSec); };
};

// locks a critical section, and unlocks it automatically
// when the lock goes out of scope
class CAutoLock
{
    CAutoLock(const CAutoLock &refAutoLock); CAutoLock &operator=(const CAutoLock &refAutoLock);
    CCritSec & m_rLock;
public:
    CAutoLock(CCritSec & rLock) : m_rLock (rLock) { m_rLock.Lock(); };
    ~CAutoLock() { m_rLock.Unlock(); };
};

// an efficient way to write COM code
// usage examples:
//  COM_function() || throw_hr ("message");
//  while ((s->Read (p, n, &m) || throw_hr ("Read failure")) == S_OK) { ... }
// is that cool or what?
struct bad_hr : public std::exception
{
    HRESULT hr;
    bad_hr (HRESULT p_hr, const char * msg) : hr (p_hr), std::exception (msg) { }
    // (only for use in || expression  --deprecated:)
    bad_hr() : std::exception (NULL) { }
    bad_hr (const char * msg) : std::exception (msg) { }
};
struct throw_hr
{
    const char * msg;
    inline throw_hr (const char * msg = NULL) : msg (msg) {}
};
inline static HRESULT operator|| (HRESULT hr, const throw_hr & e)
{
    if (SUCCEEDED (hr)) return hr;
    throw bad_hr (hr, e.msg);
}
// (old deprecated version kept for compat:)
inline static bool operator|| (HRESULT hr, const bad_hr & e) { if (SUCCEEDED (hr)) return true; throw bad_hr (hr, e.what()); }

// back-mapping of exceptions to HRESULT codes
// usage pattern: HRESULT COM_function (...) { try { exception-based function body } catch_hr_return; }
#define catch_hr_return    \
        catch (const bad_alloc &) { return E_OUTOFMEMORY; }         \
        catch (const bad_hr & e) { return e.hr; }                   \
        catch (const invalid_argument &) { return E_INVALIDARG; }   \
        catch (const runtime_error &) { return E_FAIL; }            \
        catch (const logic_error &) { return E_UNEXPECTED; }        \
        catch (const exception &) { return E_FAIL; }                \
        return S_OK;

// CoInitializeEx() wrapper to ensure CoUnintialize()
struct auto_co_initialize : noncopyable
{
    auto_co_initialize() { ::CoInitializeEx (NULL, COINIT_MULTITHREADED) || bad_hr ("auto_co_initialize: CoInitializeEx failure"); }
    ~auto_co_initialize() { ::CoUninitialize(); }
};

// auto pointer for ::CoTaskMemFree
template<class T> class auto_co_ptr : noncopyable
{
    T * p;
public:
    auto_co_ptr() : p (NULL) { }
    auto_co_ptr (T * p) : p (p) { }
    ~auto_co_ptr() { ::CoTaskMemFree (p); }
    operator T * () const { return p; }
    T * operator->() const { return p; }
    T** operator& () { assert (p == NULL); return &p; }    // must be empty when taking address
};

// represents a thread-local-storage variable
// Note: __declspec(thread) is broken on pre-Vista for delay loaded DLLs
// [http://www.nynaeve.net/?p=187]
// so instead, we need to wrap up the Win32 TLS functions ourselves.
// Note: tls instances must be allocated as static to work correctly, e.g.:
//   static tls myVal();
//   myVal = (void *) 25;
//   printf ("value is %d",(void *) myVal);

class tls
{
private:
    int tlsSlot;
public:

#ifdef UNDER_CE
    // this is from standard windows headers - seems to be missing in WINCE
    #define TLS_OUT_OF_INDEXES ((DWORD)0xFFFFFFFF)
#endif
    tls() { tlsSlot = TlsAlloc(); if (tlsSlot == TLS_OUT_OF_INDEXES) throw std::runtime_error("tls: TlsAlloc failed, out of tls slots"); }
    operator void * () { return TlsGetValue (tlsSlot); }
    void *operator = (void *val) { if (!TlsSetValue (tlsSlot,val)) throw std::runtime_error ("tls: TlsSetValue failed"); return val; }
};

};};    // namespace

#ifndef BASETYPES_NO_UNSAFECRTOVERLOAD // if on, no unsafe CRT overload functions

// ----------------------------------------------------------------------------
// overloads for "unsafe" CRT functions used in our code base
// ----------------------------------------------------------------------------

// strlen/wcslen overloads for fixed-buffer size

// Note: Careful while fixing bug related to these templates.
// In all attempted experiments, in seems all 6 definitions are required 
// below to get the correct behaviour.  Be very very careful 
// not to delete something without testing that case 5&6 have "size" deduced.
// 1. char *
// 2. char * const
// 3. const char *
// 4. const char * const
// 5. char (&) [size]
// 6. const char (&) [size]
// the following includes all headers that use strlen() and fail because of the mapping below
// to find those, change #define strlen strlen_ to something invalid e.g. strlen::strlen_
#if _MSC_VER >= 1600    // VS 2010  --TODO: fix this by correct include order instead
#include <intrin.h>     // defines strlen() as an intrinsic in VS 2010
#include <typeinfo>     // uses strlen()
#include <xlocale>      // uses strlen()
#endif
#define strlen strlen_
template<typename _T> inline __declspec(deprecated("Dummy general template, cannot be used directly")) 
size_t strlen_(_T &s) { return strnlen_s(static_cast<const char *>(s), SIZE_MAX); } // never be called but needed to keep compiler happy
template<typename _T> inline size_t strlen_(const _T &s)     { return strnlen_s(static_cast<const char *>(s), SIZE_MAX); }
template<> inline size_t strlen_(char * &s)                  { return strnlen_s(s, SIZE_MAX); }
template<> inline size_t strlen_(const char * &s)            { return strnlen_s(s, SIZE_MAX); }
template<size_t n> inline size_t strlen_(const char (&s)[n]) { return (strnlen_s(s, n)); }
template<size_t n> inline size_t strlen_(char (&s)[n])       { return (strnlen_s(s, n)); }
#define wcslen wcslen_
template<typename _T> inline __declspec(deprecated("Dummy general template, cannot be used directly")) 
size_t wcslen_(_T &s) { return wcsnlen_s(static_cast<const wchar_t *>(s), SIZE_MAX); } // never be called but needed to keep compiler happy
template<typename _T> inline size_t __cdecl wcslen_(const _T &s)        { return wcsnlen_s(static_cast<const wchar_t *>(s), SIZE_MAX); }
template<> inline size_t wcslen_(wchar_t * &s)                  { return wcsnlen_s(s, SIZE_MAX); }
template<> inline size_t wcslen_(const wchar_t * &s)            { return wcsnlen_s(s, SIZE_MAX); }
template<size_t n> inline size_t wcslen_(const wchar_t (&s)[n]) { return (wcsnlen_s(s, n)); }
template<size_t n> inline size_t wcslen_(wchar_t (&s)[n])       { return (wcsnlen_s(s, n)); }

// xscanf wrappers -- one overload for each actual use case in our code base
static inline int sscanf  (const char * buf, const char * format, int * i1)                     { return sscanf_s (buf, format, i1); }
static inline int sscanf  (const char * buf, const char * format, int * i1, int * i2)           { return sscanf_s (buf, format, i1, i2); }
static inline int sscanf  (const char * buf, const char * format, int * i1, int * i2, int * i3) { return sscanf_s (buf, format, i1, i2, i3); }
static inline int sscanf  (const char * buf, const char * format, double * f1)                  { return sscanf_s (buf, format, f1); }
static inline int swscanf (const wchar_t * buf, const wchar_t * format, int * i1)               { return swscanf_s (buf, format, i1); }
static inline int fscanf  (FILE * file, const char * format, float * f1)                        { return fscanf_s (file, format, f1); }

// ...TODO: should we pass 'count' instead of SIZE_MAX? (need to review use cases)
#define _vsnprintf _vsnprintf_
static inline int _vsnprintf_(char *buffer, size_t count, const char *format, va_list argptr)
{ return _vsnprintf_s (buffer, SIZE_MAX, count, format, argptr); }
#define _vsnwprintf _vsnwprintf_
static inline int _vsnwprintf_(wchar_t *buffer, size_t count, const wchar_t *format, va_list argptr)
{ return _vsnwprintf_s (buffer, SIZE_MAX, count, format, argptr); }

// wcsfcpy -- same as standard wcsncpy, use padded fixed-size buffer really needed
static inline void wcsfcpy (wchar_t * dest, const wchar_t * source, size_t count)
{
    while (count && (*dest++ = *source++) != 0) count--;    // copy
    if (count) while (--count) *dest++ = 0;                 // pad with zeroes
}

// cacpy -- fixed-size character array (same as original strncpy (dst, src, sizeof (dst)))
// NOTE: THIS FUNCTION HAS NEVER BEEN TESTED. REMOVE THIS COMMENT ONCE IT HAS.
template<class T, size_t n> static inline void cacpy (T (&dst)[n], const T * src)
{ for (int i = 0; i < n; i++) { dst[i] = *src; if (*src) src++; } }
// { return strncpy (dst, src, n); }   // using original C std lib function

// mappings for "unsafe" functions that are not really unsafe
#define strtok strtok_      // map to "safe" function (adds no value)
static inline /*const*/ char * strtok_(char * s, const char * delim)
{
    static msra::basetypes::tls tls_context; // see note for tls class def
    char *context = (char *) (void *) tls_context;
    char *ret = strtok_s (s, delim, &context);
    tls_context = context;
    return ret;
}

#define wcstok wcstok_      // map to "safe" function (adds no value)
static inline /*const*/ wchar_t * wcstok_(wchar_t * s, const wchar_t * delim) 
{ 
    static msra::basetypes::tls tls_context; // see note for tls class def
    wchar_t *context = (wchar_t *) (void *) tls_context;
    wchar_t *ret = wcstok_s (s, delim, &context);
    tls_context = context;
    return ret;
}

#define fopen fopen_        // map to _fsopen() (adds no value)
static inline FILE * fopen_(const char * p, const char * m) { return _fsopen (p, m, _SH_DENYWR); }
#define _wfopen _wfopen_    // map to _wfsopen() (adds no value)
static inline FILE * _wfopen_(const wchar_t * p, const wchar_t * m) { return _wfsopen (p, m, _SH_DENYWR); }

#define strerror(e) strerror_((e))      // map to "safe" function (adds no value)
static inline const char *strerror_(int e)
{   // keep a cache so we can return a pointer (to mimic the old interface)
    static msra::basetypes::CCritSec cs; static std::map<int,std::string> msgs;
    msra::basetypes::CAutoLock lock (cs);
    if (msgs.find(e) == msgs.end()) { char msg[1024]; strerror_s (msg, e); msgs[e] = msg; }
    return msgs[e].c_str();
}

#endif

// ----------------------------------------------------------------------------
// frequently missing string functions
// ----------------------------------------------------------------------------

namespace msra { namespace strfun {

#ifndef BASETYPES_NO_STRPRINTF

// [w]strprintf() -- like sprintf() but resulting in a C++ string
template<class _T> struct _strprintf : public std::basic_string<_T>
{   // works for both wchar_t* and char*
    _strprintf (const _T * format, ...)
    {
        va_list args; va_start (args, format);  // varargs stuff
        size_t n = _cprintf (format, args);     // num chars excl. '\0'
        const int FIXBUF_SIZE = 128;            // incl. '\0'
        if (n < FIXBUF_SIZE)
        {
            _T fixbuf[FIXBUF_SIZE];
            this->assign (_sprintf (&fixbuf[0], sizeof (fixbuf)/sizeof (*fixbuf), format, args), n);
        }
        else    // too long: use dynamically allocated variable-size buffer
        {
            std::vector<_T> varbuf (n + 1);     // incl. '\0'
            this->assign (_sprintf (&varbuf[0], varbuf.size(), format, args), n);
        }
    }
private:
    // helpers
    inline size_t _cprintf (const wchar_t * format, va_list args) { return _vscwprintf (format, args); }
    inline size_t _cprintf (const  char   * format, va_list args) { return _vscprintf  (format, args); }
    inline const wchar_t * _sprintf (wchar_t * buf, size_t bufsiz, const wchar_t * format, va_list args) { vswprintf_s (buf, bufsiz, format, args); return buf; }
    inline const  char   * _sprintf ( char   * buf, size_t bufsiz, const  char   * format, va_list args) { vsprintf_s  (buf, bufsiz, format, args); return buf; }
};
typedef strfun::_strprintf<char>    strprintf;  // char version
typedef strfun::_strprintf<wchar_t> wstrprintf; // wchar_t version

#endif

// string-encoding conversion functions
struct utf8 : std::string { utf8 (const std::wstring & p)    // utf-16 to -8
{
    size_t len = p.length();
    if (len == 0) { return;}    // empty string
    msra::basetypes::fixed_vector<char> buf (3 * len + 1);   // max: 1 wchar => up to 3 mb chars
    // ... TODO: this fill() should be unnecessary (a 0 is appended)--but verify
    std::fill (buf.begin (), buf.end (), 0);
    int rc = WideCharToMultiByte (CP_UTF8, 0, p.c_str(), (int) len,
                                  &buf[0], (int) buf.size(), NULL, NULL);
    if (rc == 0) throw std::runtime_error ("WideCharToMultiByte");
    (*(std::string*)this) = &buf[0];
}};
struct utf16 : std::wstring { utf16 (const std::string & p)  // utf-8 to -16
{
    size_t len = p.length();
    if (len == 0) { return;}    // empty string
    msra::basetypes::fixed_vector<wchar_t> buf (len + 1);
    // ... TODO: this fill() should be unnecessary (a 0 is appended)--but verify
    std::fill (buf.begin (), buf.end (), (wchar_t) 0);
    int rc = MultiByteToWideChar (CP_UTF8, 0, p.c_str(), (int) len,
                                  &buf[0], (int) buf.size());
    if (rc == 0) throw std::runtime_error ("MultiByteToWideChar");
    ASSERT (rc < buf.size ());
    (*(std::wstring*)this) = &buf[0];
}};

#pragma warning(push)
#pragma warning(disable : 4996) // Reviewed by Yusheng Li, March 14, 2006. depr. fn (wcstombs, mbstowcs)
static inline std::string wcstombs (const std::wstring & p)  // output: MBCS
{
    size_t len = p.length();
    msra::basetypes::fixed_vector<char> buf (2 * len + 1); // max: 1 wchar => 2 mb chars
    std::fill (buf.begin (), buf.end (), 0);
    ::wcstombs (&buf[0], p.c_str(), 2 * len + 1);
    return std::string (&buf[0]);
}
static inline std::wstring mbstowcs (const std::string & p)  // input: MBCS
{
    size_t len = p.length();
    msra::basetypes::fixed_vector<wchar_t> buf (len + 1); // max: >1 mb chars => 1 wchar
    std::fill (buf.begin (), buf.end (), (wchar_t) 0);
    OACR_WARNING_SUPPRESS(UNSAFE_STRING_FUNCTION, "Reviewed OK. size checked. [rogeryu 2006/03/21]");
    ::mbstowcs (&buf[0], p.c_str(), len + 1);
    return std::wstring (&buf[0]);
}
#pragma warning(pop)

// split and join -- tokenize a string like strtok() would, join() strings together
template<class _T> static inline std::vector<std::basic_string<_T>> split (const std::basic_string<_T> & s, const _T * delim)
{
    std::vector<std::basic_string<_T>> res;
    for (size_t st = s.find_first_not_of (delim); st != std::basic_string<_T>::npos; )
    {
        size_t en = s.find_first_of (delim, st +1);
        if (en == std::basic_string<_T>::npos) en = s.length();
        res.push_back (s.substr (st, en-st));
        st = s.find_first_not_of (delim, en +1);    // may exceed
    }
    return res;
}

template<class _T> static inline std::basic_string<_T> join (const std::vector<std::basic_string<_T>> & a, const _T * delim)
{
    std::basic_string<_T> res;
    for (int i = 0; i < (int) a.size(); i++)
    {
        if (i > 0) res.append (delim);
        res.append (a[i]);
    }
    return res;
}

// parsing strings to numbers
static inline int toint (const wchar_t * s)
{
    return _wtoi (s);   // ... TODO: check it
}
static inline int toint (const char * s)
{
    return atoi (s);    // ... TODO: check it
}
static inline int toint (const std::wstring & s) { return toint (s.c_str()); }

static inline double todouble (const char * s)
{
    char * ep;          // will be set to point to first character that failed parsing
    double value = strtod (s, &ep);
    if (*s == 0 || *ep != 0)
        throw std::runtime_error ("todouble: invalid input string");
    return value;
}

// TODO: merge this with todouble(const char*) above
static inline double todouble (const std::string & s)
{
    s.size();       // just used to remove the unreferenced warning
    
    double value = 0.0;

    // stod supposedly exists in VS2010, but some folks have compilation errors
    // If this causes errors again, change the #if into the respective one for VS 2010.
#if _MSC_VER > 1400 // VS 2010+
    size_t * idx = 0;
    value = std::stod (s, idx);
    if (idx) throw std::runtime_error ("todouble: invalid input string");
#else
    char *ep = 0;   // will be updated by strtod to point to first character that failed parsing
    value = strtod (s.c_str(), &ep);

    // strtod documentation says ep points to first unconverted character OR 
    // return value will be +/- HUGE_VAL for overflow/underflow
    if (ep != s.c_str() + s.length() || value == HUGE_VAL || value == -HUGE_VAL)
        throw std::runtime_error ("todouble: invalid input string");
#endif
    
    return value;
}

static inline double todouble (const std::wstring & s)
{
    wchar_t * endptr;
    double value = wcstod (s.c_str(), &endptr);
    if (*endptr) throw std::runtime_error ("todouble: invalid input string");
    return value;
}

// ----------------------------------------------------------------------------
// tokenizer -- utility for white-space tokenizing strings in a character buffer
// This simple class just breaks a string, but does not own the string buffer.
// ----------------------------------------------------------------------------

class tokenizer : public std::vector<char*>
{
    const char * delim;
public:
    tokenizer (const char * delim, size_t cap) : delim (delim) { reserve (cap); }
    // Usage: tokenizer tokens (delim, capacity); tokens = buf; tokens.size(), tokens[i]
    void operator= (char * buf)
    {
        resize (0);

        // strtok_s not available on all platforms - so backoff to strtok on those
#ifdef strtok_s
        char * context; // for strtok_s()
        for (char * p = strtok_s (buf, delim, &context); p; p = strtok_s (NULL, delim, &context))
            push_back (p);
#else
        for (char * p = strtok (buf, delim); p; p = strtok (NULL, delim))
            push_back (p);
#endif   
    }
};

};};    // namespace

// ----------------------------------------------------------------------------
// wrappers for some basic types (files, handles, timer)
// ----------------------------------------------------------------------------

namespace msra { namespace basetypes {

// FILE* with auto-close; use auto_file_ptr instead of FILE*.
// Warning: do not pass an auto_file_ptr to a function that calls fclose(),
// except for fclose() itself.
class auto_file_ptr
{
    FILE * f;
    FILE * operator= (auto_file_ptr &); // can't ref-count: no assignment
    auto_file_ptr (auto_file_ptr &);
    // implicit close (destructor, assignment): we ignore error
    void close() throw() { if (f) try { if (f != stdin && f != stdout && f != stderr) ::fclose (f); } catch (...) { } f = NULL; }
    void openfailed (const std::string & path) { throw std::runtime_error ("auto_file_ptr: error opening file '" + path + "': " + strerror (errno)); }
protected:
    friend int fclose (auto_file_ptr&); // explicit close (note: may fail)
    int fclose() { int rc = ::fclose (f); if (rc == 0) f = NULL; return rc; }
public:
    auto_file_ptr() : f (NULL) { }
    ~auto_file_ptr() { close(); }
    auto_file_ptr (const char * path, const char * mode) { f = fopen (path, mode); if (f == NULL) openfailed (path); }
    auto_file_ptr (const wchar_t * path, const char * mode) { f = _wfopen (path, msra::strfun::utf16 (mode).c_str()); if (f == NULL) openfailed (msra::strfun::utf8 (path)); }
    FILE * operator= (FILE * other) { close(); f = other; return f; }
    auto_file_ptr (FILE * other) : f (other) { }
    operator FILE * () const { return f; }
    FILE * operator->() const { return f; }
    void swap (auto_file_ptr & other) throw() { std::swap (f, other.f); }
};
inline int fclose (auto_file_ptr & af) { return af.fclose(); }

// auto-closing container for Win32 handles.
// Pass close function if not CloseHandle(), e.g.
// auto_handle h (FindFirstFile(...), FindClose);
// ... TODO: the close function should really be a template parameter
template<class _H> class auto_handle_t
{
    _H h;
    BOOL (WINAPI_CC * close) (HANDLE);  // close function
    auto_handle_t operator= (const auto_handle_t &);
    auto_handle_t (const auto_handle_t &);
public:
    auto_handle_t (_H p_h, BOOL (WINAPI_CC * p_close) (HANDLE) = ::CloseHandle) : h (p_h), close (p_close) {}
    ~auto_handle_t() { if (h != INVALID_HANDLE_VALUE) close (h); }
    operator _H () const { return h; }
};
typedef auto_handle_t<HANDLE> auto_handle;

// like auto_ptr but calls freeFunc_p (type free_func_t) instead of delete to clean up
// minor difference - wrapped object is T, not T *, so to wrap a 
// T *, use auto_clean<T *>
// TODO: can this be used for simplifying those other classes?
template<class T,class FR = void> class auto_clean
{
    T it;
    typedef FR (*free_func_t)(T); 
    free_func_t freeFunc;                           // the function used to free the pointer
    void free() { if (it) freeFunc(it); it = 0; }
    auto_clean operator= (const auto_clean &);      // hide to prevent copy
    auto_clean (const auto_clean &);                // hide to prevent copy
public:
    auto_clean (T it_p, free_func_t freeFunc_p) : it (it_p), freeFunc (freeFunc_p) {}
    ~auto_clean() { free(); }
    operator T () { return it; }
    operator const T () const { return it; }
    T detach () { T tmp = it; it = 0; return tmp; } // release ownership of object
};

// simple timer
// auto_timer timer; run(); double seconds = timer; // now can abandon the object
class auto_timer
{
    LARGE_INTEGER freq, start;
    auto_timer (const auto_timer &); void operator= (const auto_timer &);
public:
    auto_timer()
    {
        if (!QueryPerformanceFrequency (&freq)) // count ticks per second
            throw std::runtime_error ("auto_timer: QueryPerformanceFrequency failure");
        QueryPerformanceCounter (&start);
    }
    operator double() const     // each read gives time elapsed since start, in seconds
    {
        LARGE_INTEGER end;
        QueryPerformanceCounter (&end);
        return (end.QuadPart - start.QuadPart) / (double) freq.QuadPart;
    }
};

};};

namespace msra { namespace files {

// ----------------------------------------------------------------------------
// textreader -- simple reader for text files --we need this all the time!
// Currently reads 8-bit files, but can return as wstring, in which case
// they are interpreted as UTF-8 (without BOM).
// Note: Not suitable for pipes or typed input due to readahead (fixable if needed).
// ----------------------------------------------------------------------------

class textreader
{
    msra::basetypes::auto_file_ptr f;
    std::vector<char> buf;  // read buffer (will only grow, never shrink)
    int ch;                 // next character (we need to read ahead by one...)
    char getch() { char prevch = (char) ch; ch = fgetc (f); return prevch; }
public:
    textreader (const std::wstring & path) : f (path.c_str(), "rb") { buf.reserve (10000); ch = fgetc (f); }
    operator bool() const { return ch != EOF; } // true if still a line to read
    std::string getline()                       // get and consume the next line
    {
        if (ch == EOF) throw std::logic_error ("textreader: attempted to read beyond EOF");
        assert (buf.empty());
        // get all line's characters --we recognize UNIX (LF), DOS (CRLF), and Mac (CR) convention
        while (ch != EOF && ch != '\n' && ch != '\r') buf.push_back (getch());
        if (ch != EOF && getch() == '\r' && ch == '\n') getch();    // consume EOLN char
        std::string line (buf.begin(), buf.end());
        buf.clear();
        return line;
    }
    std::wstring wgetline() { return msra::strfun::utf16 (getline()); }
};

};};

// ----------------------------------------------------------------------------
// functional-programming style helper macros (...do this with templates?)
// ----------------------------------------------------------------------------

#define foreach_index(_i,_dat) for (int _i = 0; _i < (int) (_dat).size(); _i++)
#define map_array(_x,_expr,_y) { _y.resize (_x.size()); foreach_index(_i,_x) _y[_i]=_expr(_x[_i]); }
#define reduce_array(_x,_expr,_y) { foreach_index(_i,_x) _y = (_i==0) ? _x[_i] : _expr(_y,_x[_i]); }
template<class _A,class _F>
static void fill_array(_A & a, _F v) { ::fill (a.begin(), a.end(), v); }

// ----------------------------------------------------------------------------
// frequently missing utility functions
// ----------------------------------------------------------------------------

namespace msra { namespace util {

// to (slightly) simplify processing of command-line arguments.
// command_line args (argc, argv);
// while (args.has (1) && args[0][0] == '-') { option = args.shift(); process (option); }
// for (const wchar_t * arg = args.shift(); arg; arg = args.shift()) { process (arg); }
class command_line
{
    int num;
    (const wchar_t *) * args;
public:
    command_line (int argc, wchar_t * argv[]) : num (argc), args ((const wchar_t **) argv) { shift(); }
    inline int size() const { return num; }
    inline bool has (int left) { return size() >= left; }
    const wchar_t * shift() { if (size() == 0) return NULL; num--; return *args++; }
    const wchar_t * operator[] (int i) const { return (i < 0 || i >= size()) ? NULL : args[i]; }
};

// byte-reverse a variable --reverse all bytes (intended for integral types and float)
template<typename T> static inline void bytereverse (T & v) throw()
{   // note: this is more efficient than it looks because sizeof (v[0]) is a constant
    char * p = (char *) &v;
    const size_t elemsize = sizeof (v);
    for (int k = 0; k < elemsize / 2; k++)  // swap individual bytes
        swap (p[k], p[elemsize-1 - k]);
}

// byte-swap an entire array
template<class V> static inline void byteswap (V & v) throw()
{
    foreach_index (i, v)
        bytereverse (v[i]);
}

};};    // namespace

// ----------------------------------------------------------------------------
// frequently missing Win32 functions
// ----------------------------------------------------------------------------

// strerror() for Win32 error codes
static inline std::wstring FormatWin32Error (DWORD error)
{
    wchar_t buf[1024] = { 0 };
    ::FormatMessageW (FORMAT_MESSAGE_FROM_SYSTEM, "", error, 0, buf, sizeof (buf)/sizeof (*buf) -1, NULL);
    std::wstring res (buf);
    // eliminate newlines (and spaces) from the end
    size_t last = res.find_last_not_of (L" \t\r\n");
    if (last != std::string::npos) res.erase (last +1, res.length());
    return res;
}

// we always wanted this!
#pragma warning (push)
#pragma warning (disable: 6320) // Exception-filter expression is the constant EXCEPTION_EXECUTE_HANDLER
#pragma warning (disable: 6322) // Empty _except block
static inline void SetCurrentThreadName (const char* threadName)
{   // from http://msdn.microsoft.com/en-us/library/xcb2z8hs.aspx
    ::Sleep(10);
#pragma pack(push,8)
   struct { DWORD dwType; LPCSTR szName; DWORD dwThreadID; DWORD dwFlags; } info = { 0x1000, threadName, (DWORD) -1, 0 };
#pragma pack(pop)
   __try { RaiseException (0x406D1388, 0, sizeof(info)/sizeof(ULONG_PTR), (ULONG_PTR*)&info); }
   __except(EXCEPTION_EXECUTE_HANDLER) { }
}
#pragma warning (pop)

// return a string as a CoTaskMemAlloc'ed memory object
// Returns NULL if out of memory (we don't throw because we'd just catch it outside and convert to HRESULT anyway).
static inline LPWSTR CoTaskMemString (const wchar_t * s)
{
    size_t n = wcslen (s) + 1;  // number of chars to allocate and copy
    LPWSTR p = (LPWSTR) ::CoTaskMemAlloc (sizeof (*p) * n);
    if (p) for (size_t i = 0; i < n; i++) p[i] = s[i];
    return p;
}

template<class S> static inline void ZeroStruct (S & s) { memset (&s, 0, sizeof (s)); }

// ----------------------------------------------------------------------------
// machine dependent
// ----------------------------------------------------------------------------

#define MACHINE_IS_BIG_ENDIAN (false)

using namespace msra::basetypes;    // for compatibility

#pragma warning (pop)

#endif	// _BASETYPES_
