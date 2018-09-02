// htkfeatio.h -- helper for I/O of HTK feature files
//
// F. Seide, Nov 2010
//
// $Log: /Speech_To_Speech_Translation/dbn/dbn/htkfeatio.h $
// 
// 32    8/07/12 9:16 Fseide
// added Frank's weird sampling experiment
// 
// 31    6/21/12 9:45p V-hansu
// add a function to get state id from name for adaptation
// 
// 30    2/24/12 7:57p V-xieche
// fix a minor variable error in the funciton of reading mlf file.
// 
// 29    1/04/12 4:59p Fseide
// issilstate business changed from '1'/'0' to proper 'bool'
// 
// 28    1/02/12 10:44p F-gli
// change in parseentry(): due to some mlf file have write error, so skip
// malformed entry
// 
// 27    1/01/12 10:19p F-gli
// minor change to add log
// 
// 26    11/03/11 13:09 Fseide
// some refactoring and a new method getinfo() to support paging of
// feature chunks
// 
// 25    11/01/11 10:54 Fseide
// (spaces)
// 
// 24    10/31/11 15:05 Fseide
// new method parsedpath::numframes()
// 
// 23    10/27/11 13:53 Fseide
// (comments)
// 
// 22    8/22/11 3:08p F-gli
// add a function to htkmlfreader to return state number
// 
// 21    8/02/11 11:29a F-gli
// change index => state lookup array to index => sil state lookup array,
// and corresponding code
// 
// 20    7/25/11 6:32p F-gli
// add issilstate() function to htkmlfreader class
// 
// 19    7/23/11 7:29a Dongyu
// add support so that if time instead of frame number is used in the mlf
// file (without statemap) the tool can still work.
// 
// 18    7/22/11 10:48p F-gli
// check in unfinished code for remove sil frames for training
// 
// 17    6/29/11 19:32 Fseide
// parsedpath now allows paths for the form a=b (without time range)
// 
// 16    6/20/11 7:18 Fseide
// (minor fix of a compiler warning)
// 
// 15    5/07/11 10:07p Fseide
// (fixed a typo in a message)
// 
// 14    4/25/11 11:11a F-gli
// add code to make htkmlfreader support state list map which enable mlf
// transformation on the fly
// 
// 13    3/07/11 12:14 Fseide
// write() now writes to a temp file first to ensure we don't leave broken
// files
// 
// 12    2/12/11 10:40 Fseide
// htkmlfentry reduced from 24 to 8 bytes
// 
// 11    12/10/10 3:34p Fseide
// (added log message in read())
// 
// 10    12/03/10 3:56p Fseide
// MLF reader is now tolerant on duplicate #!MLF!# headers, so it accepts
// concatenations of MLF files
// 
// 9     11/23/10 15:29 Fseide
// added htkmlfreader and htkmlfreader
// 
// 8     11/18/10 9:55 Fseide
// completed  archive reading
// 
// 7     11/18/10 9:19 Fseide
// open() now handles archives (but the path parser does not yet)
// 
// 6     11/18/10 8:47 Fseide
// preparation of htkfeatreader to support archives
// 
// 5     11/18/10 8:04 Fseide
// read() function now validates indentical feature types for multiple
// reads
// 
// 4     11/12/10 4:49p Fseide
// commented a bug
// 
// 3     11/09/10 12:48 Fseide
// fixed forgotten byte swapping when writing output vectors
// 
// 2     11/09/10 8:56 Fseide
// completed except for reading from archives
// 
// 1     11/08/10 16:01 Fseide
// created
#pragma once

#include "basetypes.h"
#include "fileutil.h"
#include "simple_checked_arrays.h"
#include <string>
#include <regex>
#include <hash_map>
#include <fstream>
#include <numeric>


namespace msra { namespace asr {

	// ===========================================================================
	// htkfeatio -- common base class for reading and writing HTK feature files
	// ===========================================================================
	union ByteShortUnion
	{
		short shortvalue;
		byte bytearray[sizeof(short)/sizeof(byte)];
	}ByteSU;
	union ByteCharunion
	{
		unsigned char ucharvalue;
		byte bytearray[sizeof(unsigned char)/sizeof(byte)];
		char charvalue;
	}ByteCU;
	union ByteIntUnion
	{
		int intvalue;
		byte bytearray[sizeof(int)/sizeof(byte)];
	}ByteIU;
	union ByteFloatUnion
	{
		float floatvalue;
		byte bytearray[sizeof(float)/sizeof(byte)];
	}ByteFU;

	void ReadShortVectorFromArray (byte *mfcdata, int *pos, std::vector<short> &destdata, size_t n )
	{
		int j=0;
		destdata.clear();
		destdata.resize(n);
		for(size_t v=0; v < n; v++)
		{
			for(j=0;j<sizeof(short)/sizeof(byte);j++)
			{
				msra::asr::ByteSU.bytearray[j]=mfcdata[(*pos)++];
			}
			destdata[v]=msra::asr::ByteSU.shortvalue;
		}
	}
	void ReadFloatVectorFromArray(byte *mfcdata, int *pos, std::vector<float> &destdata, size_t n )
	{
		int v=0;
		int j=0;
		destdata.clear();
		destdata.resize(n);
		for(v=0;v<n;v++)
		{
			for(j=0;j<sizeof(float)/sizeof(byte);j++)
			{
				msra::asr::ByteFU.bytearray[j]=mfcdata[(*pos)++];
			}
			destdata[v]=msra::asr::ByteFU.floatvalue;
			//destdata.push_back(msra::asr::ByteFU.floatvalue);
		}
	}
	class htkfeatio
	{
	protected:
		auto_file_ptr f;
		wstring physicalpath;       // path of this file
		bool needbyteswapping;      // need to swap the bytes?

		wstring featkind;            // HTK feature-kind string
		size_t featdim;             // feature dimension
		unsigned int featperiod;    // sampling period

		// note that by default we assume byte swapping (seems to be HTK default)
		htkfeatio() : needbyteswapping (true), featdim (0), featperiod (0) {}

		// set the feature kind variables --if already set then validate that they are the same
		// Path is only for error message.
		void setkind (const wstring& kind, size_t dim, unsigned int period, const wstring & path)
		{
			if (featkind.empty())   // not set yet: just memorize them
			{
				assert (featdim == 0 && featperiod == 0);
				featkind = kind;
				featdim = dim;
				featperiod = period;
			}
			else                    // set already: check if consistent
			{
				if (featkind != kind || featdim != dim || featperiod != period)
					throw std::runtime_error (msra::strfun::strprintf ("setkind: inconsistent feature kind for file '%S'", path.c_str()));
			}
		}

		static short swapshort (short v) throw()
		{
			const unsigned char * b = (const unsigned char *) &v;
			return (short) ((b[0] << 8) + b[1]);
		}
		static int swapint (int v) throw()
		{
			const unsigned char * b = (const unsigned char *) &v;
			return (int) (((((b[0] << 8) + b[1]) << 8) + b[2]) << 8) + b[3];
		}

		struct fileheader
		{
			int nsamples;
			int sampperiod;
			short sampsize;
			short sampkind;
			void readFromArray(byte* mfcdata, int*pos)
			{
				int v=0;
				for(v=0;v<sizeof(int)/sizeof(byte);v++)
				{
					msra::asr::ByteIU.bytearray[v]=mfcdata[(*pos)++];
				}
				nsamples=msra::asr::ByteIU.intvalue;

				for(v=0;v<sizeof(int)/sizeof(byte);v++)
				{
					msra::asr::ByteIU.bytearray[v]=mfcdata[(*pos)++];
				}
				sampperiod=msra::asr::ByteIU.intvalue;

				for(v=0;v<sizeof(short)/sizeof(byte);v++)
				{
					msra::asr::ByteSU.bytearray[v]=mfcdata[(*pos)++];
				}
				sampsize=msra::asr::ByteSU.shortvalue;

				for(v=0;v<sizeof(short)/sizeof(byte);v++)
				{
					msra::asr::ByteSU.bytearray[v]=mfcdata[(*pos)++];
				}
				sampkind=msra::asr::ByteSU.shortvalue;
			}
			void read (FILE * f)
			{
				nsamples   = fgetint (f);
				sampperiod = fgetint (f);
				sampsize   = fgetshort (f);
				sampkind   = fgetshort (f);
			}
			void write (FILE * f)
			{
				fputint (f, nsamples);
				fputint (f, sampperiod);
				fputshort (f, sampsize);
				fputshort (f, sampkind);
			}
			void byteswap()
			{
				nsamples = swapint (nsamples);
				sampperiod = swapint (sampperiod);
				sampsize = swapshort (sampsize);
				sampkind = swapshort (sampkind);
			}
		};

		static const int BASEMASK = 077;
		static const int PLP = 11;
		static const int MFCC = 6;
		static const int FBANK = 7;
		static const int USER = 9;
		static const int FESTREAM = 12;
		static const int HASENERGY   = 0100;       // _E log energy included
		static const int HASNULLE    = 0200;       // _N absolute energy suppressed
		static const int HASDELTA    = 0400;       // _D delta coef appended
		static const int HASACCS    = 01000;       // _A acceleration coefs appended
		static const int HASCOMPX   = 02000;       // _C is compressed
		static const int HASZEROM   = 04000;       // _Z zero meaned
		static const int HASCRCC   = 010000;       // _K has CRC check
		static const int HASZEROC  = 020000;       // _0 0'th Cepstra included
		static const int HASVQ     = 040000;       // _V has VQ index attached
		static const int HASTHIRD = 0100000;       // _T has Delta-Delta-Delta index attached
	};

	// ===========================================================================
	// htkfeatwriter -- write HTK feature file
	// This is designed to write a single file only (no archive mode support).
	// ===========================================================================

	class htkfeatwriter : protected htkfeatio
	{
		size_t curframe;
		vector<float> tmp;
	public:
		short parsekind (const wstring & str)
		{
			vector<wstring> params = msra::strfun::split (str, L";");
			if (params.empty())
				throw std::runtime_error ("parsekind: invalid param kind string");
			vector<wstring> parts = msra::strfun::split (params[0], L"_");
			// map base kind
			short sampkind;
			wstring basekind = parts[0];
			if (basekind == L"PLP") sampkind = PLP;
			else if (basekind == L"MFCC") sampkind = MFCC;
			else if (basekind == L"FBANK") sampkind = FBANK;
			else if (basekind == L"USER") sampkind = USER;
			else throw std::runtime_error ("parsekind: unsupported param base kind");
			// map qualifiers
			for (size_t i = 1; i < parts.size(); i++)
			{
				wstring opt = parts[i];
				if (opt.length() != 1)
					throw std::runtime_error ("parsekind: invalid param kind string");
				switch (opt[0])
				{
				case 'E': sampkind |= HASENERGY; break;
				case 'D': sampkind |= HASDELTA; break;
				case 'N': sampkind |= HASNULLE; break;
				case 'A': sampkind |= HASACCS; break;
				case 'T': sampkind |= HASTHIRD; break;
				case 'Z': sampkind |= HASZEROM; break;
				case '0': sampkind |= HASZEROC; break;
				default: throw std::runtime_error ("parsekind: invalid qualifier in param kind string");
				}
			}
			return sampkind;
		}
	public:
		// open the file for writing
		htkfeatwriter (wstring path, const wstring& kind, size_t dim, unsigned int period)
		{
			setkind (kind, dim, period, path);
			// write header
			fileheader H;
			H.nsamples = 0; // unknown for now, updated in close()
			H.sampperiod = period;
			const int bytesPerValue = sizeof (float);   // we do not support compression for now
			H.sampsize = (short) featdim * bytesPerValue;
			H.sampkind = parsekind (kind);
			if (needbyteswapping)
				H.byteswap();
			f = fopenOrDie (path, L"wbS");
			H.write (f);
			curframe = 0;
		}
		// write a frame
		void write (const vector<float> & v)
		{
			if (v.size() != featdim)
				throw std::logic_error ("htkfeatwriter: inconsistent feature dimension");
			if (needbyteswapping)
			{
				tmp.resize (v.size());
				foreach_index (k, v) tmp[k] = v[k]; 
				msra::util::byteswap (tmp);
				fwriteOrDie (tmp, f);
			}
			else
				fwriteOrDie (v, f);
			curframe++;
		}
		// finish
		// This updates the header.
		// BUGBUG: need to implement safe-save semantics! Otherwise won't work reliably with -make mode.
		// ... e.g. set DeleteOnClose temporarily, and clear at the end?
		void close (size_t numframes)
		{
			if (curframe != numframes)
				throw std::logic_error ("htkfeatwriter: inconsistent number of frames passed to close()");
			fflushOrDie (f);
			// now implant the length field; it's at offset 0
			int nSamplesFile = (int) numframes;
			if (needbyteswapping)
				nSamplesFile = swapint (nSamplesFile);
			fseekOrDie (f, 0);
			fputint (f, nSamplesFile);
			fflushOrDie (f);
			f = NULL;   // this triggers an fclose() on auto_file_ptr
		}

		// [v-wenh] only output a few nodes according to its posterior 
		template <typename T> static std::vector<size_t> IndexOrdered(std::vector<T> const & values)
		{
			std::vector<size_t> indices(values.size());
			std::iota(std::begin(indices), std::end(indices), static_cast<size_t>(0));
			std::sort(std::begin(indices), std::end(indices), [&](size_t a, size_t b){return values[a]>values[b]; });
			return indices;
		}

		static void writeEMStatistic(map<size_t, float> &zeroStats, map<size_t, vector<float>> &firstStats, map<size_t, vector<float>> &secondStats,  wstring outPath)
		{
		
			size_t ppDim = zeroStats.size();
			size_t featDim = firstStats.begin()->second.size();
			wstring temppath = outPath + L"$$";
			unlinkOrDie(outPath);
			INT32 tempvalue;
			ofstream writer;
			writer.open(temppath, std::ios_base::binary);
			tempvalue = 1;
			writer.write((char*)&tempvalue, sizeof(tempvalue));
			tempvalue = ppDim*(2 * featDim + 1);
			writer.write((char*)&tempvalue, sizeof(tempvalue));
			for (map<size_t, float>::iterator iter = zeroStats.begin(); iter != zeroStats.end();++iter)
			{
				writer.write((char*)&zeroStats[iter->first], sizeof(zeroStats[iter->first]));
				for (size_t n = 0; n < featDim; n++)
				{
					writer.write((char*)&firstStats[iter->first][n], sizeof(firstStats[iter->first][n]));
				}
				for (size_t n = 0; n < featDim; n++)
				{
					writer.write((char*)&secondStats[iter->first][n], sizeof(secondStats[iter->first][n]));
				}
			}
			writer.close();
			renameOrDie(temppath, outPath);

		}
		// [v-wenh] calculate the EM statistic and write it to file
		template<class MATRIX> static void writeEMStatistics(const wstring &outPath, const wstring &sidFeatPath, const MATRIX &ppmatrix, map<size_t,size_t> selectedIndMap, size_t M)
		{
		   // read SID feature
			htkfeatreader reader;
			wstring featkind;
			unsigned int samperperiod;
			msra::dbn::matrix feat; // sid feature
			size_t frameNumber, featDim, selectedSenoneNumber;
			const auto path = reader.parse(sidFeatPath);
			reader.read(path, featkind, samperperiod, feat);
			frameNumber = feat.cols();
			featDim = feat.rows();
			selectedSenoneNumber = selectedIndMap.size();
		  // check the length of two different kinds of features
			if (ppmatrix.cols() != frameNumber)
			{
				cerr << "frame number mismatch between sid feature and posterior outputs" << endl;
			}
			// calculate the posterior for each frame
			map<size_t, float> zeroStats;
			map<size_t, vector<float>> firstStats, secondStats;
			for (map<size_t, size_t>::iterator iter = selectedIndMap.begin(); iter != selectedIndMap.end();++iter)
			{
				zeroStats.insert(std::pair<int, float>(iter->second, 0));
				firstStats.insert(std::pair<int, vector<float>>(iter->second, vector<float> (featDim, 0)));
				secondStats.insert(std::pair<int, vector<float>>(iter->second, vector<float> (featDim, 0)));
			}
			
			// temp variables
			vector<float> v(ppmatrix.rows()); // temp variable 
			for (size_t i = 0; i < frameNumber; i++)
			{
				foreach_index(k, v)
					v[k] = ppmatrix(k, i);
				vector<size_t> index(IndexOrdered(v)); // ordered by value
				if (selectedIndMap.find(index[0]) == selectedIndMap.end()) continue; // belongs to silence frame, skip this frame
				if (v[index[0]] < 0.1) continue; // the max value is too small, skip this frame
				float sum_nosil = 0;
				if (M == 0) M = index.size();
				for (size_t j = 0; j < M; j++)
				{
					if (selectedIndMap.find(index[j]) != selectedIndMap.end())
					{
						sum_nosil += v[index[j]];
					}
				}
				//for (size_t j = 0; j < M; j++)
				//cout << sum_nosil << endl;
				concurrency::parallel_for(size_t (0),M,[&](size_t j)
				{
					if (selectedIndMap.find(index[j]) != selectedIndMap.end())
					{
						float normalizedScore = v[index[j]] / sum_nosil;
						zeroStats[selectedIndMap[index[j]]] += normalizedScore;
						for (size_t n = 0; n < featDim; n++)
						{
							firstStats[selectedIndMap[index[j]]][n] += normalizedScore*feat(n, i);
							secondStats[selectedIndMap[index[j]]][n] += normalizedScore*feat(n, i) * feat(n, i);
						}
						
					}
				});
			}

			// accumuate the zeroth, first and second order statistics
			// write the statistics into file
			writeEMStatistic(zeroStats, firstStats, secondStats,outPath);
			zeroStats.clear();
			firstStats.clear();
			secondStats.clear();
			v.clear();
		}
		template<class MATRIX> static void writeOrderedFrames(const wstring & path, const wstring & kindstr, unsigned int period, const MATRIX & feat, unsigned int M)
		{
			wstring tmppath = path + L"$$"; // tmp path for make-mode compliant
			unlinkOrDie(path);             // delete if old file is already there
			// write it out
			const size_t featdim = feat.rows();
			size_t numframes = feat.cols();
			vector<float> v(featdim);
			float sum = 0;
			size_t count = 0;
			size_t shift = 0;
			htkfeatwriter W(tmppath, kindstr, 2 * M, period);
			for (size_t i = 0; i < feat.cols(); i++)
			{
				foreach_index(k, v)
					v[k] = feat(k, i);
				sum = 0;
				count = 0;
				vector<size_t> index(IndexOrdered(v));
				vector<float> indexCombinedValue(2 * M);
				vector<size_t> selectedIndex; 
				for (size_t j = 0; j<index.size(); j++)
				{
					selectedIndex.push_back(index[j]);
				}

				for (size_t j = 0; j<M; j++) sum += v[selectedIndex[j]];

				for (size_t j = 0; j<M; j++)
				{
					indexCombinedValue[2 * j] = (float)(selectedIndex[j]);  // first is the index
					indexCombinedValue[2 * j + 1] = v[selectedIndex[j]] / sum; // second is the value

				}
				W.write(indexCombinedValue);
			}

			W.close(numframes);
			// rename to final destination
			// (This would only fail in strange circumstances such as accidental multiple processes writing to the same file.)
			renameOrDie(tmppath, path);
		}



		// write as ascii format to check the data //v-wenh
		template<class MATRIX> static void writeAsASCII (const wstring & path, const wstring & kindstr, unsigned int period, const MATRIX & feat)
		{
			wstring tmppath = path + L"$$"; // tmp path for make-mode compliant
			unlinkOrDie (path);             // delete if old file is already there
			// write it out
			size_t featdim = feat.rows();
			size_t numframes = feat.cols();
			ofstream fout(tmppath);
			for (size_t i = 0; i < numframes; i++)
			{
				for(size_t k=0;k<featdim;k++)
				{
					fout<<feat(k,i)<<" ";
				}
				fout<<endl;
			}
			fout<<flush;
			fout.close();
			renameOrDie (tmppath, path);
		}

		// read an entire utterance into a matrix
		// Matrix type needs to have operator(i,j) and resize(n,m).
		// We write to a tmp file first to ensure we don't leave broken files that would confuse make mode.
		template<class MATRIX> static void write (const wstring & path, const wstring & kindstr, unsigned int period, const MATRIX & feat)
		{
			wstring tmppath = path + L"$$"; // tmp path for make-mode compliant
			unlinkOrDie (path);             // delete if old file is already there
			// write it out
			size_t featdim = feat.rows();
			size_t numframes = feat.cols();
			vector<float> v (featdim);
			htkfeatwriter W (tmppath, kindstr, feat.rows(), period);
#ifdef SAMPLING_EXPERIMENT
			for (size_t i = 0; i < numframes; i++)
			{
				foreach_index (k, v)
				{
					float val = feat(k,i) - logf((float) SAMPLING_EXPERIMENT);
					if (i % SAMPLING_EXPERIMENT == 0)
						v[k] = val;
					else
						v[k] += (float) (log (1 + exp (val - v[k])));   // log add
				}
				if (i % SAMPLING_EXPERIMENT == SAMPLING_EXPERIMENT -1)
					W.write (v);
			}
#else
			for (size_t i = 0; i < numframes; i++)
			{
				foreach_index (k, v)
					v[k] = feat(k,i);
				W.write (v);
			}
#endif
#ifdef SAMPLING_EXPERIMENT
			W.close (numframes / SAMPLING_EXPERIMENT);
#else
			W.close (numframes);
#endif
			// rename to final destination
			// (This would only fail in strange circumstances such as accidental multiple processes writing to the same file.)
			renameOrDie (tmppath, path);
		}
	};

	// ===========================================================================
	// htkfeatreader -- read HTK feature file, with archive support
	//
	// To support archives, one instance of this can (and is supposed to) be used
	// repeatedly. All feat files read on the same instance are validated to have
	// the same feature kind.
	//
	// For archives, this caches the last used file handle, in expectation that most reads
	// are sequential anyway. In conjunction with a big buffer, this makes a huge difference.
	// ===========================================================================

	class htkfeatreader : protected htkfeatio
	{
		// information on current file
		// File handle and feature type information is stored in the underlying htkfeatio object.
		size_t physicalframes;              // total number of frames in physical file
		unsigned __int64 physicaldatastart; // byte offset of first data byte
		size_t vecbytesize;                 // size of one vector in bytes

		bool compressed;        // is compressed to 16-bit values
		bool hascrcc;           // need to skip crcc
		vector<float> a, b;     // for decompression
		vector<short> tmp;      // for decompression

		size_t curframe;        // current # samples read so far
		size_t numframes;       // number of samples for current logical file

	public:

		// parser for complex a=b[s,e] syntax
		struct parsedpath
		{
		protected:
			friend class htkfeatreader;
			bool isarchive;         // true if archive (range specified)
			wstring xpath;          // original full path specification as passed to constructor (for error messages)
			wstring logicalpath;    // virtual path that this file should be understood to belong to
			wstring archivepath;    // physical path of archive file
			size_t s, e;            // first and last frame inside the archive file; (0, INT_MAX) if not given
			void malformed() const { throw std::runtime_error (msra::strfun::strprintf ("parsedpath: malformed path '%S'", xpath.c_str())); }

			// consume and return up to 'delim'; remove from 'input' (we try to avoid C++0x here for VS 2008 compat)
			wstring consume (wstring & input, const wchar_t * delim)
			{
				vector<wstring> parts = msra::strfun::split (input, delim); // (not very efficient, but does not matter here)
				if (parts.size() == 1) input.clear();   // not found: consume to end
				else input = parts[1];                  // found: break at delimiter
				return parts[0];
			}
		public:
			// constructor parses a=b[s,e] syntax and fills in the file
			// Can be used implicitly e.g. by passing a string to open().
			parsedpath (wstring xpath) : xpath (xpath)
			{
				// parse out logical path
				logicalpath = consume (xpath, L"=");
				if (xpath.empty())  // no '=' detected: pass entire file (it's not an archive)
				{
					archivepath = logicalpath;
					s = 0;
					e = INT_MAX;
					isarchive = false;
				}
				else                // a=b[s,e] syntax detected
				{
					archivepath = consume (xpath, L"[");
					if (xpath.empty())  // actually it's only a=b
					{
						s = 0;
						e = INT_MAX;
						isarchive = false;
					}
					else
					{
						s = msra::strfun::toint (consume (xpath, L","));
						if (xpath.empty()) malformed();
						e = msra::strfun::toint (consume (xpath, L"]"));
						if (!xpath.empty()) malformed();
						isarchive = true;
					}
				}
			}

			// get the physical path for 'make' test
			const wstring & physicallocation() const { return archivepath; }

			// casting to wstring yields the logical path
			operator const wstring & () const { return logicalpath; }

			// get duration in frames
			size_t numframes() const
			{
				if (!isarchive)
					throw runtime_error ("parsedpath: this mode requires an input script with start and end frames given");
				return e - s + 1;
			}
		};

	private:

		// open the physical HTK file
		// This is different from the logical (virtual) path name in the case of an archive.
		void openphysicalFromArray (byte* mfcdata, int*pos)
		{
			//wstring physpath = ppath.physicallocation();
			//auto_file_ptr f = fopenOrDie (physpath, L"rbS");
			//auto_file_ptr f = fopenOrDie (physpath, L"rb"); // removed 'S' for now, as we mostly run local anyway, and this will speed up debugging

			// read the header (12 bytes)
			int j,k;
			fileheader H;
			H.readFromArray(mfcdata,pos);
			// H.read (f);

			// take a guess as to whether we need byte swapping or not
			bool needbyteswapping = ((unsigned int) swapint (H.sampperiod) < (unsigned int) H.sampperiod);
			if (needbyteswapping)
				H.byteswap();

			// interpret sampkind
			int basekind = H.sampkind & BASEMASK;
			wstring kind;
			switch (basekind)
			{
			case PLP: kind = L"PLP"; break;
			case MFCC: kind = L"MFCC"; break;
			case FBANK: kind = L"FBANK"; break;
			case USER: kind = L"USER"; break;
			case FESTREAM: kind = L"USER"; break;    // we return this as USER type (with guid)
			default: throw std::runtime_error ("htkfeatreader:unsupported feature kind");
			}
			// add qualifiers
			if (H.sampkind & HASENERGY) kind += L"_E";
			if (H.sampkind & HASDELTA) kind += L"_D";
			if (H.sampkind & HASNULLE) kind += L"_N";
			if (H.sampkind & HASACCS) kind += L"_A";
			if (H.sampkind & HASTHIRD) kind += L"_T";
			bool compressed = (H.sampkind & HASCOMPX) != 0;
			bool hascrcc = (H.sampkind & HASCRCC) != 0;
			if (H.sampkind & HASZEROM) kind += L"_Z";
			if (H.sampkind & HASZEROC) kind += L"_0";
			if (H.sampkind & HASVQ) throw std::runtime_error ("htkfeatreader:we do not support VQ");
			// skip additional GUID in FESTREAM features
			if (H.sampkind == FESTREAM)
			{   // ... note: untested
				unsigned char guid[16];
				for(j=0;j<16;j++)
				{
					for(k=0;k<sizeof(unsigned char)/sizeof(byte);k++)
					{
						msra::asr::ByteCU.bytearray[k]=mfcdata[(*pos)++];
					}
					guid[j]=msra::asr::ByteCU.ucharvalue;
				}
				//  freadOrDie (&guid, sizeof (guid), 1, f);
				kind += L";guid=";
				for (int i = 0; i < sizeof (guid)/sizeof (*guid); i++)
					kind += msra::strfun::wstrprintf (L"%02x", guid[i]);
			}

			// other checks
			size_t bytesPerValue = compressed ? sizeof (short) : sizeof (float);
			if (H.sampsize % bytesPerValue != 0) throw std::runtime_error ("htkfeatreader:sample size not multiple of dimension");
			size_t dim = H.sampsize / bytesPerValue;

			// read the values for decompressing
			vector<float> a, b;
			if (compressed)
			{
				ReadFloatVectorFromArray(mfcdata, pos, a, dim );
				ReadFloatVectorFromArray(mfcdata, pos, b, dim );
				//freadOrDie (a, dim, f);
				//freadOrDie (b, dim, f);
				H.nsamples -= 4;      // these are counted as 4 frames--that's the space they use
				if (needbyteswapping) { msra::util::byteswap (a); msra::util::byteswap (b); }
			}

			// done: swap it in
			//__int64 bytepos = fgetpos (f);
			setkind (kind, dim, H.sampperiod, L"array");       // this checks consistency
			// this->physicalpath.swap (physpath);
			//   this->physicaldatastart = bytepos;
			this->physicalframes = H.nsamples;
			// this->f.swap (f);   // note: this will get the previous f auto-closed at the end of this function
			this->needbyteswapping = needbyteswapping;
			this->compressed = compressed;
			this->a.swap (a);
			this->b.swap (b);
			this->vecbytesize = H.sampsize;
			this->hascrcc = hascrcc;
		}
		void openphysical (const parsedpath & ppath)
		{
			wstring physpath = ppath.physicallocation();
			//auto_file_ptr f = fopenOrDie (physpath, L"rbS");
			auto_file_ptr f = fopenOrDie (physpath, L"rb"); // removed 'S' for now, as we mostly run local anyway, and this will speed up debugging

			// read the header (12 bytes)
			fileheader H;
			H.read (f);

			// take a guess as to whether we need byte swapping or not
			bool needbyteswapping = ((unsigned int) swapint (H.sampperiod) < (unsigned int) H.sampperiod);
			if (needbyteswapping)
				H.byteswap();

			// interpret sampkind
			int basekind = H.sampkind & BASEMASK;
			wstring kind;
			switch (basekind)
			{
			case PLP: kind = L"PLP"; break;
			case MFCC: kind = L"MFCC"; break;
			case FBANK: kind = L"FBANK"; break;
			case USER: kind = L"USER"; break;
			case FESTREAM: kind = L"USER"; break;    // we return this as USER type (with guid)
			default: throw std::runtime_error ("htkfeatreader:unsupported feature kind");
			}
			// add qualifiers
			if (H.sampkind & HASENERGY) kind += L"_E";
			if (H.sampkind & HASDELTA) kind += L"_D";
			if (H.sampkind & HASNULLE) kind += L"_N";
			if (H.sampkind & HASACCS) kind += L"_A";
			if (H.sampkind & HASTHIRD) kind += L"_T";
			bool compressed = (H.sampkind & HASCOMPX) != 0;
			bool hascrcc = (H.sampkind & HASCRCC) != 0;
			if (H.sampkind & HASZEROM) kind += L"_Z";
			if (H.sampkind & HASZEROC) kind += L"_0";
			if (H.sampkind & HASVQ) throw std::runtime_error ("htkfeatreader:we do not support VQ");
			// skip additional GUID in FESTREAM features
			if (H.sampkind == FESTREAM)
			{   // ... note: untested
				unsigned char guid[16];
				freadOrDie (&guid, sizeof (guid), 1, f);
				kind += L";guid=";
				for (int i = 0; i < sizeof (guid)/sizeof (*guid); i++)
					kind += msra::strfun::wstrprintf (L"%02x", guid[i]);
			}

			// other checks
			size_t bytesPerValue = compressed ? sizeof (short) : sizeof (float);
			if (H.sampsize % bytesPerValue != 0) throw std::runtime_error ("htkfeatreader:sample size not multiple of dimension");
			size_t dim = H.sampsize / bytesPerValue;

			// read the values for decompressing
			vector<float> a, b;
			if (compressed)
			{
				freadOrDie (a, dim, f);
				freadOrDie (b, dim, f);
				H.nsamples -= 4;      // these are counted as 4 frames--that's the space they use
				if (needbyteswapping) { msra::util::byteswap (a); msra::util::byteswap (b); }
			}

			// done: swap it in
			__int64 bytepos = fgetpos (f);
			setkind (kind, dim, H.sampperiod, ppath);       // this checks consistency
			this->physicalpath.swap (physpath);
			this->physicaldatastart = bytepos;
			this->physicalframes = H.nsamples;
			this->f.swap (f);   // note: this will get the previous f auto-closed at the end of this function
			this->needbyteswapping = needbyteswapping;
			this->compressed = compressed;
			this->a.swap (a);
			this->b.swap (b);
			this->vecbytesize = H.sampsize;
			this->hascrcc = hascrcc;
		}

	public:

		htkfeatreader() {}

		// helper to create a parsed-path object
		// const auto path = parse (xpath)
		parsedpath parse (const wstring & xpath) { return parsedpath (xpath); }

		// read a feature file
		// Returns number of frames in that file.
		// This understands the more complex syntax a=b[s,e] and optimizes a little

		size_t openFromArray(byte *mfcdata,int *pos)
		{
			openphysicalFromArray(mfcdata,pos);
			curframe = 0;
			numframes = physicalframes;
			return numframes;
		}
		size_t open (const parsedpath & ppath)
		{
			// do not reopen the file if it is the same; use fsetpos() instead
			if (f == NULL || ppath.physicallocation() != physicalpath)
				openphysical (ppath);

			if (ppath.isarchive)    // reading a sub-range from an archive
			{
				if (ppath.s > ppath.e)
					throw std::runtime_error (msra::strfun::strprintf ("open: start frame > end frame in '%S'", ppath.e, physicalframes, ppath.xpath.c_str()));
				if (ppath.e >= physicalframes)
					throw std::runtime_error (msra::strfun::strprintf ("open: end frame exceeds archive's total number of frames %d in '%S'", physicalframes, ppath.xpath.c_str()));

				__int64 dataoffset = physicaldatastart + ppath.s * vecbytesize;
				fsetpos (f, dataoffset);    // we assume fsetpos(), which is our own, is smart to not flush the read buffer
				curframe = 0;
				numframes = ppath.e + 1 - ppath.s;
			}
			else                    // reading a full file
			{
				curframe = 0;
				numframes = physicalframes;
				assert (fgetpos (f) == physicaldatastart);
			}
			return numframes;
		}
		// get dimension and type information for a feature file
		// This will alter the state of this object in that it opens the file. It is efficient to read it right afterwards
		void getinfo (const parsedpath & ppath, wstring & featkind, size_t & featdim, unsigned int & featperiod)
		{
			open (ppath);
			featkind = this->featkind;
			featdim = this->featdim;
			featperiod = this->featperiod;
		}

		const wstring & getfeattype() const { return featkind; }
		operator bool() const { return curframe < numframes; }
		// read a vector from the open file
		void read (std::vector<float> & v)
		{
			if (curframe >= numframes) throw std::runtime_error ("htkfeatreader:attempted to read beyond end");
			if (!compressed)        // not compressed--the easy one
			{
				freadOrDie (v, featdim, f);
				if (needbyteswapping) msra::util::byteswap (v);
			}
			else                    // need to decompress
			{
				// read into temp vector
				freadOrDie (tmp, featdim, f);
				if (needbyteswapping) msra::util::byteswap (tmp);
				// 'decompress' it
				v.resize (tmp.size());
				foreach_index (k, v)
					v[k] = (tmp[k] + b[k]) / a[k];
			}
			curframe++;
		}

		void readDataVectorFromArray(byte* mfcdata, int *pos,std::vector<float> & v)
		{
			if (curframe >= numframes) throw std::runtime_error ("htkfeatreader:attempted to read beyond end");
			if (!compressed)        // not compressed--the easy one
			{
				ReadFloatVectorFromArray(mfcdata, pos,v, featdim);
				// freadOrDie (v, featdim, f);
				if (needbyteswapping) msra::util::byteswap (v);
			}
			else                    // need to decompress
			{
				// read into temp vector
				ReadShortVectorFromArray(mfcdata, pos,tmp, featdim);
				//freadOrDie (tmp, featdim, f);
				if (needbyteswapping) msra::util::byteswap (tmp);
				// 'decompress' it
				v.resize (tmp.size());
				foreach_index (k, v)
					v[k] = (tmp[k] + b[k]) / a[k];
			}
			curframe++;
		}
		// read a sequence of vectors from the open file into a range of frames [ts,te)
		template<class MATRIX> void readMatrixFromArray(byte* mfcdata, int *pos, MATRIX & feat, size_t ts, size_t te)
		{
			vector<float> v (featdim);
			for (size_t t = ts; t < te; t++)
			{
				readDataVectorFromArray (mfcdata,pos,v);
				foreach_index (k, v)
					feat(k,t) = v[k];
			}
		}
		template<class MATRIX> void read (MATRIX & feat, size_t ts, size_t te)
		{
			// read vectors from file and push to our target structure
			vector<float> v (featdim);
			for (size_t t = ts; t < te; t++)
			{
				read (v);
				foreach_index (k, v)
					feat(k,t) = v[k];
			}
		}
		// read an entire utterance into an already allocated matrix
		// Matrix type needs to have operator(i,j)
		template<class MATRIX> void read (const parsedpath & ppath, const wstring & kindstr, const unsigned int period, MATRIX & feat)
		{
			// open the file and check dimensions
			size_t numframes = open (ppath);
			if (feat.cols() != numframes || feat.rows() != featdim)
				throw std::logic_error ("read: stripe read called with wrong dimensions");
			if (kindstr != featkind || period != featperiod)
				throw std::logic_error ("read: attempting to mixing different feature kinds");

			// read vectors from file and push to our target structure
			read (feat, 0, numframes);
		}
		// read an entire utterance into a virgen, allocatable matrix
		// Matrix type needs to have operator(i,j) and resize(n,m)

		template<class MATRIX> void readFromArray (byte* mfcdata, int *pos, wstring& kindstr, unsigned int & period, MATRIX & feat)
		{
			size_t numframes = openFromArray(mfcdata,pos);
			feat.resize (featdim, numframes); 
			readMatrixFromArray(mfcdata,pos,feat, 0, numframes);
			kindstr = featkind;
			period = featperiod;

		}
		template<class MATRIX> void read (const parsedpath & ppath, wstring & kindstr, unsigned int & period, MATRIX & feat)
		{
			// get the file
			size_t numframes = open (ppath);
			feat.resize (featdim, numframes);   // result matrix--columns are features

			// read vectors from file and push to our target structure
			read (feat, 0, numframes);

			// return file info
			kindstr = featkind;
			period = featperiod;
		}
	};

	struct htkmlfentry
	{
		unsigned int firstframe;    // range [firstframe,firstframe+numframes)
		unsigned short numframes;
		unsigned short classid;     // numeric state id

	private:
		// verify and save data
		void setdata (size_t ts, size_t te, size_t uid)
		{
			if (te < ts) throw std::runtime_error ("htkmlfentry: end time below start time??");
			// save
			firstframe = (unsigned int) ts;
			numframes = (unsigned short) (te - ts);
			classid = (unsigned short) uid;
			// check for numeric overflow
			if (firstframe != ts || firstframe + numframes != te || classid != uid)
				throw std::runtime_error ("htkmlfentry: not enough bits for one of the values");
		}
	public:

		// [ganl] parse format with htk state algin MLF and state list: change htk time to frame, map state string to index
		void parsewithstatelist (const vector<char*> & toks, const hash_map<const string, size_t> & statelisthash)
		{
			const double htkTimeToFrame = 100000.0;                                                // frame shift 10ms in htk time code
			size_t uid;
			size_t ts = (size_t) (msra::strfun::todouble (toks[0]) / htkTimeToFrame + 0.5);          // get start frame
			size_t te = (size_t) (msra::strfun::todouble (toks[1]) / htkTimeToFrame + 0.5);          // get end frame
			auto iter = statelisthash.find (toks[2]);
			if (iter == statelisthash.end())
				throw std::runtime_error (msra::strfun::strprintf("htkmlfentry: state %s not found in statelist", toks[2]));
			else
				uid = iter->second;                     // get state index
			setdata (ts, te, uid);
		}

		// ... note: this will be too simplistic for parsing more complex MLF formats. Fix when needed.
		// add support so that it can handle conditions where time instead of frame numer is used.
		void parse (const vector<char*> & toks)
		{
			const double htkTimeToFrame = 100000.0;               // frame shift 10ms in htk time code
			const double maxFrameNumber = htkTimeToFrame/2;       //if frame number is greater than this we assume it is time instead of frame
			size_t ts, te;

			// parse
			if (toks.size() != 4) throw std::runtime_error ("htkmlfentry: currently we only support 4-column format");

			double rts = msra::strfun::todouble (toks[0]);
			double rte = msra::strfun::todouble (toks[1]);
			if (rte > maxFrameNumber) // convert time to frame
			{
				ts = (size_t) (rts/htkTimeToFrame + 0.5);          // get start frame
				te = (size_t) (rte/htkTimeToFrame + 0.5);          // get end frame
			}
			else
			{
				ts = (size_t)(rts);        //msra::strfun::toint (toks[0]);
				te = (size_t)(rte);        //msra::strfun::toint (toks[1]);
			}
			size_t uid = msra::strfun::toint (toks[3]);
			setdata(ts, te, uid);
		}
	};

	template<class ENTRY>
	class htkmlfreader : public map<wstring,vector<ENTRY>>   // [key][i] the data
	{
		wstring curpath;                                     // for error messages
		hash_map<const std::string, size_t> statelistmap;    // for state <=> index

		void strtok (char * s, const char * delim, vector<char*> & toks)
		{
			toks.resize (0);
			char * context;
			for (char * p = strtok_s (s, delim, &context); p; p = strtok_s (NULL, delim, &context))
				toks.push_back (p);
		}
		void malformed (string what)
		{
			throw std::runtime_error (msra::strfun::strprintf ("htkmlfreader: %s in '%S'", what.c_str(), curpath.c_str()));
		}

		vector<char*> readlines (const wstring & path, vector<char> & buffer)
		{
			// load it into RAM in one huge chunk
			auto_file_ptr f = fopenOrDie (path, L"rb");
			size_t len = filesize (f);
			buffer.reserve (len +1);
			freadOrDie (buffer, len, f);
			buffer.push_back (0);           // this makes it a proper C string

			// parse into lines
			vector<char *> lines;
			lines.reserve (len / 20);
			strtok (&buffer[0], "\r\n", lines);
			return lines;
		}

		// determine mlf entry lines range
		// lines range: [s,e)
		size_t getnextmlfstart (vector<char*> & lines, size_t s)
		{
			// determine lines range
			size_t e;
			for (e = s ; ; e++)
			{
				if (e >= lines.size()) malformed ("unexpected end in mid-utterance");
				char * ll = lines[e];
				if (ll[0] == '.' && ll[1] == 0) // end delimiter: a single dot on a line
					break;
			}
			return (e + 1);
			// lines range: [s,e)
		}

		void parseentry (vector<char*> & lines, size_t & line)
		{
			assert (line < lines.size());
			string filename = lines[line++];
			while (filename == "#!MLF!#")   // skip embedded duplicate MLF headers (so user can 'cat' MLFs)
				filename = lines[line++];

			if (filename.length() < 3 || filename[0] != '"' || filename[filename.length()-1] != '"')
			{// due to some mlf file have write error, so skip malformed entry
				fprintf (stderr, "warning: filename entry (%s)\n", filename.c_str());
				size_t s = line;
				line = getnextmlfstart (lines, s);
				fprintf (stderr, "skip current mlf entry form line (%d) until line (%d).\n", s, line);
				return;
			}

			filename = filename.substr (1, filename.length() -2);   // strip quotes
			if (filename.find ("*/") == 0) filename = filename.substr (2);
			wstring key = msra::strfun::utf16 (regex_replace (filename, regex ("\\.[^\\.\\\\/:]*$"), string()));  // delete extension (or not if none)

			// determine lines range
			//size_t s = line;
			//size_t e;
			//for (e = s ; ; e++)
			//{
			//    if (s >= lines.size()) malformed ("unexpected end in mid-utterance");
			//    char * ll = lines[e];
			//    if (ll[0] == '.' && ll[1] == 0) // end delimiter: a single dot on a line
			//        break;
			//}
			//line = e + 1;
			// lines range: [s,e)

			// determine lines range
			size_t s = line;
			line = getnextmlfstart (lines, line);
			size_t e = line - 1;
			// lines range: [s,e)

			vector<ENTRY> & entries = (*this)[key];    // this creates a new entry
			if (!entries.empty()) malformed (msra::strfun::strprintf ("duplicate entry '%S'", key.c_str()));
			entries.resize (e-s);
			vector<char*> toks;
			for (size_t i = s; i < e; i++)
			{
				strtok (lines[i], " \t", toks);
				if (statelistmap.size() == 0)
					entries[i-s].parse (toks);
				else
					entries[i-s].parsewithstatelist (toks, statelistmap);
			}
		}

	public:

		// return if input statename is sil state (hard code to compared first 3 chars with "sil")
		bool issilstate (const string & statename) const    // (later use some configuration table)
		{
			return (statename.size() > 3 && statename.at(0) == 's' && statename.at(1) == 'i' && statename.at(2) == 'l');
		}

		vector<bool> issilstatetable;       // [state index] => true if is sil state (cached)

		// return if input stateid represent sil state (by table lookup)
		bool issilstate (const size_t id) const
		{
			assert (id < issilstatetable.size());
			return issilstatetable[id];
		}

		// constructor reads multiple MLF files
		htkmlfreader (const vector<wstring> & paths, const wstring & stateListPath = L"")
		{
			// read state list
			if (stateListPath != L"")
				readstatelist (stateListPath);

			// read MLF(s) --note: there can be multiple, so this is a loop
			foreach_index (i, paths)
				read (paths[i]);
		}

		// note: this function is not designed to be pretty but to be fast
		void read (const wstring & path)
		{
			fprintf (stderr, "htkmlfreader: reading MLF file %S ...", path.c_str());
			curpath = path;         // for error messages only

			vector<char> buffer;    // buffer owns the characters--don't release until done
			vector<char*> lines = readlines (path, buffer);

			if (lines.empty() || strcmp (lines[0], "#!MLF!#")) malformed ("header missing");

			// parse entries
			size_t line = 1;
			while (line < lines.size())
				parseentry (lines, line);

			curpath.clear();
			fprintf (stderr, " total %d entries\n", size());
		}

		// [ganl] read state list, index is from 0
		void readstatelist (const wstring & stateListPath = L"")
		{
			if (stateListPath != L"")
			{
				vector<char> buffer;    // buffer owns the characters--don't release until done
				vector<char*> lines = readlines (stateListPath, buffer);
				size_t index;
				issilstatetable.reserve (lines.size());
				for (index = 0; index < lines.size(); index++)
				{
					statelistmap[lines[index]] = index;
					issilstatetable.push_back (issilstate (lines[index]));
				}
				if (index != statelistmap.size())
					throw std::runtime_error (msra::strfun::strprintf ("readstatelist: lines (%d) not equal to statelistmap size (%d)", index, statelistmap.size()));
				if (statelistmap.size() != issilstatetable.size())
					throw std::runtime_error (msra::strfun::strprintf ("readstatelist: size of statelookuparray (%d) not equal to statelistmap size (%d)", issilstatetable.size(), statelistmap.size()));
				fprintf (stderr, "total %d state names in state list %S\n", statelistmap.size(), stateListPath.c_str());
			}
		}

		// return state num: varify the fintune layer dim
		size_t getstatenum () const
		{
			return statelistmap.size();
		}

		size_t getstateid (string statename)        // added by Hang Su adaptation
		{
			return statelistmap[statename];
		}
	};

};};    // namespaces
