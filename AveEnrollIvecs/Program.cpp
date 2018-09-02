#include "Common.h"
#include <fstream>
#include <iostream>
#include <string>
#include "HTKFile.h"
#include <map>
#include <vector>
#include <iostream>
#include <sstream>
#include "direct.h"

using namespace std;

// collect ivectors and average them
map<string, string>  readIdKeyVaulePair(string idwavFile)
{
	map<string, string> idmap;
	string line, key, value;
	ifstream myfile(idwavFile);
	if (myfile.is_open())
	{
		while (getline(myfile, line))
		{
			istringstream iss(line);
			iss >> key;
			while (iss)
			{
				iss >> value;
				idmap[key] = value;
			}
		}
		myfile.close();

	}
	else
	{
		cout << "Unable to open file" << endl;
	}
	return idmap;
}

map<string, vector<string>> readKeyValuePairs(const string speakerUttPiarFile, const bool uniqueOnly=true)
{
	map<string, vector<string>> speakerMap;
	ifstream reader(speakerUttPiarFile);
	string line;
	string key, value;
	if (reader.is_open())
	{
		while (getline(reader, line))
		{
			istringstream iss(line);
			iss >> key >>value;

			// the first utterance
			if (speakerMap.find(key) == speakerMap.end())
			{
				speakerMap[key].push_back(value);
			}
			else
			{
				// check replicate cases
				if (uniqueOnly)
				{
					if (find(speakerMap[key].begin(), speakerMap[key].end(), value) != speakerMap[key].end())
					{
						cerr << "this value is exsisted and will be ignored: " << value.c_str() << endl;
						continue;
					}
				}
					speakerMap[key].push_back(value);
				
			}
		}
		reader.close();
	}
	else
	{
		std::cerr << "can't open file: " << speakerUttPiarFile.c_str() << std::endl;
	}


	return speakerMap;
}

void printMapKeyValuesPerLine(const map<string, vector<string>> _map, const string outdir, const string suffix,const string outputFile)
{
	ofstream writer(outputFile);
	if (writer.is_open())
	{
		for ( map<string, vector<string>>::const_iterator iter = _map.begin(); iter != _map.end(); ++iter)
		{
			writer << outdir+"\\"+ iter->first+"."+suffix;
			for (vector<string>::const_iterator siter = iter->second.begin(); siter != iter->second.end(); ++siter)
			{
				writer << " "<<*siter;
			}
			writer << endl;
		}
		writer.close();
	}
	else
	{
		cerr << "can't open file: " << outputFile.c_str() << endl;
	}

}

void AverageIvectors(const map<string,vector<string>> spkIvesMap, const string outIvecDir, const string ivecSuffix)
{
	HTKhdr header;
	HTKFile ihtk;
	for (std::map<string, vector<string>>::const_iterator it = spkIvesMap.begin(); it != spkIvesMap.end(); ++it)
	{
		string outputIvePath = outIvecDir + "\\" + it->first + "." + ivecSuffix;
		int featDim = ihtk.readHTKHeader(&it->second[0][0], header);
		float *Sum = new float[featDim];
		ihtk.readHTKData(&it->second[0][0], Sum, header.nSamples);
		float *tempData = new float[featDim];
		for (int i = 1; i < it->second.size(); i++)
		{
			ihtk.readHTKData(&it->second[i][0], tempData, 1);
			for (int j = 0; j < featDim; j++)
			{
				Sum[j] += tempData[j];
			}
		}
		for (int j = 0; j < featDim; j++)
			Sum[j] /= it->second.size();

		ihtk.writeHTKFeats(&outputIvePath[0], Sum, header, featDim);
		delete[] Sum;
		delete[] tempData;
	}
}

void printKeyValuePerLine(const map<string, int>_map, const string outFile)
{
	ofstream writer(outFile);
	if (writer.is_open())
	{
		for (map<string, int>::const_iterator iter = _map.begin(); iter != _map.end(); ++iter)
		{
			writer << iter->first << " " << iter->second << endl;
		}
		writer.close();
	}
	else
	{
		cerr << "can't open file: " << outFile.c_str() << endl;
	}
}
int main(int argc, char* argv[])
{

	if (argc < 6)
	{
		cerr << "usage: exe speakerUttMap(first column is speakerId, second column is a segment name) segmentIvecFileMap(first column is segmentname, the second is path) speakerIvecFile(each line consits of speakerId and uttIvecPath) ivecdir IvecTag(suffix of ivecs, default is ivec) speaker_num_file" << endl;
		return 0;
	}
	string spkeruttfile = argv[1];
	string uttIdItemMapFile = argv[2];
	string ivecdir = argv[3];
	string ivecTag = argv[4];
	string num_utt = argv[5];
	_mkdir(ivecdir.c_str());

	map<string, vector<string>> speakerItemMaps = readKeyValuePairs(spkeruttfile);
	map<string, string> uttIdItemPair = readIdKeyVaulePair(uttIdItemMapFile);
	
	map<string, vector<string>> speakerutt;
	for (map<string, vector<string>>::iterator iter = speakerItemMaps.begin(); iter != speakerItemMaps.end(); ++iter)
	{
		for (vector<string>::iterator siter = iter->second.begin(); siter != iter->second.end(); ++siter){
			if (uttIdItemPair.find(*siter) != uttIdItemPair.end())
				speakerutt[iter->first].push_back(uttIdItemPair[*siter]);
		}
	}

	AverageIvectors(speakerutt,ivecdir,ivecTag);
	map<string, int> spkNumMap;
	for (map<string, vector<string>>::iterator iter = speakerutt.begin(); iter != speakerutt.end(); ++iter)
		spkNumMap[iter->first] = iter->second.size();
	printKeyValuePerLine(spkNumMap, num_utt);
}