#include "I_vector.h"
#include <iostream>
#include <algorithm>
#include <map>
#include <fstream>
#include <sstream>

using namespace std;

map<string, string>  readIdKeyVaulePair(char* idwavFile)
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
			iss >> value;
			idmap.insert(std::pair<string, string>(key, value));
		}
		myfile.close();

	}
	else
	{
		cout << "Unable to open file" << endl;
	}
	return idmap;
}

int main(int argc, char* argv[])
{
	if (argc < 6)
	{
		cout << "usage: exe UBMmodel TMmodel NatureOrder(F|T) rawFeatOrStatistic(feat|stat) idWavFile ivecDir suffix(gmm.ivec)";
	}
	char* UBMModel = argv[1];
	char* TModel = argv[2];
	bool natureorder = strcmp(argv[3], "F") == 0 ? false : true; //HTK feature read order T for naturereaderorder
	bool inputIsStatistis = strcmp(argv[4], "stat") == 0 ? true:false;// input is raw feature or statistic file
	string idFeatFile = argv[5];
	string ivecDir = argv[6];
	string suffix = argv[7];

	I_vector ivectorExtractor(natureorder);
	ivectorExtractor.Initialize(UBMModel, TModel);
	map<string, string> IdFeatMap = readIdKeyVaulePair(&idFeatFile[0]);
	for (std::map<string, string>::iterator iter = IdFeatMap.begin(); iter != IdFeatMap.end(); iter++)
	{
		string ivecPath = ivecDir + "\\" + iter->first +"."+ suffix;
		printf("process utterance:%s\n", iter->second.c_str());
		ivectorExtractor.IvectorEstimation(&iter->second[0], &ivecPath[0], inputIsStatistis);
	}

}