#include "evalhelper.h"
#include <iostream>
#include <fstream>
#include <sstream>

map<wstring, wstring>  readIdKeyVaulePair(wstring idwavFile)
{
	map<wstring, wstring> idmap;
	wstring line, key, value;
	wifstream myfile(idwavFile);
	if (myfile.is_open())
	{
		while (getline(myfile, line))
		{
			wstringstream iss(line);
			iss >> key;
			iss >> value;
			idmap.insert(std::pair<wstring, wstring>(key, value));
		}
		myfile.close();

	}
	else
	{
		cout << "Unable to open file" << endl;
	}
	return idmap;
}

vector<wstring> readKeyFromFile(wstring file)
{
	vector<wstring> contents;
	wifstream reader(file);
	wstring line;
	if (reader.is_open())
	{
		while (getline(reader, line))
		{
			contents.push_back(line);
		}
		reader.close();
	}
	else
	{
		cerr << "unable to open file" << endl;
	}
	return contents;
}
map<size_t, size_t> getSelectIndexTable(wstring statelist, wstring selectlist)
{
	map<size_t, size_t> dict;
	int ind = 0;
	vector<wstring> totalSenones= readKeyFromFile(statelist);
	vector<wstring> selectedSenones = readKeyFromFile(selectlist);
	for (size_t i = 0; i < totalSenones.size(); i++)
	{
		if (std::find(selectedSenones.begin(), selectedSenones.end(), totalSenones[i]) != selectedSenones.end())
		{// existed
			dict.insert(std::pair<size_t, size_t>(i, ind));
			ind++;
		}
	}
	return dict;
}

size_t obtainSenoneToStateMaping(wstring & senone2stateMapFile, map<size_t, size_t> &senone2stateMap)
{
	wifstream reader(senone2stateMapFile);
	
	map<wstring, size_t> stateIndexMap;
	wstring line, state, senone;
	if (reader.is_open())
	{
		size_t lineIndex = 0;
		while (getline(reader, line))
		{
			wstringstream wss(line);
			wss >> senone >> state;
			if (stateIndexMap.count(state) == 0)
			{
				stateIndexMap[state] = stateIndexMap.size();
			}
			senone2stateMap[lineIndex++] = stateIndexMap[state];
		}
	}
	else
	{
		cerr << "unable to open file" << endl;
	}
	return stateIndexMap.size();
}
void forwardPropagation_OutputFirstN(onlineeval *model, wstring featFile, wstring outputFile, unsigned int firstN)
{
	wstring featkind;
	unsigned int sampperiod;
	msra::dbn::matrix feat;
	msra::asr::htkfeatreader reader;
	auto path = reader.parse(featFile);
	reader.read(path, featkind, sampperiod, feat);

	model->evalOrderedFirstNToFile(feat, sampperiod, outputFile,firstN);
}
void EMStatistic(onlineeval *model, wstring featFile,wstring sidFeatFile, wstring outputFile, map<size_t,size_t> selectedIndMap, unsigned int firstN)
{
	wstring featkind;
	unsigned int sampperiod;
	msra::dbn::matrix feat;
	msra::asr::htkfeatreader reader;
	auto path = reader.parse(featFile);
	reader.read(path, featkind, sampperiod, feat);
	model->forwardPropagationGetEMStatistics(feat, sidFeatFile, selectedIndMap, firstN, outputFile);
}

void PosteriorForClusteredStates(onlineeval *model, wstring &featFile, wstring &outputFile, map<size_t, size_t> &senone2StateMap, size_t stateNumber)
{
	wstring featkind;
	unsigned int sampperiod;
	msra::dbn::matrix feat;
	msra::asr::htkfeatreader reader;
	auto path = reader.parse(featFile);
	reader.read(path, featkind, sampperiod, feat);
	model->forwardPropagationGetClusteredStatePosterior(feat, sampperiod, senone2StateMap, stateNumber, outputFile);
	//model->forwardPropagationGetEMStatistics(feat, senone2StateMap, outputFile);
}

void extractDnnBasedEMStatistics(wstring& szDNNModelPath, wstring &statelist, wstring &selectedList, wstring &idAsrFeatFile, wstring &idSidFeatFile, unsigned int firstN, wstring &outputDir, wstring &suffix)
{
	msra::dbn::model *s_pDNNModel = NULL;
	onlineeval *s_pEval = NULL;
	// load dnn model
	size_t cores = msra::parallel::determine_num_cores();
	msra::parallel::set_cores(cores);
	std::cout << "number of cores:" << cores << endl;
	s_pDNNModel = new msra::dbn::model(szDNNModelPath);
	s_pDNNModel->entercomputation(0);
	s_pEval = new onlineeval(*s_pDNNModel, false);

	map<wstring, wstring> idAsrFeatMap = readIdKeyVaulePair(idAsrFeatFile);
	map<wstring, wstring> idSidFeatMap = readIdKeyVaulePair(idSidFeatFile);
	map<size_t, size_t> selectedIndMap = getSelectIndexTable(statelist, selectedList);

	for (std::map<wstring, wstring>::iterator iter = idAsrFeatMap.begin(); iter != idAsrFeatMap.end(); iter++)
	{
		if (idSidFeatMap.find(iter->first) != idSidFeatMap.end())
		{
			wstring sidFeatFile = idSidFeatMap[iter->first];
			wstring outputFile = outputDir + L"\\" + iter->first + L"." + suffix;
			EMStatistic(s_pEval, iter->second, sidFeatFile, outputFile, selectedIndMap, firstN);
			// process one utterance
			//	forwardPropagation_OutputFirstN(s_pEval, iter->second, outputFile, firstN);
		}
	}
	s_pDNNModel->exitcomputation();
}

void obtainClusteredDnnPosterior(wstring&dnnModelPath, wstring &senone2stateMapFile, wstring &idFeatFile, wstring &outDir, wstring &suffix)
{
	msra::dbn::model *s_pDNNModel = NULL;
	onlineeval *s_pEval = NULL;
	// load dnn model
	size_t cores = msra::parallel::determine_num_cores();
	msra::parallel::set_cores(cores);
	std::cout << "number of cores:" << cores << endl;
	s_pDNNModel = new msra::dbn::model(dnnModelPath);
	s_pDNNModel->entercomputation(0);
	s_pEval = new onlineeval(*s_pDNNModel, false);

	map<wstring, wstring> idFeatList = readIdKeyVaulePair(idFeatFile);
	map<size_t, size_t> senone2stateMap;
	size_t totalStateNumber=obtainSenoneToStateMaping(senone2stateMapFile,senone2stateMap);

	for (std::map<wstring, wstring>::iterator iter = idFeatList.begin(); iter != idFeatList.end(); ++iter)
	{
		wstring outputFile = outDir + L"\\" + iter->first + L"." + suffix;
		PosteriorForClusteredStates(s_pEval, iter->second, outputFile, senone2stateMap, totalStateNumber);
	}

}
int wmain(int argc, wchar_t* argv[])
{


	wstring command = argv[1];
	if (command.compare(L"--dnnEMStats") == 0)
	{
		if (argc < 9)
		{
			cout << "usage: exe --dnnEMStats dnnmodel statelist selectedStates  idAsrFeatFile idSidFeatFile firstN (0) outputDir suffix(stat0)";
			return 0;
		}
		wstring szDNNModelPath = argv[2]; // dnn model path
		wstring statelist = argv[3]; // statelist
		wstring selectedList = argv[4]; // selected senone list
		wstring idAsrFeatFile = argv[5];  // asr feature file
		wstring idSidFeatFile = argv[6]; // sid feature file
		unsigned int firstN = (unsigned int)_wtoi(argv[7]); // output only first N posteriors
		wstring outputDir = argv[8]; // output EM statistic
		wstring suffix = argv[9]; //stat0
		extractDnnBasedEMStatistics(szDNNModelPath, statelist, selectedList, idAsrFeatFile, idSidFeatFile, firstN, outputDir, suffix);
	}
	else if (command.compare(L"--clusteredDnnPosterior") == 0)
	{
		if (argc < 6)
		{
			cout << "usage: exe --clusteredDnnPosterior dnnmodel senone2stateMap idFeatFile outDir suffix" << endl;
			return 0;
		}
		wstring szDnnModelPath = argv[2];
		wstring senone2stateMap = argv[3];
		wstring idFeatFile = argv[4];
		wstring outDir = argv[5];
		wstring suffix = argv[6];
		obtainClusteredDnnPosterior(szDnnModelPath,senone2stateMap,idFeatFile,outDir,suffix);
	}
	else
	{
		cout << "unsupported command" << endl;
		cout << "usage: exe --dnnEMStats dnnmodel statelist selectedStates  idAsrFeatFile idSidFeatFile firstN (0) outputDir suffix(stat0)";
		cout << "usage: exe --clusteredDnnPosterior dnnmodel senone2stateMap idFeatFile outDir suffix" << endl;
		return 0;
	}



}