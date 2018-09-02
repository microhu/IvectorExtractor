#pragma once
#include "Common.h"


typedef struct {              /* HTK File Header */
	int nSamples;
	int sampPeriod;
	short sampSize;
	short sampKind;
} HTKhdr;

class HTKFile
{
public:
	HTKFile();
	~HTKFile();
	int readHTKHeader(const char* featpath, HTKhdr &header);
	void readHTKData(const char* featpath, float* data, int m);
	void writeHTKFeats(char* featpath, float data[], HTKhdr header, int featDim);

};

