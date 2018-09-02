#include "HTKFile.h"
#include <vector>
#include <iostream>

using namespace::std;

HTKFile::HTKFile()
{
}


HTKFile::~HTKFile()
{
}

int HTKFile::readHTKHeader(const char* featpath, HTKhdr &header)
{
	FILE *fp;
	bool naturalReadOrder;
	bool compressed;
	int featDim;
	if (fopen_s(&fp, featpath, "rb"))
	{
		printf("can't open file %s\n", featpath);
		return 0;
	}
	fread_s(&header, sizeof(HTKhdr), sizeof(HTKhdr), 1, fp);



	int tempPeriod = header.sampPeriod;
	int* temppoint = &tempPeriod;
	Swap32(temppoint);
	naturalReadOrder = header.sampPeriod > tempPeriod;

	if (!naturalReadOrder)
	{
		Swap32(&header.nSamples);
		Swap32(&header.sampPeriod);
		Swap16(&header.sampSize);
		Swap16(&header.sampKind);
	}

	compressed = (header.sampKind & (int)1024) != 0;
	featDim = compressed ? header.sampSize / sizeof(short) : header.sampSize / sizeof(float);
	if (compressed)
	{
		header.nSamples -= 4;
	}
	fclose(fp);
	return featDim;
}
void HTKFile::readHTKData(const char* featpath, float* data, int m)
{
	FILE *fp;
	HTKhdr header;
	bool naturalReadOrder, compressed;
	int featDim;
	vector<float> frameData,a,b;
	vector<short> shortData;

	if (fopen_s(&fp, featpath, "rb"))
	{
		printf("can't open file %s\n", featpath);
		return;
	}
	fread_s(&header, sizeof(HTKhdr), sizeof(HTKhdr), 1, fp);
	int tempPeriod = header.sampPeriod;
	int* temppoint = &tempPeriod;
	Swap32(temppoint);
	naturalReadOrder = header.sampPeriod > tempPeriod;


	if (!naturalReadOrder)
	{
		Swap32(&header.nSamples);
		Swap32(&header.sampPeriod);
		Swap16(&header.sampSize);
		Swap16(&header.sampKind);
	}

	compressed = (header.sampKind & (int)1024) != 0;
	featDim = compressed ? header.sampSize / sizeof(short) : header.sampSize / sizeof(float);
	frameData.resize(featDim);

	if (compressed)
	{
		a.resize(header.sampSize);
		b.resize(header.sampSize);
		shortData.resize(header.sampSize);

		header.nSamples -= 4;
		// a vector value
		if (fread_s(&frameData[0], header.sampSize, sizeof(float), featDim, fp) != featDim)
		{
			printf("Error loading feature frame\n");
		}
		if (!naturalReadOrder)
		{
			int *temp = (int*)(&frameData[0]);
			for (int j = 0; j < featDim; j++)
			{
				Swap32(temp + j);
				a[j] = frameData[j];
			}
		}
		// b vector value
		if (fread_s(&frameData[0], header.sampSize, sizeof(float), featDim, fp) != featDim)
		{
			printf("Error loading feature frame\n");
		}
		if (!naturalReadOrder)
		{
			int *temp = (int*)(&frameData[0]);
			for (int j = 0; j < featDim; j++)
			{
				Swap32(temp + j);
				b[j] = frameData[j];
			}
		}
	}


	if (1 != header.nSamples)
	{
		printf("the sample number not equals to 1! %s\n", featpath);
		fclose(fp);
		return;
	}


	for (int i = 0; i < header.nSamples; i++)
	{
		if (compressed)
		{
			if (fread_s(&shortData[0], header.sampSize, sizeof(short), featDim, fp) != featDim)
			{
				printf("Error loading feature frame\n");
			}
			if (!naturalReadOrder)
			{
				short *temp = (short*)(&shortData[0]);
				for (int j = 0; j < featDim; j++)
				{
					Swap16(temp + j);
					data[i*featDim+ j]= (shortData[j] + b[j]) / a[j];
				}
			}

		}
		else
		{

			if (fread_s(&frameData[0], header.sampSize, sizeof(float), featDim, fp) != featDim)
			{
				printf("Error loading feature frame\n");
			}
			if (!naturalReadOrder)
			{
				int *temp = (int*)(&frameData[0]);
				for (int j = 0; j < featDim; j++)
				{
					Swap32(temp + j);
					data[i*featDim + j] = frameData[j];
				//	printf("%f\n", data[i*featDim + j]);
				}
			}
		}
	}

	fclose(fp);
	frameData.clear();
	if (compressed) shortData.clear();
}
void HTKFile::writeHTKFeats(char* featpath, float data[], HTKhdr header, int featDim)
{
	FILE *fp;
	int number;
	vector<float> swapvec;
	swapvec.resize(featDim);
	for (size_t i = 0; i < featDim; ++i)
	{
		swapvec[i] = data[i];
		Swap32((int *)&swapvec[i]);
	}
	Swap32(&header.nSamples);
	Swap32(&header.sampPeriod);
	Swap16(&header.sampSize);
	Swap16(&header.sampKind);

	fp = fopen(featpath, "wb");

	if (fwrite(&header, sizeof(header), 1, fp) != 1)
	{
		printf("write falied!\n");
	}
	for (int i = 0; i < featDim; i++)
	{
		if (fwrite(&swapvec[i], sizeof(swapvec[i]), 1, fp) != 1)
		{
			printf("write falied\n");
		}
	}
	fclose(fp);
	swapvec.clear();
}