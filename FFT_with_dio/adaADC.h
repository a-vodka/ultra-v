//---------------------------------------------------------------------------

#ifndef adaADCH
#define adaADCH
//---------------------------------------------------------------------------
#include <vcl.h>
#include <basetyps.h>
#include <windows.h>
#include <assert.h>
#include "ADA1406DLL5_0.h"

#pragma hdrstop

class AdaADC
{

int nGetVer;

UINT Fdisc; // частота дискретизации
UCHAR NumbPhysCh; // число каналов
UCHAR CntCh;    // Канал для счетчика частоты
public:
AdaADC();
~AdaADC();

UCHAR getNumberDevices();
void openDevice(UCHAR NumberDevice=0);
void closeDevice();

void getDeviceInfo(AnsiString &manufact, AnsiString &product, AnsiString &sn);
void readData(short *ReadBuf, DWORD dwBytesRead);

double getFreq();
int getNumChannels();
void stopADC();

void stopCnt();
void startCnt();
unsigned long getCnt();
void resetCnt();
};


#endif
