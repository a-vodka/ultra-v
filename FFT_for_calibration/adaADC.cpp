//---------------------------------------------------------------------------

#include <basetyps.h>
#include <windows.h>
#include <assert.h>

#include "ADA1406DLL5_0.h"
#pragma hdrstop

#include "adaADC.h"
#include "defines.h"
//---------------------------------------------------------------------------

#pragma package(smart_init)


void AdaADC::getDeviceInfo(AnsiString &manufact, AnsiString &product, AnsiString &sn)
{
    wchar_t str[64];
    BOOL succ = FALSE;

    ZeroMemory(str,sizeof(str));
    succ = ADC14USB_GetInformationDevice(MANUFACTURER, ENGLISH_US, str);
    assert(succ);
    manufact = str;

    ZeroMemory(str,sizeof(str));
    succ = ADC14USB_GetInformationDevice(PRODUCT, ENGLISH_US, str);
    assert(succ);
    product = str;

    ZeroMemory(str,sizeof(str));
    succ = ADC14USB_GetInformationDevice(SERIAL_NUMBER, ENGLISH_US, str);
    assert(succ);

    sn = str;

}

void AdaADC::readData(short int* ReadBuf, DWORD dwBytesRead)
{
	#ifdef NOADC
        Sleep(100);
	#else
		UCHAR ReadLogicCh;
		ReadLogicCh = ADC14USB_ReadLogicChannels_1(ReadBuf, &dwBytesRead);
		assert(ReadLogicCh==0);
	#endif
}

void AdaADC::openDevice(UCHAR NumberDevice)
{
    assert(ADC14USB_OpenDevice(NumberDevice)==0);

	UCHAR DataPhysCh[4] = {0x64, 0x65, 0x66, 0x67 }; // (X0,X1 усиление-1,однопроводный)
	NumbPhysCh = 2; // число каналов
	Fdisc = 5e4; // частота дискретизации

	#ifndef NOADC
	BOOL suc = ADC14USB_WriteSwitchStack(DataPhysCh, NumbPhysCh);
	assert(suc);

	suc = ADC14USB_SetFreqADC(Fdisc);

	assert(suc);
	#endif
}

void AdaADC::closeDevice()
{
    assert(ADC14USB_CloseDevice());
}

UCHAR AdaADC::getNumberDevices()
{
    UCHAR bCountRet;
    assert( ADC14USB_GetNumberDevices(&bCountRet) );
    return bCountRet;

}

AdaADC::AdaADC()
{
	nGetVer = ADC14USB_GetVersion();
	assert(nGetVer == 0x0400); // Not the correct version
}

AdaADC::~AdaADC()
{
 //	stopADC();
 //   closeDevice();
}

double AdaADC::getFreq()
{
 	int mult = (int)(ADA_BASE_FREQ/(double)Fdisc);
	double freq = (ADA_BASE_FREQ_C/(double)mult);

	return freq;
}

int AdaADC::getNumChannels()
{
    return NumbPhysCh;
}

void AdaADC::stopADC()
{
	#ifndef NOADC
		assert(ADC14USB_StartStopADC14(0));
	#endif
}
