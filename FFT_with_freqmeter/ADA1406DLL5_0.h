#ifndef __ADC14USB_H
#define __ADC14USB_H
#ifndef __ADC14USB__
#define __ADC14USBLIB__ __declspec(dllimport)
#else
#define __ADC14USBLIB__ __declspec(dllexport)
#endif

#include <setupapi.h> // Структуры поиска имени устройства.
typedef unsigned short USHORT;


#define	F_MAX	350000

#define		MANUFACTURER			1				// Индекс дескриптора строки описывающий изготовителя.
#define		PRODUCT						2				// Индекс дескриптора строки описывающий продукт.
#define		SERIAL_NUMBER			3				// Индекс дескриптора строки содержащий серийный номер.
#define		ENGLISH_US				0x0409	// Идентификатор англиского США.
#define		RUSSIAN						0x0419	// Идентификатор руского языка.
#define		SER_NUMBER_EEPROM	1				// Серийный номер EEPROM.

extern "C"
{
	USHORT	__ADC14USBLIB__	__stdcall ADC14USB_GetVersion();
	BOOL		__ADC14USBLIB__	__stdcall ADC14USB_OpenDevice(UCHAR NumberDevice);
	BOOL		__ADC14USBLIB__ __stdcall ADC14USB_GetNumberDevices(UCHAR* pbCounteRet);
	BOOL		__ADC14USBLIB__	__stdcall ADC14USB_CloseDevice();
	BOOL		__ADC14USBLIB__	__stdcall ADC14USB_GetInformationDevice(UCHAR nInd, USHORT nLang, PWCHAR pString);
	BOOL		__ADC14USBLIB__	__stdcall ADC14USB_GetInformationDeviceANSI(UCHAR nInd, USHORT nLang, PUCHAR pString);
	BOOL		__ADC14USBLIB__	__stdcall ADC14USB_ReadFirmwareVersion(UCHAR nInd, UCHAR* pInBuf);
	BOOL		__ADC14USBLIB__	__stdcall ADC14USB_ReadEE(UCHAR nInd, UCHAR* pInBuf);
	BOOL		__ADC14USBLIB__	__stdcall ADC14USB_StartStopADC14(UCHAR nInd);
	BOOL		__ADC14USBLIB__	__stdcall ADC14USB_DMAMode(UCHAR nInd);
	BOOL		__ADC14USBLIB__	__stdcall ADC14USB_ReadData(UCHAR* pInBuf, DWORD* pBytesRead);
	BOOL		__ADC14USBLIB__	__stdcall ADC14USB_WriteSwitchStack(UCHAR* LogChan, USHORT Number);
	UCHAR		__ADC14USBLIB__	__stdcall ADC14USB_ReadLogicChannels(UCHAR* pInBuf, DWORD* pBytesRead, DWORD Timeout=INFINITE);
	UCHAR		__ADC14USBLIB__	__stdcall ADC14USB_ReadLogicChannels_1(short int* pInBuf, DWORD* pBytesRead, DWORD Timeout=INFINITE);
	BOOL		__ADC14USBLIB__	__stdcall ADC14USB_SetFreqADC(UINT Fdisc);
	BOOL		__ADC14USBLIB__	__stdcall ADC14USB_ProcessDioSingleRead(UCHAR nInd, UCHAR* pData);
	BOOL		__ADC14USBLIB__	__stdcall ADC14USB_ProcessDioRead(UCHAR* pInBuf, DWORD BytesRet);
	BOOL		__ADC14USBLIB__	__stdcall ADC14USB_ProcessDioWrite(UCHAR Data);
	BOOL		__ADC14USBLIB__	__stdcall ADC14USB_ProcessDioSingleWrite(UCHAR bIndex, USHORT bValue);
	BOOL		__ADC14USBLIB__	__stdcall ADC14USB_ProcessDACSingle(UCHAR bIndex, USHORT bValue,
																							BOOL bVoltage=0, short nVoltage=0);
	BOOL		__ADC14USBLIB__	__stdcall ADC14USB_ProcessDACWrite(USHORT nVoltage, UCHAR Dac);
	BOOL		__ADC14USBLIB__	__stdcall ADC14USB_SOFTReadADC(short int* pData, UCHAR PhChan);
	BOOL		__ADC14USBLIB__	__stdcall ADC14USB_QuickReadDIN(UCHAR* pInBuf, DWORD* pBytesRead);
	BOOL		__ADC14USBLIB__ __stdcall ADC14USB_StopQkRdDIN();
	BOOL		__ADC14USBLIB__ __stdcall ADC14USB_ReadCntN(UCHAR n, unsigned long *pData);
	BOOL		__ADC14USBLIB__ __stdcall ADC14USB_ResetCntN(UCHAR n);
	BOOL		__ADC14USBLIB__ __stdcall ADC14USB_SetupCntN(UCHAR n,BOOL gated,BOOL level,BOOL dir);
	BOOL		__ADC14USBLIB__ __stdcall ADC14USB_StartStopCntN(UCHAR n,BOOL mode);
}


class ADC14UCB  
{
public:
	BOOL		fStartReadDIN;
	BOOL		StartStopADC14(UCHAR nInd);
	BOOL		OpenDevice(UCHAR NumberDevice);
	ADC14UCB();
	virtual ~ADC14UCB();
	UCHAR		Mode_1;
	UCHAR		Mode_2;
	UCHAR		CCNT0;
	UCHAR		CCNT1;
	USHORT		DiskBak;
	BOOL		fWriteSwitchStack;
	HANDLE	hReadPipe2;
	BOOL		WritePipe1(UCHAR* pInBuf, DWORD* pBytesWrite);
	HANDLE	hWritePipe1;
	BOOL		ReadPipe1(PVOID pInBuf, DWORD* pBytesRead);
	HANDLE	hReadPipe1;
	BOOL		OpenReadWrite(HANDLE *hDeviceReadWrite, char Pipe[32], DWORD dwFlagsAtr);
	BOOL		fOpen;
	BOOL		fStart;
	HANDLE	hDevice;
	char*		pDeviceName;
	BOOL		PLDMode(UCHAR bInd, UCHAR bValue);
};

#endif
