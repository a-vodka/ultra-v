//---------------------------------------------------------------------------
#include "FFT_struct.h"
#ifndef PostProcThH
#define PostProcThH
//---------------------------------------------------------------------------
#include <Classes.hpp>
#include <list>
#include <deque>
#include <stdio.h>

 using namespace std;
//---------------------------------------------------------------------------
class TPostProcThread : public TThread
{
  typedef struct tagTHREADNAME_INFO
  {
    DWORD dwType;     // must be 0x1000
    LPCSTR szName;    // pointer to name (in user addr space)
    DWORD dwThreadID; // thread ID (-1=caller thread)
    DWORD dwFlags;    // reserved for future use, must be zero
  } THREADNAME_INFO;
private:
  void SetName();
  void makeAccel(UINT n);
  void makeVibro(UINT n);
  void makeDispl(UINT n);
  void makeADC(UINT n);
  void makeVoltage(UINT n);
  channels startVelo;
  channels startDisp;
  double TmaxV;
  double TmaxD;
  int resultType;
  void setResType(int);

  FILE *pFile;
  
protected:
	void __fastcall Execute();
	void __fastcall CalibrateInAccel(channels *ch);
	void __fastcall CalibrateInV(channels *ch);
	void __fastcall ConvertTo(UINT n);
public:
   __fastcall TPostProcThread(bool CreateSuspended);

   void OpenFile(AnsiString S);
   bool writeEnabled;
   void closeFile();
   void WriteToFile(channels ch);
   deque<channels> resultList;
   deque<channels> fourierList;
   __property int resType = { read = resultType, write=setResType};


};
//---------------------------------------------------------------------------
#endif
