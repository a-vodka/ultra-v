//---------------------------------------------------------------------------
#include "FFT_struct.h"
#ifndef PostProcThH
#define PostProcThH
//---------------------------------------------------------------------------
#include <Classes.hpp>
#include <list>
#include <deque>
#include <fstream>

using namespace std;

#define F_OPEN WM_USER + 1
#define F_CLOSE WM_USER + 2

//---------------------------------------------------------------------------
class TPostProcThread : public TThread
{
 void __fastcall ThUpdate (tagMSG &);
 /*BEGIN_MESSAGE_MAP
 MESSAGE_HANDLER ( F_OPEN , TMessage , ThUpdate )
 MESSAGE_HANDLER ( F_CLOSE , TMessage , ThUpdate )

 END_MESSAGE_MAP(TThread)*/

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
  AnsiString filename;
  ofstream pFile;

protected:
	void __fastcall Execute();
	void __fastcall CalibrateInAccel(channels *ch);
	void __fastcall CalibrateInV(channels *ch);
	void __fastcall ConvertTo(UINT n);
	void OpenFile(AnsiString S);
	void closeFile();

public:
   __fastcall TPostProcThread(bool CreateSuspended);
   bool isOpenFile();
   volatile  bool writeEnabled;
   void WriteToFile(channels ch);
   deque<channels> resultList;
   deque<channels> fourierList;
   __property int resType = { read = resultType, write=setResType};
   void setFileName(AnsiString s);

};
//---------------------------------------------------------------------------
#endif
