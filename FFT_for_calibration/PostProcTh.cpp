//---------------------------------------------------------------------------

#include <vcl.h>
#include <math.hpp>
#include <iomanip>
#pragma hdrstop

#include "PostProcTh.h"
#include "myThread.h"
#include "procHolder.h"
#pragma package(smart_init)
//---------------------------------------------------------------------------

//   Important: Methods and properties of objects in VCL can only be
//   used in a method called using Synchronize, for example:
//
//      Synchronize(UpdateCaption);
//
//   where UpdateCaption could look like:
//
//      void __fastcall TPostProcThread::UpdateCaption()
//      {
//        Form1->Caption = "Updated in a thread";
//      }
//---------------------------------------------------------------------------

__fastcall TPostProcThread::TPostProcThread(bool CreateSuspended)
    : TThread(CreateSuspended)
{
	resultType = 0;
	writeEnabled = false;
}
//---------------------------------------------------------------------------
void TPostProcThread::SetName()
{
    THREADNAME_INFO info;
    info.dwType = 0x1000;
    info.szName = "PostProcThread";
    info.dwThreadID = -1;
    info.dwFlags = 0;

    __try
    {
         RaiseException( 0x406D1388, 0, sizeof(info)/sizeof(DWORD),(DWORD*)&info );
    }
    __except (EXCEPTION_CONTINUE_EXECUTION)
    {
    }
}
//---------------------------------------------------------------------------
void __fastcall TPostProcThread::Execute()
{
    SetName();

	while ( !Terminated )
	{
		int n = myTh->adaTh->vect.size();
		if (n < 3 ) continue;

		EnterCriticalSection(&myTh->CS);
		EnterCriticalSection(&myTh->CS2);

		ConvertTo(n);

		LeaveCriticalSection(&myTh->CS);
		LeaveCriticalSection(&myTh->CS2);
	}
}
//---------------------------------------------------------------------------

void __fastcall  TPostProcThread::ConvertTo(UINT n)
{
	for (UINT i = 0; i < n; i++)
	{
		channels ch = myTh->adaTh->vect.front();

		if (writeEnabled)
			WriteToFile(ch);

		switch (resultType)
		{
			case 0: // ���������
				CalibrateInAccel(&ch);
				break;
			case 2: // ������
				CalibrateInV(&ch);
				break;

			default:
			case 1:  // ��� ���
				break;

		}
		resultList.push_back(ch);
		fourierList.push_back(ch);
		myTh->adaTh->vect.pop_front();
	}

}

void __fastcall TPostProcThread::CalibrateInAccel(channels *ch)
{
	CalibrateInV(ch);
//	ch->A = (ch->A - AZeroOffset)/ASensetivity;
	ch->B = (ch->B - BZeroOffset)/BSensetivity;
}

void __fastcall TPostProcThread::CalibrateInV(channels *ch)
{
	ch->A = TO_VOLT*ch->A;
	ch->B = TO_VOLT*ch->B;
}

void TPostProcThread::setResType(int type)
{
	resultType = type;
}

void TPostProcThread::OpenFile(AnsiString S)
{
	pFile.open(S.c_str(), ios::out);
	pFile << resetiosflags(ios_base::scientific);
	pFile << setprecision(10);
}
bool TPostProcThread::isOpenFile()
{
	return	pFile.is_open();
}

void TPostProcThread::WriteToFile(channels ch)
{
	pFile << (int)ch.A << "\t"
		  << (int)ch.B << "\t"
		  << ch.t 	   << endl;
//	 fprintf_s(pFile,"%d\t%d\t%d\t%e\n", (int)ch.A, (int)ch.B, (int)ch.C, ch.t);
}

void TPostProcThread::closeFile()
{
	if (isOpenFile())
		pFile.close();
}
