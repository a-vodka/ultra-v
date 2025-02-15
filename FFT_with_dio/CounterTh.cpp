//---------------------------------------------------------------------------

#include <vcl.h>
#pragma hdrstop
#include "procHolder.h"
#include "CounterTh.h"
#pragma package(smart_init)
//---------------------------------------------------------------------------

//   Important: Methods and properties of objects in VCL can only be
//   used in a method called using Synchronize, for example:
//
//      Synchronize(&UpdateCaption);
//
//   where UpdateCaption could look like:
//
//      void __fastcall TCounter::UpdateCaption()
//      {
//        Form1->Caption = "Updated in a thread";
//      }
//---------------------------------------------------------------------------

__fastcall TCounter::TCounter(bool CreateSuspended)
	: TThread(CreateSuspended)
{
}
//---------------------------------------------------------------------------
void __fastcall TCounter::Execute()
{
	NameThreadForDebugging("CounterThread");
	double time_s, time_e;
	myTh->adaTh->ada.startCnt();
	while ( !Terminated )
	{
		time_e = time_s;
		time_s = GetTime();
		double dt = 24*3600*(time_s - time_e);
 		EnterCriticalSection(&myTh->counter_cs);
			unsigned long ncount = myTh->adaTh->ada.getCnt();
			myTh->adaTh->ada.resetCnt();
		LeaveCriticalSection(&myTh->counter_cs);
		myTh->adaTh->inp_freq = ncount ;
		Sleep(500);


    }
}
//---------------------------------------------------------------------------
