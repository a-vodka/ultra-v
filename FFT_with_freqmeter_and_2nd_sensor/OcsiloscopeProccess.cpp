//---------------------------------------------------------------------------

#include <vcl.h>
#include <deque>
#pragma hdrstop

#include "fft.h"
#include "procHolder.h"
#include "OcsiloscopeProccess.h"
#pragma package(smart_init)
using namespace std;
//---------------------------------------------------------------------------

//   Important: Methods and properties of objects in VCL can only be
//   used in a method called using Synchronize, for example:
//
//      Synchronize(&UpdateCaption);
//
//   where UpdateCaption could look like:
//
//      void __fastcall TOcsiloscopeProccess::UpdateCaption()
//      {
//        Form1->Caption = "Updated in a thread";
//      }
//---------------------------------------------------------------------------

__fastcall TOcsiloscopeProccess::TOcsiloscopeProccess(bool CreateSuspended)
	: TThread(CreateSuspended)
{
}
//---------------------------------------------------------------------------
void __fastcall TOcsiloscopeProccess::Execute()
{
	NameThreadForDebugging("OcsiloscopeProccess");

	while (!Terminated)
	{
		Synchronize(&plot);
		Sleep(100);
	}
}

void __fastcall TOcsiloscopeProccess::plot()
{
	deque<channels> *pDeque = & myTh->postTh->resultList;
	static int numadd = 0; // ������������ ����� �� �������
	EnterCriticalSection(&myTh->CS2);
	int n = pDeque->size();
	for (int i = 0; i < n; i++)
	{
		channels ch = pDeque->front();
		if (i % 20 == 0 ) // ��������� ������������ -- ����� ������ �������
		{
		   Form1->Series1->AddXY(ch.t,ch.A);
		   Form1->Series2->AddXY(ch.t,ch.B);
		   Form1->D_Series->AddXY(ch.t,ch.D);
		   Form1->E_Series->AddXY(ch.t,ch.E);
		   Form1->F_Series->AddXY(ch.t,ch.F);
		   numadd++;
		}
		pDeque->pop_front();
	}
	LeaveCriticalSection(&myTh->CS2);
	int numpoint = 200*(Form1->TrackBar1->Position);

	while (numadd > numpoint)
	{
		Form1->Series1->Delete(0);
		Form1->Series2->Delete(0);
		Form1->D_Series->Delete(0);
		Form1->E_Series->Delete(0);
		Form1->F_Series->Delete(0);
		numadd--;
	}

}
//---------------------------------------------------------------------------
