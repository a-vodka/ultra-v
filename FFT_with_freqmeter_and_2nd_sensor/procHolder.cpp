//---------------------------------------------------------------------------

#pragma hdrstop

#include "procHolder.h"
//---------------------------------------------------------------------------
#pragma package(smart_init)

TProccessHolder *myTh;

TProccessHolder::TProccessHolder()
{
	InitializeCriticalSection(&CS);
	InitializeCriticalSection(&CS2);

	m_adaTh = new MyThread(true);
	m_postTh = new TPostProcThread(true);
	m_fourierTh = new TFourierProcces(true);
	m_osciloscope = new TOcsiloscopeProccess(true);

	m_postTh->Start();
	m_fourierTh->Start();
	m_osciloscope->Start();

}

TProccessHolder::~TProccessHolder()
{
	m_postTh->Terminate();
	m_postTh->WaitFor();

	m_fourierTh->Terminate();
	m_fourierTh->WaitFor();

	m_osciloscope->Terminate();
	m_osciloscope->WaitFor();

	m_adaTh->Resume();
	m_adaTh->Terminate();
	m_adaTh->WaitFor();


	delete m_adaTh;
	delete m_postTh;
	delete m_fourierTh;
	delete m_osciloscope;

	DeleteCriticalSection(&CS);
	DeleteCriticalSection(&CS2);
}
