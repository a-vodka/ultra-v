//---------------------------------------------------------------------------


#ifndef procHolderH
#define procHolderH

#include "myThread.h"
#include "PostProcTh.h"
#include "FourierProccess.h"
#include "OcsiloscopeProccess.h"
#include "CounterTh.h"

class TProccessHolder
{
	MyThread* m_adaTh;
	TPostProcThread* m_postTh;
	TFourierProcces* m_fourierTh;
	TOcsiloscopeProccess* m_osciloscope;
	TCounter* m_counterTh;
public:

	CRITICAL_SECTION CS;
	CRITICAL_SECTION CS2;

	CRITICAL_SECTION counter_cs;

	TProccessHolder();
	~TProccessHolder();

	__property MyThread* adaTh  = { read= m_adaTh};
	__property TPostProcThread* postTh  = { read= m_postTh};
	__property TFourierProcces* fourierTh  = { read= m_fourierTh};
	__property TCounter* counterTh  = { read= m_counterTh};
};

extern PACKAGE TProccessHolder *myTh;

//---------------------------------------------------------------------------
#endif
