//---------------------------------------------------------------------------


#ifndef procHolderH
#define procHolderH

#include "myThread.h"
#include "PostProcTh.h"
#include "FourierProccess.h"
#include "OcsiloscopeProccess.h"

class TProccessHolder
{
	MyThread* m_adaTh;
	TPostProcThread* m_postTh;
	TFourierProcces* m_fourierTh;
	TOcsiloscopeProccess* m_osciloscope;

public:

	CRITICAL_SECTION CS;
	CRITICAL_SECTION CS2;

	TProccessHolder();
	~TProccessHolder();

	__property MyThread* adaTh  = { read= m_adaTh};
	__property TPostProcThread* postTh  = { read= m_postTh};
	__property TFourierProcces* fourierTh  = { read= m_fourierTh};
};

extern PACKAGE TProccessHolder *myTh;

//---------------------------------------------------------------------------
#endif
