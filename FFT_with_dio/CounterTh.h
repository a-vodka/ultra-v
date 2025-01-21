//---------------------------------------------------------------------------

#ifndef CounterThH
#define CounterThH
//---------------------------------------------------------------------------
#include <System.Classes.hpp>
//---------------------------------------------------------------------------
class TCounter : public TThread
{
protected:
	void __fastcall Execute();
public:
	__fastcall TCounter(bool CreateSuspended);
};
//---------------------------------------------------------------------------
#endif
