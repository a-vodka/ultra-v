//---------------------------------------------------------------------------

#ifndef OcsiloscopeProccessH
#define OcsiloscopeProccessH
//---------------------------------------------------------------------------
#include <System.Classes.hpp>
//---------------------------------------------------------------------------
class TOcsiloscopeProccess : public TThread
{
protected:
	void __fastcall Execute();
	void __fastcall plot();
public:
	__fastcall TOcsiloscopeProccess(bool CreateSuspended);
};
//---------------------------------------------------------------------------
#endif
