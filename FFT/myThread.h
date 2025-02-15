//---------------------------------------------------------------------------

#ifndef myThreadH
#define myThreadH
//---------------------------------------------------------------------------
#include <Classes.hpp>
#include "adaADC.h"
#include "fft_struct.h"
#include <list>


#include <iostream>
 using namespace std;




//---------------------------------------------------------------------------
class MyThread : public TThread
{
private:
protected:
    void __fastcall Execute();
public:
    __fastcall MyThread(bool CreateSuspended);
    AdaADC ada;
    list<channels> vect;
    TDateTime dt;
};
//---------------------------------------------------------------------------
#endif
