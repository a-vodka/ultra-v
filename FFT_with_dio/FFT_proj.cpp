//---------------------------------------------------------------------------

#include <vcl.h>
#include "procHolder.h"
#pragma hdrstop
//---------------------------------------------------------------------------















USEFORM("FFT.cpp", Form1);
//---------------------------------------------------------------------------
WINAPI WinMain(HINSTANCE, HINSTANCE, LPSTR, int)
{
		try
		{
				 Application->Initialize();
				 Application->CreateForm(__classid(TForm1), &Form1);
		myTh = new TProccessHolder();
				 Application->Run();
				 delete myTh;
		}
        catch (Exception &exception)
        {
                 Application->ShowException(&exception);
        }
        catch (...)
        {
                 try
                 {
                         throw Exception("");
                 }
                 catch (Exception &exception)
                 {
                         Application->ShowException(&exception);
                 }
        }
        return 0;
}
//---------------------------------------------------------------------------
