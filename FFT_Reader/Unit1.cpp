//---------------------------------------------------------------------------

#include <vcl.h>
#include <fstream.h>
#pragma hdrstop

#include "Unit1.h"
//---------------------------------------------------------------------------
#pragma package(smart_init)
#pragma link "Chart"
#pragma link "Series"
#pragma link "TeEngine"
#pragma link "TeeProcs"
#pragma resource "*.dfm"
TForm1 *Form1;
//---------------------------------------------------------------------------
__fastcall TForm1::TForm1(TComponent* Owner)
	: TForm(Owner)
{
}
//---------------------------------------------------------------------------
void __fastcall TForm1::Button1Click(TObject *Sender)
{
	if (OpenDialog1->Execute())
	{
		ifstream f(OpenDialog1->FileName.c_str());
		int a,b,c;
		double t;
		while (!f.eof())
		{
			f>>a>>b>>c>>t;
			Series1->AddXY(t,a);
			Series2->AddXY(t,b);
			Series3->AddXY(t,c);
		}
		f.close();
	}


}
//---------------------------------------------------------------------------
