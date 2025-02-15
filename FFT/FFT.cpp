﻿//---------------------------------------------------------------------------

#include <vcl.h>
#include <math.hpp>

#pragma hdrstop

#include "FFT.h"
#include "procHolder.h"


//---------------------------------------------------------------------------
#pragma package(smart_init)
#pragma resource "*.dfm"
TForm1 *Form1;
//---------------------------------------------------------------------------
__fastcall TForm1::TForm1(TComponent* Owner)
        : TForm(Owner)
{
	StringGrid1->Cells[1][0] = "A";
	StringGrid1->Cells[2][0] = "B";
	StringGrid1->Cells[0][1] = "СКЗ";
	StringGrid1->Cells[0][2] = "СКЗ*sqrt(2)";
	StringGrid1->Cells[0][3] = "Макс. частота";
	StringGrid1->Cells[0][4] = "Ампл на макс част.";
	StringGrid1->Cells[0][5] = "Ампл (ф)";
	StringGrid1->Cells[0][6] = "Ампл*2 (ф)";
	StringGrid1->Cells[0][7] = "Ампл (и)";
	StringGrid1->Cells[0][8] = "Ампл*2 (и)";

}
//---------------------------------------------------------------------------
__fastcall TForm1::~TForm1()
{
}

void __fastcall TForm1::FCalc(TObject *Sender)
{
	#define digits -4
	StringGrid1->Cells[1][1] = FloatToStr(RoundTo(myTh->fourierTh->rms_a, digits));
	StringGrid1->Cells[1][2] = FloatToStr(RoundTo(myTh->fourierTh->rms_a*sqrt(2.0), digits));

	StringGrid1->Cells[2][1] = FloatToStr(RoundTo(myTh->fourierTh->rms_b, digits));
	StringGrid1->Cells[2][2] = FloatToStr(RoundTo(myTh->fourierTh->rms_b*sqrt(2.0), digits));

	StringGrid1->Cells[1][3] = FloatToStr(RoundTo(myTh->fourierTh->max_freq_a, digits));;
	StringGrid1->Cells[2][3] = FloatToStr(RoundTo(myTh->fourierTh->max_freq_b, digits));

	StringGrid1->Cells[1][4] = FloatToStr(RoundTo(myTh->fourierTh->max_freq_value_a, digits));
	StringGrid1->Cells[2][4] = FloatToStr(RoundTo(myTh->fourierTh->max_freq_value_b, digits));

	StringGrid1->Cells[1][5] = FloatToStr(RoundTo(myTh->fourierTh->amp_a, digits));
	StringGrid1->Cells[2][5] = FloatToStr(RoundTo(myTh->fourierTh->amp_b, digits));

	StringGrid1->Cells[1][6] = FloatToStr(RoundTo(myTh->fourierTh->amp_a*2.0, digits));
	StringGrid1->Cells[2][6] = FloatToStr(RoundTo(myTh->fourierTh->amp_b*2.0, digits));

	StringGrid1->Cells[1][7] = FloatToStr(RoundTo(myTh->fourierTh->amp_a_src, digits));
	StringGrid1->Cells[2][7] = FloatToStr(RoundTo(myTh->fourierTh->amp_b_src, digits));

	StringGrid1->Cells[1][8] = FloatToStr(RoundTo(myTh->fourierTh->amp_a_src*2.0, digits));
	StringGrid1->Cells[2][8] = FloatToStr(RoundTo(myTh->fourierTh->amp_b_src*2.0, digits));

}

//---------------------------------------------------------------------------
void __fastcall TForm1::Button2Click(TObject *Sender)
{
   AnsiString capt;

   if (myTh->adaTh->Suspended)
   {
		myTh->adaTh->Resume();
		capt = "Стоп";
   }
   else
   {
		myTh->adaTh->Suspend();
		capt = "Старт";
   }

   Button2->Caption = capt;

}

//---------------------------------------------------------------------------

void __fastcall TForm1::RadioGroup1Click(TObject *Sender)
{
	if (myTh->postTh->resType != RadioGroup1->ItemIndex)
		myTh->postTh->resType = RadioGroup1->ItemIndex;
}
//---------------------------------------------------------------------------

void __fastcall TForm1::Button1Click(TObject *Sender)
{
	if (OpenDialog1->Execute())
	{
		AnsiString S = OpenDialog1->FileName;
		if (FileExists(S))
		{
			MessageDlg("Файл уже существует", mtError, TMsgDlgButtons() << mbOK,0);
			return;
		}
		myTh->postTh->closeFile();
		myTh->postTh->OpenFile(S);
		Edit3->Text = S;
    }
}
//---------------------------------------------------------------------------


void __fastcall TForm1::Button3Click(TObject *Sender)
{
	if (!myTh->postTh->isOpenFile())
	{
		MessageDlg("Не выбран файл", mtError, TMsgDlgButtons() << mbOK, 0);
		return;
	}

	myTh->postTh->writeEnabled = !myTh->postTh->writeEnabled;
	Button1->Enabled = !myTh->postTh->writeEnabled;

	AnsiString S = myTh->postTh->writeEnabled? "Остановить запись в файл":"Начать запись в файл";
	Button3->Caption = S;

	if (myTh->postTh->writeEnabled == false)
	{
		myTh->postTh->closeFile();
	}

}
//---------------------------------------------------------------------------

void __fastcall TForm1::N1Click(TObject *Sender)
{
	freqDim = FreqDim::frHz;
}
//---------------------------------------------------------------------------

void __fastcall TForm1::N2Click(TObject *Sender)
{
	freqDim = FreqDim::frRadPS;
}
//---------------------------------------------------------------------------

void __fastcall TForm1::N3Click(TObject *Sender)
{
	freqDim = FreqDim::frRPM;
}
//---------------------------------------------------------------------------


void __fastcall TForm1::FormShow(TObject *Sender)
{
	myTh->fourierTh->OnCalc = FCalc;
}
//---------------------------------------------------------------------------


void __fastcall TForm1::CheckBox1Click(TObject *Sender)
{
	if (CheckBox1->Checked)
	{
		Series1->Visible = true;
		Series2->Visible = true;
		Series7->Visible = false;
		Series8->Visible = false;
	}
	else
	{
		Series1->Visible = false;
		Series2->Visible = false;
		Series7->Visible = true;
		Series8->Visible = true;

    }
}
//---------------------------------------------------------------------------

void __fastcall TForm1::CheckBox2Click(TObject *Sender)
{
	myTh->fourierTh->suppress_noise = CheckBox2->Checked;
	if (CheckBox2->Checked == false)
	{
	   CheckBox1->Checked = true;
	   CheckBox1->Enabled = false;
	}
	else
	{
	  CheckBox1->Enabled = true;
	}
}
//---------------------------------------------------------------------------

