//---------------------------------------------------------------------------

#ifndef FFTH
#define FFTH
//---------------------------------------------------------------------------
#include <Classes.hpp>
#include <Controls.hpp>
#include <StdCtrls.hpp>
#include <Forms.hpp>
#include <ExtCtrls.hpp>
#include <ComCtrls.hpp>
#include <Dialogs.hpp>
#include <Menus.hpp>
#include <VCLTee.Chart.hpp>
#include <VCLTee.Series.hpp>
#include <VCLTee.TeEngine.hpp>
#include <VCLTee.TeeProcs.hpp>
#include <Vcl.XPMan.hpp>
#include <Vcl.ActnColorMaps.hpp>
#include <Vcl.ActnMan.hpp>
#include <Vcl.Grids.hpp>

#include <list>
#include <deque>

#include "myThread.h"
#include "PostProcTh.h"
#include "FourierProccess.h"

using namespace std;
//---------------------------------------------------------------------------

enum FreqDim {frHz, frRPM, frRadPS};
//---------------------------------------------------------------------------

class TForm1 : public TForm
{
__published:	// IDE-managed Components
	TOpenDialog *OpenDialog1;
	TPopupMenu *PopupMenu1;
	TMenuItem *N1;
	TMenuItem *N2;
	TMenuItem *N3;
	TXPManifest *XPManifest1;
	TPanel *Panel1;
	TRadioGroup *RadioGroup1;
	TStringGrid *StringGrid1;
	TButton *Button2;
	TButton *Button3;
	TButton *Button1;
	TEdit *Edit3;
	TPanel *Panel2;
	TGroupBox *GroupBox2;
	TChart *Chart2;
	TGroupBox *GroupBox1;
	TChart *Chart1;
	TFastLineSeries *Series1;
	TFastLineSeries *Series2;
	TTrackBar *TrackBar1;
	TSplitter *Splitter1;
	TBarSeries *Series3;
	TBarSeries *Series4;
	TPointSeries *Series5;
	TPointSeries *Series6;
	TCheckBox *CheckBox1;
	TFastLineSeries *Series7;
	TFastLineSeries *Series8;
	TCheckBox *CheckBox2;
	TEdit *Edit1;
	TEdit *Edit2;
	TLabel *Label1;
	TLabel *Label2;
	TProgressBar *ProgressBar1;
	TBarSeries *Series9;
	TBarSeries *Series10;
	TFastLineSeries *D_Series;
	TFastLineSeries *E_Series;
	TFastLineSeries *F_Series;
	TBarSeries *D_Spectrum_series;
	TBarSeries *E_Spectrum_series;
	TBarSeries *F_Spectrum_series;

	void __fastcall Button2Click(TObject *Sender);
    void __fastcall RadioGroup1Click(TObject *Sender);
	void __fastcall Button1Click(TObject *Sender);
	void __fastcall Button3Click(TObject *Sender);
	void __fastcall N1Click(TObject *Sender);
	void __fastcall N2Click(TObject *Sender);
	void __fastcall N3Click(TObject *Sender);
	void __fastcall FormShow(TObject *Sender);
	void __fastcall CheckBox1Click(TObject *Sender);
	void __fastcall CheckBox2Click(TObject *Sender);

private:	// User declarations
		FreqDim freqDim;
		void __fastcall FCalc(TObject *Sender);

public:		// User declarations
		__fastcall TForm1(TComponent* Owner);
		__fastcall ~TForm1();

};
//---------------------------------------------------------------------------
extern PACKAGE TForm1 *Form1;
//---------------------------------------------------------------------------


#endif
