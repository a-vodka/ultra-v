//---------------------------------------------------------------------------
#include "FFT_class.h"
#ifndef FourierProccessH
#define FourierProccessH
//---------------------------------------------------------------------------
#include <System.Classes.hpp>
//---------------------------------------------------------------------------
class TFourierProcces : public TThread
{
protected:
	void __fastcall Execute();

	void __fastcall plotFFT();
	void __fastcall getMinMax(const vector<FFT::Complex>& arr, double &max, double &min, int &maxindex, int &minindex);
	void __fastcall getMinMax(const vector<FFT::Complex>& arr, double &max, double &min, int &maxindex, int &minindex, int limit);

	void __fastcall getMax(const vector<FFT::Complex>& arr, double &max, int &maxindex, int limit);

	void __fastcall FFTPostProc( vector<FFT::Complex>& bufA, vector<FFT::Complex>& bufB);
	void __fastcall CorrectSpectrum( vector<FFT::Complex>& bufA, vector<FFT::Complex>& bufB);

	vector<FFT::Complex> A_FFT,iA_FFT;
	vector<FFT::Complex> B_FFT,iB_FFT;

	double m_rms_a, m_rms_b;  // ÑÊÇ
	double m_maxfreq_a, m_maxfreq_b, m_maxfreq_value_a, m_maxfreq_value_b;
	double m_amp_a, m_amp_b, m_amp_a_src, m_amp_b_src;

	TNotifyEvent FOnCalc;

public:
	__fastcall TFourierProcces(bool CreateSuspended);
	__property TNotifyEvent OnCalc = {read=FOnCalc, write=FOnCalc};
	__property double rms_a = {read = m_rms_a} ;
	__property double rms_b = {read = m_rms_b} ;

	__property double max_freq_a = {read = m_maxfreq_a} ;
	__property double max_freq_b = {read = m_maxfreq_b} ;

	__property double max_freq_value_a = {read = m_maxfreq_value_a} ;
	__property double max_freq_value_b = {read = m_maxfreq_value_b} ;
	__property double amp_a = {read = m_amp_a} ;
	__property double amp_b = {read = m_amp_b} ;
	__property double amp_a_src = {read = m_amp_a_src} ;
	__property double amp_b_src = {read = m_amp_b_src} ;

   bool suppress_noise;
};
//---------------------------------------------------------------------------
#endif
