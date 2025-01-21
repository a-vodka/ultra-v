//---------------------------------------------------------------------------

#include <vcl.h>
#include <algorithm>

#include <math.h>
#include <stdlib.h>
#pragma hdrstop
#include "fft.h"
#include "FourierProccess.h"
#include "procHolder.h"
#include "defines.h"

#pragma package(smart_init)

using namespace std;
//---------------------------------------------------------------------------

//   Important: Methods and properties of objects in VCL can only be
//   used in a method called using Synchronize, for example:
//
//      Synchronize(&UpdateCaption);
//
//   where UpdateCaption could look like:
//
//      void __fastcall TFourierProcces::UpdateCaption()
//      {
//        Form1->Caption = "Updated in a thread";
//      }
//---------------------------------------------------------------------------

__fastcall TFourierProcces::TFourierProcces(bool CreateSuspended)
	: TThread(CreateSuspended)
{
	suppress_noise = true;
}
//---------------------------------------------------------------------------
void __fastcall TFourierProcces::Execute()
{
	NameThreadForDebugging("FourierProcces");

	int n, i;
	vector<FFT::Complex> A(FUR_MAX);
	vector<FFT::Complex> B(FUR_MAX);
	vector<FFT::Complex> D(FUR_MAX);
	vector<FFT::Complex> E(FUR_MAX);
	vector<FFT::Complex> F(FUR_MAX);
	FFT dft(FUR_MAX), idft(FUR_MAX, true);
	while (!Terminated)
	{
		n = myTh->postTh->fourierList.size();
		if (n < FUR_MAX)
		{
			Sleep(25);
			continue;
		}
		EnterCriticalSection(&myTh->CS2);
		for (i = 0; i < FUR_MAX; i++)
		{
			channels ch = myTh->postTh->fourierList.front();
			A[i] = ch.A;
			B[i] = ch.B;
			D[i] = ch.D;
			E[i] = ch.E;
			F[i] = ch.F;

			myTh->postTh->fourierList.pop_front();
		}
		myTh->postTh->fourierList.clear();
		LeaveCriticalSection(&myTh->CS2);

		double max = 0, min = 0;
		int max_indx = 0 , min_indx = 0;

		getMinMax(A, max, min, max_indx, min_indx);
		m_amp_a_src = (max - min)/2.0;

		getMinMax(B, max, min, max_indx, min_indx);
		m_amp_b_src = (max - min)/2.0;

		A_FFT =	dft.transform2(A); // Выполняем БПФ
		B_FFT = dft.transform2(B);
		D_FFT =	dft.transform2(D); // Выполняем БПФ
		E_FFT = dft.transform2(E);
		F_FFT = dft.transform2(F);

		m_amp_a = m_amp_b = 0;

		if (suppress_noise) // включено шумоподавление?
		{
			FFTPostProc(A_FFT, B_FFT, D_FFT, E_FFT, F_FFT); // Очищаем шум

			iA_FFT = idft.transform2(A_FFT);   // Обратное  БПФ
			iB_FFT = idft.transform2(B_FFT);

			iD_FFT = idft.transform2(D_FFT);   // Обратное  БПФ
			iE_FFT = idft.transform2(E_FFT);
			iF_FFT = idft.transform2(F_FFT);

			getMinMax(iA_FFT, max, min, max_indx, min_indx);
			m_amp_a = (max - min)/2.0;

			getMinMax(iB_FFT, max, min, max_indx, min_indx);
			m_amp_b = (max - min)/2.0;
	   }

		Synchronize(&plotFFT) ;

		if (FOnCalc)
			FOnCalc(this);

	}


}
//---------------------------------------------------------------------------

void __fastcall TFourierProcces::getMinMax(const vector<FFT::Complex>& arr, double &max, double &min, int &maxindex, int &minindex, int limit )
{

	if (arr.empty())
		return;

	assert(CROP_SIZE < 1 && CROP_SIZE > 0);
	assert(limit < arr.size() );
	/* в связи с наличием каревых эффектов при обратоном БПФ целесообразно
	 отбросить CROP_SIZE от размера массива с каждого края */

	int n = limit == 0 ? arr.size() : limit;
	int start = CROP_SIZE*n;
	int end = (1.0 - CROP_SIZE)*n;

	max = min = real(arr[0]);
	for (int i = start; i < end; i++)
	{
		 if ( max < real(arr[i]))
		 {
			max = real(arr[i]);
			maxindex = i;
		 }

		 if ( min > real(arr[i]))
		 {
			min = real(arr[i]);
			minindex = i;
		 }
	}

}

void __fastcall TFourierProcces::getMinMax(const vector<FFT::Complex>& arr, double &max, double &min, int &maxindex, int &minindex)
{
	getMinMax(arr, max, min, maxindex, minindex, 0);
}

void __fastcall TFourierProcces::getMax(const vector<FFT::Complex>& arr, double &max, int &maxindex, int limit = 0)
{
	max = abs(arr[1]);
	maxindex = 1;
	int n = limit == 0 ? arr.size(): limit;

	assert(limit < arr.size());

	for (int i = 1; i < n; i++)
	{
	  if ( max < abs(arr[i]))
	  {
		  max = abs(arr[i]);
		  maxindex = i;
	  }
	}
}

double kb(double val)
{

	const double x[] = {0, 5.0, 6.3, 8.0, 10.0, 12.5, 16.0, 20.0, 25.0, 31.0, 40.0,
				 50.0, 63.0, 80.0, 100.0, 125.0, 160.0, 200.0, 250.0, 315.0, 400.0};
	const double k[] = {5.20E-02, 5.20E-02, 5.13E-02, 5.14E-02, 5.17E-02, 5.19E-02,
					5.17E-02, 5.20E-02, 5.22E-02, 5.23E-02, 5.25E-02, 5.27E-02,
					5.35E-02, 5.45E-02, 5.52E-02, 5.67E-02, 5.88E-02, 6.05E-02,
					6.26E-02, 4.92E-02, 3.31E-02 };
	assert(sizeof(x) == sizeof(k));
	int n = sizeof(x)/sizeof(x[0]);
	int j = 0;
	for (int i = 0; i < n - 1; i++)
	{
		if (val >= x[i] && val < x[i + 1])
		{
			 j = i;
		}
	}
	double res = k[j] + (k[j + 1] - k[j]) * (val - x[j]) / (x[j + 1] - x[j]);
	return res;
}


double ka(double val)
{

	const double x[] = {0, 5.0, 6.3, 8.0, 10.0, 12.5, 16.0, 20.0, 25.0, 31.0, 40.0,
				 50.0, 63.0, 80.0, 100.0, 125.0, 160.0, 200.0, 250.0, 315.0, 400.0};
	const double k[] = {3.33E-02, 3.33E-02, 3.28E-02, 3.30E-02, 3.29E-02, 3.27E-02,
				3.28E-02, 3.27E-02, 3.29E-02,3.31E-02, 3.35E-02, 3.40E-02,
				3.40E-02, 3.52E-02, 3.67E-02, 3.92E-02, 4.35E-02, 4.58E-02,
				4.81E-02, 4.27E-02, 4.31E-02 };
	assert(sizeof(x) == sizeof(k));
	int n = sizeof(x)/sizeof(x[0]);
	int j = 0;
	for (int i = 0; i < n - 1; i++)
	{
		if (val >= x[i] && val < x[i + 1])
		{
			 j = i;
		}
	}
	double res = k[j] + (k[j + 1] - k[j]) * (val - x[j]) / (x[j + 1] - x[j]);
	return res;
}

double kf(double val)
{

	const double x[] = {0, 16.0, 20.0, 25.0, 31.0, 40.0,
				 50.0, 63.0, 80.0, 100.0, 125.0, 160.0, 200.0, 250.0, 315.0, 400.0};
	const double k[] = { 2.20E-02, 2.20E-02, 2.21E-02, 2.08E-02, 2.10E-02, 2.12E-02, 2.11E-02,
	2.13E-02, 2.18E-02, 1.59E-02, 1.73E-02, 1.68E-02, 1.70E-02, 1.70E-02, 1.17E-02, 1.02E-02
};
	assert(sizeof(x) == sizeof(k));
	int n = sizeof(x)/sizeof(x[0]);
	int j = 0;
	for (int i = 0; i < n - 1; i++)
	{
		if (val >= x[i] && val < x[i + 1])
		{
			 j = i;
		}
	}
	double res = k[j] + (k[j + 1] - k[j]) * (val - x[j]) / (x[j + 1] - x[j]);
	return res;
}

double kd(double val)
{

	const double x[] = {0,10.0, 12.5, 16.0, 20.0, 25.0, 31.0, 40.0,
				 50.0, 63.0, 80.0, 100.0, 125.0, 160.0, 315.0, 400.0};
	const double k[] = {
	2.31E-02, 2.31E-02, 2.11E-02, 2.17E-02, 2.25E-02, 2.14E-02, 2.14E-02, 2.23E-02, 2.28E-02,
	1.79E-02, 1.83E-02, 1.81E-02, 1.71E-02, 1.89E-02, 1.37E-02, 1.49E-02 };

	assert(sizeof(x) == sizeof(k));

	int n = sizeof(x)/sizeof(x[0]);
	int j = 0;
	for (int i = 0; i < n - 1; i++)
	{
		if (val >= x[i] && val < x[i + 1])
		{
			 j = i;
		}
	}
	double res = k[j] + (k[j + 1] - k[j]) * (val - x[j]) / (x[j + 1] - x[j]);
	return res;
}

double ke(double val)
{

	const double x[] = {0,10.0, 12.5, 16.0, 20.0, 25.0, 31.0, 40.0,
				 50.0, 63.0, 80.0, 100.0, 125.0, 160.0, 200.0,  250.0, 315.0, 400.0};
	const double k[] = {
	1.99E-02, 1.99E-02, 2.18E-02, 2.22E-02, 2.19E-02, 2.19E-02, 2.12E-02, 2.12E-02, 2.24E-02,
2.17E-02, 1.89E-02, 2.00E-02, 2.02E-02, 2.25E-02, 1.90E-02, 2.21E-02, 2.88E-02, 3.33E-02 };

	assert(sizeof(x) == sizeof(k));

	int n = sizeof(x)/sizeof(x[0]);
	int j = 0;
	for (int i = 0; i < n - 1; i++)
	{
		if (val >= x[i] && val < x[i + 1])
		{
			 j = i;
		}
	}
	double res = k[j] + (k[j + 1] - k[j]) * (val - x[j]) / (x[j + 1] - x[j]);
	return res;
}

inline double  kfreq(double x)
{
	double a =	-0.96864463;
	double b =	620.93631;
	double c =	-0.0037144418;
	double d =	-0.0008899866;

    return (a+b*x)/(1+x*(c+d*x));
}

void __fastcall TFourierProcces::CorrectSpectrum(CxVector& bufA, CxVector& bufB, CxVector& bufD,CxVector& bufE, CxVector& bufF)
{
	assert(myTh->postTh->resType == 0);
	assert(bufA.size() == bufB.size());
	int n = bufA.size();
	double freq =myTh->adaTh->ada.getFreq()/myTh->adaTh->ada.getNumChannels()/(double)FUR_MAX;
	int maxfreqindex = F_MAX/freq;


	for ( int i = 1; i < n; i++)
	{
		if (i <= maxfreqindex)
		{
			bufA[i] *=  ASensetivity / ka((double)i*freq)*2.0;
			bufB[i] *=  BSensetivity / kb((double)i*freq)*2.0;

			bufD[i] *=  DSensetivity / kd((double)i*freq)*2.0;
			bufE[i] *=  ESensetivity / ke((double)i*freq)*2.0;
			bufF[i] *=  FSensetivity / kf((double)i*freq)*2.0;
		}
		else
		{
			bufA[i] = 0;
			bufB[i] = 0;
		}
	}
}

void __fastcall TFourierProcces::FFTPostProc(CxVector& bufA, CxVector& bufB, CxVector& bufD,CxVector& bufE, CxVector& bufF)
{
	double rate = 0;
	switch (myTh->postTh->resType)
	{
		case 0: // Ускорения
				rate = NOICE_RATE_ACCEL;
				CorrectSpectrum(bufA, bufB, bufD, bufE, bufF);
				break;
			case 2: // Вольты
                rate = NOICE_RATE_VOLT;
				break;

			default:
			case 1:  // Код АЦП
				rate = NOICE_RATE_ADC;
				break;
	}

	if (bufA.empty() || bufB.empty())
		return;

	assert(bufA.size() == bufB.size());
	int n = bufA.size();

	for ( int i = 0; i < n; i++)
	{
		if (abs(bufA[i]) < rate)
			bufA[i] = 0;
		if (abs(bufB[i]) < rate)
			bufB[i] = 0;
	}
}

void __fastcall TFourierProcces::plotFFT()
{

	Form1->Series3->Clear();
	Form1->Series4->Clear();
	Form1->Series5->Clear();
 	Form1->Series6->Clear();
	Form1->Series7->Clear();
	Form1->Series8->Clear();
	Form1->Series9->Clear();
	Form1->Series10->Clear();

	Form1->D_Spectrum_series->Clear();
	Form1->E_Spectrum_series->Clear();
	Form1->F_Spectrum_series->Clear();

	double rms_a = 0, rms_b = 0;
	double a, b, f;
	m_maxfreq_value_a = m_maxfreq_value_b = 0;
	m_maxfreq_a = m_maxfreq_b = 0;

	double freq =myTh->adaTh->ada.getFreq()/myTh->adaTh->ada.getNumChannels()/(double)FUR_MAX;
	int maxfreqindex = F_MAX/freq;
	assert(maxfreqindex < FUR_MAX/2.0);

	if (suppress_noise)
	{
		int n = iA_FFT.size();
		for (int i = 0; i < n; i+=15)
		{
			f = (double)i / freq / FUR_MAX;
			Form1->Series7->AddXY(f, real(iA_FFT[i]));
			Form1->Series8->AddXY(f, real(iB_FFT[i]));
 		}
	}
	// выводим результаты

	double dmax = 0;
	int max_indx = 0;

	getMax(A_FFT, dmax, max_indx, maxfreqindex);

	m_maxfreq_value_a = dmax;
	m_maxfreq_a = freq * max_indx;

	getMax(B_FFT, dmax, max_indx, maxfreqindex);

	m_maxfreq_value_b = dmax;
	m_maxfreq_b = freq * max_indx;


	for (int i = 1; i < maxfreqindex; i++)
	{
		f = i * freq;
		a = abs(A_FFT[i]);
		b = abs(B_FFT[i]);

		rms_a += a*a;
		rms_b += b*b;

		a = a < m_maxfreq_value_a / 100.0 ? 0 : a;
		b = b < m_maxfreq_value_b / 100.0 ? 0 : b;
		Form1->Series3->AddXY(f, a);
		Form1->Series4->AddXY(f, b);

		Form1->D_Spectrum_series->AddXY(f, abs(D_FFT[i]));
		Form1->E_Spectrum_series->AddXY(f, abs(E_FFT[i]));
		Form1->F_Spectrum_series->AddXY(f, abs(F_FFT[i]));
	}

	Form1->Series5->AddXY(m_maxfreq_a, m_maxfreq_value_a);
	Form1->Series6->AddXY(m_maxfreq_b, m_maxfreq_value_b);

	m_freq = kfreq(myTh->adaTh->freq_ADC*TO_VOLT);

	double max_amp_for_freq = max(m_maxfreq_value_a,m_maxfreq_value_b);

	for (int i = 1; i <= 10; i++)
	{
        Form1->Series9->AddXY(i*m_freq/60.0, max_amp_for_freq);
	}

	 for (int i = 1; i <= 4; i++)
	{
		if (7.0*i*m_freq/60 < 420)
			Form1->Series10->AddXY(7.0*i*m_freq/60.0, max_amp_for_freq);
	}


	m_rms_a =  sqrt(rms_a/2.0);
	m_rms_b =  sqrt(rms_b/2.0);

	//myTh->adaTh->freq_rpm;

}
  /*
void __fastcall TFourierProcces::FFT( double *dIn, double *amp, const nn )
{
	int i, j, n, m, mmax, istep;
	double tempr, tempi, wtemp, theta, wpr, wpi, wr, wi;

	int isign = -1;
	double* data = new double [ nn * 2 + 1 ];

	for( i = 0; i < nn; i++ )
	{
		data[ i * 2 ] = 0;
		data[ i * 2 + 1 ] = dIn[ i ];
	}

	n = nn << 1;
	j = 1;
	i = 1;
	while( i < n )
	{
        if( j > i )
		{
            tempr = data[ i ]; data[ i ] = data[ j ]; data[ j ] = tempr;
			tempr = data[ i + 1 ]; data[ i + 1 ] = data[ j + 1 ]; data[ j + 1 ] = tempr;
		}
        m = n >> 1;
		while( ( m >= 2 ) && ( j > m ) )
		{
            j = j - m;
            m = m >> 1;
        }
        j = j + m;
        i = i + 2;
    }
    mmax = 2;
	while( n > mmax )
    {
        istep = 2 * mmax;
        theta = 2.0 * M_PI / ( isign * mmax );
        wtemp = sin( 0.5 * theta );
        wpr = -2.0 * wtemp * wtemp;
        wpi = sin( theta );
		wr = 1.0;
        wi = 0.0;
        m = 1;
        while( m < mmax )
        {
            i = m;
            while( i < n )
			{
                j = i + mmax;
				tempr = wr * data[ j ] - wi * data[ j + 1 ];
				tempi = wr * data[ j + 1 ] + wi * data[ j ];
				data[ j ] = data[ i ] - tempr;
				data[ j + 1 ] = data[ i + 1 ] - tempi;
                data[ i ] = data[ i ] + tempr;
                data[ i + 1 ] = data[ i + 1 ] + tempi;
                i = i + istep;
            }
            wtemp = wr;
			wr = wtemp * wpr - wi * wpi + wr;
            wi = wi * wpr + wtemp * wpi + wi;
            m = m + 2;
        }
        mmax = istep;
    }

	for( i = 0; i < ( nn / 2 ); i++ )
	{
		amp[ i ] = sqrt( data[ i * 2 ] * data[ i * 2 ] + data[ i * 2 + 1 ] * data[ i * 2 + 1 ] );
	}
	delete []data;

}          */
