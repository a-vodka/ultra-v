#include "defines.h"
#ifndef FFT_struct
#define FFT_struct

struct channels
{
	double t;
	double A;
	double B;
	double C;
	double D,E,F;
	void reset();
};

void channels::reset()
{
   A=0;
   B=0;
   C=0;
   D=E=F=0;
   t=0;
}



#endif
