#include <iostream>
#include <stdio.h>
#include <math.h>
#include <complex>

#ifdef _OPENMP
#include <omp.h>
#endif

// Complex number i
const std::complex<double> I(0.0, 1.0);

// Fast Fourier Transform, based on Cooley-Tukey FFT
void FFT(std::complex<double> fs[], std::complex<double> f_hats[], long N) {
    if (N == 1) {
        f_hats[0] = fs[0];
    }
    else {
        std::complex<double> f_even[N/2]; 
        std::complex<double> f_odd[N/2];
        std::complex<double> fs_even[N/2]; 
        std::complex<double> fs_odd[N/2];
        for (long i = 0; i < N/2; i++) {
            fs_even[i] = fs[2*i];
            fs_odd[i] = fs[1+2*i];
        }
        FFT(fs_even, f_even, N/2);
        FFT(fs_odd, f_odd, N/2);
        std::complex<double> p;
        std::complex<double> q;
        for (long i = 0; i < N/2; i++) {
            p = f_even[i];
            q = exp(-2. * M_PI * i / N * I) * f_odd[i];
            f_hats[i] = p+q;
            f_hats[i+N/2] = p-q;
        }
    }
}

// Inverse Fast Fourier Transform, based on Cooley-Tukey FFT
void IFFT(std::complex<double> fs[], std::complex<double> f_hats[], long N) {
    if (N == 1) {
        f_hats[0] = fs[0];
    }
    else {
        std::complex<double> f_even[N/2]; 
        std::complex<double> f_odd[N/2];
        std::complex<double> fs_even[N/2]; 
        std::complex<double> fs_odd[N/2];
        for (long i = 0; i < N/2; i++) {
            fs_even[i] = fs[2*i];
            fs_odd[i] = fs[1+2*i];
        }
        IFFT(fs_even, f_even, N/2);
        IFFT(fs_odd, f_odd, N/2);
        std::complex<double> p;
        std::complex<double> q;
        for (long i = 0; i < N/2; i++) {
            p = f_even[i];
            q = exp(2. * M_PI * i / N * I) * f_odd[i];
            f_hats[i] = p+q;
            f_hats[i+N/2] = p-q;
        }
    }
}

double err(double* x, std::complex<double> y[], long N) {
  double error = 0;
  for (long i = 0; i < N; i++) error = std::max(error, fabs(x[i]-y[i].real()));
  return error;
}

int main() {
    long N = 20;
    std::complex<double> fs[N]; 
    std::complex<double> f_hats[N]; 
    double xs[N]; 
    double ys[N]; 
    for (long j = 0; j < N; j++) {
        double x = -1. + j*0.1;
        double y = cos(x * M_PI);
        xs[j] = x;
        ys[j] = y;
        fs[j] = y;
        f_hats[j] = 0.;
    }

    FFT(fs, f_hats, N);
    IFFT(fs, f_hats, N);
    for (long j = 0; j < N; j++) {
        printf("%10f,", xs[j]);
    }
    printf("\n");
    for (long j = 0; j < N; j++) {
        printf("%10f,", fs[j].real());
    }
    printf("\n");
    printf("%10f\n", err(ys, fs, N));

    return 0;
}