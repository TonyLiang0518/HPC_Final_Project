#include <iostream>
#include <stdio.h>
#include <math.h>
#include <complex>
#include "utils.h"

#ifdef _OPENMP
#include <omp.h>
#endif

// Complex number i
const std::complex<double> I(0.0, 1.0);

// Fast Fourier Transform, based on Cooley-Tukey FFT
void FFT(std::complex<double> fs[], std::complex<double> f_hats[], long N) {
    // Base case
    if (N == 1) {
        f_hats[0] = fs[0];
    }
    else {
        std::complex<double> f_even[N/2]; 
        std::complex<double> f_odd[N/2];
        std::complex<double> fs_even[N/2]; 
        std::complex<double> fs_odd[N/2];
        // Fit in even terms and odd terms
        for (long i = 0; i < N/2; i++) {
            fs_even[i] = fs[2*i];
            fs_odd[i] = fs[1+2*i];
        }
        // Recursion
        FFT(fs_even, f_even, N/2);
        FFT(fs_odd, f_odd, N/2);
        std::complex<double> p;
        std::complex<double> q;
        // Compute coefficients
        for (long i = 0; i < N/2; i++) {
            p = f_even[i];
            q = exp(2. * M_PI * i / N * I) * f_odd[i];
            f_hats[i] = p+q;
            f_hats[i+N/2] = p-q;
        }
    }
}

// Inverse Fast Fourier Transform, based on Cooley-Tukey FFT
void IFFT(std::complex<double> fs[], std::complex<double> f_hats[], long N) {
    // Base case
    if (N == 1) {
        f_hats[0] = fs[0];
    }
    else {
        std::complex<double> f_even[N/2]; 
        std::complex<double> f_odd[N/2];
        std::complex<double> fs_even[N/2]; 
        std::complex<double> fs_odd[N/2];
        // Fit in even terms and odd terms
        for (long i = 0; i < N/2; i++) {
            fs_even[i] = fs[2*i];
            fs_odd[i] = fs[1+2*i];
        }
        // Recursion
        IFFT(fs_even, f_even, N/2);
        IFFT(fs_odd, f_odd, N/2);
        std::complex<double> p;
        std::complex<double> q;
        // Compute coefficients
        for (long i = 0; i < N/2; i++) {
            p = f_even[i];
            q = exp(-2. * M_PI * i / N * I) * f_odd[i];
            f_hats[i] = (p+q);
            f_hats[i+N/2] = (p-q);
        }
    }
}

unsigned int bitReverse(unsigned int x, int log2n)
{
    int n = 0;
    for (int i = 0; i < log2n; i++)
    {
        n <<= 1;
        n |= (x & 1);
        x >>= 1;
    }
    return n;
}
 
// Iterative FFT function to compute the DFT
// of given coefficient vector
void FFT_ite(std::complex<double> fs[], std::complex<double> f_hats[], long N, long log2N)
{
    // bit reversal of the given array
    for (unsigned int i = 0; i < N; ++i) {
        int rev = bitReverse(i, log2N);
        f_hats[i] = fs[rev];
    }
 
    for (int s = 1; s <= log2N; ++s) {
        int m = 1 << s; // 2 power s
        int m2 = m >> 1; // m2 = m/2 -1
        std::complex<double> w(1, 0);
 
        // principle root of nth complex
        // root of unity.
        std::complex<double> wm = exp(I * (M_PI / m2));
        for (int j = 0; j < m2; ++j) {
            for (int k = j; k < N; k += m) {
 
                // t = twiddle factor
                std::complex<double> t = w * f_hats[k + m2];
                std::complex<double> u = f_hats[k];
 
                // similar calculating y[k]
                f_hats[k] = u + t;
 
                // similar calculating y[k+n/2]
                f_hats[k + m2] = u - t;
            }
            w *= wm;
        }
    }
}

void IFFT_ite(std::complex<double> fs[], std::complex<double> f_hats[], long N, long log2N)
{
    // bit reversal of the given array
    for (unsigned int i = 0; i < N; ++i) {
        int rev = bitReverse(i, log2N);
        f_hats[i] = fs[rev];
    }
 
    for (int s = 1; s <= log2N; ++s) {
        int m = 1 << s; // 2 power s
        int m2 = m >> 1; // m2 = m/2 -1
        std::complex<double> w(1, 0);
 
        // principle root of nth complex
        // root of unity.
        std::complex<double> wm = exp(-I * (M_PI / m2));
        for (int j = 0; j < m2; ++j) {
            for (int k = j; k < N; k += m) {
 
                // t = twiddle factor
                std::complex<double> t = w * f_hats[k + m2];
                std::complex<double> u = f_hats[k];
 
                // similar calculating y[k]
                f_hats[k] = u + t;
 
                // similar calculating y[k+n/2]
                f_hats[k + m2] = u - t;
            }
            w *= wm;
        }
    }
}

// Compute largest error i.e. max norm
double err(double* x, std::complex<double> y[], long N) {
  double error = 0;
  for (long i = 0; i < N; i++) error = std::max(error, fabs(x[i]-y[i].real()));
  return error;
}

void ex1(double alpha, double beta, std::complex<double> fs[], long N) {

}

int main(int argc, char* argv[]) {
    clock_t start, end;
    double time_taken;
    long log2N = read_option<long>("-n", argc, argv);

    // Test with cosine function
    long N = long(pow(2, log2N));
    // Test with cosine function
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
    // Check forward and inverse transformation
    start = clock();
    FFT(fs, f_hats, N);
    IFFT(f_hats, fs, N);
    // FFT_ite(fs, f_hats, N, log2N);
    // IFFT_ite(f_hats, fs, N, log2N);
    for (long j = 0; j < N; j++) {
        fs[j] = fs[j].real() / N;
    }
    end = clock();
    time_taken = (double)(end - start)/CLOCKS_PER_SEC;
    // for (long j = 0; j < N; j++) {
    //     printf("%10f,", xs[j]);
    // }
    // printf("\n");
    // for (long j = 0; j < N; j++) {
    //     printf("%10f,", fs[j].real());
    // }
    // printf("\n");
    // for (long j = 0; j < N; j++) {
    //     printf("%10f,", ys[j]);
    // }
    // printf("\n");
    printf("Time taken for FFT and IFFT for size %ld is: %f s\n", N, time_taken);
    printf("Transform function check: %10f\n", err(ys, fs, N));
    printf("----------------------------------------------------------\n");

    // This part solves the PDE u_xx + alpha*u_x - beta*u = f, with periodic BC u(0)=u(2pi)=0
    // test with exact solution u = sin(cos(x)) --> u_x = -cos(cos(x))sin(x) --> u_xx = -sin^2(x)sin(cos(x))-cos(x)cos(cos(x))
    double h = 2.*M_PI / (N);
    double alpha = 1.;
    double beta = 1.;
    double us[N];
    //std::complex<double> fs[N]; 
    std::complex<double> Fs[N]; 
    std::complex<double> u_hats[N]; 
    std::complex<double> Us[N]; 
    // initialize
    start = clock();
    for (long j = 0; j < N; j++) {
        double x = h*(j);
        double u_true = sin(cos(x));
        double u_x = -cos(cos(x))*sin(x);
        double u_xx = -sin(x)*sin(x)*sin(cos(x)) - cos(x)*cos(cos(x));
        xs[j] = x;
        us[j] = u_true;
        fs[j] = u_xx + alpha*u_x - beta*u_true;
        u_hats[j] = 0.;
        Fs[j] = 0.;
    }
    FFT_ite(fs, Fs, N, log2N);
    std::complex<double> coeff[N];
    // wave frequency shifted
    for (long j = 0; j < N/2; j++) {
        coeff[j] = j;
        coeff[N-j-1] = -1-j;
    }
    std::complex<double> div;
    for (long j = 0; j < N; j++) {
        div = (-beta + I*alpha*coeff[j] - coeff[j]*coeff[j]);
        Us[j] = Fs[j].real() / div.real();
    }
    IFFT_ite(Us, u_hats, N, log2N);
    for (long j = 0; j < N; j++) {
        u_hats[j] = u_hats[j].real() / N;
    }
    for (long i = 0; i < N; i++) {
        printf("%10f ", u_hats[i].real());
    }
    
    end = clock();
    time_taken = (double)(end - start)/CLOCKS_PER_SEC;
    printf("\nTime taken for solving PDE with size %ld is: %f s\n", N, time_taken);
    printf("1-D second order PDE check: %10f\n", err(us, u_hats, N));


    return 0;
}
