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
#pragma omp parallel for shared(fs, f_hats, N, log2N)
    for (unsigned int i = 0; i < N; ++i)
    {
        int rev = bitReverse(i, log2N);
        f_hats[i] = fs[rev];
    }
#pragma omp parallel shared(fs, f_hats, N, log2N)
    for (int s = 1; s <= log2N; ++s)
    {
        int m = 1 << s;  // 2 power s
        int m2 = m >> 1; // m2 = m/2 -1
        std::complex<double> w(1, 0);

        // principle root of nth complex
        // root of unity.
        std::complex<double> wm = exp(I * (M_PI / m2));

        for (int j = 0; j < m2; ++j)
        {
#pragma omp for
            for (int k = j; k < N; k += m)
            {

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
#pragma omp parallel for shared(fs, f_hats, N, log2N)
    for (unsigned int i = 0; i < N; ++i)
    {
        int rev = bitReverse(i, log2N);
        f_hats[i] = fs[rev];
    }
#pragma omp parallel shared(fs, f_hats, N, log2N)
    for (int s = 1; s <= log2N; ++s)
    {
        int m = 1 << s;  // 2 power s
        int m2 = m >> 1; // m2 = m/2 -1
        std::complex<double> w(1, 0);

        // principle root of nth complex
        // root of unity.
        std::complex<double> wm = exp(-I * (M_PI / m2));
        for (int j = 0; j < m2; ++j)
        {
#pragma omp for
            for (int k = j; k < N; k += m)
            {

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
double err(double *x, std::complex<double> y[], long N)
{
    double error = 0;
    for (long i = 0; i < N; i++)
        error = std::max(error, fabs(x[i] - y[i].real()));
    return error;
}

int main()
{
    // Test with cosine function
    long log2N = 17;

    for (int i = 1; i <= log2N; i++)
    {
        long N = pow(2, i);
        std::complex<double> fs[N];
        std::complex<double> f_hats[N];
        double xs[N];
        double ys[N];
        for (long j = 0; j < N; j++)
        {
            double x = -1. + j * 0.1;
            double y = cos(x * M_PI);
            xs[j] = x;
            ys[j] = y;
            fs[j] = y;
            f_hats[j] = 0.;
        }
        printf("i=%d:\n", i);
        Timer timer;
        timer.tic();
        FFT_ite(fs, f_hats, N, i);
        IFFT_ite(f_hats, fs, N, i);
        printf("time: %3f\n", timer.toc());
        for (long j = 0; j < N; j++)
        {
            fs[j] = fs[j].real() / N;
        }
        printf("error: %10f\n\n", err(ys, fs, N));
    }

    return 0;
}