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
    for (unsigned int i = 0; i < N; ++i)
    {
        int rev = bitReverse(i, log2N);
        f_hats[i] = fs[rev];
    }
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
    for (unsigned int i = 0; i < N; ++i)
    {
        int rev = bitReverse(i, log2N);
        f_hats[i] = fs[rev];
    }
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

// Iterative version with openMP
void FFT_ite_omp(std::complex<double> fs[], std::complex<double> f_hats[], long N, long log2N)
{

// bit reversal of the given array
#pragma omp parallel for shared(fs, f_hats, N, log2N)
    for (unsigned int i = 0; i < N; ++i)
    {
        int rev = bitReverse(i, log2N);
        f_hats[i] = fs[rev];
    }
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

void IFFT_ite_omp(std::complex<double> fs[], std::complex<double> f_hats[], long N, long log2N)
{
// bit reversal of the given array
#pragma omp parallel for shared(fs, f_hats, N, log2N)
    for (unsigned int i = 0; i < N; ++i)
    {
        int rev = bitReverse(i, log2N);
        f_hats[i] = fs[rev];
    }
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

// Recursive version
void FFT_rec(std::complex<double> fs[], std::complex<double> f_hats[], long N) {
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
            FFT_rec(fs_even, f_even, N/2);
            FFT_rec(fs_odd, f_odd, N/2);
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

void IFFT_rec(std::complex<double> fs[], std::complex<double> f_hats[], long N) {
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
        IFFT_rec(fs_even, f_even, N/2);
        IFFT_rec(fs_odd, f_odd, N/2);
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

// Recursive with parallel for
void FFT_rec_para(std::complex<double> fs[], std::complex<double> f_hats[], long N, long threshold) {
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
        #pragma omp parallel
        {
            #pragma omp for
            for (long i = 0; i < N/2; i++) {
                fs_even[i] = fs[2*i];
                fs_odd[i] = fs[1+2*i];
            }
            #pragma omp single
            {
                // Recursion
                long next_N = N / 2;
                if (next_N < threshold) {
                    FFT_rec(fs_even, f_even, next_N);
                    FFT_rec(fs_odd, f_odd, next_N);
                }
                else {
                    FFT_rec_para(fs_even, f_even, next_N, threshold);
                    FFT_rec_para(fs_odd, f_odd, next_N, threshold); 
                }
            }
            std::complex<double> p;
            std::complex<double> q;
            // Compute coefficients
            #pragma omp for
            for (long i = 0; i < N/2; i++) {
                p = f_even[i];
                q = exp(2. * M_PI * i / N * I) * f_odd[i];
                f_hats[i] = p+q;
                f_hats[i+N/2] = p-q;
            }
        }
    }
}

void FFT_rec_1(std::complex<double> fs[], std::complex<double> f_hats[], long N, long threshold) {
    if (N > threshold)
        FFT_rec_para(fs, f_hats, N, threshold);
    else
        FFT_rec(fs, f_hats, N);
}

void IFFT_rec_para(std::complex<double> fs[], std::complex<double> f_hats[], long N, long threshold) {
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
        #pragma omp parallel
        {
            #pragma omp for
            for (long i = 0; i < N/2; i++) {
                fs_even[i] = fs[2*i];
                fs_odd[i] = fs[1+2*i];
            }

            #pragma omp single
            {
                // Recursion
                long next_N = N / 2;
                if (next_N < threshold) {
                    IFFT_rec(fs_even, f_even, next_N);
                    IFFT_rec(fs_odd, f_odd, next_N);
                }
                else {
                    IFFT_rec_para(fs_even, f_even, next_N, threshold);
                    IFFT_rec_para(fs_odd, f_odd, next_N, threshold); 
                }
            }
            std::complex<double> p;
            std::complex<double> q;
            // Compute coefficients
            #pragma omp for
            for (long i = 0; i < N/2; i++) {
                p = f_even[i];
                q = exp(-2. * M_PI * i / N * I) * f_odd[i];
                f_hats[i] = (p+q);
                f_hats[i+N/2] = (p-q);
            }
        }
    }
}

void IFFT_rec_1(std::complex<double> fs[], std::complex<double> f_hats[], long N, long threshold) {
    if (N > threshold)
        IFFT_rec_para(fs, f_hats, N, threshold);
    else
        IFFT_rec(fs, f_hats, N);
}

// Recursive with task
void FFT_rec_task(std::complex<double> fs[], std::complex<double> f_hats[], long N, int depth, int threshold) {
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
        if (depth < threshold) {
            #pragma omp task shared(f_even)
            FFT_rec_task(fs_even, f_even, N/2, depth + 1, threshold);
            #pragma omp task shared(f_odd)
            FFT_rec_task(fs_odd, f_odd, N/2, depth + 1, threshold);
        }
        else {
            #pragma omp task shared(f_even)
            FFT_rec(fs_even, f_even, N/2);
            #pragma omp task shared(f_odd)
            FFT_rec(fs_odd, f_odd, N/2);
        }
        
        #pragma omp taskwait
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

void FFT_rec_2(std::complex<double> fs[], std::complex<double> f_hats[], long N, int threshold) {
    #pragma omp parallel
    #pragma omp single
    FFT_rec_task(fs, f_hats, N, 0, threshold);
}

void IFFT_rec_task(std::complex<double> fs[], std::complex<double> f_hats[], long N, int depth, int threshold) {
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
        if (depth < threshold) {
            #pragma omp task shared(f_even)
            IFFT_rec_task(fs_even, f_even, N/2, depth + 1, threshold);
            #pragma omp task shared(f_odd)
            IFFT_rec_task(fs_odd, f_odd, N/2, depth + 1, threshold);
        }
        else {
            #pragma omp task shared(f_even)
            IFFT_rec(fs_even, f_even, N/2);
            #pragma omp task shared(f_odd)
            IFFT_rec(fs_odd, f_odd, N/2);
        }
        
        #pragma omp taskwait
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

void IFFT_rec_2(std::complex<double> fs[], std::complex<double> f_hats[], long N, int threshold) {
    #pragma omp parallel
    #pragma omp single
    IFFT_rec_task(fs, f_hats, N, 0, threshold);
}

// Compute largest error i.e. max norm
double err(double *x, std::complex<double> y[], long N)
{
    double error = 0;
    for (long i = 0; i < N; i++)
        error = std::max(error, fabs(x[i] - y[i].real()));
    return error;
}

int main(int argc, char **argv)
{
    // Test with cosine function
    long log2N = read_option<long>("-n", argc, argv);
    //double speedup[log2N];

    for (int i = 1; i <= log2N; i++)
    {
        double rec_time = 0.0;
        double rec_para_time = 0.0;
        double rec_task_time = 0.0;
        double ite_time = 0.0;
        double ite_para_time= 0.0;
        for (int k = 0; k < 20; k++)
        {

            long N = pow(2, i);
            std::complex<double> fs[N];
            std::complex<double> f_hats[N];
            double xs[N];
            double ys[N];

            // Run recursive
            for (long j = 0; j < N; j++)
            {
                double x = -1. + j * 0.1;
                double y = cos(x * M_PI);
                xs[j] = x;
                ys[j] = y;
                fs[j] = y;
                f_hats[j] = 0.;
            }
            Timer timer, timer2;
            timer.tic();
            FFT_rec(fs, f_hats, N);
            IFFT_rec(f_hats, fs, N);
            rec_time += timer.toc(); 

            // Run recursive parallel
            long threshold = 1024;
            for (long j = 0; j < N; j++)
            {
                double x = -1. + j * 0.1;
                double y = cos(x * M_PI);
                xs[j] = x;
                ys[j] = y;
                fs[j] = y;
                f_hats[j] = 0.;
            }
            timer.tic();
            FFT_rec_1(fs, f_hats, N, threshold);
            IFFT_rec_1(f_hats, fs, N, threshold);
            rec_para_time += timer.toc(); 

            // Run recursive task
            long threshold_depth = i <= 10 ? 0 : i - 10;
            for (long j = 0; j < N; j++)
            {
                double x = -1. + j * 0.1;
                double y = cos(x * M_PI);
                xs[j] = x;
                ys[j] = y;
                fs[j] = y;
                f_hats[j] = 0.;
            }
            timer.tic();
            FFT_rec_2(fs, f_hats, N, threshold_depth);
            IFFT_rec_2(f_hats, fs, N, threshold_depth);
            rec_task_time += timer.toc(); 

            // Run iterative
            for (long j = 0; j < N; j++)
            {
                double x = -1. + j * 0.1;
                double y = cos(x * M_PI);
                xs[j] = x;
                ys[j] = y;
                fs[j] = y;
                f_hats[j] = 0.;
            }
            timer.tic();
            FFT_ite(fs, f_hats, N, i);
            IFFT_ite(f_hats, fs, N, i);
            ite_time += timer.toc();

            // Run iterative parallel
            for (long j = 0; j < N; j++)
            {
                double x = -1. + j * 0.1;
                double y = cos(x * M_PI);
                xs[j] = x;
                ys[j] = y;
                fs[j] = y;
                f_hats[j] = 0.;
            }
            timer2.tic();
            FFT_ite(fs, f_hats, N, i);
            IFFT_ite(f_hats, fs, N, i);
            ite_para_time += timer2.toc();

            
        }
        //speedup[i] = omp_time/seq_time;
        printf("--------%d---------\n", i);
        printf("rec time: %3f\n", rec_time/20);
        printf("rec with parallelfor time: %3f\n", rec_para_time/20); 
        printf("rec with task time: %3f\n", rec_task_time/20); 
        printf("ite time: %3f\n", ite_time/20);
        printf("ite with parallelfor time: %3f\n", ite_para_time/20);
        //printf("N: %d; speedup: %3f\n", i, speedup[i]);

    }

    return 0;
}