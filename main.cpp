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
void FFT_seq(std::complex<double> fs[], std::complex<double> f_hats[], long N) {
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
            FFT_seq(fs_even, f_even, N/2);
            FFT_seq(fs_odd, f_odd, N/2);
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

void FFT_omp(std::complex<double> fs[], std::complex<double> f_hats[], long N, long threshold) {
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
                    FFT_seq(fs_even, f_even, next_N);
                    FFT_seq(fs_odd, f_odd, next_N);
                }
                else {
                    FFT_omp(fs_even, f_even, next_N, threshold);
                    FFT_omp(fs_odd, f_odd, next_N, threshold); 
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

void FFT(std::complex<double> fs[], std::complex<double> f_hats[], long N, long threshold) {
    if (N > threshold)
        FFT_omp(fs, f_hats, N, threshold);
    else
        FFT_seq(fs, f_hats, N);
}

// Inverse Fast Fourier Transform, based on Cooley-Tukey FFT
void IFFT_seq(std::complex<double> fs[], std::complex<double> f_hats[], long N) {
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
        IFFT_seq(fs_even, f_even, N/2);
        IFFT_seq(fs_odd, f_odd, N/2);
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

void IFFT_omp(std::complex<double> fs[], std::complex<double> f_hats[], long N, long threshold) {
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
                    IFFT_seq(fs_even, f_even, next_N);
                    IFFT_seq(fs_odd, f_odd, next_N);
                }
                else {
                    IFFT_omp(fs_even, f_even, next_N, threshold);
                    IFFT_omp(fs_odd, f_odd, next_N, threshold); 
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

void IFFT(std::complex<double> fs[], std::complex<double> f_hats[], long N, long threshold) {
    if (N > threshold)
        IFFT_omp(fs, f_hats, N, threshold);
    else
        IFFT_seq(fs, f_hats, N);
}

// Compute largest error i.e. max norm
double err(double* x, std::complex<double> y[], long N) {
  double error = 0;
  for (long i = 0; i < N; i++) error = std::max(error, fabs(x[i]-y[i].real()));
  return error;
}

int main() {
    // Test with cosine function
    long N = 32768;
    long threshold = 1024;
    long log2N = 16;
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
    FFT(fs, f_hats, N, threshold);
    IFFT(f_hats, fs, N, threshold); 
    // FFT_ite(fs, f_hats, N, log2N);
    // IFFT_ite(f_hats, fs, N, log2N);
    for (long j = 0; j < N; j++) {
        fs[j] = fs[j].real() / N;
    }
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
    printf("%10f\n", err(ys, fs, N));

    return 0;
}