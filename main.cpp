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
void FFT_ser(std::complex<double> fs[], std::complex<double> f_hats[], long N) {
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
        FFT_ser(fs_even, f_even, N/2);
        FFT_ser(fs_odd, f_odd, N/2);
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

void FFT_par(std::complex<double> fs[], std::complex<double> f_hats[], long N, int depth, int threshold) {
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
            FFT_par(fs_even, f_even, N/2, depth + 1, threshold);
            #pragma omp task shared(f_odd)
            FFT_par(fs_odd, f_odd, N/2, depth + 1, threshold);
        }
        else {
            #pragma omp task shared(f_even)
            FFT_ser(fs_even, f_even, N/2);
            #pragma omp task shared(f_odd)
            FFT_ser(fs_odd, f_odd, N/2);
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

void FFT(std::complex<double> fs[], std::complex<double> f_hats[], long N, int threshold) {
    #pragma omp parallel
    #pragma omp single
    FFT_par(fs, f_hats, N, 0, threshold);
}

// Inverse Fast Fourier Transform, based on Cooley-Tukey FFT
void IFFT_ser(std::complex<double> fs[], std::complex<double> f_hats[], long N) {
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
        IFFT_ser(fs_even, f_even, N/2);
        IFFT_ser(fs_odd, f_odd, N/2);
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

void IFFT_par(std::complex<double> fs[], std::complex<double> f_hats[], long N, int depth, int threshold) {
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
            IFFT_par(fs_even, f_even, N/2, depth + 1, threshold);
            #pragma omp task shared(f_odd)
            IFFT_par(fs_odd, f_odd, N/2, depth + 1, threshold);
        }
        else {
            #pragma omp task shared(f_even)
            IFFT_ser(fs_even, f_even, N/2);
            #pragma omp task shared(f_odd)
            IFFT_ser(fs_odd, f_odd, N/2);
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

void IFFT(std::complex<double> fs[], std::complex<double> f_hats[], long N, int threshold) {
    #pragma omp parallel
    #pragma omp single
    IFFT_par(fs, f_hats, N, 0, threshold);
}

// Compute largest error i.e. max norm
double err(double* x, std::complex<double> y[], long N) {
  double error = 0;
  for (long i = 0; i < N; i++) error = std::max(error, fabs(x[i]-y[i].real()));
  return error;
}

int main() {
    // Test with cosine function
    long N = 8192;
    long log2N = 13;
    int threshold = 10;
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