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

int main(int argc, char* argv[]) {
    unsigned int log2N; // input number
    if(argc != 2)
    {
        fprintf(stderr, "sequential prime factorization\n");
        fprintf(stderr, "num = input number\n");
        exit(1);
    }

    log2N = (unsigned int) atoi(argv[1]);
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
    FFT(fs, f_hats, N);
    IFFT(f_hats, fs, N);
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