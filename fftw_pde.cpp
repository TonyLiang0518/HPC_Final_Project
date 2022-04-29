#include <fftw3.h>
#include <math.h>
#include "utils.h"
#include <complex>

int main(int argc, char **argv)
{

    long log2N = read_option<long>("-n", argc, argv);
    long times = 20;
    double time_arr[log2N];

    long N = pow(2, log2N);
    fftw_complex *in, *out, *back_out, *back_in;
    fftw_plan p, q;
    in = (fftw_complex *)fftw_malloc(sizeof(fftw_complex) * N);
    out = (fftw_complex *)fftw_malloc(sizeof(fftw_complex) * N);
    back_in = (fftw_complex *)fftw_malloc(sizeof(fftw_complex) * N);
    back_out = (fftw_complex *)fftw_malloc(sizeof(fftw_complex) * N);
    double h = 2. * M_PI / (N);
    double alpha = 1.;
    double beta = 1.;
    for (long j = 0; j < N; j++)
    {
        double x = h * (j);
        double u_true = sin(cos(x));
        double u_x = -cos(cos(x)) * sin(x);
        double u_xx = -sin(x) * sin(x) * sin(cos(x)) - cos(x) * cos(cos(x));
        in[j][0] = u_xx + alpha * u_x - beta * u_true;
        in[j][1] = 0.;
    }
    Timer timer;
    timer.tic();
    p = fftw_plan_dft_1d(N, in, out, FFTW_FORWARD, FFTW_ESTIMATE);

    fftw_execute(p); /* repeat as needed */

    std::complex<double> coeff[N];
    // wave frequency shifted
    for (long j = 0; j < N/2; j++) {
        coeff[j] = j;
        coeff[N-j-1] = -1-j;
    }
    const std::complex<double> I(0.0, 1.0);
    std::complex<double> div;
    for (long j = 0; j < N; j++)
    {
        div = (-beta + I * alpha * coeff[j] - coeff[j] * coeff[j]);
        back_in[j][0] = out[j][0] / div.real();
    }
    q = fftw_plan_dft_1d(N, back_in, back_out, FFTW_BACKWARD, FFTW_ESTIMATE);

    fftw_execute(q);

    for (long i = 0; i < N; i++)
    {

        printf("%10f ", back_out[i][0] / N);
    }
    fftw_destroy_plan(p);
    fftw_destroy_plan(q);
    fftw_free(in);
    fftw_free(out);
    fftw_free(back_out);

    return 0;
}
