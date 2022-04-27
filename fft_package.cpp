#include <fftw3.h>
#include <math.h>
#include "utils.h"

int main(int argc, char **argv)
{

    long log2N = read_option<long>("-n", argc, argv);
    for (int i = 1; i <= log2N; i++)
    {
        long N = pow(2, i);
        fftw_complex *in, *out, *back_out;
        fftw_plan p, q;
        in = (fftw_complex *)fftw_malloc(sizeof(fftw_complex) * N);
        out = (fftw_complex *)fftw_malloc(sizeof(fftw_complex) * N);
        back_out = (fftw_complex *)fftw_malloc(sizeof(fftw_complex) * N);
        for (long j = 0; j < N; j++)
        {
            double x = -1. + j * 0.1;
            double y = cos(x * M_PI);
            in[j][0] = x;
            in[j][1] = y;
        }
        Timer timer;
        timer.tic();
        p = fftw_plan_dft_1d(N, in, out, FFTW_FORWARD, FFTW_ESTIMATE);
        q = fftw_plan_dft_1d(N, out, back_out, FFTW_BACKWARD, FFTW_ESTIMATE);
        fftw_execute(p); /* repeat as needed */
        fftw_execute(q);
        printf("N: %d\n", i);
        printf("run time: %3f\n", timer.toc());
        double err = 0.0;
        for (int i = 0; i < N; i++)
        {
            // printf("%3f %3f ", in[i][0], in[i][1]);
            // printf("%3f %3f ", back_out[i][0], back_out[i][1]);
            // printf("%3f ", in[i][0] - back_out[i][0]/N);
            err += in[i][0] - back_out[i][0] / N;
        }
        printf("error: %3f\n", err);
        fftw_destroy_plan(p);
        fftw_destroy_plan(q);
        fftw_free(in);
        fftw_free(out);
        fftw_free(back_out);
    }

    return 0;
}
