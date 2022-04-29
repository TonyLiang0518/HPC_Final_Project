#include <fftw3.h>
#include <math.h>
#include "utils.h"

int main(int argc, char **argv)
{

    long log2N = read_option<long>("-n", argc, argv);
    long times = 20;
    double time_arr[log2N];

    for (int i = 1; i <= log2N; i++)
    {
        // printf("N: %d\n", i);
        double total_time = 0.0;
        for (int k = 0; k < times; k++)
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

            total_time += timer.toc();
            // printf("run time: %3f\n", timer.toc());
            double err = 0.0;
            for (int i = 0; i < N; i++)
            {
                err += in[i][0] - back_out[i][0] / N;
            }
            // printf("error: %3f\n", err);
            fftw_destroy_plan(p);
            fftw_destroy_plan(q);
            fftw_free(in);
            fftw_free(out);
            fftw_free(back_out);
        }
        time_arr[i-1] = total_time / times;
        // printf("run time: %3f\n", total_time / times);
    }
    printf("fftw run time\n");
    for (int i = 0; i < log2N; i++) {
        printf("%.20g ", time_arr[i]);
    }
    

    return 0;
}
