#include <stdlib.h>
#include <math.h>
#include <lapacke.h>

void s_det_double(const double *a, const int *stridesA,
                  double *res, const int *stridesRes,
                  const int *shape, int rank) {
    if (a == NULL || res == NULL || rank < 2 || rank > 8) return;

    int n = shape[rank - 1];
    int stack_elements = 1;
    for (int i = 0; i < rank - 2; i++) stack_elements *= shape[i];

    double *aCopy = (double *)malloc(n * n * sizeof(double));
    int *ipiv = (int *)malloc(n * sizeof(int));

    int coord[8] = {0};
    int offsetA = 0, offsetRes = 0;

    for (int el = 0; el < stack_elements; el++) {
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                aCopy[i * n + j] = a[offsetA + i * stridesA[rank - 2] + j * stridesA[rank - 1]];
            }
        }

        lapack_int info = LAPACKE_dgetrf(101, n, n, aCopy, n, ipiv);
        
        double detValue = 1.0;
        if (info > 0) {
            detValue = 0.0;
        } else if (info < 0) {
            detValue = NAN;
        } else {
            for (int i = 0; i < n; i++) {
                detValue *= aCopy[i * n + i];
            }
            int swaps = 0;
            for (int i = 0; i < n; i++) {
                if (ipiv[i] != i + 1) swaps++;
            }
            if (swaps % 2 != 0) detValue = -detValue;
        }

        res[offsetRes] = detValue;

        for (int d = rank - 3; d >= 0; d--) {
            coord[d]++;
            if (coord[d] < shape[d]) {
                offsetA += stridesA[d];
                offsetRes += stridesRes[d];
                break;
            }
            coord[d] = 0;
            offsetA -= (shape[d] - 1) * stridesA[d];
            offsetRes -= (shape[d] - 1) * stridesRes[d];
        }
    }

    free(aCopy);
    free(ipiv);
}
