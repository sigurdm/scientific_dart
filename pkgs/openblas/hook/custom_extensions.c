#define HAVE_STDINT_H
#include <stdint.h>
#define int32_t int32_t
#include <lapacke.h>

void* get_dgetrf_ptr(void) {
    return (void*)&LAPACKE_dgetrf;
}

void* get_sgetrf_ptr(void) {
    return (void*)&LAPACKE_sgetrf;
}

void* get_zgetrf_ptr(void) {
    return (void*)&LAPACKE_zgetrf;
}

void* get_cgetrf_ptr(void) {
    return (void*)&LAPACKE_cgetrf;
}
