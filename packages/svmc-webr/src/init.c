#include <R.h>
#include <Rinternals.h>
#include <R_ext/Rdynload.h>

/* Fortran wrapper entry points (symbol names lowered by F77_NAME) */
extern void F77_NAME(r_alloc_h2)(
    double *, double *, double *, double *, double *,
    double *, double *, double *, double *,
    double *, double *, double *,
    double *, double *, double *,
    double *, double *,
    double *, double *, double *, double *,
    double *, double *, double *,
    double *, double *, double *,
    int *, double *, double *,
    double *, double *,
    int *, int *);

extern void F77_NAME(r_invert_alloc)(
    double *, double *, double *, double *,
    double *, double *, double *,
    double *, double *, double *, double *,
    double *, double *, double *,
    double *, double *, double *,
    int *, double *, double *,
    double *, double *,
    int *, int *);

static const R_FortranMethodDef FortranEntries[] = {
    {"r_alloc_h2",      (DL_FUNC) &F77_NAME(r_alloc_h2),      34},
    {"r_invert_alloc",  (DL_FUNC) &F77_NAME(r_invert_alloc),  24},
    {NULL, NULL, 0}
};

void R_init_SVMCwebr(DllInfo *dll) {
    R_registerRoutines(dll, NULL, NULL, FortranEntries, NULL);
    R_useDynamicSymbols(dll, FALSE);
}
