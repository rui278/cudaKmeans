#ifndef cudaKmeans_macros_h
#define cudaKmeans_macros_h

#define data_t int

#define X_(p,d)       (x[(p) + (d) * numPoints])
#define DIST_(p, c)   (dist[(p) * numCent + (c)])
#define CENT_(c, d)   (cent[(c) + (d) * numCent])
#define CENT_R_(c,d)  (cent_r[(c) + (d) * numCent])
#define LOCAL_TOT(c)  (local_tot[blockIdx.x * numCents + (c)])

#endif
