#ifndef cudaKmeans_macros_h
#define cudaKmeans_macros_h

#define data_t int

#define X_(p,d)       (x[(p) * dims + (d)])
#define DIST_(p, c)   (dist[(p) * numCent + (c)])
#define CENT_(c, d)   (cent[(c) * dims + (d)])
#define CENT_R_(c,d)  (cent_r[(c) * dims + (d)])
#define LOCAL_TOT(c)  (local_tot[blockIdx.x * numCents + (c)])

#define SHARED_CENT_R(c, e)		(temp_cent_r[(c) * (dims+1) + (e)])
#define SHARED_TOTAL(c)			(temp_cent_r[(c) * (dims+1) + (dims)])

#endif
