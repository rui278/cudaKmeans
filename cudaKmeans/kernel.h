//
//  kernel.h
//  cudaKmeans
//
//  Created by Rui on 22/05/14.
//  Copyright (c) 2014 __Grupo215AAC__. All rights reserved.
//

#ifndef cudaKmeans_kernel_h
#define cudaKmeans_kernel_h

#define data_t int

#define X_(p,d)       (x[p * dims + d])
#define DIST_(p, c)   (dist[p * numCent + c])
#define CENT_(c, d)   (cent[c * dims + d])
#define CENT_R_(c,d)  (cent_r[c * dims + d])
#define LOCAL_TOT(c)  ()

/* Executes K-Means algorithm on the CPU (host) */
void hostKmeans(int NUMIT, int numPoints, int numCent, int dims, data_t * x, data_t * dist, data_t * cent, data_t * mean, int * bestCent, data_t * cent_r);

/* Executes K-Means algorithm on the GPU (device) */
__global__ void kernel(int NUMIT, int numPoints, int numCent, int dims, data_t * x, data_t * dist, data_t * cent, data_t * mean, int * bestCent, data_t * cent_r, int pointsPerThread, int* local_tot, int* tot);

#endif
