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
#define LOCAL_TOT(c)  (local_tot[blockIdx.x * numCents + c])

/* Executes K-Means algorithm on the CPU (host) */
void hostKmeans(int NUMIT, int numPoints, int numCent, int dims, data_t * x, data_t * dist, data_t * cent, data_t * mean, int * bestCent, data_t * cent_r);

/* Executes K-Means algorithm on the GPU (device) */
__global__ void classifyPoints(int NUMIT, int numPoints, int numCent, int dims, data_t * x, data_t * dist, data_t * cent, data_t * mean, int * bestCent, data_t * cent_r, int pointsPerThread, int* local_tot, int* tot);

__global__ void
	clearVars(int numCent, int dims,
	int numCentPerThread,
	data_t * cent_r,	/* Temporary accumulation cells for each centroid's mean coordinates. All the threads will accumulate here, so they must use atomicAdd. Length: dims * numCent. */
	int* local_tot,		/* Number of total points of each centroid, summed from all CUDA blocks. */
	int* tot			/* Number of points of each centroid, from the points belonging to a certain CUDA block. */
	);

__global__ void accumulateTotals(int NUMIT, int numPoints, int numCent, int dims, data_t * x, data_t * dist, data_t * cent, data_t * mean, int * bestCent, data_t * cent_r, int pointsPerThread, int* local_tot, int* tot);

__global__ void reduceTotals(int NUMIT, int numPoints, int numCent, int dims, data_t * x, data_t * dist, data_t * cent, data_t * mean, int * bestCent, data_t * cent_r, int pointsPerThread, int* local_tot, int* tot);

__global__ void calculateCentroids(int NUMIT, int numPoints, int numCent, int dims, data_t * x, data_t * dist, data_t * cent, data_t * mean, int * bestCent, data_t * cent_r, int pointsPerThread, int* local_tot, int* tot);

#endif
