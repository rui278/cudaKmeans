//
//  kernel.h
//  cudaKmeans
//
//  Created by Rui on 22/05/14.
//  Copyright (c) 2014 __Grupo215AAC__. All rights reserved.
//

#ifndef cudaKmeans_kernel_h
#define cudaKmeans_kernel_h

#include "macros.h"

/* Executes K-Means algorithm on the GPU (device) */
__global__ void classifyPoints(int NUMIT,
							   int numPoints,
							   int numCent,
							   int dims,
							   data_t * x,
							   data_t * cent,
							   int * bestCent,
							   data_t * cent_r,
							   int pointsPerThread,
							   int* tot);

__global__ void clearVars(int numCent,
						  int dims,
						  int numCentPerThread,
						  data_t * cent_r,	/* Temporary accumulation cells for each centroid's mean coordinates. All the threads will accumulate here, so they must use atomicAdd. Length: dims * numCent. */
						  int* tot			/* Number of points of each centroid, from the points belonging to a certain CUDA block. */
	);

__global__ void calculateCentroids(int NUMIT,
								   int numPoints,
								   int numCent,
								   int dims,
								   data_t * x,
								   data_t * dist,
								   data_t * cent,
								   data_t * mean,
								   int * bestCent,
								   data_t * cent_r,
								   int numCentPerThread,
								   int* tot);

#endif
