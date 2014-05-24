//
//  kernel.c
//  cudaKmeans
//
//  Created by Rui on 22/05/14.
//  Copyright (c) 2014 __Grupo215AAC__. All rights reserved.
//

#include <stdio.h>
#include <string.h>
#include <limits.h>

#include "kernel.h"

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

/* Classify points to nearest centroid. Part 1 */
__global__ void
	classifyPoints(int NUMIT, int numPoints, int numCent, int dims,
	data_t * x, data_t * dist, data_t * cent,
	data_t * mean,
	int * bestCent,
	data_t * cent_r,	/* Temporary accumulation cells for each centroid's mean coordinates. All the threads will accumulate here, so they must use atomicAdd. Length: dims * numCent. */
	int pointsPerThread,
	int* local_tot,		/* Number of total points of each centroid, summed from all CUDA blocks. */
	int* tot			/* Number of points of each centroid, from the points belonging to a certain CUDA block. */
	)
{
	int pointOffset = (blockDim.x * blockIdx.x + threadIdx.x) * pointsPerThread;

	int j, k;

	/* calculate distance matrix and minimum */
	data_t rMin;

	for (j=0; j<pointsPerThread; j++)
	{
		/* Last threads of last block may not have a round number of points */
		if (pointOffset + j >= numPoints)
			return;

		rMin=INT_MAX;
		for (k=0; k < numCent; k++)
		{
			int e;
			DIST_(j + pointOffset, k)=0;
			for (e=0; e<dims; e++)
			{
				DIST_(j+pointOffset, k) += (X_(j+pointOffset, e) - CENT_(k, e)) * (X_(j+pointOffset, e) - CENT_(k, e));
			}
			if (DIST_(j+pointOffset, k) < rMin)
			{
				bestCent[j+pointOffset]=k;
				rMin=DIST_(j+pointOffset, k);
			}
		}


	}
}



/* Clears the vars cent_r, local_tot and tot */
__global__ void
	clearVars(int numCent, int dims,
	int numCentPerThread,
	data_t * cent_r,	/* Temporary accumulation cells for each centroid's mean coordinates. All the threads will accumulate here, so they must use atomicAdd. Length: dims * numCent. */
	int* local_tots,	/* Number of points of each centroid, from the points belonging to a certain CUDA block. */
	int* tot			/* Number of total points of each centroid, summed from all CUDA blocks. */
	)
{
	int centOffset = (blockDim.x * blockIdx.x + threadIdx.x) * numCentPerThread;

	// Clear the shared variables for this block
	int c;
	for (c = 0; c < numCentPerThread; c++)
	{
		if (centOffset + c >= numCent)
			return;

		int j;
		for (j = 0; j < dims; j++)
		{
			cent_r[(threadIdx.x+c) * dims + j] = 0;
		}
		local_tots[centOffset + c] = 0;

		// Clear the global variable of this centroid for all blocks
		tot[threadIdx.x + c] = 0;
	}


}




__global__ void
	accumulateTotals(int NUMIT, int numPoints, int numCent, int dims,
	data_t * x, data_t * dist, data_t * cent,
	data_t * mean,
	int * bestCent,
	data_t * cent_r,	/* Temporary accumulation cells for each centroid's mean coordinates. All the threads will accumulate here, so they must use atomicAdd. Length: dims * numCent. */
	int pointsPerThread,
	int* local_tot,		/* Number of total points of each centroid, summed from all CUDA blocks. */
	int* tot			/* Number of points of each centroid, from the points belonging to a certain CUDA block. */
	)
{
	/* Reestimate centroids */
	/* Accumulate centroid coordinates */
	
	int pointOffset = (blockDim.x * blockIdx.x + threadIdx.x) * pointsPerThread;
	
	int j;
	/* Count number of points belonging to this centroid, and accumulate coordinates */

	for (j=0; j<pointsPerThread; j++)
	{
		if (pointOffset + j >= numPoints)
			return;

		int k = bestCent[j+pointOffset];

		int e;
		for (e=0; e<dims; e++)
			atomicAdd(&(CENT_R_(k, e)), X_(j+pointOffset,e));

		atomicAdd(&local_tot[k], 1);
	}
}






__global__ void
	reduceTotals(int NUMIT, int numPoints, int numCent, int dims,
	data_t * x, data_t * dist, data_t * cent,
	data_t * mean,
	int * bestCent,
	data_t * cent_r,	/* Temporary accumulation cells for each centroid's mean coordinates. All the threads will accumulate here, so they must use atomicAdd. Length: dims * numCent. */
	int numCentPerThread,
	int* local_tot,		/* Number of total points of each centroid, summed from all CUDA blocks. */
	int* tot			/* Number of points of each centroid, from the points belonging to a certain CUDA block. */
	)
{

	// Reduce all counts among all blocks
	// TODO

	int centOffset = (blockDim.x * blockIdx.x + threadIdx.x) * numCentPerThread;

	// Clear the shared variables for this block
	int c;
	for (c = 0; c < numCentPerThread; c++)
	{
		if (centOffset + c >= numCent)
			return;

		atomicAdd();
	}
}

__global__ void
	calculateCentroids(int NUMIT, int numPoints, int numCent, int dims,
	data_t * x, data_t * dist, data_t * cent,
	data_t * mean,
	int * bestCent,
	data_t * cent_r,	/* Temporary accumulation cells for each centroid's mean coordinates. All the threads will accumulate here, so they must use atomicAdd. Length: dims * numCent. */
	int pointsPerThread,
	int* local_tot,		/* Number of total points of each centroid, summed from all CUDA blocks. */
	int* tot			/* Number of points of each centroid, from the points belonging to a certain CUDA block. */
	)
{
	int k;
	for (k=0; k<numCent; k++)
	{
		/* If centroid has more than 0 points associated (normal), relocate it to mean of its points. */
		if (tot[k] > 0)
		{
			int e;
			for (e=0; e<dims; e++)
				CENT_(k, e)=CENT_R_(k, e)/tot[k];
		}
		/* Else, relocate it to the mean of the other centroids (put it near points) */
		else
		{
			int e;
			for (e=0; e<dims; e++)
				CENT_(k, e)=mean[e];
		}
	}


}