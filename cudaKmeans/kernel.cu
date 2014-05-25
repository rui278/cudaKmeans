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




/* Clears the vars cent_r, local_tot and tot */
__global__ void
	clearVars(
	int numCent,
	int dims,
	int numCentPerThread,
	data_t * cent_r,	/* Temporary accumulation cells for each centroid's mean coordinates. All the threads will accumulate here, so they must use atomicAdd. Length: dims * numCent. */
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
			cent_r[(centOffset + c) * dims + j] = 0;
		}

		// Clear the global variable of this centroid for all blocks
		tot[centOffset + c] = 0;
	}
}






/* Classify points to nearest centroid. Part 1 */
__global__ void
	classifyPoints(int NUMIT, int numPoints, int numCent, int dims,
	data_t * x, data_t * dist, data_t * cent,
	int * bestCent,
	data_t * cent_r,	/* Temporary accumulation cells for each centroid's mean coordinates. All the threads will accumulate here, so they must use atomicAdd. Length: dims * numCent. */
	int pointsPerThread,
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
		{
			return;
		}

		// Find this point's nearest centroid
		rMin=INT_MAX;
		for (k=0; k < numCent; k++)
		{
			int e;
			DIST_(j + pointOffset, k) = 0;
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

		int e;
		for (e = 0; e < dims; e++)
		{
			atomicAdd(&CENT_R_(bestCent[j+pointOffset], e), X_(j+pointOffset, e));
		}
		atomicAdd( &(tot[bestCent[j+pointOffset]]) , 1 );
	}
}







__global__ void
	calculateCentroids(int NUMIT, int numPoints, int numCent, int dims,
	data_t * x,
	data_t * dist,
	data_t * cent,
	data_t * mean,
	int * bestCent,
	data_t * cent_r,	/* Temporary accumulation cells for each centroid's mean coordinates. All the threads will accumulate here, so they must use atomicAdd. Length: dims * numCent. */
	int numCentPerThread,
	int* tot			/* Number of points of each centroid, from the points belonging to a certain CUDA block. */
	)
{
	int centOffset = (blockDim.x * blockIdx.x + threadIdx.x) * numCentPerThread;

	int k;
	for (k=0; k<numCentPerThread; k++)
	{
		if (centOffset + k >= numCent)
			return;

		/* If centroid has more than 0 points associated (normal), relocate it to mean of its points. */
		if (tot[k + centOffset] > 0)
		{
			int e;
			for (e=0; e<dims; e++)
				CENT_(k + centOffset, e)=CENT_R_(k + centOffset, e) / tot[k + centOffset];
		}
		/* Else, relocate it to the mean of the other centroids (put it near points) */
		else
		{
			int e;
			for (e=0; e<dims; e++)
				CENT_(k + centOffset, e)=mean[e];
		}
	}


}