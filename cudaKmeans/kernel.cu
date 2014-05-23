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

__global__ void
	kernel(int NUMIT, int numPoints, int numCent, int dims,
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

    int it;
    int j, k;

	if (pointOffset > numPoints)
		return;
    
    for (it=0; it<NUMIT; it++)
    {
        /* calculate distance matrix and minimum */
        data_t rMin;
        
        for (j=0; j<pointsPerThread; j++)
        {
            rMin=INT_MAX;
            for (k=0; k < numCent; k++)
            {
                int e;
                DIST_(j + pointOffset, k)=0;
                for (e=0; e<dims; e++)
                {
                    DIST_(j+pointOffset, k) += (X_(j+pointOffset, e) - CENT_(k, e)) * (X_(j+pointOffset, e) - CENT_(k, e));
                    /* printf("   dist=%f\n",dist[j][k]);*/
                }
                /* printf("x[j]=(%f,%f), cent[k]=(%f,%f), dist[%d][%d]=%f\n",x[j][0],x[j][1],cent[k][0],cent[k][1],j,k,dist[j][k]); */
                if (DIST_(j+pointOffset, k) < rMin)
                {
                    bestCent[j+pointOffset]=k;
                    rMin=DIST_(j+pointOffset, k);
                }
            }
        }

		// Clear the shared variables for this block
		if (pointOffset == 0)
		{
			int i, j;
			for (i = 0; i < numCent; i++)
			{
				for (j = 0; j < dims; j++)
				{
					cent_r[i * dims + j] = 0;
				}
				local_tot[i] = 0;
			}

			//memset(cent_r, 0, dims * numCent * sizeof(data_t));
			//memset(local_tot, 0, numCent * sizeof(int));
		}

		// Clear the global variables for all blocks
		if (blockIdx.x == 0 && threadIdx.x == 0)
		{
			int i;
			for (i = 0; i < numCent; i++)
				tot[i] = 0;

			// memset(tot, 0, numCent * sizeof(int));
		}

		__syncthreads();
        
        /* reestimate centroids */
        for (k=0; k<numCent; k++)
        {
            /* Count number of points belonging to this centroid, and accumulate coordinates */
            
            // memset(CENT_R(k+), 0, sizeof(data_t) * dims);

			for (j=0; j<pointsPerThread; j++)
            {
				if (bestCent[j+pointOffset]==k)
                {
                    int e;
                    for (e=0; e<dims; e++)
						atomicAdd(&(CENT_R_(k, e)), X_(j+pointOffset,e));
                        // cent_r[e]+=X_(j,e);

					atomicAdd(&local_tot[k], 1);
                }
            }
        }

		__syncthreads();

		// Reduce all counts among all blocks
		// TODO
		if (blockIdx.x == 0 && threadIdx.x == 0)
		{
			for (k=0; k<numCent; k++)
			{
				tot[k] += local_tot[k];
			}

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

		__syncthreads();
    }
}