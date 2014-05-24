/*********************************************************/
/*                                                       */
/*                     k-means.c                         */
/*                     =========                         */
/*                                                       */
/*        C programme to demonstrate k-means             */
/*                clustering on 2D data                  */
/*                                                       */
/*          Written for EE3J2 Data Mining                */
/*                                                       */
/*        Version 1: Martin Russell 26/02/04             */
/*                                                       */
/* Dept. Electronic, Electrical & Computer Engineering   */
/*            University of Birmingham                   */
/*                                                       */
/*    To compile under linux:                            */
/*                 gcc -lm k-means.c                     */
/*                 mv a.out k-means                      */
/*                                                       */
/*    To run:                                            */
/*                 k-means ipFile centroids opFile numIt */
/*                                                       */
/*                                                       */
/*                  CUDA COMPATIBILITY                   */
/*                  ==================                   */
/*                                                       */
/*                                                       */
/*  Changed by:                                          */
/*    Rui Albuquerque                                    */
/*    Artur Gonçalves                                    */
/*    Daniel Filipe                                      */
/*                                                       */
/*    for: Advanced Computer Architectures Class         */
/*        Electrical and Computer Engineering Department */
/*        Instituto Superior Técnico                     */
/*        Spring Semester                                */
/*        2013/1014                                      */
/*                                                       */
/*  To run:                                              */
/*  kmeans numPoints numCents Dims Range RandSeed <numIt>*/
/*                                                       */
/*                                                       */
/*********************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <limits.h>
#include <math.h>
#include <string.h>
#include <math.h>
#include <ctype.h>
#include <time.h>

#include "cuda_runtime.h"

#include "kernel.h"

#ifdef COUNTTIME
/* Calculates the difference between two times, in seconds. */
double timeDiff(struct timespec tStart, struct timespec tEnd);
#endif

void mySrand(int seed);
int myRand();

int main(int argc, char *argv[])
{

	int j;
	int k;
	int n;

	/*Host data*/
	data_t *x;		/* Max values: x[numPoints * Dimensions]. Coordinate of point P, all 'dims' of them. Indexed as [P * dims + dim]. */
	data_t *cent;	/* Coordinates of each centroid. Max value: [numCent * dims]. Layout is [cx1 cy1 cx2 cy2 cx3 cy3 ...] in case dims=2  */
	data_t *mean;	/* Length: dims */
	data_t *dist;	/* Distance between Point P and Centroid C. Indexed as [P][C]. */
	int *bestCent;	/* Length: numPoints */
	data_t *cent_r; /* Length: dims. Used as temporary variable. */

	/*Device data*/
	/* TODO: put device_x in constant texture memory - should be faster */
	data_t *device_x;		/* Max value: [numPoints * dims]. Layout is [x1 y1 x2 y2 x3 y3 ...] in case dims=2 */
	data_t *device_cent;	/* Max value: [numCent * dims]. Layout is [cx1 cy1 cx2 cy2 cx3 cy3 ...] in case dims=2 */
	data_t *device_mean;	/* Max value: [dims]. */
	data_t *device_dist;	/* Max value: [numPoints * numCent]. Distance between Point P and Centroid C. Indexed as [P * numCent + C]. */
	int *device_bestCent;	/* Length: numPoints */
	data_t *device_cent_r;	/* Length: dims * numCent. Used as temporary variables for accumulation. */
	int *device_cent_tot;	/* Number of points for each centroid, global to all blocks. Length: numCent */
	int *device_cent_partial_tots;	/* Number of points for each centroid, local to one block. Length: numCent */

	int numPoints;	/* Number of points */
	int dims;		/* Dimensions */
	int numCent;	/* Number of centroids */
	int NUMIT = 100;

	int range;
	int randSeed;

	int numThreadsPerBlock;	/* Number of threads per block */
	int numPointsPerThread;	/* Number of points per thread */
	int numBlocks;			/* Number of blocks */
	int numCentPerThread;

	void * hostErr [6];	// Errors for malloc
	cudaError_t err[8];	// Errors for cudaMalloc and cudaMemcpy

#ifdef COUNTTIME
	struct timespec hostStartTime;
	struct timespec hostEndTime;

	struct timespec startTime;
	struct timespec endTime;

	struct timespec startCommTime;
	struct timespec endCommTime;

	struct timespec memBackStartTime;
	struct timespec memBackEndTime;
#endif

	/* Check correct number of input parameters */
	if ((argc!=6)&&(argc!=7))
	{
		printf("format: k-means numPoints numCents Dims Range RandSeed <numIt>\n");
		exit(1);
	}

	printf("Reading Arguments\n");

	numPoints	= atoi(argv[1]);
	numCent		= atoi(argv[2]);
	dims		= atoi(argv[3]);
	range		= atoi(argv[4]);
	randSeed	= atoi(argv[5]);

	if(argc == 7)
		NUMIT = atoi(argv[6]);

	if(numCent > numPoints)
	{
		printf("Number of Centroids must be smaller than the number of points");
		exit(1);
	}

	printf("Starting with arguments:\n\
		   numPoints = %d\n\
		   numCents  = %d\n\
		   Dims      = %d\n\
		   Range     = %d\n\
		   RandSeed  = %d\n\
		   numIt     = %d\n\n",
		   numPoints, numCent, dims, range, randSeed, NUMIT);

	printf("Allocating host memory\n");

	/* allocate memory */
	hostErr[0] = x = (data_t *)calloc(numPoints * dims,sizeof(data_t *));
	hostErr[1] = cent=(data_t *)calloc(numCent * dims,sizeof(data_t *));
	hostErr[2] = mean=(data_t *)calloc(dims,sizeof(data_t));
	hostErr[3] = dist=(data_t *)calloc(numPoints * numCent,sizeof(data_t *));
	hostErr[4] = bestCent=(int *)calloc(numPoints,sizeof(int));
	hostErr[5] = cent_r=(data_t *)calloc(dims,sizeof(data_t));
	
	for(n = 0; n < 6; n++) {
		if(hostErr[n] == NULL){
			printf("Error allocating memory on host (error code %s). Exiting.", cudaGetErrorString(err[n]));
			exit(0);
		}
	}

	/* Calculate kernel parameters */

	cudaDeviceProp props;
	err[0] = cudaGetDeviceProperties (&props, 0);
	if (err[0] != cudaSuccess)
	{
		printf("Error, could not find device 0: %s\n", cudaGetErrorString(err[0]));
		exit(1);
	}

	// int maxThreads = props.maxThreadsPerMultiProcessor * props

	// For the classifyPoints kernel
	// Work to be done: numPoints * numCent

	//numPointsPerThread = numPoints / 4096; // Assuming 4096 threads, and numPoints > 4096
	//if (numPoints % 4096 != 0)
	//	numPointsPerThread++; // Always round up

	//numThreadsPerBlock = 1024;
	//
	//numBlocks = 4096 / numThreadsPerBlock;

	numPointsPerThread = numPoints / 4096;
		if (numPoints % 4096 != 0)
			numPointsPerThread++; // Always round up

	numBlocks = 4;
	numThreadsPerBlock = 1024;

	numCentPerThread = numCent / 4096;
		if (numCent % 4096 != 0)
			numCentPerThread++; // Always round up
	


	/* Memory allocation on the CUDA device */

	/* Touch the device once to initialize it */
	err[0] = cudaFree(0);
	if (err[0] != cudaSuccess)
	{
		printf("Error on first touch (cudaFree(0)): %s\nExiting.\n", cudaGetErrorString(err[0]));
		exit(1);
	}

	/*Allocate device memory*/
	printf("Allocating device memory\n");

	err[0] = cudaMalloc ((void **) &device_x, numPoints * dims * sizeof(data_t));
	err[1] = cudaMalloc ((void **) &device_cent, numCent * dims * sizeof(data_t));
	err[2] = cudaMalloc ((void **) &device_mean, dims * sizeof(data_t));
	err[3] = cudaMalloc ((void **) &device_dist, numPoints * numCent * sizeof(data_t));
	err[4] = cudaMalloc ((void **) &device_bestCent, numPoints * sizeof(int));
	err[5] = cudaMalloc ((void **) &device_cent_r, dims * numCent * sizeof(data_t));
	
	err[6] = cudaMalloc ((void **) &device_cent_tot, numCent * sizeof(int));
	err[7] = cudaMalloc ((void **) &device_cent_partial_tots, numCent * numBlocks * sizeof(int));

	for(n = 0; n < 8; n++) {
		if(err[n] != cudaSuccess){
			printf("Error allocating memory on device: %s\nExiting.\n", cudaGetErrorString(err[n]));
			exit(1);
		}
	}

	/*Generate random test set according to user specification*/
	printf("Generating Test Set\n");

	mySrand(randSeed);

	for(n = 0; n < numPoints; n ++)
	{
		for(j = 0; j < dims; j++)
		{
			X_(n,j) = myRand() % range;
			mean[j] += X_(n,j);

			if(n < numCent){
				CENT_(n, j) = myRand() % range;
			}

		}
	}

	for(n = 0; n < dims; n++)
	{
		mean[n] = mean[n]/numPoints;
	}

	/* Sending Data to Device*/

#ifdef COUNTTIME
	clock_gettime(CLOCK_REALTIME, &startCommTime);
#endif

	err[0] = cudaMemcpy (device_x, x, numPoints * dims * sizeof(data_t), cudaMemcpyHostToDevice);
	err[1] = cudaMemcpy (device_cent, cent, numCent * dims * sizeof(data_t), cudaMemcpyHostToDevice);
	err[2] = cudaMemcpy (device_mean, mean, dims * sizeof(data_t), cudaMemcpyHostToDevice);
	err[3] = cudaMemcpy (device_dist, dist, numPoints * numCent * sizeof(data_t), cudaMemcpyHostToDevice);
	err[4] = cudaMemcpy (device_bestCent,bestCent, numPoints * sizeof(int), cudaMemcpyHostToDevice);
	err[5] = cudaMemcpy (device_cent_r, cent_r, dims * numCent * sizeof(data_t), cudaMemcpyHostToDevice);

	err[6] = cudaMemset (device_cent_tot, 0, numCent * sizeof(int));
	err[7] = cudaMemset (device_cent_partial_tots, 0, numCent * numBlocks * sizeof(int));

	for(n = 0; n < 8; n++) {
		if(err[n] != cudaSuccess){
			printf("Error allocating memory on device (error code %s). Exiting.", cudaGetErrorString(err[n]));
			exit(0);
		}
	}

#ifdef COUNTTIME
	clock_gettime(CLOCK_REALTIME, &endCommTime);
#endif

	printf("Starting host calculation.\n");

#ifdef COUNTTIME
	clock_gettime(CLOCK_REALTIME, &hostStartTime);
#endif
	
	// Run algorithm on host for correctness check
	hostKmeans(NUMIT, numPoints, numCent, dims,
			x, dist, cent,
			mean, bestCent, cent_r);

#ifdef COUNTTIME
	clock_gettime(CLOCK_REALTIME, &hostEndTime);
#endif

	printf("Starting device calculation.\n");

	size_t heap_size;
	cudaDeviceGetLimit(&heap_size, cudaLimitMallocHeapSize);

#ifdef COUNTTIME
	clock_gettime(CLOCK_REALTIME, &startTime);
#endif

	int it;
	for (it = 0; it < NUMIT; it++)
	{
		classifyPoints<<<numBlocks, numThreadsPerBlock>>>(NUMIT, numPoints, numCent, dims,
						device_x, device_dist, device_cent,
						device_mean, device_bestCent, device_cent_r,
						numPointsPerThread,
						device_cent_partial_tots,
						device_cent_tot
						);

		// TODO 
		clearVars<<<numBlocks, numThreadsPerBlock>>>(
						numCent, dims,
						numCentPerThread,
						device_cent_r,
						device_cent_partial_tots,
						device_cent_tot
						);

		accumulateTotals<<<numBlocks, numThreadsPerBlock>>>(NUMIT, numPoints, numCent, dims,
						device_x, device_dist, device_cent,
						device_mean, device_bestCent, device_cent_r,
						numPointsPerThread,
						device_cent_partial_tots,
						device_cent_tot
						);

		reduceTotals<<<numBlocks, numThreadsPerBlock>>>(NUMIT, numPoints, numCent, dims,
						device_x, device_dist, device_cent,
						device_mean, device_bestCent, device_cent_r,
						numCentPerThread,
						device_cent_partial_tots,
						device_cent_tot
						);

		calculateCentroids<<<1, 1>>>(NUMIT, numPoints, numCent, dims,
						device_x, device_dist, device_cent,
						device_mean, device_bestCent, device_cent_r,
						numPointsPerThread,
						device_cent_partial_tots,
						device_cent_tot
						);

	}
	cudaDeviceSynchronize();

	err[0] = cudaGetLastError();
	if (err[0] != cudaSuccess)
	{
		printf("Oh shit, shit happened: %s\n", cudaGetErrorString(err[0]));
		exit(1);
	}

#ifdef COUNTTIME
	clock_gettime(CLOCK_REALTIME, &endTime);
#endif

	printf("End of device calculation.\n");

	/* Pull results back from device */
	int *cudaBestCent = (int *) calloc(numPoints, sizeof(int));
	if (cudaBestCent == NULL)
	{
		printf("Failed to allocate extra space for device's result.\n");
		exit(1);
	}

	data_t *cudaCent = (data_t *) calloc(numCent * dims, sizeof(data_t));
	if (cudaCent == NULL)
	{
		printf("Failed to allocate extra space for device's result.\n");
		exit(1);
	}

#ifdef COUNTTIME
	clock_gettime(CLOCK_REALTIME, &memBackStartTime);
#endif

	err[0] = cudaMemcpy (cudaBestCent, device_bestCent, numPoints * sizeof(int), cudaMemcpyDeviceToHost);
	err[1] = cudaMemcpy (cudaCent, device_cent, numCent * dims * sizeof(int), cudaMemcpyDeviceToHost);
	
	if (err[0] != cudaSuccess || err[1] != cudaSuccess)
	{
		printf("Failed to transfer device's result: %s\n", cudaGetErrorString(err[0]));
		exit(1);
	}

#ifdef COUNTTIME
	clock_gettime(CLOCK_REALTIME, &memBackEndTime);
#endif

	/* Verify device results */
	for (j = 0; j < numPoints; j++)
	{
		if (cudaBestCent[j] != bestCent[j])
		{
			printf("Error: Host and device bestCent results do not match.\n");
			printf("Error at %d\n\tHost has %d\n\tDevice has %d\n", j, bestCent[j], cudaBestCent[j]);
			exit(1);
		}
	}

	for (k = 0; k < numCent; k++)
	{
		int e;
		for (e = 0; e < dims; e++)
		{
			if (cent[k * dims + e] != cudaCent[k * dims + e])
			{
				printf("Error: Host and device cent results do not match.\n");
				printf("Error at centroid %d\n", k);
				exit(1);
			}
		}
	}

	///* write clusters to screen */
	//printf("\nDevice results\n=========\n");

	//for (k=0; k<numCent; k++)
	//{
	//	printf("\nCluster %d\n=========\n",k);
	//	for (j=0; j<numPoints; j++)
	//	{
	//		if (cudaBestCent[j]==k)
	//			printf("point %d\n",j);
	//	}
	//}

#ifdef COUNTTIME

	double hostTime = timeDiff(hostStartTime, hostEndTime);
	double deviceTime = timeDiff(startTime, endTime);

	printf("\nTime Report\n=======\n\n");
	printf("Communication Host->Device time: %f s\n", timeDiff(startCommTime, endCommTime));
	printf("Algorithm Host Computation:      %f s\n", hostTime);
	printf("Algorithm Device Computation:    %f s\n", deviceTime);
	printf("Communication Device->Host time: %f s\n", timeDiff(memBackStartTime, memBackEndTime));
	printf("\nSpeed-up: %f\n", hostTime / deviceTime);
#endif

	printf("\nAll OK.\n");
	exit(0);
}

#ifdef COUNTTIME
/**
* timeDiff
*
* Computes the difference (in ns) between the start and end time
*/
double timeDiff(struct timespec tStart, struct timespec tEnd)
{
	struct timespec diff;

	diff.tv_sec  = tEnd.tv_sec  - tStart.tv_sec  - (tEnd.tv_nsec<tStart.tv_nsec?1:0);
	diff.tv_nsec = tEnd.tv_nsec - tStart.tv_nsec + (tEnd.tv_nsec<tStart.tv_nsec?1000000000:0);

	return ((double) diff.tv_sec) + ((double) diff.tv_nsec)/1e9;
}
#endif

int randState;

/** Predictable pseudo-random function to always generate the same output, regardless of running platform. */
int myRand()
{
	int const a = 1103515245;
	int const c = 12345;
	randState = a * randState + c;
	return (randState >> 16) & 0x7FFF;
}

void mySrand(int seed)
{
	randState = seed;
}
