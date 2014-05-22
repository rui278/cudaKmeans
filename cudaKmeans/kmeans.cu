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

// __global__ void kernel(int NUMIT, int numPoints, int numCent, int dims, data_t ** x, data_t ** dist, data_t ** cent, data_t * mean, int * bestCent, data_t * cent_r);


#ifdef COUNTTIME
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
	data_t **cent;	/* Coordinates of each centroid */
	data_t *mean;
	data_t **dist;	/* Distance between Point P and Centroid C. Indexed as [P][C]. */
	int *bestCent;
    data_t *cent_r;
    
    /*device data*/
    data_t *device_x;		/* Max value: [numPoints * dims]. Layout is [x1 y1 x2 y2 x3 y3 ...] in case dims=2 */
	data_t *device_cent;	/* Max value: [numCent * dims]. Layout is [cx1 cy1 cx2 cy2 cx3 cy3 ...] in case dims=2 */
	data_t *device_mean;	/* Max value: [dims]. */
	data_t *device_dist;	/* Max value: [numPoints * numCent]. Distance between Point P and Centroid C. Indexed as [P * dims + C]. */
	int *device_bestCent;
    data_t *device_cent_r;
    
	int numPoints;	/* Number of points */
	int dims;		/* Dimensions */
	int numCent;	/* Number of centroids */
	int NUMIT = 100;
    
	int range;
	int randSeed;
    
    cudaError_t err[] = {cudaSuccess, cudaSuccess,cudaSuccess,cudaSuccess,cudaSuccess,cudaSuccess,cudaSuccess,cudaSuccess,cudaSuccess}; //9 errors for malloc and memcpy
    
#ifdef COUNTTIME
    struct timespec startTime;
    struct timespec endTime;
    struct timespec startCommTime;
    struct timespec endCommTime;
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
    
	printf("Allocating memory\n");
    
	/* allocate memory */
	x=(data_t *)calloc(numPoints * dims,sizeof(data_t *));
	if (x == NULL) printf("Error on calloc.\n");

	cent=(data_t **)calloc(numCent,sizeof(data_t *));
	if (cent == NULL) printf("Error on calloc.\n");
	for (j=0; j<numCent; j++)
	{
		cent[j]=(data_t *)calloc(dims,sizeof(data_t));
		if (cent[j] == NULL) printf("Error on calloc.\n");
	}

	mean=(data_t *)calloc(dims,sizeof(data_t));

	dist=(data_t **)calloc(numPoints,sizeof(data_t *));
	if (dist == NULL) printf("Error on calloc.\n");

	for (j=0; j<numPoints; j++)
	{
		dist[j]=(data_t *)calloc(numCent,sizeof(data_t));
		if (dist[j] == NULL) printf("Error on calloc.\n");
	}

	bestCent=(int *)calloc(numPoints,sizeof(int));
    
    cent_r=(data_t *)calloc(dims,sizeof(data_t));
    
	printf("Allocated host memory.\n");






	/* Memory allocation on the CUDA device */

	/* Touch the device once to initialize it */
    err[0] = cudaFree(0);
	if (err[0] != cudaSuccess)
	{
		printf("Error: %d\n", cudaGetErrorString(err[0]));
	}

    /*Allocate device memory*/
    
 //   err[0] = cudaMalloc ((void **) &device_x, numPoints * dims * sizeof(data_t));
	//err[1] = cudaMalloc ((void **) &device_cent, numCent * dims * sizeof(data_t));

	//err[2] = cudaMalloc ((void **) &device_mean, dims * sizeof(data_t));
	//err[3] = cudaMalloc ((void **) &device_dist, numPoints * numCent * sizeof(data_t));

	//err[4] = cudaMalloc ((void**) &device_bestCent, numPoints * sizeof(int));
	//err[5] = cudaMalloc ((void**) &device_cent_r, dims * sizeof(data_t));

 //   
 //   for(n = 0; n < 6; n++) {
 //       if(err[n] != cudaSuccess){
 //           printf("Error allocating memory on device (error code %s). Exiting.", cudaGetErrorString(err[n]));
 //           exit(0);
 //       }
 //   }

	printf("Allocated memory on device.\n");
    
	/*Generate Test set according to user specification*/
	printf("Generating Test Set\n");
    
    
	//x[Lines][Columns] = x[numPoints][Dimensions]
    
	mySrand(randSeed);
    
	for(n = 0; n < numPoints; n ++)
	{
		for(j = 0; j < dims; j++)
		{
			X_(n,j) = myRand() % range;
			mean[j] += X_(n,j);
            
			if(n < numCent){
				cent[n][j] = myRand() % range;
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
    
////  err[0] = cudaMemcpy (device_x, x, numPoints * sizeof(data_t *));
////    
////	err[1] = cudaMemcpy (device_dist,dist, numPoints * sizeof(data_t *));
//
//	err[0] = cudaMemcpy(device_x, x, numPoints * dims * sizeof(data_t), cudaMemcpyHostToDevice);
//    
//	err[2] = cudaMemcpy (device_mean,mean, dims * sizeof(data_t), cudaMemcpyHostToDevice);
//    
//    err[5] = cudaMemcpy (device_bestCent,bestCent, numPoints * sizeof(int), cudaMemcpyHostToDevice);
//    
//    err[8] = cudaMemcpy (device_cent_r, cent_r,dims * sizeof(data_t), cudaMemcpyHostToDevice);
//    
//	//for (j=0; j<numPoints; j++)
//	//{
//		// err[3] = cudaMemcpy (device_x[j],x[j], dims * sizeof(data_t), cudaMemcpyHostToDevice);
//        //err[6] = cudaMemcpy (device_dist[j] , dist[j],numCent * sizeof(data_t), cudaMemcpyHostToDevice);
//	//}
//    
//    
////	err[4] = cudaMemcpy (device_cent,cent, numCent * sizeof(data_t *));
//    
//	for (j=0; j<numCent; j++)
//	{
//        //err[7] = cudaMemcpy (device_cent[j],cent[j], dims * sizeof(data_t), cudaMemcpyHostToDevice);
//	}
    
#ifdef COUNTTIME
    clock_gettime(CLOCK_REALTIME, &endCommTime);
#endif
    
    /*Start Computaton*/
	printf("Starting computation on algorithm\n");
    
#ifdef COUNTTIME
    clock_gettime(CLOCK_REALTIME, &startTime);
#endif
    
	// Try to run on host
	kernel(NUMIT, numPoints, numCent, dims,
			x, dist, cent,
			mean, bestCent, cent_r);

    // kernel<<<1, 1>>>(NUMIT, numPoints, numCent, dims, device_x, device_dist, device_cent, device_mean, device_bestCent, device_cent_r);
    
#ifdef COUNTTIME
    clock_gettime(CLOCK_REALTIME, &endTime);
#endif
    
    //kernel();
    
	/* write clusters to screen */
	printf("Results\n=======\n\n");
    
	for (k=0; k<numCent; k++)
	{
		printf("\nCluster %d\n=========\n",k);
		for (j=0; j<numPoints; j++)
		{
			if (bestCent[j]==k) printf("point %d\n",j);
		}
	}
    
#ifdef COUNTTIME
	printf("\nTime Report\n=======\n\n");
    printf("Algorithm Computation: %fs", timeDiff(startTime, endTime));
#endif
	
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

//
//  kernel.c
//  cudaKmeans
//
//  Created by Rui on 22/05/14.
//  Copyright (c) 2014 __Grupo215AAC__. All rights reserved.
//
//
//#include <stdio.h>
//#include <string.h>
//#include <limits.h>
//
//#include "kernel.h"
//
//#include "cuda_runtime.h"
//
//__global__ void kernel(int NUMIT, int numPoints, int numCent, int dims, data_t ** x, data_t ** dist, data_t ** cent, data_t * mean, int * bestCent, data_t * cent_r){
//    
//    
//    int it;
//    int j, k;
//    
//    
//    for (it=0; it<NUMIT; it++)
//    {
//        /* calculate distance matrix and minimum */
//        data_t rMin;
//        int count;
//        
//        count=0;
//        
//        for (j=0; j<numPoints; j++)
//        {
//            rMin=INT_MAX;
//            for (k=0; k < numCent; k++)
//            {
//                int e;
//                dist[j][k]=0;
//                for (e=0; e<dims; e++)
//                {
//                    dist[j][k]+=(x[j][e]-cent[k][e])*(x[j][e]-cent[k][e]);
//                    /* printf("   dist=%f\n",dist[j][k]);*/
//                }
//                /* printf("x[j]=(%f,%f), cent[k]=(%f,%f), dist[%d][%d]=%f\n",x[j][0],x[j][1],cent[k][0],cent[k][1],j,k,dist[j][k]); */
//                if (dist[j][k] < rMin)
//                {
//                    bestCent[j]=k;
//                    rMin=dist[j][k];
//                }
//            }
//            count++;
//        }
//        
//        /* reestimate centroids */
//        for (k=0; k<numCent; k++)
//        {
//            
//            int tot;
//            
//            /* Count number of points belonging to this centroid, and accumulate coordinates */
//            
//            
//            memset(cent_r, 0, sizeof(data_t) * dims);
//            
//            tot=0;
//            for (j=0; j<numPoints; j++)
//            {
//                if (bestCent[j]==k)
//                {
//                    int e;
//                    for (e=0; e<dims; e++)
//                        cent_r[e]+=x[j][e];
//                    tot++;
//                }
//            }
//            
//            /* If centroid has more than 0 points associated (normal), relocate it to mean of its points. */
//            if (tot > 0)
//            {
//                int e;
//                for (e=0; e<dims; e++)
//                    cent[k][e]=cent_r[e]/tot;
//            }
//            /* Else, relocate it to the mean of the other centroids (put it near points) */
//            else
//            {
//                int e;
//                for (e=0; e<dims; e++)
//                    cent[k][e]=mean[e];
//            }
//        }
//    }
//}