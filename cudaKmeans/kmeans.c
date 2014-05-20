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
/*    Artur Gon�alves                                    */
/*    Daniel Filipe                                      */
/*                                                       */
/*    for: Advanced Computer Architectures Class         */
/*        Electrical and Computer Engineering Department */
/*        Instituto Superior T�cnico                     */
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

//#include "cuda_runtime.h"

/* Type of numbers to use. int or float. */
#define data_t int

#ifdef COUNTTIME
double timeDiff(struct timespec tStart, struct timespec tEnd);
#endif

void mySrand(int seed);
int myRand();

int main(int argc, char *argv[])
{
    int i;
	int j;
	int k;
	int n;
    int e;
	data_t **x;		/* x[Lines][Columns]. Max values: x[numPoints][Dimensions] */
	data_t **cent;	/* Coordinates of each centroid */
	data_t *mean;	
	data_t **dist;	
	int *bestCent;
	int *totals = 0;
    data_t **cent_r;
	int it;			/* Current iteration */
	int numPoints;	/* Number of points */
	int dims;		/* Dimensions */
	int numCent;	/* Number of centroids */
	int NUMIT = 100;
    
	int range;
	int randSeed;
    
#ifdef COUNTTIME
    struct timespec startTime;
    struct timespec endTime;
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
	x=(data_t **)calloc(numPoints,sizeof(data_t *));
    
	dist=(data_t **)calloc(numPoints,sizeof(data_t *));
    
	mean=(data_t *)calloc(dims,sizeof(data_t));
    
    
	for (j=0; j<numPoints; j++)
	{
		x[j]=(data_t *)calloc(dims,sizeof(data_t));
	}
    
    
	cent=(data_t **)calloc(numCent,sizeof(data_t *));
    cent_r=(data_t **)calloc(numCent, sizeof(data_t *));
    
	bestCent=(int *)calloc(numPoints,sizeof(int));
    totals = (int*)calloc(numCent, sizeof(int));
	
    
    for (j=0; j<numPoints; j++)
	{
		dist[j]=(data_t *)calloc(numCent,sizeof(data_t));
	}
    
	for (j=0; j<numCent; j++)
	{
		cent[j]=(data_t *)calloc(dims,sizeof(data_t));
        cent_r[j]=(data_t *)calloc(dims,sizeof(data_t));

	}
    
	/*Generates Test set according to user specification*/
	printf("Generating Test Set\n");
    
    
	//x[Lines][Columns] = x[numPoints][Dimensions]
    
	mySrand(randSeed);
    
	for(n = 0; n < numPoints; n ++)
	{
		for(j = 0; j < dims; j++)
		{
			x[n][j] = myRand() % range;
			mean[j] += x[n][j];
            
			if(n < numCent){
				cent[n][j] = myRand() % range;
			}
            
		}
	}
    
	for(n = 0; n < dims; n++)
	{
		mean[n] = mean[n]/numPoints;
	}
    
    
	printf("Starting computation on algorithm\n");
    
#ifdef COUNTTIME
    clock_gettime(CLOCK_REALTIME, &startTime);
#endif
    
    for(j = 0; j < numCent; j++)
    {
        memset(cent_r[j], 0, sizeof(data_t) * dims);
    }
    
	for (it=0; it<NUMIT; it++)
	{
		/* calculate distance matrix and minimum */
		data_t rMin;

		for (j=0; j<numPoints; j++)
		{
			rMin=INT_MAX;
			for (k=0; k < numCent; k++)
			{
				int e;
				
                dist[j][k]=0;
				
                for (e=0; e<dims; e++)
				{
					dist[j][k]+=(x[j][e]-cent[k][e])*(x[j][e]-cent[k][e]);
					/* printf("   dist=%f\n",dist[j][k]);*/
				}
				/* printf("x[j]=(%f,%f), cent[k]=(%f,%f), dist[%d][%d]=%f\n",x[j][0],x[j][1],cent[k][0],cent[k][1],j,k,dist[j][k]); */
				if (dist[j][k] < rMin)
				{
					bestCent[j]=k;
					rMin=dist[j][k];
				}
			}
            
            for(e = 0; e < dims; e ++)
            {
                int bestC = bestCent[j];
                
                cent_r[bestC][e] += x[j][e];
            
                totals[bestC]++;
            }
		}
        
        for(i = 0; i < numCent; i ++)
        {
            if(totals[i] != 0)
            {
                for(e = 0; e < dims; e ++)
                {
                    cent[i][e] = cent_r[i][e]/totals[i];
                }
            }
            else
            {
                for(e = 0; e < dims; e ++)
                {
                    cent[i][e] = mean[e];
                }

            }
        }
        
        
		/* reestimate centroids */
//		for (k=0; k<numCent; k++)
//		{
//			
//			int tot;
//            
//			/* Count number of points belonging to this centroid, and accumulate coordinates */
//            
//            
//			memset(cent_r, 0, sizeof(data_t) * dims);
//            
//			tot=0;
//			for (j=0; j<numPoints; j++)
//			{
//				if (bestCent[j]==k)
//				{
//					int e;
//					for (e=0; e<dims; e++)
//						cent_r[e]+=x[j][e];
//					tot++;
//				}
//			}
//            
//			/* If centroid has more than 0 points associated (normal), relocate it to mean of its points. */
//			if (tot > 0)
//			{
//				int e;
//				for (e=0; e<dims; e++)
//					cent[k][e]=cent_r[e]/tot;
//			}
//			/* Else, relocate it to the mean of the other centroids (put it near points) */
//			else
//			{
//				int e;
//				for (e=0; e<dims; e++)
//					cent[k][e]=mean[e];
//			}
//		}
	}
#ifdef COUNTTIME
    clock_gettime(CLOCK_REALTIME, &endTime);
#endif
    
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