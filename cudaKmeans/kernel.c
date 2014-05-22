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

__global__ void kernel(int NUMIT, int numPoints, int numCent, int dims, data_t ** x, data_t ** dist, data_t ** cent, data_t * mean, int * bestCent, data_t * cent_r){


    int it;
    int j, k;

    
    for (it=0; it<NUMIT; it++)
    {
        /* calculate distance matrix and minimum */
        data_t rMin;
        int count;
        
        count=0;
        
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
            count++;
        }
        
        /* reestimate centroids */
        for (k=0; k<numCent; k++)
        {
            
            int tot;
            
            /* Count number of points belonging to this centroid, and accumulate coordinates */
            
            
            memset(cent_r, 0, sizeof(data_t) * dims);
            
            tot=0;
            for (j=0; j<numPoints; j++)
            {
                if (bestCent[j]==k)
                {
                    int e;
                    for (e=0; e<dims; e++)
                        cent_r[e]+=x[j][e];
                    tot++;
                }
            }
            
            /* If centroid has more than 0 points associated (normal), relocate it to mean of its points. */
            if (tot > 0)
            {
                int e;
                for (e=0; e<dims; e++)
                    cent[k][e]=cent_r[e]/tot;
            }
            /* Else, relocate it to the mean of the other centroids (put it near points) */
            else
            {
                int e;
                for (e=0; e<dims; e++)
                    cent[k][e]=mean[e];
            }
        }
    }
}