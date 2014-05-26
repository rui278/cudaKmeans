//
//  hostKernel.h
//  cudaKmeans
//
//  Created by Artur on 25/05/14.
//  Copyright (c) 2014 __Grupo215AAC__. All rights reserved.
//

#ifndef cudaKmeans_hostKmeans_h
#define cudaKmeans_hostKmeans_h

#include "macros.h"

/* Executes K-Means algorithm on the CPU (host) */
void hostKmeans(int NUMIT, int numPoints, int numCent, int dims,
				data_t * x, data_t * dist, data_t * cent, data_t * mean, int * bestCent, data_t * cent_r);

#endif
