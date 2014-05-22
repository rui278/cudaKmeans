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

__global__ void kernel(int NUMIT, int numPoints, int numCent, int dims, data_t ** x, data_t ** dist, data_t ** cent, data_t * mean, int * bestCent, data_t * cent_r);

#endif
