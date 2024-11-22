#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>

__global__ void mandelKernel(int* DeviceMemory, int resX, float stepX, float stepY, float lowerX, float lowerY, int maxIterations) {
    // To avoid error caused by the floating number, use the following pseudo code
    //
    // float x = lowerX + thisX * stepX;
    // float y = lowerY + thisY * stepY;
    int thisX = blockIdx.x * blockDim.x + threadIdx.x;
    int thisY = blockIdx.y * blockDim.y + threadIdx.y;

    float c_re = lowerX + thisX * stepX;
    float c_im = lowerY + thisY * stepY;
    float z_re = c_re, z_im = c_im;

    int i;
    for (i = 0; i < maxIterations; ++i){

        if (z_re * z_re + z_im * z_im > 4.f) break;

        float new_re = z_re * z_re - z_im * z_im;
        float new_im = 2.f * z_re * z_im;
        z_re = c_re + new_re;
        z_im = c_im + new_im;
    }
    DeviceMemory[thisY * resX + thisX] = i;
}

// Host front-end function that allocates the memory and launches the GPU kernel
void hostFE (float upperX, float upperY, float lowerX, float lowerY, int* img, int resX, int resY, int maxIterations)
{
    float stepX = (upperX - lowerX) / resX;
    float stepY = (upperY - lowerY) / resY;
    int size = resX * resY * sizeof(int);

    //int *HostMemory = (int *)malloc(size);
    int *DeviceMemory;
    cudaMalloc(&DeviceMemory, size);

    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks(resX / threadsPerBlock.x, resY / threadsPerBlock.y);
    mandelKernel<<<numBlocks, threadsPerBlock>>>(DeviceMemory, resX, stepX, stepY, lowerX, lowerY, maxIterations);

    cudaMemcpy(img, DeviceMemory, size, cudaMemcpyDeviceToHost);
    //memcpy(img, HostMemory, size);
    
    cudaFree(DeviceMemory);
    //free(HostMemory);
}
