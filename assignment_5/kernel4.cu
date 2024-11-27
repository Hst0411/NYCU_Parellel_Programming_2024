#pragma optimize(3, "Ofast", "inline")
#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>

template <int maxIterations>

__global__ void mandelKernel(int* DeviceMemory, int resX, int resY, float lowerX, float lowerY, float dx, float dy){
    // To avoid error caused by the floating number, use the following pseudo code
    //
    // float x = lowerX + thisX * stepX;
    // float y = lowerY + thisY * stepY;
	int thisX = blockIdx.x * blockDim.x + threadIdx.x;
	int thisY = blockIdx.y * blockDim.y + threadIdx.y;

    if (thisX >= resX || thisY >= resY) return;

    float c_re = lowerX + thisX * dx;
	float c_im = lowerY + thisY * dy;
	float z_re = c_re, z_im = c_im;

    int i;
    #pragma unroll
    for (i = 0; i < maxIterations; ++i) {

        if (z_re * z_re + z_im * z_im > 4.f) break;

        float new_re = z_re * z_re - z_im * z_im;
        float new_im = 2.f * z_re * z_im;
        z_re = c_re + new_re;
        z_im = c_im + new_im;
    }
    DeviceMemory[thisX + thisY * resX] = i;
}

__global__ void mandelKernelDefault(int* DeviceMemory, int resX, int resY, float lowerX, float lowerY, float dx, float dy, int maxIterations){
    // To avoid error caused by the floating number, use the following pseudo code
    //
    // float x = lowerX + thisX * stepX;
    // float y = lowerY + thisY * stepY;
    int thisX = blockIdx.x * blockDim.x + threadIdx.x;
	int thisY = blockIdx.y * blockDim.y + threadIdx.y;

    if (thisX >= resX || thisY >= resY) return;

    float c_re = lowerX + thisX * dx;
	float c_im = lowerY + thisY * dy;
	float z_re = c_re, z_im = c_im;

    int i;
    #pragma unroll
    for (i = 0; i < maxIterations; ++i) {
        
        if (z_re * z_re + z_im * z_im > 4.f) break;

        float new_re = z_re * z_re - z_im * z_im;
        float new_im = 2.f * z_re * z_im;
        z_re = c_re + new_re;
        z_im = c_im + new_im;
    }
    DeviceMemory[thisX + thisY * resX] = i;
}

void hostFE (float upperX, float upperY, float lowerX, float lowerY, int* img, int resX, int resY, int maxIterations)
{
	float stepX = (upperX - lowerX) / resX;
    float stepY = (upperY - lowerY) / resY;
    int size = resX * resY * sizeof(int);

    //int *HostMemory = (int *)malloc(size);
    int *DeviceMemory;
    cudaMalloc(&DeviceMemory, size);

    dim3 blockSize(16, 16);
    dim3 numBlocks(resX / blockSize.x, resY / blockSize.y);

    switch (maxIterations) {
        case 256:
            mandelKernel<256><<<numBlocks, blockSize>>>(DeviceMemory, resX, resY, lowerX, lowerY, stepX, stepY);
            break;
        case 1000:
            mandelKernel<1000><<<numBlocks, blockSize>>>(DeviceMemory, resX, resY, lowerX, lowerY, stepX, stepY);
            break;
        case 10000:
            mandelKernel<10000><<<numBlocks, blockSize>>>(DeviceMemory, resX, resY, lowerX, lowerY, stepX, stepY);
            break;
        case 100000:
            mandelKernel<100000><<<numBlocks, blockSize>>>(DeviceMemory, resX, resY, lowerX, lowerY, stepX, stepY);
            break;
        default:
            mandelKernelDefault<<<numBlocks, blockSize>>>(DeviceMemory, resX, resY, lowerX, lowerY, stepX, stepY, maxIterations);
            break;
    }
    cudaMemcpy(img, DeviceMemory, size, cudaMemcpyDeviceToHost);
    //memcpy(img, HostMemory, size);
    
    cudaFree(DeviceMemory);
    //free(HostMemory);
}
