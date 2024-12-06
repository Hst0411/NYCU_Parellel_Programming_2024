#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>
#include "helper.h"
extern "C"{
#include "hostFE.h"
}

__global__ void convolution(int filterWidth, float *filter, int imageHeight, int imageWidth, float *inputImage, float *outputImage) {
    // Iterate over the rows of the source image
    int halffilterSize = filterWidth / 2;
    float sum = 0; // Reset sum for new source pixel
    int thisX = blockIdx.x * blockDim.x + threadIdx.x;  // x index -> imgWidth
    int thisY = blockIdx.y * blockDim.y + threadIdx.y;  // y index -> imgHeight

    // Apply the filter to the neighborhood
    // k: row, l: col
    for (int k = -halffilterSize; k <= halffilterSize; k++)
    {
        for (int l = -halffilterSize; l <= halffilterSize; l++)
        {
            if (thisY + k >= 0 && thisY + k < imageHeight &&
                thisX + l >= 0 && thisX + l < imageWidth)
            {
                sum += inputImage[(thisY + k) * imageWidth + thisX + l] *
                        filter[(k + halffilterSize) * filterWidth +
                                l + halffilterSize];
            }
        }
    }
    outputImage[thisY * imageWidth + thisX] = sum;
}

extern "C"
// Host front-end function that allocates the memory and launches the GPU kernel
void hostFE (int filterWidth, float *filter, int imageHeight, int imageWidth,
            float *inputImage, float *outputImage, cl_device_id *device,
            cl_context *context, cl_program *program)
{
    int filterSize = sizeof(float) * filterWidth * filterWidth;
    int inputImageSize = sizeof(float) * imageHeight * imageWidth;

    // host and device memory
    float *filterDeviceMemory, *inputDeviceMemory, *outputDeviceMemory;
    cudaMalloc(&filterDeviceMemory, filterSize);
    cudaMalloc(&inputDeviceMemory, inputImageSize);
    cudaMalloc(&outputDeviceMemory, inputImageSize);
    
    // copy data from host to device
    cudaMemcpy(filterDeviceMemory, filter, filterSize, cudaMemcpyHostToDevice);
    cudaMemcpy(inputDeviceMemory, inputImage, inputImageSize, cudaMemcpyHostToDevice);

    // workgroup
    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks(imageWidth / threadsPerBlock.x, imageHeight / threadsPerBlock.y);
    convolution<<<numBlocks, threadsPerBlock>>>(filterWidth, filterDeviceMemory, imageHeight,  imageWidth, inputDeviceMemory, outputDeviceMemory);

    cudaMemcpy(outputImage, outputDeviceMemory, inputImageSize, cudaMemcpyDeviceToHost);

    cudaFree(filterDeviceMemory);
    cudaFree(inputDeviceMemory);
    cudaFree(outputDeviceMemory);
}
