#include <stdio.h>
#include <stdlib.h>
#include "hostFE.h"
#include "helper.h"

void hostFE(int filterWidth, float *filter, int imageHeight, int imageWidth,
            float *inputImage, float *outputImage, cl_device_id *device,
            cl_context *context, cl_program *program)
{
    int filterSize = sizeof(float) * filterWidth * filterWidth;
    int inputImageSize = sizeof(float) * imageHeight * imageWidth;

    //  create command queue
    cl_command_queue command_queue = clCreateCommandQueue(*context, *device, 0, NULL);

    // allocate device memory
    cl_mem filterMem = clCreateBuffer(*context, CL_MEM_READ_ONLY, filterSize, NULL, NULL);
    cl_mem inputMem = clCreateBuffer(*context, CL_MEM_READ_ONLY, inputImageSize, NULL, NULL);
    cl_mem outputMem = clCreateBuffer(*context, CL_MEM_WRITE_ONLY, inputImageSize, NULL, NULL);
    
    // Transfer data from host to device
    clEnqueueWriteBuffer(command_queue, filterMem, CL_TRUE, 0, filterSize, filter, 0, NULL, NULL);
    clEnqueueWriteBuffer(command_queue, inputMem, CL_TRUE, 0, inputImageSize, inputImage, 0, NULL, NULL);

    // create kernel function
    cl_kernel kernel = clCreateKernel(*program, "convolution", NULL);

    // set kernel arguments
    clSetKernelArg(kernel, 0, sizeof(cl_int), &filterWidth);
    clSetKernelArg(kernel, 1, sizeof(cl_mem), &filterMem);
    clSetKernelArg(kernel, 2, sizeof(cl_int), &imageHeight);
    clSetKernelArg(kernel, 3, sizeof(cl_int), &imageWidth);
    clSetKernelArg(kernel, 4, sizeof(cl_mem), &inputMem);
    clSetKernelArg(kernel, 5, sizeof(cl_mem), &outputMem);

    // workgroup
    size_t localSize[2] = {25, 25};
    size_t globalSize[2] = {imageWidth, imageHeight};

    // execute kernel
    clEnqueueNDRangeKernel(command_queue, kernel, 2, 0, globalSize, localSize, 0, NULL, NULL);
    // copy data from the device to the host
    clEnqueueReadBuffer(command_queue, outputMem, CL_TRUE, 0, inputImageSize, outputImage, NULL, NULL, NULL);

    // release memory
    clReleaseCommandQueue(command_queue);
    clReleaseMemObject(filterMem);
    clReleaseMemObject(inputMem);
    clReleaseMemObject(outputMem);
    clReleaseKernel(kernel);
}