#include <stdio.h>
#include <stdlib.h>
#include "hostFE.h"
#include "helper.h"

void hostFE(int filterWidth, float *filter, int imageHeight, int imageWidth,
            float *inputImage, float *outputImage, cl_device_id *device,
            cl_context *context, cl_program *program)
{
    int filter_size = sizeof(float) * filterWidth * filterWidth;
    int image_size = sizeof(float) * imageHeight * imageWidth;
    
    cl_command_queue thequeue = clCreateCommandQueue(*context, *device, 0, NULL);
    cl_kernel kernel = clCreateKernel(*program, "convolution", NULL);

    cl_mem filter_mem = clCreateBuffer(*context, CL_MEM_READ_ONLY, filter_size, NULL, NULL);
    cl_mem inputImage_mem = clCreateBuffer(*context, CL_MEM_READ_ONLY, image_size, NULL, NULL);
    cl_mem outputImage_mem = clCreateBuffer(*context, CL_MEM_WRITE_ONLY, image_size, NULL, NULL);

    // Write data to the buffers
    clEnqueueWriteBuffer(thequeue, filter_mem, CL_TRUE, 0, filter_size, filter, 0, NULL, NULL);
    clEnqueueWriteBuffer(thequeue, inputImage_mem, CL_TRUE, 0, image_size, inputImage, 0, NULL, NULL);
    
    int localSize = 8;

    clSetKernelArg(kernel, 0, sizeof(cl_int), &filterWidth);
    clSetKernelArg(kernel, 1, sizeof(cl_mem), &filter_mem);
    clSetKernelArg(kernel, 2, sizeof(cl_int), &imageHeight);
    clSetKernelArg(kernel, 3, sizeof(cl_int), &imageWidth);
    clSetKernelArg(kernel, 4, sizeof(cl_mem), &inputImage_mem);
    clSetKernelArg(kernel, 5, sizeof(cl_mem), &outputImage_mem);
    clSetKernelArg(kernel, 6, sizeof(cl_int), &localSize);

    size_t global_size[2] = {imageWidth, imageHeight};
    size_t local_size[2] = {localSize, localSize};

    clEnqueueNDRangeKernel(thequeue, kernel, 2, 0, global_size, local_size, 0, NULL, NULL);
    clFinish(thequeue);
    clEnqueueReadBuffer(thequeue, outputImage_mem, CL_TRUE, 0, image_size, outputImage, NULL, NULL, NULL);

    clReleaseCommandQueue(thequeue);
    clReleaseKernel(kernel);
    clReleaseMemObject(filter_mem);
    clReleaseMemObject(inputImage_mem);
    clReleaseMemObject(outputImage_mem);
}
