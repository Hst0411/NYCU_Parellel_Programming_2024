__kernel void convolution(int filterWidth, __constant float *filter, int imageHeight, int imageWidth,
                          __global float *inputImage, __global float *outputImage, int localSize) 
{
    int i = get_global_id(0);
    int j = get_global_id(1);

    int halffilterSize = filterWidth / 2;
    float sum = 0.0f;

    int start_k = (-halffilterSize + j >= 0) ? -halffilterSize : -j;
    int end_k = (halffilterSize + j < imageHeight) ? halffilterSize : imageHeight - 1 - j;
    int start_l = (-halffilterSize + i >= 0) ? -halffilterSize : -i;
    int end_l = (halffilterSize + i < imageWidth) ? halffilterSize : imageWidth - 1 - i;

    int temp1 = (j + start_k) * imageWidth + i;
    int temp2 = (start_k + halffilterSize) * filterWidth + halffilterSize;

    for (int k = start_k; k <= end_k; ++k)
    {
        for (int l = start_l; l <= end_l; ++l)
        {
            sum += inputImage[temp1 + l] * filter[temp2 + l];
        }

        temp1 += imageWidth;
        temp2 += filterWidth;
    }

    outputImage[j * imageWidth + i] = sum;
}