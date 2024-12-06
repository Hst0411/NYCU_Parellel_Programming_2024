__kernel void convolution(int filterWidth, __constant float *filter, int imageHeight, int imageWidth, 
                        __global float *inputImage, __global float *outputImage)
{
    // Iterate over the rows of the source image
    int halffilterSize = filterWidth / 2;
    float sum = 0; // Reset sum for new source pixel
    int ix = get_global_id(0);  // x index -> imgWidth
    int iy = get_global_id(1);  // y index -> imgHeight

    // Apply the filter to the neighborhood
    // k: row, l: col
    for (int k = -halffilterSize; k <= halffilterSize; k++)
    {
        for (int l = -halffilterSize; l <= halffilterSize; l++)
        {
            if (iy + k >= 0 && iy + k < imageHeight &&
                ix + l >= 0 && ix + l < imageWidth)
            {
                sum += inputImage[(iy + k) * imageWidth + ix + l] *
                        filter[(k + halffilterSize) * filterWidth +
                                l + halffilterSize];
            }
        }
    }
    outputImage[iy * imageWidth + ix] = sum;
}
