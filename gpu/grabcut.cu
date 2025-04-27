#include <cstring>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cmath>
#include <iterator>
#include "grabcut.h"
#include "graph.hpp"
#include <opencv2/opencv.hpp>
#include <iostream>
#include <omp.h>
using namespace std;

#define COMPONENT_COUNT 5
#define NUM_GPU_STREAMS 4


int cpu_dot_diff(pixel_t *a, pixel_t *b) {
    int red = (int)(a->r) - (int)(b->r);
    int green = (int)(a->g) - (int)(b->g);
    int blue = (int)(a->b) - (int)(b->b);
    return (red * red) + (green * green) + (blue * blue);
}

/*
static float calcBeta(image_t *img)
{
    float beta = 0.0;
    for (int y = 0; y < img->rows; y++)
    {
        for (int x = 0; x < img->cols; x++)
        {
            pixel_t *color = img_at(img, y, x);
            if (x > 0) // left
            {
                beta += cpu_dot_diff(color, img_at(img, y, x - 1));
            }
            if (y > 0 && x > 0) // upleft
            {
                beta += cpu_dot_diff(color, img_at(img, y - 1, x - 1));
            }
            if (y > 0) // up
            {
                beta += cpu_dot_diff(color, img_at(img, y - 1, x));
            }
            if (y > 0 && x < img->cols - 1) // upright
            {
                beta += cpu_dot_diff(color, img_at(img, y - 1, x + 1));
            }
        }
    }
    cout << "Real Beta Int: " << beta << endl;

    if (beta <= 0.0000001f)
        beta = 0;
    else
        beta = 1.f / (2 * beta / (4 * img->cols * img->rows - 3 * img->cols - 3 * img->rows + 2));

    cout << "Real Beta: " << beta << endl;
    return beta;
}

static void calcNWeights(image_t *img, weight_t leftW, weight_t upleftW, weight_t upW, weight_t uprightW, double beta, double gamma)
{
    double gammaDivSqrt2 = gamma / sqrt(2.0);
    uint64_t num_pixels = img->rows * img->cols;
    leftW = (weight_t)calloc(num_pixels, sizeof(double));
    upleftW = (weight_t)calloc(num_pixels, sizeof(double));
    upW = (weight_t)calloc(num_pixels, sizeof(double));
    uprightW = (weight_t)calloc(num_pixels, sizeof(double));

    for (int y = 0; y < img->rows; y++)
    {
        int row_index = y * img->cols;
        for (int x = 0; x < img->cols; x++)
        {
            pixel_t *color = img_at(img, y, x);
            leftW[row_index + x] = (x - 1 > 0) ? // left
                                       gamma * exp(-beta * dot_diff(color, img_at(img, y, x - 1)))
                                               : 0;
            upleftW[row_index + x] = (x - 1 >= 0 && y - 1 >= 0) ? // upleft
                                         gammaDivSqrt2 * exp(-beta * dot_diff(color, img_at(img, y - 1, x - 1)))
                                                                : 0;
            upW[row_index + x] = (y - 1 > 0) ? // up
                                     gamma * exp(-beta * dot_diff(color, img_at(img, y - 1, x)))
                                             : 0;
            uprightW[row_index + x] = (x + 1 < img->cols && y - 1 >= 0) ? // upright
                                          gammaDivSqrt2 * exp(-beta * dot_diff(color, img_at(img, y - 1, x + 1)))
                                                                        : 0;
        }
    }
}
*/


// Calculate the first row on the CPU because it wouldn't utilize GPU shared
// memory and otherwise the CPU would be sitting idle
float cpuCalcBetaRowZero(pixel_t *pixels, uint64_t cols, weight_t leftW) {
    float beta = 0;
    float diff;
    for (int i = 1; i < cols; i++) {
        diff = cpu_dot_diff(&pixels[i], &pixels[i-1]);
        beta += diff;
        leftW[i] = diff;
    }

    return beta;
}

__device__ float gpu_dot_diff(pixel_t *a, pixel_t *b) {
    int red = (int)(a->r) - (int)(b->r);
    int green = (int)(a->g) - (int)(b->g);
    int blue = (int)(a->b) - (int)(b->b);
    return (float)((red * red) + (green * green) + (blue * blue));
}

__global__ void fastCalcBeta(
        pixel_t *pixels, uint64_t rows, uint64_t cols,
        weight_t leftW, weight_t upleftW, weight_t upW, weight_t uprightW,
        float *globalBeta
) {
    extern __shared__ pixel_t shared_mem[];

    // image_t img = {.rows = rows, .cols = cols, .array = pixels};
    int id = threadIdx.x;

    float beta = 0.0;
    // Row 0 will be done on CPU
    for (int y = blockIdx.x + 1; y < rows; y+= gridDim.x) {

        // Copy in 2 rows of the image into shared memory
        for (int i = id; i < 2 * cols; i += blockDim.x) {
            shared_mem[i] = pixels[(y-1)*cols + i];
        }

        // Process the two rows
        int row_index = y * cols;
        for (int x = id; x < cols; x += blockDim.x)
        {
            pixel_t *color = &shared_mem[cols+x];
            float diff;
            if (x > 0) // left
            {
                // diff = gpu_dot_diff(color, img_at(&img, y, x-1));
                diff = gpu_dot_diff(color, color - 1);
                beta += diff;
                leftW[row_index + x] = diff;
            }
            if (y > 0 && x > 0) // upleft
            {
                // diff = gpu_dot_diff(color, img_at(&img, y-1, x-1));
                diff = gpu_dot_diff(color, &shared_mem[x-1]);
                beta += diff;
                upleftW[row_index + x] = diff;
            }
            if (y > 0) // up
            {
                // diff = gpu_dot_diff(color, img_at(&img, y-1, x));
                diff = gpu_dot_diff(color, &shared_mem[x]);
                beta += diff;
                upW[row_index + x] = diff;
            }
            if (y > 0 && x < cols - 1) // upright
            {
                // diff = gpu_dot_diff(color, img_at(&img, y-1, x+1));
                diff = gpu_dot_diff(color, &shared_mem[x+1]);
                beta += diff;
                uprightW[row_index + x] = diff;
            }
        }
    }

    // __reduce_add_sync is undefined
    for (int i=blockDim.x / 2; i >= 1; i/=2)
        beta += __shfl_down_sync(0xffffffff, beta, i);

    if (id == 0) atomicAdd(globalBeta, beta);
}

__global__ void fastCalcWeights(weight_t w, double beta, double gamma, uint64_t size)
{
    int id = threadIdx.x + blockDim.x * blockIdx.x;
    for (int i = id; i < size; i += blockDim.x * gridDim.x) {
        w[i] = gamma * exp(-beta * w[i]);
    }
}

void fastCalcConsts(image_t *img, weight_t leftW, weight_t upleftW, weight_t upW, weight_t uprightW, double gamma)
{
    double st, et;

    weight_t gpuLeftW, gpuUpLeftW, gpuUpW, gpuUpRightW;
    pixel_t *gpuPixels;
    float *gpuBeta, beta;
    cudaStream_t streams[NUM_GPU_STREAMS];

    uint64_t num_pixels = img->rows * img->cols;

    cudaError_t err;
    err = cudaMalloc(&gpuPixels, num_pixels * sizeof(pixel_t));
    if (err != cudaSuccess){
      cout<<"Pixel Memory not allocated"<<endl;
      exit(-1);
    }
    err = cudaMalloc(&gpuLeftW, num_pixels * sizeof(double));
    if (err != cudaSuccess){
      cout<<"Left Memory not allocated"<<endl;
      exit(-1);
    }
    err = cudaMalloc(&gpuUpLeftW, num_pixels * sizeof(double));
    if (err != cudaSuccess){
      cout<<"UpLeft Memory not allocated"<<endl;
      exit(-1);
    }
    err = cudaMalloc(&gpuUpW, num_pixels * sizeof(double));
    if (err != cudaSuccess){
      cout<<"Up Memory not allocated"<<endl;
      exit(-1);
    }
    err = cudaMalloc(&gpuUpRightW, num_pixels * sizeof(double));
    if (err != cudaSuccess){
      cout<<"UpRight Memory not allocated"<<endl;
      exit(-1);
    }
    err = cudaMalloc(&gpuBeta, sizeof(float));
    if (err != cudaSuccess){
      cout<<"UpRight Memory not allocated"<<endl;
      exit(-1);
    }

    for (int i = 0; i < NUM_GPU_STREAMS; i++)
        cudaStreamCreateWithFlags(&streams[i], cudaStreamNonBlocking);

    cudaMemcpy(gpuPixels, img->array, num_pixels * sizeof(pixel_t), cudaMemcpyHostToDevice);
    cudaMemset(gpuBeta, 0, sizeof(float));

    st = omp_get_wtime();

    // CalcBeta (potentially slower but more work)
    fastCalcBeta<<<320,256, 2 * img->cols * sizeof(pixel_t), streams[0]>>>(gpuPixels,
            img->rows, img->cols, gpuLeftW, gpuUpLeftW, gpuUpW, gpuUpRightW, gpuBeta);
    cudaMemcpyAsync(&beta, gpuBeta, sizeof(int), cudaMemcpyDeviceToHost, streams[0]);

    double tmpBeta = cpuCalcBetaRowZero(img->array, img->cols, leftW);
    cudaDeviceSynchronize();

    beta += tmpBeta;
    cout << "Int Beta: " << beta << endl;

    if (beta <= 0.0000001f)
        beta = 0;
    else
        beta = 1.f / (2 * beta / (4 * img->cols * img->rows - 3 * img->cols - 3 * img->rows + 2));


    cudaMemcpyAsync(gpuLeftW, leftW, img->cols * sizeof(double), cudaMemcpyHostToDevice, streams[0]);
    cudaMemsetAsync(gpuUpLeftW, 0, img->cols * sizeof(float), streams[1]);
    cudaMemsetAsync(gpuUpW, 0, img->cols * sizeof(float), streams[2]);
    cudaMemsetAsync(gpuUpRightW, 0, img->cols * sizeof(float), streams[3]);

    cout << "Beta: " << beta << endl;
    et = omp_get_wtime();
    cout<< "GPU calcBeta ran for " <<(et-st)<< " seconds" <<endl;

    // CalcNWeights (definitely faster)
    st = omp_get_wtime();
    double gammaDivSqrt2 = gamma / sqrt(2.0);
    fastCalcWeights<<<80,256,0,streams[0]>>>(gpuLeftW, beta, gamma, num_pixels);
    fastCalcWeights<<<80,256,0,streams[1]>>>(gpuUpLeftW, beta, gammaDivSqrt2, num_pixels);
    fastCalcWeights<<<80,256,0,streams[2]>>>(gpuUpW, beta, gamma, num_pixels);
    fastCalcWeights<<<80,256,0,streams[3]>>>(gpuUpRightW, beta, gammaDivSqrt2, num_pixels);

    cudaMemcpyAsync(leftW, gpuLeftW, num_pixels * sizeof(double), cudaMemcpyDeviceToHost, streams[0]);
    cudaMemcpyAsync(upleftW, gpuUpLeftW, num_pixels * sizeof(double), cudaMemcpyDeviceToHost, streams[1]);
    cudaMemcpyAsync(upW, gpuUpW, num_pixels * sizeof(double), cudaMemcpyDeviceToHost, streams[2]);
    cudaMemcpyAsync(uprightW, gpuUpRightW, num_pixels * sizeof(double), cudaMemcpyDeviceToHost, streams[3]);

    cudaDeviceSynchronize();
    et = omp_get_wtime();
    cout<< "GPU calcNWeights ran for " <<(et-st)<< " seconds" <<endl;

    cudaFree(gpuLeftW);
    cudaFree(gpuUpLeftW);
    cudaFree(gpuUpW);
    cudaFree(gpuUpRightW);

    for (int i = 0; i < NUM_GPU_STREAMS; i++)
        cudaStreamDestroy(streams[i]);
}

int main()
{
    cv::Mat image = cv::imread("../dataset/large/flower.jpg");

    if (image.empty())
    {
        std::cerr << "Image not loaded!" << std::endl;
        return -1;
    }

    std::cout << "Loaded Image" << std::endl;

    image_t *img = (image_t *)malloc(sizeof(image_t));
    img->rows = image.rows;
    img->cols = image.cols;

    std::cout << "image dimensions: " << img->rows << " " << img->cols << std::endl;
    img->array = (pixel_t *)malloc(img->rows * img->cols * sizeof(pixel_t));
    for (int r = 0; r < img->rows; r++)
    {
        for (int c = 0; c < img->cols; c++)
        {
            cv::Vec3b color = image.at<cv::Vec3b>(r, c);
            img->array[r * img->cols + c].r = color[2];
            img->array[r * img->cols + c].g = color[1];
            img->array[r * img->cols + c].b = color[0];
        }
    }


    uint64_t num_pixels = img->rows * img->cols;
    double *leftW, *upleftW, *upW, *uprightW;

    leftW = (weight_t)malloc(num_pixels * sizeof(double));
    upleftW = (weight_t)malloc(num_pixels *  sizeof(double));
    upW = (weight_t)malloc(num_pixels * sizeof(double));
    uprightW = (weight_t)malloc(num_pixels * sizeof(double));

    fastCalcConsts(img, leftW, upleftW, upW, uprightW, 50);

    free(leftW);
    free(upleftW);
    free(upW);
    free(uprightW);

    free(img->array);
    free(img);
    return 0;
}
