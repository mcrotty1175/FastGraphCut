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
#define THREADS_PER_BLOCK 512


int cpu_dot_diff(pixel_t *a, pixel_t *b) {
    int red = (int)(a->r) - (int)(b->r);
    int green = (int)(a->g) - (int)(b->g);
    int blue = (int)(a->b) - (int)(b->b);
    return (red * red) + (green * green) + (blue * blue);
}

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
                                       gamma * exp(-beta * cpu_dot_diff(color, img_at(img, y, x - 1)))
                                               : 0;
            upleftW[row_index + x] = (x - 1 >= 0 && y - 1 >= 0) ? // upleft
                                         gammaDivSqrt2 * exp(-beta * cpu_dot_diff(color, img_at(img, y - 1, x - 1)))
                                                                : 0;
            upW[row_index + x] = (y - 1 > 0) ? // up
                                     gamma * exp(-beta * cpu_dot_diff(color, img_at(img, y - 1, x)))
                                             : 0;
            uprightW[row_index + x] = (x + 1 < img->cols && y - 1 >= 0) ? // upright
                                          gammaDivSqrt2 * exp(-beta * cpu_dot_diff(color, img_at(img, y - 1, x + 1)))
                                                                        : 0;
        }
    }
}


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
            shared_mem[i] = pixels[(y-1)*cols + i];
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
                diff = gpu_dot_diff(color, color-1);
                beta += diff;
                leftW[row_index + x] = diff;
            }
            if (y > 0 && x > 0) // upleft
            {
                // diff = gpu_dot_diff(color, img_at(&img, y-1, x-1));
                diff = gpu_dot_diff(color, color-cols-1);
                beta += diff;
                upleftW[row_index + x] = diff;
            }
            if (y > 0) // up
            {
                // diff = gpu_dot_diff(color, img_at(&img, y-1, x));
                diff = gpu_dot_diff(color, color-cols);
                beta += diff;
                upW[row_index + x] = diff;
            }
            if (y > 0 && x < cols - 1) // upright
            {
                // diff = gpu_dot_diff(color, img_at(&img, y-1, x+1));
                diff = gpu_dot_diff(color, color-cols+1);
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
    int t = blockDim.x * gridDim.x;

    uint64_t pixels_per_thread = (size + t - 1) / t;
    uint64_t start = id * pixels_per_thread;
    uint64_t count = min(pixels_per_thread, size-start);

    uint64_t iters = (count + 7) / 8;
    uint64_t i = start;
    switch (count % 8) {
        case 0: do {    w[i] = gamma * exp(-beta * w[i]); i++;
        case 7:         w[i] = gamma * exp(-beta * w[i]); i++;
        case 6:         w[i] = gamma * exp(-beta * w[i]); i++;
        case 5:         w[i] = gamma * exp(-beta * w[i]); i++;
        case 4:         w[i] = gamma * exp(-beta * w[i]); i++;
        case 3:         w[i] = gamma * exp(-beta * w[i]); i++;
        case 2:         w[i] = gamma * exp(-beta * w[i]); i++;
        case 1:         w[i] = gamma * exp(-beta * w[i]); i++;
                } while (--iters > 0);
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
    fastCalcBeta<<<320,THREADS_PER_BLOCK, 2 * img->cols * sizeof(pixel_t), streams[0]>>>(gpuPixels,
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

    cout << "Beta: " << beta << endl;
    et = omp_get_wtime();
    cout<< "GPU calcBeta ran for " <<(et-st)<< " seconds" <<endl;

    cudaMemcpyAsync(gpuLeftW, leftW, img->cols * sizeof(double), cudaMemcpyHostToDevice, streams[0]);
    cudaMemsetAsync(gpuUpLeftW, 0, img->cols * sizeof(float), streams[1]);
    cudaMemsetAsync(gpuUpW, 0, img->cols * sizeof(float), streams[2]);
    cudaMemsetAsync(gpuUpRightW, 0, img->cols * sizeof(float), streams[3]);


    // CalcNWeights (definitely faster)
    st = omp_get_wtime();
    double gammaDivSqrt2 = gamma / sqrt(2.0);

    fastCalcWeights<<<320,THREADS_PER_BLOCK,0,streams[0]>>>(gpuLeftW, beta, gamma, num_pixels);
    fastCalcWeights<<<320,THREADS_PER_BLOCK,0,streams[1]>>>(gpuUpLeftW, beta, gammaDivSqrt2, num_pixels);
    fastCalcWeights<<<320,THREADS_PER_BLOCK,0,streams[2]>>>(gpuUpW, beta, gamma, num_pixels);
    fastCalcWeights<<<320,THREADS_PER_BLOCK,0,streams[3]>>>(gpuUpRightW, beta, gammaDivSqrt2, num_pixels);

    cudaDeviceSynchronize();

    et = omp_get_wtime();
    cout<< "GPU calcNWeights ran for " <<(et-st)<< " seconds" <<endl;

    cudaMemcpyAsync(leftW, gpuLeftW, num_pixels * sizeof(double), cudaMemcpyDeviceToHost, streams[0]);
    cudaMemcpyAsync(upleftW, gpuUpLeftW, num_pixels * sizeof(double), cudaMemcpyDeviceToHost, streams[1]);
    cudaMemcpyAsync(upW, gpuUpW, num_pixels * sizeof(double), cudaMemcpyDeviceToHost, streams[2]);
    cudaMemcpyAsync(uprightW, gpuUpRightW, num_pixels * sizeof(double), cudaMemcpyDeviceToHost, streams[3]);

    cudaDeviceSynchronize();

    cudaFree(gpuLeftW);
    cudaFree(gpuUpLeftW);
    cudaFree(gpuUpW);
    cudaFree(gpuUpRightW);

    for (int i = 0; i < NUM_GPU_STREAMS; i++)
        cudaStreamDestroy(streams[i]);
}

int main(int argc, char **argv)
{
    double st, et;

    string file_path = "../dataset/large/flower.jpg";
    if (argc == 2) {
        file_path = argv[1];
    }

    cv::Mat image = cv::imread(file_path);

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

    st = omp_get_wtime();
    const double beta = calcBeta(img);
    et = omp_get_wtime();
    cout<< "Original calcBeta ran for " <<(et-st)<< " seconds" <<endl;

    st = omp_get_wtime();
    calcNWeights(img, leftW, upleftW, upW, uprightW, beta, 50);
    et = omp_get_wtime();
    cout<< "Original calcNWeights ran for " <<(et-st)<< " seconds" <<endl;

    fastCalcConsts(img, leftW, upleftW, upW, uprightW, 50);

    free(leftW);
    free(upleftW);
    free(upW);
    free(uprightW);

    free(img->array);
    free(img);
    return 0;
}
