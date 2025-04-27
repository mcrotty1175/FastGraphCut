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

#define NUM_GPU_STREAMS 4
// Max 640
#define NUM_THREAD_BLOCKS 640
// Max like 4*256 = 1024?
#define THREADS_PER_BLOCK 32
#define MAX_SHARED_MEM (64000/16)
#define MAX_COLS (MAX_SHARED_MEM/2/sizeof(pixel_t))


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
        pixel_t *pixels, uint64_t rows, uint64_t cols, int tile_width,
        weight_t leftW, weight_t upleftW, weight_t upW, weight_t uprightW,
        float *globalBeta
) {
    extern __shared__ pixel_t shared_mem[];

    // image_t img = {.rows = rows, .cols = cols, .array = pixels};
    int id = threadIdx.x;
    // Need overlap on both left & right side
    uint64_t horizontal_tiles = (cols + tile_width - 3) / (tile_width - 2);
    // printf("Horizontal Tiles: %lu\n", horizontal_tiles);

    float beta = 0.0;
    // Row 0 will be done on CPU
    for (uint64_t y = blockIdx.x + 1; y < rows; y+= gridDim.x) {

        for (uint64_t j = 0; j < horizontal_tiles; j++) {
            // Copy in 2 rows of the image into shared memory
            uint64_t rel_col = j * tile_width - 2 * j;
            for (uint64_t i = id; i < tile_width; i += blockDim.x) {
                if (rel_col + i < cols) {
                    shared_mem[i] = pixels[(y-1)*cols + rel_col + i];
                    shared_mem[i+tile_width] = pixels[y*cols + rel_col + i];
                }
            }
            uint64_t start_row = y - 1;
            uint64_t end = min(rel_col + tile_width - 1, cols-1);
            /*
            if (y < 5)
                printf("Brought in: (%lu,%lu) - (%lu, %lu)\n", start_row, rel_col, y, end);
                */

            // Process the two rows
            uint64_t row_index = y * cols + rel_col;
            for (uint64_t x = id; x < tile_width-1; x += blockDim.x)
            {
                uint64_t real_x = j > 0 ? rel_col + x + 1: rel_col + x;
                if (real_x < cols) {
                    // printf("Col %lu\t", real_x);
                    pixel_t *color = &shared_mem[tile_width+x];
                    if (j > 0) color += 1;
                    float diff;
                    if (rel_col + x > 0) // left
                    {
                        // printf("Left\t");
                        // diff = gpu_dot_diff(color, img_at(&img, y, x-1));
                        diff = gpu_dot_diff(color, color-1);
                        beta += diff;
                        leftW[row_index + x] = diff;
                    } else {
                        leftW[row_index + x] = 0;
                    }
                    if (rel_col + x > 0) // upleft
                    {
                        // printf("Up Left\t\t");
                        // diff = gpu_dot_diff(color, img_at(&img, y-1, x-1));
                        diff = gpu_dot_diff(color, color-tile_width-1);
                        beta += diff;
                        upleftW[row_index + x] = diff;
                    } else {
                        upleftW[row_index + x] = 0;
                    }

                    // Up - Always Happens
                    // diff = gpu_dot_diff(color, img_at(&img, y-1, x));
                    // printf("Up\t");
                    diff = gpu_dot_diff(color, color-tile_width);
                    beta += diff;
                    upW[row_index + x] = diff;

                    if (rel_col + x < cols - 1) // upright
                    {
                        //printf("Up Right\n");
                        // diff = gpu_dot_diff(color, img_at(&img, y-1, x+1));
                        diff = gpu_dot_diff(color, color-tile_width+1);
                        beta += diff;
                        uprightW[row_index + x] = diff;
                    } else {
                        uprightW[row_index + x] = 0;
                    }
                }
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


    uint64_t tile_width = min(img->cols, MAX_COLS);
    uint64_t shared_mem_size = 2 * tile_width * sizeof(pixel_t);

    st = omp_get_wtime();
    // CalcBeta (potentially slower but more work)
    fastCalcBeta<<<NUM_THREAD_BLOCKS,THREADS_PER_BLOCK, shared_mem_size, streams[0]>>>(
            gpuPixels, img->rows, img->cols, tile_width,
            gpuLeftW, gpuUpLeftW, gpuUpW, gpuUpRightW, gpuBeta);
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
