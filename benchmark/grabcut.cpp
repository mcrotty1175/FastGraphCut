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
#include <chrono>
#include <filesystem>

#define COMPONENT_COUNT 5
using namespace std;

typedef struct
{
    double *model;

    double *coefs;
    double *mean;
    double *cov;

    double inverseCovs[COMPONENT_COUNT][3][3];
    double covDeterms[COMPONENT_COUNT];

    double sums[COMPONENT_COUNT][3];
    double prods[COMPONENT_COUNT][3][3];
    int sampleCounts[COMPONENT_COUNT];
    int totalSampleCount;
} GMM_t;

void initLearning(GMM_t *gmm);
void addSample(GMM_t *gmm, int ci, pixel_t color);
void endLearning(GMM_t *gmm);
int whichComponent(GMM_t *gmm, pixel_t color);

void calcInverseCovAndDeterm(GMM_t *gmm, int ci, double singularFix);

void initEmptyGMM(GMM_t *gmm)
{
    int modelSize = 3 /*mean*/ + 9 /*covariance*/ + 1 /*component weight*/;
    if (gmm == NULL)
        return;

    // gmm = (GMM_t *)malloc(sizeof(GMM_t));
    gmm->model = (double *)calloc(modelSize * COMPONENT_COUNT, sizeof(double));
    if (gmm->model == NULL) {
        std::cerr << "Memory allocation failed!" << std::endl;
        return; //DO WE NEED THIS?
    }

    gmm->coefs = gmm->model;
    gmm->mean = gmm->coefs + COMPONENT_COUNT;
    gmm->cov = gmm->mean + 3 * COMPONENT_COUNT;

    // Pretty sure this doesn't do anything for a new array
    for (int ci = 0; ci < COMPONENT_COUNT; ci++)
        calcInverseCovAndDeterm(gmm, ci, 0.0);

    gmm->totalSampleCount = 0;
}

double getComponent(GMM_t *gmm, int ci, pixel_t color)
{
    double result = 0;
    if (gmm->coefs[ci] > 0)
    {
        double *m = gmm->mean + 3 * ci;
        double diff[3] = {color.r - m[0], color.g - m[1], color.b - m[2]};
        double mult = diff[0] * (diff[0] * gmm->inverseCovs[ci][0][0] + diff[1] * gmm->inverseCovs[ci][1][0] + diff[2] * gmm->inverseCovs[ci][2][0]) + diff[1] * (diff[0] * gmm->inverseCovs[ci][0][1] + diff[1] * gmm->inverseCovs[ci][1][1] + diff[2] * gmm->inverseCovs[ci][2][1]) + diff[2] * (diff[0] * gmm->inverseCovs[ci][0][2] + diff[1] * gmm->inverseCovs[ci][1][2] + diff[2] * gmm->inverseCovs[ci][2][2]);
        result = 1.0f / sqrt(gmm->covDeterms[ci]) * exp(-0.5f * mult);
    }
    return result;
}

double doSomething(GMM_t *gmm, pixel_t color)
{
    double res = 0;
    for (int ci = 0; ci < COMPONENT_COUNT; ci++)
        res += gmm->coefs[ci] * getComponent(gmm, ci, color);
    return res;
}

int whichComponent(GMM_t *gmm, pixel_t color)
{
    int k = 0;
    double max = 0;

    for (int ci = 0; ci < COMPONENT_COUNT; ci++)
    {
        double p = getComponent(gmm, ci, color);
        if (p > max)
        {
            k = ci;
            max = p;
        }
    }

    return k;
}

void initLearning(GMM_t *gmm)
{
    if (gmm == NULL)
        return;

    for (int ci = 0; ci < COMPONENT_COUNT; ci++)
    {
        gmm->sums[ci][0] = gmm->sums[ci][1] = gmm->sums[ci][2] = 0;
        gmm->prods[ci][0][0] = gmm->prods[ci][0][1] = gmm->prods[ci][0][2] = 0;
        gmm->prods[ci][1][0] = gmm->prods[ci][1][1] = gmm->prods[ci][1][2] = 0;
        gmm->prods[ci][2][0] = gmm->prods[ci][2][1] = gmm->prods[ci][2][2] = 0;
        gmm->sampleCounts[ci] = 0;
    }
    gmm->totalSampleCount = 0;
}

void addSample(GMM_t *gmm, int ci, pixel_t color)
{
    if (gmm == NULL)
    {
        cout << "gmm is null in addsample\n";
        return;
    }

    if (ci < 0 || ci >= COMPONENT_COUNT) {
        std::cerr << "Invalid component index in addSample: " << ci << std::endl;
        return;
    }

    gmm->sums[ci][0] += color.r;
    gmm->sums[ci][1] += color.g;
    gmm->sums[ci][2] += color.b;
    gmm->prods[ci][0][0] += color.r * color.r;
    gmm->prods[ci][0][1] += color.r * color.g;
    gmm->prods[ci][0][2] += color.r * color.b;
    gmm->prods[ci][1][0] += color.g * color.r;
    gmm->prods[ci][1][1] += color.g * color.g;
    gmm->prods[ci][1][2] += color.g * color.b;
    gmm->prods[ci][2][0] += color.b * color.r;
    gmm->prods[ci][2][1] += color.b * color.g;
    gmm->prods[ci][2][2] += color.b * color.b;
    gmm->sampleCounts[ci]++;
    gmm->totalSampleCount++;
}

void endLearning(GMM_t *gmm)
{
    if (gmm == NULL)
        return;
    for (int ci = 0; ci < COMPONENT_COUNT; ci++)
    {
        int n = gmm->sampleCounts[ci];
        if (n == 0)
            gmm->coefs[ci] = 0;
        else
        {
            double inv_n = 1.0 / n;
            gmm->coefs[ci] = (double)n / gmm->totalSampleCount;

            double *m = gmm->mean + 3 * ci;
            m[0] = gmm->sums[ci][0] * inv_n;
            m[1] = gmm->sums[ci][1] * inv_n;
            m[2] = gmm->sums[ci][2] * inv_n;

            double *c = gmm->cov + 9 * ci;
            c[0] = gmm->prods[ci][0][0] * inv_n - m[0] * m[0];
            c[1] = gmm->prods[ci][0][1] * inv_n - m[0] * m[1];
            c[2] = gmm->prods[ci][0][2] * inv_n - m[0] * m[2];
            c[3] = gmm->prods[ci][1][0] * inv_n - m[1] * m[0];
            c[4] = gmm->prods[ci][1][1] * inv_n - m[1] * m[1];
            c[5] = gmm->prods[ci][1][2] * inv_n - m[1] * m[2];
            c[6] = gmm->prods[ci][2][0] * inv_n - m[2] * m[0];
            c[7] = gmm->prods[ci][2][1] * inv_n - m[2] * m[1];
            c[8] = gmm->prods[ci][2][2] * inv_n - m[2] * m[2];

            calcInverseCovAndDeterm(gmm, ci, 0.01);
        }
    }
    // Print GMM means
    // std::cout << "GMM Means:" << std::endl;
    for (int ci = 0; ci < COMPONENT_COUNT; ci++)
    {
        double *m = gmm->mean + 3 * ci;
        // std::cout << "Component " << ci << ": (" << m[0] << ", " << m[1] << ", " << m[2] << ")" << std::endl;
    }

    // Print GMM covariance matrices
    // std::cout << "GMM Covariance Matrices:" << std::endl;
    for (int ci = 0; ci < COMPONENT_COUNT; ci++)
    {
        double *c = gmm->cov + 9 * ci;
        // std::cout << "Component " << ci << ":" << std::endl;
        // std::cout << "[" << c[0] << ", " << c[1] << ", " << c[2] << "]" << std::endl;
        // std::cout << "[" << c[3] << ", " << c[4] << ", " << c[5] << "]" << std::endl;
        // std::cout << "[" << c[6] << ", " << c[7] << ", " << c[8] << "]" << std::endl;
    }

}

void calcInverseCovAndDeterm(GMM_t *gmm, int ci, double singularFix)
{
    if (gmm == NULL)
        return;

    if (gmm->coefs[ci] > 0)
    {
        double *c = gmm->cov + 9 * ci;
        double dtrm = c[0] * (c[4] * c[8] - c[5] * c[7]) - c[1] * (c[3] * c[8] - c[5] * c[6]) + c[2] * (c[3] * c[7] - c[4] * c[6]);
        if (dtrm <= 1e-6 && singularFix > 0)
        {
            // Adds the white noise to avoid singular covariance matrix.
            c[0] += singularFix;
            c[4] += singularFix;
            c[8] += singularFix;
            dtrm = c[0] * (c[4] * c[8] - c[5] * c[7]) - c[1] * (c[3] * c[8] - c[5] * c[6]) + c[2] * (c[3] * c[7] - c[4] * c[6]);
        }
        gmm->covDeterms[ci] = dtrm;

        double inv_dtrm = 1.0 / dtrm;
        gmm->inverseCovs[ci][0][0] = (c[4] * c[8] - c[5] * c[7]) * inv_dtrm;
        gmm->inverseCovs[ci][1][0] = -(c[3] * c[8] - c[5] * c[6]) * inv_dtrm;
        gmm->inverseCovs[ci][2][0] = (c[3] * c[7] - c[4] * c[6]) * inv_dtrm;
        gmm->inverseCovs[ci][0][1] = -(c[1] * c[8] - c[2] * c[7]) * inv_dtrm;
        gmm->inverseCovs[ci][1][1] = (c[0] * c[8] - c[2] * c[6]) * inv_dtrm;
        gmm->inverseCovs[ci][2][1] = -(c[0] * c[7] - c[1] * c[6]) * inv_dtrm;
        gmm->inverseCovs[ci][0][2] = (c[1] * c[5] - c[2] * c[4]) * inv_dtrm;
        gmm->inverseCovs[ci][1][2] = -(c[0] * c[5] - c[2] * c[3]) * inv_dtrm;
        gmm->inverseCovs[ci][2][2] = (c[0] * c[4] - c[1] * c[3]) * inv_dtrm;
    }
}

static double calcBeta(image_t *img)
{
    double beta = 0;
    for (int y = 0; y < img->rows; y++)
    {
        for (int x = 0; x < img->cols; x++)
        {
            pixel_t *color = img_at(img, y, x);
            if (x > 0) // left
            {
                beta += dot_diff(color, img_at(img, y, x - 1));
            }
            if (y > 0 && x > 0) // upleft
            {
                beta += dot_diff(color, img_at(img, y - 1, x - 1));
            }
            if (y > 0) // up
            {
                beta += dot_diff(color, img_at(img, y - 1, x));
            }
            if (y > 0 && x < img->cols - 1) // upright
            {
                beta += dot_diff(color, img_at(img, y - 1, x + 1));
            }
        }
    }

    if (beta <= 0.0000001f)
        beta = 0;
    else
        beta = 1.f / (2 * beta / (4 * img->cols * img->rows - 3 * img->cols - 3 * img->rows + 2));

    return beta;
}

static void calcNWeights(image_t *img, double *leftW, double *upleftW, double *upW, double *uprightW, double beta, double gamma)
{
    double gammaDivSqrt2 = gamma / sqrt(2.0);
    uint64_t num_pixels = img->rows * img->cols;

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

// Technically should have a checkMask fn

static void initMaskWithRect(mask_t *mask, rect_t rect, image_t *img)
{
    mask->rows = img->rows;
    mask->cols = img->cols;
    mask->array = (MaskVal *)calloc(img->rows * img->cols, sizeof(MaskVal));

    int start_x = rect.x;
    int start_y = rect.y;

    int remaining_width = img->cols - start_x;
    int width = (rect.width < remaining_width) ? rect.width : remaining_width;

    int remaining_height = img->rows -start_y;
    int end_y = rect.height < remaining_height ? rect.height : remaining_height;
    end_y += start_y;
    int margin = 15;
    for (int r = start_y; r < end_y; r++)
    {
        for (int c = start_x; c < start_x + width; c++)
        {
            // if (r > start_y + margin && r < end_y - margin && c > start_x + margin && c < start_x + width - margin)
            //     mask->array[r * img->cols + c] = GC_FGD;
            // else
            //     mask->array[r * img->cols + c] = GC_PR_FGD;

            mask->array[r * img->cols + c] = GC_PR_FGD;
        }
    }
}

void kmeans(pixel_t *pixels, int num_pixels, int k, int num_clusters, int max_iters, int *labels)
{
    // labels = (int *)malloc(num_pixels * sizeof(int));
    // Allocate centroids
    Centroid *centroids = (Centroid *)malloc(num_clusters * sizeof(Centroid));
    Centroid *new_centroids = (Centroid *)malloc(num_clusters * sizeof(Centroid));
    int *counts = (int *)malloc(num_clusters * sizeof(int));

    // Set initial cluster centers randomly
    for (int i = 0; i < num_clusters; ++i)
    {
        int idx = rand() % num_pixels;
        centroids[i].r = pixels[idx].r;
        centroids[i].g = pixels[idx].g;
        centroids[i].b = pixels[idx].b;
    }

    for (int iter = 0; iter < max_iters; ++iter)
    {
        // Reset accumulators
        for (int i = 0; i < num_clusters; ++i)
        {
            new_centroids[i].r = 0;
            new_centroids[i].g = 0;
            new_centroids[i].b = 0;
            counts[i] = 0;
        }

        // Assign labels based on nearest centroid
        for (int i = 0; i < num_pixels; ++i)
        {
            float min_dist = INFINITY;
            int label = 0;
            for (int j = 0; j < num_clusters; ++j)
            {
                float dist = distance_squared(pixels[i], centroids[j]);
                if (dist < min_dist)
                {
                    min_dist = dist;
                    label = j;
                }
            }
            labels[i] = label;
            new_centroids[label].r += pixels[i].r;
            new_centroids[label].g += pixels[i].g;
            new_centroids[label].b += pixels[i].b;
            counts[label]++;
        }

        // Update centroids
        int converged = 1;
        for (int i = 0; i < num_clusters; ++i)
        {
            if (counts[i] == 0)
                continue; // avoid division by zero

            Centroid updated = {
                new_centroids[i].r / counts[i],
                new_centroids[i].g / counts[i],
                new_centroids[i].b / counts[i]};

            // Check if centroid has changed significantly
            // pixel_t estimate_center = {(uint8_t)centroids[i].r, (uint8_t)centroids[i].g, (uint8_t)centroids[i].b};
            // float shift = distance_squared(estimate_center, updated);

            float shift =
                (centroids[i].r - updated.r) * (centroids[i].r - updated.r) +
                (centroids[i].g - updated.g) * (centroids[i].g - updated.g) +
                (centroids[i].b - updated.b) * (centroids[i].b - updated.b);

            if (shift > 1e-4f)
            {
                converged = 0;
            }

            centroids[i] = updated;
        }

        if (converged)
            break;
    }

    free(centroids);
    free(new_centroids);
    free(counts);
}

/*
  Initialize GMM background and foreground models using kmeans algorithm.
*/
static void initGMMs(image_t *img, mask_t *mask, GMM_t *bgdGMM, GMM_t *fgdGMM)
{
    int kMeansItCount = 10;
    int k = 5;
    // cout << "in init gmms\n";
    std::vector<pixel_t> bgdSamples;
    std::vector<pixel_t> fgdSamples;
    for (int r = 0; r < img->rows; r++)
    {
        for (int c = 0; c < img->cols; c++)
        {
            MaskVal m = mask_at(mask, r, c);
            if (m == GC_BGD || m == GC_PR_BGD)
                bgdSamples.push_back(*img_at(img, r, c));
            else // GC_FGD | GC_PR_FGD
                fgdSamples.push_back(*img_at(img, r, c));
        }
    }
    // cout << "before kmeans\n";
    // cout << "bgd samples size: " << bgdSamples.size() << "\n";
    // cout << "fgd samples size: " << fgdSamples.size() << "\n";

    int *bgdLabels = (int*)malloc(img->rows * img->cols * sizeof(int));
    int *fgdLabels = (int*)malloc(img->rows * img->cols * sizeof(int));
    // cout << "first for loop\n";

    // replace with kmeans kernel - maybe use streams?


    {
        int num_clusters = COMPONENT_COUNT;
        num_clusters = std::min(num_clusters, (int)bgdSamples.size());
        auto start = std::chrono::high_resolution_clock::now();
        kmeans(bgdSamples.data(), bgdSamples.size(), k, num_clusters, kMeansItCount,
               bgdLabels);
        auto end = std::chrono::high_resolution_clock::now();

        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        std::cout << "K-means for background took: " << duration.count() << " microseconds" << std::endl;
    }

    {
        int num_clusters = COMPONENT_COUNT;
        num_clusters = std::min(num_clusters, (int)fgdSamples.size());

        auto start = std::chrono::high_resolution_clock::now();
        kmeans(fgdSamples.data(), fgdSamples.size(), k, num_clusters, kMeansItCount,
               fgdLabels);
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        std::cout << "K-means for foreground took: " << duration.count() << " microseconds" << std::endl;
    }

    // cout << "done with kmeans?\n";

    // can use streams? one for fg and one for bg
    initLearning(bgdGMM);
    for (int i = 0; i < (int)bgdSamples.size(); i++)
    {
        addSample(bgdGMM, bgdLabels[i], bgdSamples[i]);
    }
    // std::cout << "BGD GMM means weights after initGMMs" << std::endl;
    endLearning(bgdGMM);

    initLearning(fgdGMM);
    for (int i = 0; i < (int)fgdSamples.size(); i++)
        addSample(fgdGMM, fgdLabels[i], fgdSamples[i]);
    // std::cout << "FGD GMM means weights after initGMMs" << std::endl;
    endLearning(fgdGMM);
}

static void assignGMMsComponents(image_t *img, mask_t *mask, GMM_t *bgdGMM, GMM_t *fgdGMM, int *compIdxs)
{
    for (int r = 0; r < img->rows; r++)
    {
        int row_index = r * img->cols;
        for (int c = 0; c < img->cols; c++)
        {
            pixel_t color = *img_at(img, r, c);
            MaskVal m = mask_at(mask, r, c);
            compIdxs[row_index + c] = (m == GC_BGD || m == GC_PR_BGD) ? whichComponent(bgdGMM, color) : whichComponent(fgdGMM, color);
        }
    }
}

/*
  Learn GMMs parameters.
*/
static void learnGMMs(image_t *img, mask_t *mask, int *compIdxs, GMM_t *bgdGMM, GMM_t *fgdGMM, int iter)
{
    initLearning(bgdGMM);
    initLearning(fgdGMM);
    for (int ci = 0; ci < COMPONENT_COUNT; ci++)
    {
        for (int r = 0; r < img->rows; r++)
        {
            int row_index = r * img->cols;
            for (int c = 0; c < img->cols; c++)
            {
                if (compIdxs[row_index + c] == ci)
                {
                    MaskVal m = mask_at(mask, r, c);
                    if (iter == 0) {
                        if (m == GC_BGD || m == GC_PR_BGD)
                            addSample(bgdGMM, ci, *img_at(img, r, c));
                        else if (m == GC_FGD || m == GC_PR_FGD)
                            addSample(fgdGMM, ci, *img_at(img, r, c));
                    } else {
                        if (m == GC_BGD || m == GC_PR_BGD)
                            addSample(bgdGMM, ci, *img_at(img, r, c));
                        else
                            addSample(fgdGMM, ci, *img_at(img, r, c));
                    }
                }
            }
        }
    }
    // std::cout << "BGD GMM means weights after learning:" << std::endl;
    endLearning(bgdGMM);
    // std::cout << "FGD GMM means weights after learning:" << std::endl;
    endLearning(fgdGMM);
}

static void constructGCGraph(image_t *img, mask_t *mask, GMM_t *bgdGMM, GMM_t *fgdGMM, double lambda,
                             weight_t leftW, weight_t upleftW, weight_t upW, weight_t uprightW,
                             GCGraph<double> &graph)
{
    if (img == NULL || mask == NULL || bgdGMM == NULL || fgdGMM == NULL)
        return;

    int vtxCount = img->cols * img->rows,
        edgeCount = 2 * (4 * img->cols * img->rows - 3 * (img->cols + img->rows) + 2);

    // cout << "vertex count: " << vtxCount << "\n";
    graph.create(vtxCount, edgeCount);
    // std::cout << "Graph created with " << vtxCount << " vertices and " << edgeCount << " edges." << std::endl;


    // cout << "created graph in construct function\n";
    for (int r = 0; r < img->rows; r++)
    {
        int row_index = r * img->cols;
        for (int c = 0; c < img->cols; c++)
        {
            // add node
            int vtxIdx = graph.addVtx();
            pixel_t color = *img_at(img, r, c);

            // set t-weights
            double fromSource, toSink;
            MaskVal m = mask_at(mask, r, c);
            if (m == GC_PR_BGD || m == GC_PR_FGD)
            {
                fromSource = -log(doSomething(bgdGMM, color) + 1e-6);
                toSink = -log(doSomething(fgdGMM, color) + 1e-6);
            }
            else if (m == GC_BGD)
            {
                fromSource = 0;
                toSink = lambda;
            }
            else // GC_FGD
            {
                fromSource = lambda;
                toSink = 0;
            }
            graph.addTermWeights(vtxIdx, fromSource, toSink);

            // set n-weights
            if (c > 0)
            {
                double w = leftW[row_index + c];
                graph.addEdges(vtxIdx, vtxIdx - 1, w, w);
            }
            if (c > 0 && r > 0)
            {
                double w = upleftW[row_index + c];
                graph.addEdges(vtxIdx, vtxIdx - img->cols - 1, w, w);
            }
            if (r > 0)
            {
                double w = upW[row_index + c];
                graph.addEdges(vtxIdx, vtxIdx - img->cols, w, w);
            }
            if (c < img->cols - 1 && r > 0)
            {
                double w = uprightW[row_index + c];
                graph.addEdges(vtxIdx, vtxIdx - img->cols + 1, w, w);
            }
        }
    }

}

static void estimateSegmentation(GCGraph<double>& graph, mask_t *mask)
{
    int flow = graph.maxFlow();
    // cout << "Max flow: " << flow << "\n";
    for (int r = 0; r < mask->rows; r++)
    {
        for (int c = 0; c < mask->cols; c++)
        {
            MaskVal m = mask_at(mask, r, c);
            if (m == GC_PR_BGD || m == GC_PR_FGD)
            {
                if (graph.inSourceSegment(r * mask->cols + c /*vertex index*/)) {
                    // cout << "mask[" << r << "][" << c << "] = " << m;
                    mask_set(mask, r, c, GC_PR_FGD);
                    // cout << " mask[" << r << "][" << c << "] = GC_PR_FGD\n";
                }
                else {
                    // cout << "mask[" << r << "][" << c << "] = " << m;
                    mask_set(mask, r, c, GC_PR_BGD);
                    // cout << " mask[" << r << "][" << c << "] = GC_PR_BGD\n";
                }

            }
        }
    }
}

void displayImage(image_t *img) {
    cv::Mat displayImg(img->rows, img->cols, CV_8UC3);
    // std::cout << "create display image mat\n";
    for (int r = 0; r < img->rows; r++)
    {
        for (int c = 0; c < img->cols; c++)
        {
            pixel_t *color = img_at(img, r, c);
            displayImg.at<cv::Vec3b>(r, c) = cv::Vec3b(color->b, color->g, color->r);
        }
    }
    cv::imshow("Image", displayImg);
    cv::waitKey(0);
}

void gettingOutput(image_t *img, mask_t *mask, image_t *foreground, image_t *background)
{
    int fg = 0, bg = 0;
    for (int r = 0; r < img->rows; r++)
    {
        for (int c = 0; c < img->cols; c++)
        {
            pixel_t *color = img_at(img, r, c);
            if (mask_at(mask, r, c) == GC_FGD || mask_at(mask, r, c) == GC_PR_FGD)
            {
                foreground->array[r * img->cols + c] = *color;
                background->array[r * img->cols + c] = {0, 0, 0};
                fg++;
            }
            else
            {
                background->array[r * img->cols + c] = *color;
                foreground->array[r * img->cols + c] = {0, 0, 0};
                bg++;
            }
        }
    }
    // std::cout << "Segmentation result: " << fg << " foreground, " << bg << " background pixels." << std::endl;

}


void grabCut(image_t *img, rect_t rect, image_t *foreground, image_t *background, int iterCount)
{
    int num_pixels = img->rows * img->cols;
    // std::cout << "grabCut called\n";

    GMM_t *bgdGMM, *fgdGMM;
    bgdGMM = (GMM_t *)malloc(sizeof(GMM_t));
    fgdGMM = (GMM_t *)malloc(sizeof(GMM_t));
    mask_t *mask = (mask_t *)malloc(sizeof(mask_t));

    initEmptyGMM(bgdGMM);
    initEmptyGMM(fgdGMM);

    // std::cout << "init GMMs\n";
    int *compIdxs = (int *)malloc(num_pixels * sizeof(int));

    initMaskWithRect(mask, rect, img);
    // gettingOutput(img, mask, foreground, background);
    // displayImage(foreground);
    // displayImage(background);
    // cout << "After init mask with rect\n";
    initGMMs(img, mask, bgdGMM, fgdGMM);
    // cout << "init gmms again\n";

    if (iterCount <= 0)
        return;

    const double gamma = 50;
    const double lambda = 9 * gamma;

    // how to copy image over to the gpu
    const double beta = calcBeta(img);
    std::cout << "Beta: " << beta << std::endl;

    double *leftW, *upleftW, *upW, *uprightW;
    leftW = (double*)calloc(num_pixels, sizeof(double));
    upleftW = (double*)calloc(num_pixels, sizeof(double));
    upW = (double*)calloc(num_pixels, sizeof(double));
    uprightW = (double*)calloc(num_pixels, sizeof(double));

    // how to copy image over to the gpu

    auto start = std::chrono::high_resolution_clock::now();
    const double beta = calcBeta(img);
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    std::cout << "calcBeta took: " << duration.count() << " microseconds" << std::endl;

    start = std::chrono::high_resolution_clock::now();
    calcNWeights(img, leftW, upleftW, upW, uprightW, beta, gamma);
    end = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    std::cout << "calcNWeights took: " << duration.count() << " microseconds" << std::endl;


    // std::cout << "Left edge weights sample:" << std::endl;
    // for (int y = 0; y < 5; ++y) {
    //     for (int x = 0; x < 5; ++x) {
    //         std::cout << leftW[x + (img->cols)*y] << " ";
    //     }
    //     std::cout << std::endl;
    // }

    // cout << "After calc nweights\n";
    // std::cout << "Gamma: " << gamma << std::endl;


    // for (int i = 0; i < iterCount; i++) //i< iterCount
    // {
    //     GCGraph<double> graph;
    //     assignGMMsComponents(img, mask, bgdGMM, fgdGMM, compIdxs);
    //     learnGMMs(img, mask, compIdxs, bgdGMM, fgdGMM, i);
    //     constructGCGraph(img, mask, bgdGMM, fgdGMM, lambda, leftW, upleftW, upW, uprightW, graph);
    //     estimateSegmentation(graph, mask);
    // }
    // gettingOutput(img, mask, foreground, background);

    // displayImage(foreground);
    // displayImage(background);
    // cout << "after lop\n";
}

int main(int argc, char **argv)
{
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

    std::cout << "Loaded Image " << file_path << " " << std::endl;

    image_t *img = (image_t *)malloc(sizeof(image_t));
    img->rows = image.rows;
    img->cols = image.cols;

    /*
    image_t *foreground = (image_t *)malloc(sizeof(image_t));
    foreground->rows = image.rows;
    foreground->cols = image.cols;
    foreground->array = (pixel_t *)malloc(img->rows * img->cols * sizeof(pixel_t));

    image_t *background = (image_t *)malloc(sizeof(image_t));
    background->rows = image.rows;
    background->cols = image.cols;
    background->array = (pixel_t *)malloc(img->rows * img->cols * sizeof(pixel_t));
    */

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
    // std::cout << "generated image struct\n";

    // displayImage(img);

    // 24077.jpg 1 1 98 79
    // grabCut(img, {1, 1, 98, 79}, 5);
    // grabCut(img, {21, 12, 104, 40}, foreground, background, 5); //small 86016

    // grabCut(img, {78, 188, 240, 390}, foreground, background, 5); // 78 188 240 390 large 304074

    double *leftW, *upleftW, *upW, *uprightW;
    int num_pixels = img->rows * img->cols;
    leftW = (double*)calloc(num_pixels, sizeof(double));
    upleftW = (double*)calloc(num_pixels, sizeof(double));
    upW = (double*)calloc(num_pixels, sizeof(double));
    uprightW = (double*)calloc(num_pixels, sizeof(double));


    const double beta = calcBeta(img);
    cout << "Beta: " << beta << endl;

    calcNWeights(img, leftW, upleftW, upW, uprightW, beta, 50);

    // grabCut(img, {103, 59, 477, 401}, foreground, background, 5);
    //103 59 477 401 large flower
    // cv::imshow("Loaded Image", img.array);
    // cv::waitKey(0);
    free(img->array);
    free(img);
    return 0;
}
