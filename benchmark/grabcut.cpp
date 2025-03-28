#include <cstring>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cmath>
#include <iterator>
#include "grabcut.h"
#include "graph.hpp"

#define COMPONENT_COUNT 5

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
    if (gmm != NULL)
        return;

    gmm = (GMM_t *)malloc(sizeof(GMM_t));
    gmm->model = (double *)calloc(modelSize * COMPONENT_COUNT, sizeof(double));

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
}

void calcInverseCovAndDeterm(GMM_t *gmm, int ci, double singularFix)
{
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

// Technically should have a checkMask fn

static void initMaskWithRect(mask_t *mask, rect_t rect, image_t *img)
{
    mask = (mask_t *)malloc(sizeof(mask_t));

    mask->rows = img->rows;
    mask->cols = img->cols;
    mask->array = (MaskVal *)calloc(img->rows * img->cols, sizeof(MaskVal));

    int start_x = rect.x;
    int remaining_width = img->cols - rect.x;
    int width = (rect.width > remaining_width) ? rect.width : remaining_width;
    int start_y = rect.y;
    int remaining_height = img->rows - rect.y;
    int end_y = rect.height < remaining_height ? rect.height : remaining_height;
    end_y += start_y;

    for (int r = start_y; r < end_y; r++)
    {
        int row_index = r * img->cols;
        memset(&(mask->array[row_index + start_x]), GC_PR_FGD, width);
    }
}

void kmeans(pixel_t *pixels, int num_pixels, int k, int num_clusters, int max_iters, int *labels)
{
    labels = (int *)malloc(num_pixels * sizeof(int));
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
            for (int j = 0; j < k; ++j)
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
            pixel_t estimate_center = {(uint8_t)centroids[i].r, (uint8_t)centroids[i].g, (uint8_t)centroids[i].b};
            float shift = distance_squared(estimate_center, updated);
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

    int *bgdLabels = NULL, *fgdLabels = NULL;
    std::vector<pixel_t> bgdSamples, fgdSamples;
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

    // replace with kmeans kernel - maybe use streams?
    {
        int num_clusters = COMPONENT_COUNT;
        num_clusters = std::min(num_clusters, (int)bgdSamples.size());
        kmeans((pixel_t *)&bgdSamples[0], (int)bgdSamples.size(), k, num_clusters, kMeansItCount,
               bgdLabels);
    }

    {
        int num_clusters = COMPONENT_COUNT;
        num_clusters = std::min(num_clusters, (int)fgdSamples.size());
        kmeans((pixel_t *)&fgdSamples[0], (int)fgdSamples.size(), k, num_clusters, kMeansItCount,
               fgdLabels);
    }

    // can use streams? one for fg and one for bg
    initLearning(bgdGMM);
    for (int i = 0; i < (int)bgdSamples.size(); i++)
        addSample(bgdGMM, bgdLabels[i], bgdSamples[i]);
    endLearning(bgdGMM);

    initLearning(fgdGMM);
    for (int i = 0; i < (int)fgdSamples.size(); i++)
        addSample(fgdGMM, fgdLabels[i], fgdSamples[i]);
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
static void learnGMMs(image_t *img, mask_t *mask, int *compIdxs, GMM_t *bgdGMM, GMM_t *fgdGMM)
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
                    if (m == GC_BGD || m == GC_PR_BGD)
                        addSample(bgdGMM, ci, *img_at(img, r, c));
                    else
                        addSample(fgdGMM, ci, *img_at(img, r, c));
                }
            }
        }
    }
    endLearning(bgdGMM);
    endLearning(fgdGMM);
}

// static void constructGCGraph(image_t *img, mask_t *mask, GMM_t *bgdGMM, GMM_t *fgdGMM, double lambda,
//                              weight_t leftW, weight_t upleftW, weight_t upW, weight_t uprightW,
//                              GCGraph<double> &graph)
// {
//     int vtxCount = img->cols * img->rows,
//         edgeCount = 2 * (4 * img->cols * img->rows - 3 * (img->cols + img->rows) + 2);
//     graph.create(vtxCount, edgeCount);
//     for (int r = 0; r < img->rows; r++)
//     {
//         int row_index = r * img->cols;
//         for (int c = 0; c < img->cols; c++)
//         {
//             // add node
//             int vtxIdx = graph.addVtx();
//             pixel_t color = *img_at(img, r, c);

//             // set t-weights
//             double fromSource, toSink;
//             MaskVal m = mask_at(mask, r, c);
//             if (m == GC_PR_BGD || m == GC_PR_FGD)
//             {
//                 fromSource = -log(doSomething(bgdGMM, color));
//                 toSink = -log(doSomething(fgdGMM, color));
//             }
//             else if (m == GC_BGD)
//             {
//                 fromSource = 0;
//                 toSink = lambda;
//             }
//             else // GC_FGD
//             {
//                 fromSource = lambda;
//                 toSink = 0;
//             }
//             graph.addTermWeights(vtxIdx, fromSource, toSink);

//             // set n-weights
//             if (c > 0)
//             {
//                 double w = leftW[row_index + c];
//                 graph.addEdges(vtxIdx, vtxIdx - 1, w, w);
//             }
//             if (c > 0 && r > 0)
//             {
//                 double w = upleftW[row_index + c];
//                 graph.addEdges(vtxIdx, vtxIdx - img->cols - 1, w, w);
//             }
//             if (r > 0)
//             {
//                 double w = upW[row_index + c];
//                 graph.addEdges(vtxIdx, vtxIdx - img->cols, w, w);
//             }
//             if (c < img->cols - 1 && r > 0)
//             {
//                 double w = uprightW[row_index + c];
//                 graph.addEdges(vtxIdx, vtxIdx - img->cols + 1, w, w);
//             }
//         }
//     }
// }

// static void estimateSegmentation(GCGraph<double> &graph, mask_t *mask)
// {
//     graph.maxFlow();
//     for (int r = 0; r < mask->rows; r++)
//     {
//         for (int c = 0; c < mask->cols; c++)
//         {
//             MaskVal m = mask_at(mask, r, c);
//             if (m == GC_PR_BGD || m == GC_PR_FGD)
//             {
//                 if (graph.inSourceSegment(r * mask->cols + c /*vertex index*/))
//                     mask_set(mask, r, c, GC_PR_FGD);
//                 else
//                     mask_set(mask, r, c, GC_PR_BGD);
//             }
//         }
//     }
// }

void grabCut(image_t *img, rect_t rect, int iterCount)
{
    GMM_t *bgdGMM, *fgdGMM;
    mask_t *mask;

    initEmptyGMM(bgdGMM);
    initEmptyGMM(fgdGMM);
    int *compIdxs = (int *)malloc(img->rows * img->cols * sizeof(int));

    initMaskWithRect(mask, rect, img);
    initGMMs(img, mask, bgdGMM, fgdGMM);

    if (iterCount <= 0)
        return;

    const double gamma = 50;
    const double lambda = 9 * gamma;

    // how to copy image over to the gpu
    const double beta = calcBeta(img);

    double *leftW, *upleftW, *upW, *uprightW;
    calcNWeights(img, leftW, upleftW, upW, uprightW, beta, gamma);

    for (int i = 0; i < iterCount; i++)
    {
        GCGraph<double> graph;
        assignGMMsComponents(img, mask, bgdGMM, fgdGMM, compIdxs);
        learnGMMs(img, mask, compIdxs, bgdGMM, fgdGMM);
        // constructGCGraph(img, mask, bgdGMM, fgdGMM, lambda, leftW, upleftW, upW, uprightW, graph);
        // estimateSegmentation(graph, mask);
    }
}

int main()
{
    // read image
    //  read bounding box
    //  call grabcut
    grabCut(NULL, {0, 0, 0, 0}, 0);
    return 0;
}