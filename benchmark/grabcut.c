#include <string.h>
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <math.h>
#include "grabcut.h"

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
void addSample(GMM_t *gmm, int ci, const pixel_t color);
void endLearning(GMM_t *gmm);
const int whichComponent(GMM_t *gmm, const pixel_t color);

void calcInverseCovAndDeterm(GMM_t *gmm, int ci, double singularFix);

void initEmptyGMM(GMM_t *gmm)
{
    const int modelSize = 3 /*mean*/ + 9 /*covariance*/ + 1 /*component weight*/;
    if (gmm != NULL)
        return;

    gmm = malloc(sizeof(GMM_t));
    gmm->model = calloc(modelSize * COMPONENT_COUNT, sizeof(double));

    gmm->coefs = gmm->model;
    gmm->mean = gmm->coefs + COMPONENT_COUNT;
    gmm->cov = gmm->mean + 3 * COMPONENT_COUNT;

    // Pretty sure this doesn't do anything for a new array
    for (int ci = 0; ci < COMPONENT_COUNT; ci++)
        calcInverseCovAndDeterm(gmm, ci, 0.0);

    gmm->totalSampleCount = 0;
}

const double getComponent(GMM_t *gmm, int ci, const pixel_t color)
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

const int whichComponent(GMM_t *gmm, const pixel_t color)
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

void addSample(GMM_t *gmm, int ci, const pixel_t color)
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

void calcInverseCovAndDeterm(GMM_t *gmm, int ci, const double singularFix)
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

static double calcBeta(const image_t *img)
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

static void calcNWeights(const image_t *img, weight_t leftW, weight_t upleftW, weight_t upW, weight_t uprightW, double beta, double gamma)
{
    const double gammaDivSqrt2 = gamma / sqrt(2.0);
    uint64_t num_pixels = img->rows * img->cols;
    leftW = calloc(num_pixels, sizeof(double));
    upleftW = calloc(num_pixels, sizeof(double));
    upW = calloc(num_pixels, sizeof(double));
    uprightW = calloc(num_pixels, sizeof(double));

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

static void initMaskWithRect(mask_t mask, rect_t rect, image_t *img)
{
    mask = calloc(img->rows * img->cols, sizeof(MaskVal));

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
        memset(&mask[row_index + start_x], GC_PR_FGD, width);
    }
}

/*
  Initialize GMM background and foreground models using kmeans algorithm.
*/
static void initGMMs(const Mat &img, const Mat &mask, GMM &bgdGMM, GMM &fgdGMM)
{
    const int kMeansItCount = 10;
    const int kMeansType = KMEANS_PP_CENTERS;

    Mat bgdLabels, fgdLabels;
    std::vector<Vec3f> bgdSamples, fgdSamples;
    Point p;
    for (p.y = 0; p.y < img.rows; p.y++)
    {
        for (p.x = 0; p.x < img.cols; p.x++)
        {
            if (mask.at<uchar>(p) == GC_BGD || mask.at<uchar>(p) == GC_PR_BGD)
                bgdSamples.push_back((Vec3f)img.at<Vec3b>(p));
            else // GC_FGD | GC_PR_FGD
                fgdSamples.push_back((Vec3f)img.at<Vec3b>(p));
        }
    }
    CV_Assert(!bgdSamples.empty() && !fgdSamples.empty());
    {
        Mat _bgdSamples((int)bgdSamples.size(), 3, CV_32FC1, &bgdSamples[0][0]);
        int num_clusters = GMM::componentsCount;
        num_clusters = std::min(num_clusters, (int)bgdSamples.size());
        kmeans(_bgdSamples, num_clusters, bgdLabels,
               TermCriteria(TermCriteria::MAX_ITER, kMeansItCount, 0.0), 0, kMeansType);
    }
    {
        Mat _fgdSamples((int)fgdSamples.size(), 3, CV_32FC1, &fgdSamples[0][0]);
        int num_clusters = GMM::componentsCount;
        num_clusters = std::min(num_clusters, (int)fgdSamples.size());
        kmeans(_fgdSamples, num_clusters, fgdLabels,
               TermCriteria(TermCriteria::MAX_ITER, kMeansItCount, 0.0), 0, kMeansType);
    }

    bgdGMM.initLearning();
    for (int i = 0; i < (int)bgdSamples.size(); i++)
        bgdGMM.addSample(bgdLabels.at<int>(i, 0), bgdSamples[i]);
    bgdGMM.endLearning();

    fgdGMM.initLearning();
    for (int i = 0; i < (int)fgdSamples.size(); i++)
        fgdGMM.addSample(fgdLabels.at<int>(i, 0), fgdSamples[i]);
    fgdGMM.endLearning();
}
