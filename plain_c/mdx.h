/* MDX-Net (UVR_MDXNET_KARA_2) forward pass in plain C99. */
#ifndef MDX_H
#define MDX_H

#include <stdint.h>

#define MDX_MAGIC 0x4E58444Du /* "MDXN" */
#define MDX_VERSION 1
#define MDX_HEADER_SIZE 64
#define MDX_MAX_SCALES 8
#define MDX_MAX_TFC 4

typedef struct {
    uint32_t dim_c;     /* input/output channels: L.re, L.im, R.re, R.im */
    uint32_t dim_f;     /* frequency bins seen by the model */
    uint32_t dim_t;     /* time frames per chunk */
    uint32_t n_scales;  /* encoder/decoder levels */
    uint32_t growth;    /* channels added per level */
    uint32_t n_tfc;     /* conv3x3 layers per TFC */
    uint32_t bn_factor; /* TDF bottleneck: F -> F / bn_factor */
    float bn_eps;
} MdxConfig;

/* convolution: w is [Cout][Cin][k][k] (ConvTranspose: [Cin][Cout][k][k]), b is [Cout] */
typedef struct { const float *w, *b; } MdxConv;

/* batch norm, each [C] */
typedef struct { const float *scale, *bias, *mean, *var; } MdxBN;

/* TFC-TDF block: n_tfc x (conv3x3 + relu), then two Linear+BN+ReLU along F, residual */
typedef struct {
    MdxConv tfc[MDX_MAX_TFC];
    const float *tdf1; /* [F][F/bn_factor] */
    MdxBN tdf1_bn;
    const float *tdf2; /* [F/bn_factor][F] */
    MdxBN tdf2_bn;
} MdxBlock;

/* field order below == order of tensors in the file == order the forward pass uses them */
typedef struct {
    MdxConv first;                   /* 1x1, dim_c -> growth */
    MdxBlock enc[MDX_MAX_SCALES];
    MdxConv down[MDX_MAX_SCALES];    /* 2x2 stride 2, C -> C + growth */
    MdxBlock mid;
    MdxConv up[MDX_MAX_SCALES];      /* ConvTranspose 2x2 stride 2, C -> C - growth */
    MdxBN up_bn[MDX_MAX_SCALES];
    MdxBlock dec[MDX_MAX_SCALES];
    MdxConv final;                   /* 1x1, growth -> dim_c */
} MdxWeights;

typedef struct {
    MdxConfig config;
    MdxWeights weights;
    float *data;          /* all weights, one allocation; MdxWeights points into it */
    uint64_t n_floats;
} MdxModel;

/* Returns 0 on success; on failure prints the reason to stderr and returns -1. */
int mdx_load(MdxModel *m, const char *path);
void mdx_free(MdxModel *m);

/* ------------------------------------------------------------------------- */
/* kernels
 *
 * Every tensor is a dense row-major float array [C][H][W] (batch 1).
 * y is the output, x the input; y and x never alias unless noted "in place".
 * Formulas are in PLAN.md section 2 and in the docstrings of tools/reference.py.
 */

/* y[o,h,w] = b[o] + sum_i W[o,i] x[i,h,w]                        W [cout][cin][1][1] */
void mdx_conv1x1(float *y, const float *x, const float *W, const float *b, int cin, int cout, int H, int Wd);

/* y[o,h,w] = b[o] + sum_i,ky,kx W[o,i,ky,kx] x[i,h+ky-1,w+kx-1], zero outside   W [cout][cin][3][3] */
void mdx_conv3x3(float *y, const float *x, const float *W, const float *b, int cin, int cout, int H, int Wd);

/* y[o,h,w] = b[o] + sum_i,ky,kx W[o,i,ky,kx] x[i,2h+ky,2w+kx]; x is [cin][H][Wd], y [cout][H/2][Wd/2]   W [cout][cin][2][2] */
void mdx_conv2x2_s2(float *y, const float *x, const float *W, const float *b, int cin, int cout, int H, int Wd);

/* y[o,2h+ky,2w+kx] = b[o] + sum_i W[i,o,ky,kx] x[i,h,w]; x is [cin][H][Wd], y [cout][2H][2Wd]   W [cin][cout][2][2] */
void mdx_convT2x2_s2(float *y, const float *x, const float *W, const float *b, int cin, int cout, int H, int Wd);

/* y[r,j] = sum_f x[r,f] W[f,j]; x is [rows][fin], y [rows][fout]   W [fin][fout]
 * (rows = C*T: the Linear acts on the last axis only) */
void mdx_matmul_lastdim(float *y, const float *x, const float *W, int rows, int fin, int fout);

/* in place: x[c,i] = (x[c,i] - mean[c]) / sqrt(var[c] + eps) * scale[c] + bias[c], i over hw */
void mdx_batchnorm(float *x, const MdxBN *bn, float eps, int C, int hw);

/* in place: x = max(x, 0) */
void mdx_relu(float *x, long n);

/* in place: y += a */
void mdx_add(float *y, const float *a, long n);

/* in place: y *= a */
void mdx_mul(float *y, const float *a, long n);

/* y[c,j,i] = x[c,i,j]; x is [C][H][Wd], y [C][Wd][H] */
void mdx_transpose_last2(float *y, const float *x, int C, int H, int Wd);

#endif
