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

#endif
