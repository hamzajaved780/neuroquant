#ifndef PARAMETERS_H_
#define PARAMETERS_H_

#include <complex>
#ifndef __INTELFPGA_COMPILER__
#include "ref/ac_int.h"
#include "ref/ac_fixed.h"
#else
#include "HLS/ac_int.h"
#include "HLS/ac_fixed.h"
#endif
#include "nnet_utils/nnet_dense.h"
//#include "nnet_utils/nnet_conv.h"
//#include "nnet_utils/nnet_conv2d.h"
#include "nnet_utils/nnet_activation.h"
//#include "nnet_utils/nnet_common.h"
//#include "nnet_utils/nnet_batchnorm.h"
//#include "nnet_utils/nnet_pooling.h"
//#include "nnet_utils/nnet_merge.h"

//hls-fpga-machine-learning insert numbers
typedef ac_fixed<16,6> accum_default_t;
typedef ac_fixed<16,6> weight_default_t;
typedef ac_fixed<16,6> bias_default_t;
typedef ac_fixed<16,6> input_t;
typedef ac_fixed<16,6> result_t;
typedef ac_fixed<16,6> beta_default_t;
typedef ac_fixed<16,6> mean_default_t;
typedef ac_fixed<16,6> scale_default_t;

#define N_INPUT_1_1 16
#define N_LAYER_2 64
#define N_LAYER_4 32
#define N_LAYER_6 32
#define N_LAYER_8 5

//hls-fpga-machine-learning insert layer-precision
typedef ac_fixed<16,6> layer1_t;
typedef ac_fixed<16,6> layer2_t;
typedef ac_fixed<16,6> layer3_t;
typedef ac_fixed<16,6> layer4_t;
typedef ac_fixed<16,6> layer5_t;
typedef ac_fixed<16,6> layer6_t;
typedef ac_fixed<16,6> layer7_t;
typedef ac_fixed<16,6> layer8_t;
typedef ac_fixed<16,6> layer9_t;

//hls-fpga-machine-learning insert layer-config
struct config2 : nnet::dense_config {
    static const unsigned n_in = N_INPUT_1_1;
    static const unsigned n_out = N_LAYER_2;
    static const unsigned io_type = nnet::io_parallel;
    static const unsigned reuse_factor = 1;
    static const unsigned n_zeros = 0;
    static const bool store_weights_in_bram = false;
    typedef accum_default_t accum_t;
    typedef bias_default_t bias_t;
    typedef weight_default_t weight_t;
};

struct relu_config3 : nnet::activ_config {
    static const unsigned n_in = N_LAYER_2;
    static const unsigned table_size = 512;
    static const unsigned io_type = nnet::io_parallel;
};

struct config4 : nnet::dense_config {
    static const unsigned n_in = N_LAYER_2;
    static const unsigned n_out = N_LAYER_4;
    static const unsigned io_type = nnet::io_parallel;
    static const unsigned reuse_factor = 1;
    static const unsigned n_zeros = 0;
    static const bool store_weights_in_bram = false;
    typedef accum_default_t accum_t;
    typedef bias_default_t bias_t;
    typedef weight_default_t weight_t;
};

struct relu_config5 : nnet::activ_config {
    static const unsigned n_in = N_LAYER_4;
    static const unsigned table_size = 512;
    static const unsigned io_type = nnet::io_parallel;
};

struct config6 : nnet::dense_config {
    static const unsigned n_in = N_LAYER_4;
    static const unsigned n_out = N_LAYER_6;
    static const unsigned io_type = nnet::io_parallel;
    static const unsigned reuse_factor = 1;
    static const unsigned n_zeros = 0;
    static const bool store_weights_in_bram = false;
    typedef accum_default_t accum_t;
    typedef bias_default_t bias_t;
    typedef weight_default_t weight_t;
};

struct relu_config7 : nnet::activ_config {
    static const unsigned n_in = N_LAYER_6;
    static const unsigned table_size = 512;
    static const unsigned io_type = nnet::io_parallel;
};

struct config8 : nnet::dense_config {
    static const unsigned n_in = N_LAYER_6;
    static const unsigned n_out = N_LAYER_8;
    static const unsigned io_type = nnet::io_parallel;
    static const unsigned reuse_factor = 1;
    static const unsigned n_zeros = 0;
    static const bool store_weights_in_bram = false;
    typedef accum_default_t accum_t;
    typedef bias_default_t bias_t;
    typedef weight_default_t weight_t;
};

struct softmax_config9 : nnet::activ_config {
    static const unsigned n_in = N_LAYER_8;
    static const unsigned table_size = 512;
    static const unsigned io_type = nnet::io_parallel;
};


#endif 
