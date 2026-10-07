#pragma once

#include <ap_fixed.h>

#ifndef DATA_W
#define DATA_W 32
#endif

#ifndef DATA_I
#define DATA_I 8
#endif

#ifndef ACC_W
#define ACC_W 48
#endif

#ifndef ACC_I
#define ACC_I 16
#endif

// The paper's first fixed-point candidate. Both C simulation and synthesized
// hardware use convergent rounding and saturation at every named storage
// boundary.
using dt = ap_fixed<DATA_W, DATA_I, AP_RND_CONV, AP_SAT>;
using acc_t = ap_fixed<ACC_W, ACC_I, AP_RND_CONV, AP_SAT>;
