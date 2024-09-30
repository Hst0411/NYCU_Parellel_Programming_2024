#include "PPintrin.h"


// implementation of absSerial(), but it is vectorized using PP intrinsics
void absVector(float *values, float *output, int N)
{
  __pp_vec_float x;
  __pp_vec_float result;
  __pp_vec_float zero = _pp_vset_float(0.f);
  __pp_mask maskAll, maskIsNegative, maskIsNotNegative;

  //  Note: Take a careful look at this loop indexing.  This example
  //  code is not guaranteed to work when (N % VECTOR_WIDTH) != 0.
  //  Why is that the case?
  for (int i = 0; i < N; i += VECTOR_WIDTH)
  {

    // All ones
    maskAll = _pp_init_ones();

    // All zeros
    maskIsNegative = _pp_init_ones(0);

    // Load vector of values from contiguous memory addresses
    _pp_vload_float(x, values + i, maskAll); // x = values[i];

    // Set mask according to predicate
    _pp_vlt_float(maskIsNegative, x, zero, maskAll); // if (x < 0) { 標記 values < 0 的 index
						     // ex: values = [2, 3, -5, 6, -7]
						     //     maskIsNegative = 00101

    // Execute instruction using mask ("if" clause)
    _pp_vsub_float(result, zero, x, maskIsNegative); //   output[i] = -x;

    // Inverse maskIsNegative to generate "else" mask
    maskIsNotNegative = _pp_mask_not(maskIsNegative); // } else { 
						      // ex: maskIsNotNegative = 11010

    // Execute instruction ("else" clause)
    _pp_vload_float(result, values + i, maskIsNotNegative); //   output[i] = x; }

    // Write results back to memory
    _pp_vstore_float(output + i, result, maskAll);
  }
}

void clampedExpVector(float *values, int *exponents, float *output, int N){
    //
    // PP STUDENTS TODO: Implement your vectorized version of
    // clampedExpSerial() here.
    //
    // Your solution should work for any value of
    // N and VECTOR_WIDTH, not just when VECTOR_WIDTH divides N

    __pp_vec_float value, result;
    __pp_vec_int exponent;
    __pp_mask max_mask;
    __pp_mask exponent_mask;
    __pp_vec_float max_result = _pp_vset_float(9.999999f);
    __pp_mask maskAll = _pp_init_ones();
    __pp_vec_int zero = _pp_vset_int(0);
    __pp_vec_int one = _pp_vset_int(1);
    bool over = false;

    int i;
    for (i = 0;; i += VECTOR_WIDTH) {

      if (i + VECTOR_WIDTH > N){
        maskAll = _pp_init_ones(N - i);
        over = true;
      }
      // result = 1
      result = _pp_vset_float(1.0f);

      // x = values[i]
      _pp_vload_float(value, values + i, maskAll);
      // y = exponents[i]
      _pp_vload_int(exponent, exponents + i, maskAll);
      // If exponent == 0, ex: exp = [ 2, 2, 0], n_mask = 110
      _pp_veq_int(exponent_mask, exponent, zero, maskAll);
      exponent_mask = _pp_mask_not(exponent_mask);

      while (_pp_cntbits(exponent_mask) > 0) {
        // result *= x;
        _pp_vmult_float(result, result, value, exponent_mask);
        // count--;
        _pp_vsub_int(exponent, exponent, one, exponent_mask);
        // count > 0
        _pp_vgt_int(exponent_mask, exponent, zero, exponent_mask);
      }

      // result > 9.999999f
      _pp_vgt_float(max_mask, result, max_result, maskAll);
      // result = 9.999999f
      _pp_vmove_float(result, max_result, max_mask);
      _pp_vstore_float(output + i, result, maskAll);
      if (over) break;
    }
}

// returns the sum of all elements in values
// You can assume N is a multiple of VECTOR_WIDTH, N = n*VECTOR_WIDTH
// You can assume VECTOR_WIDTH is a power of 2, VECTOR_WIDTH = 2^n
float arraySumVector(float *values, int N)
{

  //
  // PP STUDENTS TODO: Implement your vectorized version of arraySumSerial here
  //
  
  // N = 8, VECTOR_WIDTH = 4
  // ex: [1, 2, 3, 4, 5, 6, 7, 8]
  float sum = 0;

  __pp_vec_float value, result;
  __pp_mask maskAll = _pp_init_ones();

  // sum = 0
  result = _pp_vset_float(0.f);
  for (int i = 0; i < N; i += VECTOR_WIDTH){
    // value = values[i]
    _pp_vload_float(value, values + i, maskAll);
    // sum += values[i]
    _pp_vadd_float(result, result, value, maskAll);
  }
  // i = 0
  // result = [1, 2, 3, 4]
  // i = 4
  // result = [1, 2, 3, 4] + [5, 6, 7, 8] = [6, 8, 10, 12]


  // VECTOR_WIDTH = 2^k, ex: 4 = 2^2
  // n = 4
  int n = VECTOR_WIDTH;
  while(n > 1){
    _pp_hadd_float(result, result);
    _pp_interleave_float(result, result);
    
    n /= 2;
  }
  // n = 4
  // result = [14, 14, 22, 22]
  // result = [14, 22, 14, 22]
  // n = 2
  // result = [36, 36, 36, 36]
  // result = [36, 36, 36, 36]

  sum = result.value[0];
  return sum;
}
