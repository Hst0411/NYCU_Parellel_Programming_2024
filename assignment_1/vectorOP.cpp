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

void clampedExpVector(float *values, int *exponents, float *output, int N)
{
  //
  // PP STUDENTS TODO: Implement your vectorized version of
  // clampedExpSerial() here.
  //
  // Your solution should work for any value of
  // N and VECTOR_WIDTH, not just when VECTOR_WIDTH divides N

  __pp_vec_float x, result;
  __pp_vec_int exponent;
  __pp_mask maskAll, n_mask;
  __pp_vec_int zero = _pp_vset_int(0);
  __pp_vec_int one = _pp_vset_int(1);
  __pp_vec_float max_result = _pp_vset_float(9.999999f);

  for (int i = 0; i < N; i += VECTOR_WIDTH)
  {
    maskAll = _pp_init_ones();
    // ex: x = values[i]
    _pp_vload_float(x, values + i, maskAll);
    // ex: y = exponents[i]
    _pp_vload_int(exponent, exponents + i, maskAll);
    // If exponent == 0
    _pp_veq_int(n_mask, exponent, zero, maskAll)
    // n_mask = 0
    n_mask = _pp_mask_not(n_mask); // ex: exp = [ 2, 2, 0], n_mask = 110

    // result = 1;
    _pp_vset_float(result, 1, maskAll);
    // while (count > 0), n_mask = 110
    while(_pp_cntbits(n_mask) > 0){
      // result *= x
      _pp_vmult_float(result, result, x, n_mask);
      // count--, if n_mask == 1
      _pp_vsub_int(exponent, exponent, one, n_mask);
      // if exponent > 0, n_mask--
      _pp_vgt_int(n_mask, exponent, one, n_mask);
    }
    // if (result > 9.999999f)
    _pp_vgt_int(result, max_result);
    

  }
}

// returns the sum of all elements in values
// You can assume N is a multiple of VECTOR_WIDTH
// You can assume VECTOR_WIDTH is a power of 2
float arraySumVector(float *values, int N)
{

  //
  // PP STUDENTS TODO: Implement your vectorized version of arraySumSerial here
  //

  for (int i = 0; i < N; i += VECTOR_WIDTH)
  {
    
  }

  return 0.0;
}
