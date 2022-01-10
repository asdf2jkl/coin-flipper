// Copyright (c) 2022 Peter Lafreniere ( 23lafrenip@student.hpts.us ) all rights
// reserved.
//
// Xoshiro128++ algorithm by Sebastiano Vigna and David Blackman (
// http://prng.di.unimi.it/ )

#pragma OPENCL EXTENSION cl_khr_int64_base_atomics : enable

// Note: explicit simd brings no benefit on Nvidia Turing, but I'm leaving it here, 
//  despite the fact that it makes the cpu have to generate four times as many values for the seed.
typedef struct xoshiro128plusplus {
  uint4 s[4];
} xoshiro128plusplus;

static const inline uint4 rng_next(xoshiro128plusplus *restrict x) {
  const uint4 result = rotate(x->s[0] + x->s[3], (uint)7) + x->s[0];

  const uint4 t = x->s[1] << 9;

  x->s[2] ^= x->s[0];
  x->s[3] ^= x->s[1];
  x->s[1] ^= x->s[2];
  x->s[0] ^= x->s[3];

  x->s[2] ^= t;

  x->s[3] = rotate(x->s[3], (uint)11);

  return result;
}

kernel void flipper(const uint count_per_thread,
                    global xoshiro128plusplus *restrict rng,
                    global ulong *restrict output,
                    local ulong *restrict local_output) {
  uint accum = 0;

  for (uint i = 0; i < (count_per_thread / (32 * 4)); i++) {
    uint4 a = popcount(rng_next(&rng[get_global_id(0)]));
    accum += a.x + a.y + a.z + a.w;
  }
  if (count_per_thread % (32 * 4) != 0) {
    uint4 a = popcount(rng_next(&rng[get_global_id(0)]));
    accum += a.x;
    if (count_per_thread % (32 * 3) != 0) {
      accum += a.y;
      if (count_per_thread % (32 * 2) != 0) {
        accum += a.z;
        if (count_per_thread % 32 != 0) {
          accum += a.w;
        }
      }
    }
  }

  accum += popcount(rng_next(&rng[get_global_id(0)]).x &
                    ((-(uint)0) << (count_per_thread % 32)));

  if (get_local_id(0) == 0) {
    *local_output = 0;
  }
  barrier(CLK_LOCAL_MEM_FENCE);

  atomic_add(local_output, (ulong)accum);

  barrier(CLK_LOCAL_MEM_FENCE);
  // TODO: Write each group's result to their own output location, allowing the cpu to do the final reduction
  if (get_local_id(0) == 0) {
    atomic_add(output, *local_output);
  }
}
