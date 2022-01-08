// Copyright (c) 2022 Peter Lafreniere ( 23lafrenip@student.hpts.us ) all rights
// reserved.
//
// Xoshiro128++ algorithm by Sebastiano Vigna and David Blackman (
// http://prng.di.unimi.it/ )

#pragma OPENCL EXTENSION cl_khr_int64_base_atomics : enable

typedef struct xoshiro128plusplus {
  uint s[4];
} xoshiro128plusplus;

static const uint rng_next(xoshiro128plusplus *restrict x) {
  const uint result = rotate(x->s[0] + x->s[3], (uint)7) + x->s[0];

  const uint t = x->s[1] << 9;

  x->s[2] ^= x->s[0];
  x->s[3] ^= x->s[1];
  x->s[1] ^= x->s[2];
  x->s[0] ^= x->s[3];

  x->s[2] ^= t;

  x->s[3] = rotate(x->s[3], (uint)11);

  return result;
}

kernel void flipper(const uint count_per_thread, global xoshiro128plusplus *rng,
                    global ulong *output, local ulong *local_output) {
  uint accum = 0;

  for (uint i = 0; i < (count_per_thread / 32); i++) {
    accum += popcount(rng_next(&rng[get_global_id(0)]));
  }
  accum += popcount(rng_next(&rng[get_global_id(0)]) &
                    ((-(uint)0) << (count_per_thread % 32)));

  if (get_local_id(0) == 0) {
    *local_output = 0;
  }
  barrier(CLK_LOCAL_MEM_FENCE);

  atomic_add(local_output, (ulong)accum);

  barrier(CLK_LOCAL_MEM_FENCE);
  if (get_local_id(0) == 0) {
    atomic_add(output, *local_output);
  }
}
