#![feature(stdsimd)]
#![feature(portable_simd)]

use rand::{prelude::*, rngs::SmallRng};
use std::alloc::dealloc;
use std::hint::unreachable_unchecked;
use std::mem::size_of_val;
use std::ptr::slice_from_raw_parts_mut;
use std::{
    alloc::{alloc, Layout},
    mem::size_of,
};

struct Allocation {
    ptr: *mut u8,
    layout: Layout,
}

struct Flipper {
    name: &'static str,
    function: fn(count: u64) -> u64,
}

const GENERATORS_TO_TEST: [Flipper; 10] = [
    Flipper {
        name: "Naive Loop",
        function: naive_loop,
    },
    Flipper {
        name: "Naive Loop + threadrng:",
        function: naive_loop_threadrng,
    },
    Flipper {
        name: "Naive Loop + threadrng + SmallRng:",
        function: naive_loop_smallrng,
    },
    Flipper {
        name: "Vector buffer count simple (usize):",
        function: fill_buf_count_u64,
    },
    Flipper {
        name: "Vector buffer count + unsafe:",
        function: fill_buf_count_w_unsafe,
    },
    Flipper {
        name: "Vector buffer count noalloc:",
        function: fill_buff_count_noalloc,
    },
    Flipper {
        name: "Vector buffer count noalloc + union:",
        function: fill_buff_count_noalloc_union,
    },
    Flipper {
        name: "Count usize noalloc nobuffer:",
        function: gen_nobuf_count_usize,
    },
    Flipper {
        name: "Count noalloc nobuffer vectored:",
        function: gen_nobuf_count_usize_vectorized,
    }, /*
       Flipper {
           name: "Count noalloc nobuffer avx512:",
           function: vectored_xoshiro::gen_avx512_intrinsics,
       }, */
     Flipper {
         name: "Count noalloc nobuffer u64x4 generic",
         function: vectored_xoshiro::gen_portable_simd,
     },
];

fn main() {
    let coins_flipped = std::env::args()
        .next_back()
        .unwrap()
        .parse::<u64>()
        .unwrap();

    for generator in GENERATORS_TO_TEST {
        println!("{}", generator.name);
        let start = std::time::Instant::now();
        let result = (generator.function)(coins_flipped);
        let duration = start.elapsed();
        println!("  Heads: {}", result);
        println!("  Time elapsed: {} µs\n", duration.as_micros());
    }
}

fn naive_loop(count: u64) -> u64 {
    let mut count_heads = 0;
    for _ in 0..count {
        count_heads += rand::random::<bool>() as u64;
    }
    count_heads
}

fn naive_loop_threadrng(count: u64) -> u64 {
    let mut rng = thread_rng();
    let mut count_heads = 0;
    for _ in 0..count {
        count_heads += rng.gen::<bool>() as u64;
    }
    count_heads
}

fn naive_loop_smallrng(count: u64) -> u64 {
    let mut rng = SmallRng::from_entropy();
    let mut count_heads = 0;
    for _ in 0..count {
        count_heads += rng.gen::<bool>() as u64;
    }
    count_heads
}

fn fill_buf_count_u64(count: u64) -> u64 {
    let buf_len = ((count / 64) + 1).try_into().unwrap();
    let mut buffer: Vec<u64> = Vec::with_capacity(buf_len);
    buffer.resize_with(buf_len, Default::default);
    let mut rng = SmallRng::from_entropy();
    rng.fill(buffer.as_mut_slice());

    let remainder = buffer.pop().unwrap_or_else(|| unreachable!());
    buffer.iter().fold(0, |accumulator, element| {
        accumulator + element.count_ones() as u64
    }) + ((remainder & !(!0 << (count % 64))).count_ones() as u64)
}

fn fill_buf_count_w_unsafe(count: u64) -> u64 {
    if count == 0 {
        return 0;
    }

    let allocation = {
        let layout =
            Layout::from_size_align((count / 8).try_into().unwrap(), std::mem::align_of::<u64>())
                .unwrap_or_else(|_| unsafe { unreachable_unchecked() })
                .pad_to_align();

        Allocation {
            ptr: unsafe { alloc(layout) },
            layout,
        }
    };

    let buffer =
        unsafe { &mut *slice_from_raw_parts_mut(allocation.ptr, (count / 8).try_into().unwrap()) };

    let mut rng = SmallRng::from_entropy();

    rng.fill_bytes(buffer);

    let remainder = rng.gen::<u64>();

    let result = unsafe {
        &*slice_from_raw_parts_mut(allocation.ptr as *mut u64, (count / 64).try_into().unwrap())
    }
    .iter()
    .fold(0, |accumulator, element| {
        accumulator + element.count_ones() as u64
    });

    unsafe { dealloc(allocation.ptr, allocation.layout) };

    result + ((remainder & !(!0 << (count % 64))).count_ones() as u64)
}

fn fill_buff_count_noalloc(count: u64) -> u64 {
    const LEN: u64 = 16; // Make a good size.
    type Buffer = [u64; LEN as usize];

    if count == 0 {
        return 0;
    };

    let mut buffer = [0u64;//unsafe { std::mem::MaybeUninit::<ElementSize>::uninit().assume_init()};
        LEN as usize];
    let mut accumulator = 0;
    let mut rng = SmallRng::from_entropy();

    for _ in 0..(count / (LEN * 64)) {
        rng.fill_bytes(unsafe {
            &mut *slice_from_raw_parts_mut(buffer.as_mut_ptr().cast::<u8>(), LEN as usize)
        });
        accumulator += buffer.iter().fold(0, |accumulator, element| {
            accumulator + element.count_ones() as u64
        })
    }

    let rem_buffer = &mut buffer[..((((count % (64 * LEN)) / 64) + 1) as usize)];
    let remainder = (count % 64) - (64 * LEN);
    rng.fill(rem_buffer);

    accumulator
        + rem_buffer[..(rem_buffer.len() - 1)]
            .iter()
            .fold(0, |accumulator, element| {
                accumulator + element.count_ones() as u64
            })
        + (*rem_buffer
            .last()
            .unwrap_or_else(|| unsafe { unreachable_unchecked() })
            & !(!0 << (remainder % 64)))
            .count_ones() as u64
}

fn fill_buff_count_noalloc_union(count: u64) -> u64 {
    const LEN: u64 = 16; // Make a good size.
    type Buffer = [u64; LEN as usize];

    if count == 0 {
        return 0;
    };

    union Data {
        buffer: Buffer,
        bytes: [u8; LEN as usize * 8],
        init: bool,
    }

    let mut buffer = Data { init: false };
    let mut accumulator = 0;
    let mut rng = SmallRng::from_entropy();

    for _ in 0..(count / (LEN * 64)) {
        unsafe {
            rng.fill_bytes(&mut buffer.bytes);
            accumulator += buffer.buffer.iter().fold(0, |accumulator, element| {
                accumulator + element.count_ones() as u64
            })
        }
    }

    let rem_buffer =
        &mut unsafe { buffer.buffer }[..((((count % (64 * LEN)) / (64)) + 1) as usize)];
    let remainder = (count % 64) - (64 * LEN);
    rng.fill(rem_buffer);

    accumulator
        + rem_buffer[..((rem_buffer.len() - 1) as usize)]
            .iter()
            .fold(0, |accumulator, element| {
                accumulator + element.count_ones() as u64
            })
        + (*rem_buffer
            .last()
            .unwrap_or_else(|| unsafe { unreachable_unchecked() })
            & !(!0 << (remainder % 64)))
            .count_ones() as u64
}

fn gen_nobuf_count_usize(count: u64) -> u64 {
    if count == 0 {
        0
    } else {
        let mut rng = SmallRng::from_entropy();
        let mut count_heads = 0;
        for _ in 0..(count / (size_of::<usize>() * 8) as u64) {
            count_heads += rng.gen::<usize>().count_ones() as u64;
        }
        count_heads
            + ((rng.gen::<usize>() & !(!0 << (count % (size_of::<usize>() * 8) as u64)))
                .count_ones() as u64)
    }
}

fn gen_nobuf_count_usize_vectorized(count: u64) -> u64 {
    let mut thread_rng = thread_rng();
    let mut rng = [
        SmallRng::from_rng(&mut thread_rng).unwrap(),
        SmallRng::from_rng(&mut thread_rng).unwrap(),
        SmallRng::from_rng(&mut thread_rng).unwrap(),
        SmallRng::from_rng(&mut thread_rng).unwrap(),
    ];
    let mut count_heads = 0;
    for _ in 0..(count / (size_of::<usize>() * 8 * 4) as u64) {
        count_heads += rng[0].gen::<usize>().count_ones() as u64;
        count_heads += rng[1].gen::<usize>().count_ones() as u64;
        count_heads += rng[2].gen::<usize>().count_ones() as u64;
        count_heads += rng[3].gen::<usize>().count_ones() as u64;
    }
    for _ in 0..((count % (size_of::<usize>() * 8 * 4) as u64) / (size_of::<usize>() * 8) as u64) {
        count_heads += rng[0].gen::<usize>().count_ones() as u64;
    }
    count_heads
        + ((rng[0].gen::<usize>() & !(!0 << (count % (size_of::<usize>() * 8) as u64))).count_ones()
            as u64)
}

mod vectored_xoshiro {
    use std::{
        arch::x86_64::*,
        simd::{mask64x4, u64x4},
    };

    pub struct Avx512Xoshiro256plusPlus {
        s: [__m512i; 4],
    }

    impl Avx512Xoshiro256plusPlus {
        //type Seed = [u8; 32 * 4];

        pub fn from_entropy() -> Self {
            let mut seed = [u8::default(); 32 * 4];
            if let Err(err) = getrandom::getrandom(seed.as_mut()) {
                panic!("from_entropy failed: {}", err);
            }
            Self::from_seed(seed)
        }

        /// Create a new `Xoshiro256PlusPlus`.  If `seed` is entirely 0, it will be
        /// mapped to a different seed.
        #[inline]
        fn from_seed(seed: [u8; 32 * 4]) -> Avx512Xoshiro256plusPlus {
            if seed.iter().all(|&x| x == 0) {
                //return Self::seed_from_u64(0);
            }
            let state = unsafe {
                [
                    _mm512_loadu_si512(seed.as_ptr() as *const i32),
                    _mm512_loadu_si512(seed.as_ptr().add(64) as *const i32),
                    _mm512_loadu_si512(seed.as_ptr().add(64 * 2) as *const i32),
                    _mm512_loadu_si512(seed.as_ptr().add(64 * 3) as *const i32),
                ]
            };
            Avx512Xoshiro256plusPlus { s: state }
        }

        /*
        /// Create a new `Xoshiro256PlusPlus` from a `u64` seed.
        ///
        /// This uses the SplitMix64 generator internally.
        fn seed_from_u64(mut state: u64) -> Self {
            const PHI: u64 = 0x9e3779b97f4a7c15;
            let mut seed = Self::Seed::default();
            for chunk in seed.as_mut().chunks_mut(8) {
                state = state.wrapping_add(PHI);
                let mut z = state;
                z = (z ^ (z >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
                z = (z ^ (z >> 27)).wrapping_mul(0x94d049bb133111eb);
                z = z ^ (z >> 31);
                chunk.copy_from_slice(&z.to_le_bytes());
            }
            Self::from_seed(seed)
        }
        */

        #[inline]
        pub fn next_u64_x8(&mut self) -> __m512i {
            unsafe {
                let result_plusplus = _mm512_add_epi64(
                    _mm512_rol_epi64::<23>(_mm512_add_epi64(self.s[0], self.s[3])),
                    self.s[0],
                );

                let t = _mm512_rol_epi64::<17>(self.s[1]);

                self.s[2] = _mm512_xor_epi64(self.s[2], self.s[0]);
                self.s[3] = _mm512_xor_epi64(self.s[3], self.s[1]);
                self.s[1] = _mm512_xor_epi64(self.s[1], self.s[2]);
                self.s[0] = _mm512_xor_epi64(self.s[0], self.s[3]);

                self.s[2] = _mm512_xor_epi64(self.s[2], t);

                self.s[3] = _mm512_rol_epi64::<45>(self.s[3]);

                std::mem::transmute(result_plusplus)
            }
        }

        /*  #[inline]
        fn next_u64(&mut self) -> u64 {
            let result_plusplus = self.s[0]
                .wrapping_add(self.s[3])
                .rotate_left(23)
                .wrapping_add(self.s[0]);

            let t = self.s[1] << 17;

            self.s[2] ^= self.s[0];
            self.s[3] ^= self.s[1];
            self.s[1] ^= self.s[2];
            self.s[0] ^= self.s[3];

            self.s[2] ^= t;

            self.s[3] = self.s[3].rotate_left(45);

            result_plusplus
        }
        */
    }

    pub fn gen_avx512_intrinsics(count: u64) -> u64 {
        if count == 0 {
            0
        } else {
            let mut rng = Avx512Xoshiro256plusPlus::from_entropy();
            let mut count_heads = 0;
            for _ in 0..(count / 512) {
                unsafe {
                    count_heads +=
                        _mm512_reduce_add_epi64(_mm512_popcnt_epi64(rng.next_u64_x8())) as u64
                }
            }

            count_heads += unsafe {
                _mm512_reduce_add_epi32(_mm512_maskz_popcnt_epi32(
                    !((!0) << ((count % 512) / 8)),
                    rng.next_u64_x8(),
                ))
            } as u64;

            let last_result =
                unsafe { std::intrinsics::transmute::<__m512i, [u32; 16]>(rng.next_u64_x8()) };

            count_heads + (last_result[0] & !(!0 << (count % 8))).count_ones() as u64
        }
    }

    pub struct Xoshiro256PlusPlusX4 {
        s: [u64x4; 4],
    }

    impl Xoshiro256PlusPlusX4 {
        pub fn from_entropy() -> Self {
            let mut seed = [u8::default(); 32];
            if let Err(err) = getrandom::getrandom(seed.as_mut()) {
                panic!("from_entropy failed: {}", err);
            }
            Self::from_seed(seed)
        }

        #[inline]
        fn from_seed(seed: [u8; 32]) -> Xoshiro256PlusPlusX4 {
            if seed.iter().all(|&x| x == 0) {
                //return Self::seed_from_u64(0);
            }
            let state = unsafe {
                [
                    u64x4::from_slice(std::slice::from_raw_parts(seed.as_ptr() as *const u64, 4)),
                    u64x4::from_slice(std::slice::from_raw_parts(
                        seed.as_ptr().add(16) as *const u64,
                        4,
                    )),
                    u64x4::from_slice(std::slice::from_raw_parts(
                        seed.as_ptr().add(32) as *const u64,
                        4,
                    )),
                    u64x4::from_slice(std::slice::from_raw_parts(
                        seed.as_ptr().add(48) as *const u64,
                        4,
                    )),
                ]
            };

            Xoshiro256PlusPlusX4 { s: state }
        }

        #[inline]
        fn next_u64_x4(&mut self) -> [u64; 4] {
            let result_plusplus = {
                let temp = self.s[0] + self.s[3];
                let overflow = temp >> u64x4::splat(41);
                ((temp << u64x4::splat(23)) | overflow) + self.s[0]
            };

            let t = self.s[1] << u64x4::splat(17);

            self.s[2] ^= self.s[0];
            self.s[3] ^= self.s[1];
            self.s[1] ^= self.s[2];
            self.s[0] ^= self.s[3];

            self.s[2] ^= t;

            self.s[3] = (self.s[3] << u64x4::splat(19)) | (self.s[3] >> u64x4::splat(45));

            result_plusplus.as_array().to_owned()
        }
    }

    pub fn gen_portable_simd(count: u64) -> u64 {
        if count == 0 {
            0
        } else {
            let mut rng = Xoshiro256PlusPlusX4::from_entropy();
            let mut count_heads = 0;
            for _ in 0..(count / 256) {
                count_heads += &rng
                    .next_u64_x4()
                    .iter()
                    .map(|&x| x.count_ones() as u64)
                    .sum::<u64>()
            }

            // TODO: use mask types
            count_heads += rng.next_u64_x4()[0..((count % 256) / 64) as usize]
                .iter()
                .map(|&x| x.count_ones() as u64)
                .sum::<u64>();

            // TODO: unaligned to multiple of 256
            count_heads + (rng.next_u64_x4()[0] & !(!0 << (count % 8))).count_ones() as u64
        }
    }
}
