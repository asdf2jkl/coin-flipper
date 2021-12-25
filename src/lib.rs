#![feature(portable_simd)]
#![feature(available_parallelism)]

use rand::prelude::*;
use std::{num::NonZeroUsize, sync::mpsc::channel};
use vectored_xoshiro::*;

pub fn threaded_wrapper(count: u64) -> u64 {
    if count == 0 {
        0
    } else {
        // Magic number for threading, will be determined experimentally later.
        if count < 10000000 {
            gen_portable_simd_x8_inner(count, Xoshiro256PlusPlusX8::from_entropy())
        } else {
            let num_cpus: usize = std::thread::available_parallelism()
                .unwrap_or(NonZeroUsize::new(1).unwrap())
                .into();

            let mut rng = thread_rng();
            let threaded_count = count / num_cpus as u64;

            let (tx, rx) = channel();

            // TODO: Determine available thread count.
            for _ in 0..(num_cpus - 1) {
                let mut seed_buffer = [0u8; 64];
                rng.fill_bytes(&mut seed_buffer);
                let new_rng = Xoshiro256PlusPlusX8::from_seed(seed_buffer);
                let channel = tx.clone();

                std::thread::spawn(move || {
                    channel
                        .send(gen_portable_simd_x8_inner(threaded_count, new_rng))
                        .unwrap();
                });
            }
            drop(tx);

            let mut accumulator = gen_portable_simd_x8_inner(
                (count / num_cpus as u64) + (count % num_cpus as u64),
                {
                    let mut seed_buffer = [0u8; 64];
                    rng.fill_bytes(&mut seed_buffer);
                    vectored_xoshiro::Xoshiro256PlusPlusX8::from_seed(seed_buffer)
                },
            );

            loop {
                if let Ok(partial) = rx.recv() {
                    accumulator += partial;
                } else {
                    break;
                }
            }

            accumulator
        }
    }
}

pub mod vectored_xoshiro {
    use std::simd::u64x8;

    pub struct Xoshiro256PlusPlusX8 {
        s: [u64x8; 4],
    }

    impl Xoshiro256PlusPlusX8 {
        pub fn from_entropy() -> Self {
            let mut seed = [u8::default(); 64];
            if let Err(err) = getrandom::getrandom(seed.as_mut()) {
                panic!("from_entropy failed: {}", err);
            }
            Self::from_seed(seed)
        }

        #[inline]
        pub fn from_seed(seed: [u8; 64]) -> Xoshiro256PlusPlusX8 {
            if seed.iter().all(|&x| x == 0) {
                //return Self::seed_from_u64(0);
            }
            let state = unsafe {
                [
                    u64x8::from_slice(std::slice::from_raw_parts(seed.as_ptr() as *const u64, 8)),
                    u64x8::from_slice(std::slice::from_raw_parts(
                        seed.as_ptr().add(32) as *const u64,
                        8,
                    )),
                    u64x8::from_slice(std::slice::from_raw_parts(
                        seed.as_ptr().add(64) as *const u64,
                        8,
                    )),
                    u64x8::from_slice(std::slice::from_raw_parts(
                        seed.as_ptr().add(96) as *const u64,
                        8,
                    )),
                ]
            };

            Xoshiro256PlusPlusX8 { s: state }
        }

        #[inline]
        fn next_u64_x8(&mut self) -> [u64; 8] {
            let result_plusplus = {
                let temp = self.s[0] + self.s[3];
                let overflow = temp >> u64x8::splat(41);
                ((temp << u64x8::splat(23)) | overflow) + self.s[0]
            };

            let t = self.s[1] << u64x8::splat(17);

            self.s[2] ^= self.s[0];
            self.s[3] ^= self.s[1];
            self.s[1] ^= self.s[2];
            self.s[0] ^= self.s[3];

            self.s[2] ^= t;

            self.s[3] = (self.s[3] << u64x8::splat(19)) | (self.s[3] >> u64x8::splat(45));

            result_plusplus.as_array().to_owned()
        }
    }

    pub fn gen_portable_simd_x8_inner(count: u64, mut generator: Xoshiro256PlusPlusX8) -> u64 {
        let mut count_heads = 0;
        for _ in 0..(count / 512) {
            count_heads += generator
                .next_u64_x8()
                .iter()
                .map(|&x| x.count_ones() as u64)
                .sum::<u64>()
        }

        // TODO: use mask types
        count_heads += generator.next_u64_x8()[0..((count % 512) / 64) as usize]
            .iter()
            .map(|&x| x.count_ones() as u64)
            .sum::<u64>();

        // TODO: unaligned to multiple of 256
        count_heads + (generator.next_u64_x8()[0] & !(!0 << (count % 64))).count_ones() as u64
    }
}
