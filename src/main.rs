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
    function: fn(count: usize) -> usize,
}

const GENERATORS_TO_TEST: [Flipper; 6] = [
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
        function: fill_buf_count_usize,
    },
    Flipper {
        name: "Vector buffer count + unsafe:",
        function: fill_buf_count_w_unsafe,
    },
    Flipper {
        name: "Vector buffer count noalloc:",
        function: fill_buff_count_noalloc,
    },
];

fn main() {
    let coins_flipped = std::env::args()
        .next_back()
        .unwrap()
        .parse::<usize>()
        .unwrap();

    for generator in GENERATORS_TO_TEST {
        println!("{}", generator.name);
        let start = std::time::Instant::now();
        println!("  Heads: {}", (generator.function)(coins_flipped));
        println!("  Time elapsed: {} µs\n", start.elapsed().as_micros());
    }
}

fn naive_loop(count: usize) -> usize {
    let mut count_heads = 0;
    for _ in 0..count {
        count_heads += rand::random::<bool>() as usize;
    }
    count_heads
}

fn naive_loop_threadrng(count: usize) -> usize {
    let mut rng = thread_rng();
    let mut count_heads = 0;
    for _ in 0..count {
        count_heads += rng.gen::<bool>() as usize;
    }
    count_heads
}

fn naive_loop_smallrng(count: usize) -> usize {
    let mut rng = SmallRng::from_entropy();
    let mut count_heads = 0;
    for _ in 0..count {
        count_heads += rng.gen::<bool>() as usize;
    }
    count_heads
}

fn fill_buf_count_usize(count: usize) -> usize {
    type ElementSize = usize;

    let buf_len = (count / (size_of::<ElementSize>() * 8)) + 1;
    let mut buffer: Vec<ElementSize> = Vec::with_capacity(buf_len);
    buffer.resize_with(buf_len, Default::default);
    let mut rng = SmallRng::from_entropy();
    rng.fill(buffer.as_mut_slice());

    let remainder = buffer.pop().unwrap_or_else(|| unreachable!());
    buffer.iter().fold(0, |accumulator, element| {
        accumulator + element.count_ones() as usize
    }) + ((remainder & !(!0 << (count % (size_of::<ElementSize>() * 8)))).count_ones() as usize)
}

fn fill_buf_count_w_unsafe(count: usize) -> usize {
    type ElementSize = usize;
    const SIZEOF: usize = size_of::<ElementSize>();

    if count == 0 {
        return 0;
    }

    let allocation = {
        let layout = Layout::from_size_align(
            count / (size_of::<ElementSize>()),
            std::mem::align_of::<ElementSize>(),
        )
        .unwrap_or_else(|_| unsafe { unreachable_unchecked() })
        .pad_to_align();

        Allocation {
            ptr: unsafe { alloc(layout) },
            layout,
        }
    };

    let buffer = unsafe { &mut *slice_from_raw_parts_mut(allocation.ptr, count / 8) };

    let mut rng = SmallRng::from_entropy();

    rng.fill_bytes(buffer);

    let remainder: ElementSize = rng.gen();

    let result = unsafe {
        &*slice_from_raw_parts_mut(allocation.ptr as *mut ElementSize, count / (8 * SIZEOF))
    }
    .iter()
    .fold(0, |accumulator, element| {
        accumulator + element.count_ones() as usize
    });

    unsafe { dealloc(allocation.ptr, allocation.layout) };

    result + ((remainder & !(!0 << (count % (size_of::<ElementSize>() * 8)))).count_ones() as usize)
}

// TODO update by using unionsS
fn fill_buff_count_noalloc(count: usize) -> usize {
    type ElementSize = usize;
    const LEN: usize = 16; // Make a good size.
    type Buffer = [ElementSize; LEN / size_of::<ElementSize>()];

    if count == 0 {
        return 0;
    };

    let mut buffer = [0usize;//unsafe { std::mem::MaybeUninit::<ElementSize>::uninit().assume_init()};
        LEN / size_of::<ElementSize>()];
    let mut accumulator = 0;
    let mut rng = SmallRng::from_entropy();

    for _ in 0..(count / (size_of::<Buffer>() * 8)) {
        rng.fill_bytes(unsafe {
            &mut *slice_from_raw_parts_mut(buffer.as_mut_ptr().cast::<u8>(), LEN)
        });
        accumulator += buffer.iter().fold(0, |accumulator, element| {
            accumulator + element.count_ones() as usize
        })
    }

    let rem_buffer =
        &mut buffer[..(((count % (size_of::<Buffer>() * 8)) / (size_of::<ElementSize>() * 8)) + 1)];
    let remainder = (count % size_of::<ElementSize>()) - (size_of_val(rem_buffer) * 8);
    rng.fill(rem_buffer);

    accumulator
        + rem_buffer[..(rem_buffer.len() - 1)]
            .iter()
            .fold(0, |accumulator, element| {
                accumulator + element.count_ones() as usize
            })
        + (*rem_buffer
            .last()
            .unwrap_or_else(|| unsafe { unreachable_unchecked() })
            & !(!0 << (remainder % (size_of::<ElementSize>() * 8))))
            .count_ones() as usize
}
