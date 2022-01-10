/*
    TODO:
    - Add support for rng buffer reuse on inputs with count / device_threads > 2e32
    - Optimize final reduction (currently it's done on gpu, but it could be done on cpu (idk if the reduction is faster on cpu than on gpu, will have to benchmark))
    - Runtime selection of vectored vs scalar kernel (based on device properties)
    - Async execution of remainder on cpu
    - Add support for multiple devices (meh, not sure if it's worth bothering)
    - Optimze for seed generation (could use simd intrinsics, all it takes is a new chacha implementation)
    - Cleanup, document, and validate both host and device code
    - Properly handle errors (currently it's just a panic)
    - Probably a lot more
*/


use std::{ffi::c_void, ptr, simd::u32x4};

use opencl3::{
    self,
    command_queue::CommandQueue,
    context::Context,
    device::Device,
    kernel::Kernel,
    memory::{Buffer, CL_MEM_COPY_HOST_PTR, CL_MEM_READ_WRITE},
    platform::{platform, Platform},
    program::Program,
};
use rand::{thread_rng, RngCore};

use crate::threaded_wrapper;

static KERNEL_SOURCE: &'static str = include_str!("kernel.cl");

#[repr(C)]
union xoshiro128plusplus {
    s: [u32x4; 4],
    b: [u8; 64],
}

pub fn gpu_executor(count: u64) -> Option<u64> {
    if count < 100_000_000_000 {
        return threaded_wrapper(count);
    }

    // TODO: Error propagation.
    let context = if let Ok(ids) = platform::get_platform_ids() {
        match Context::from_device(&Device::new(
            *Platform::new(*ids.get(0).unwrap())
                .get_devices(opencl3::device::CL_DEVICE_TYPE_GPU)
                .unwrap()
                .get(0)
                .unwrap(),
        )) {
            Ok(context) => context,
            Err(_) => return None,
        }
    } else {
        return None;
    };

    let threads = {
        let device_threads = Device::max_compute_units(&Device::new(context.default_device()))
            .unwrap() as u64
            * 64
            * 8;
        if count / device_threads > u32::MAX as u64 {
            device_threads * ((count / device_threads) / u32::MAX as u64 + 1)
        } else {
            device_threads
        }
    };

    let program = Program::create_and_build_from_source(&context, KERNEL_SOURCE, "").unwrap();

    let kernel = Kernel::create(&program, "flipper").unwrap();

    let mut host_rng_buffer = Vec::<xoshiro128plusplus>::with_capacity(threads as usize);
    {
        let mut rng = thread_rng();
        for _ in 0..threads {
            unsafe {
                let mut s = xoshiro128plusplus { b: [0; 64] };
                rng.fill_bytes(&mut s.b);
                host_rng_buffer.push(s);
            }
        }
    }

    let rng_buffer = Buffer::<xoshiro128plusplus>::create(
        &context,
        CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
        threads.try_into().unwrap(),
        host_rng_buffer.as_mut_ptr() as *mut c_void,
    )
    .unwrap();
    kernel.set_arg(1, &rng_buffer).unwrap();
    let output = Buffer::<u64>::create(&context, CL_MEM_READ_WRITE, 1, ptr::null_mut()).unwrap();
    kernel.set_arg(2, &output).unwrap();
    kernel.set_arg_local_buffer(3, 8).unwrap();

    let queue = CommandQueue::create(&context, context.default_device(), 0).unwrap();

    kernel
        .set_arg::<u32>(0, &((count / threads).try_into().unwrap()))
        .unwrap();
    queue
        .enqueue_nd_range_kernel(
            kernel.get(),
            1,
            ptr::null(),
            [threads as usize].as_ptr(),
            [64 as usize].as_ptr(),
            &[],
        )
        .unwrap();
    let mut result_buffer: [u64; 1] = [0];
    queue
        .enqueue_read_buffer(&output, true as u32, 0, &mut result_buffer, &[])
        .unwrap();

    Some(result_buffer[0] + threaded_wrapper(count % threads as u64).unwrap())
}
