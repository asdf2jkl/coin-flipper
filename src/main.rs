use coin_flipper::{gpgpu::gpu_executor, threaded_wrapper};

struct Flipper {
    name: &'static str,
    function: fn(count: u64) -> Option<u64>,
}

const GENERATORS_TO_TEST: [Flipper; 2] = [
    Flipper {
        name: "Count noalloc nobuffer u64x4 generic + threaded",
        function: threaded_wrapper,
    },
    Flipper {
        name: "OpenCL naive gpu",
        function: gpu_executor,
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
        if let Some(result) = result {
            println!("  Heads: {}", result);
        } else {
            println!("Failed to generate result");
        }
        println!("  Time elapsed: {}s", duration.as_secs_f32());
        println!(
            "  Rate: {} Gflips\n",
            coins_flipped as f32 / (duration.as_secs_f32() * 1e9)
        );
    }
}
