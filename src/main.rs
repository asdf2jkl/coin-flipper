use coin_flipper::threaded_wrapper;

struct Flipper {
    name: &'static str,
    function: fn(count: u64) -> u64,
}

const GENERATORS_TO_TEST: [Flipper; 1] = [Flipper {
    name: "Count noalloc nobuffer u64x4 generic + threaded",
    function: threaded_wrapper,
}];

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
        println!("  Time elapsed: {}s", duration.as_secs_f32());
        println!("  Rate: {} Gflips\n", coins_flipped as f32 / (duration.as_secs_f32() * 1e9));
    }
}
