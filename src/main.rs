//! GPU compute smoke test for the pruning/training backend (separate
//! concern from the `core` repo's llama.cpp inference path, already
//! validated there). Confirms burn's WGPU (Vulkan) backend can
//! actually run a computation on this machine's GPU before any real
//! pruning/fine-tuning work depends on it.
use burn::tensor::Tensor;

fn print_actual_adapter() {
    let instance = wgpu::Instance::default();
    for adapter in pollster::block_on(instance.enumerate_adapters(wgpu::Backends::all())) {
        println!("wgpu sees adapter: {:?}", adapter.get_info());
    }
}

fn main() {
    print_actual_adapter();

    // Device type resolved implicitly by the active backend feature
    // (wgpu is the only one enabled here), matching burn's own doc
    // example rather than naming a concrete device type by hand.
    let device = Default::default();
    println!("Using device: {device:?}");

    let a = Tensor::<2>::from_data([[1.0, 2.0], [3.0, 4.0]], &device);
    let b = Tensor::<2>::from_data([[5.0, 6.0], [7.0, 8.0]], &device);
    let c = a.matmul(b);

    println!("Result:\n{c}");
}
