use cuda_builder::CudaBuilder;

fn main() {
    CudaBuilder::new("../../gpu/pi/")
        .copy_to("../../resources/pi.ptx")
        .build()
        .unwrap();
}
