use cuda_builder::CudaBuilder;

fn main() {
    CudaBuilder::new("../../gpu/fractals/")
        .copy_to("../../resources/fractals.ptx")
        .build()
        .unwrap();
}
