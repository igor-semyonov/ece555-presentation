use cuda_builder::CudaBuilder;

fn main() {
    CudaBuilder::new("../../gpu/mandelbrot/")
        .copy_to("../../resources/mandelbrot.ptx")
        .build()
        .unwrap();
}
