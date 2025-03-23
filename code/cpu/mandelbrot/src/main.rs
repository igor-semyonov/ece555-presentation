use anyhow::{Context, Result, anyhow};
use cust::function::{BlockSize, GridSize};
use cust::prelude::*;
use image::GrayImage;
use ndarray as nd;
use ndarray::parallel::prelude::*;
use num::complex::Complex64 as c64;
use vek::Vec2;

static PTX: &str =
    include_str!("../../../resources/mandelbrot.ptx");

const NROWS: usize = 32;
const NCOLS: usize = 32;

fn main() -> Result<()> {
    let zn_limit = 100;
    let x_min = -2.0;
    let x_max = 1.0;
    let y_min = -1.0;
    let y_max = 1.0;
    let x_range = x_max - x_min;
    let y_range = y_max - y_min;

    let points = nd::Array2::from_shape_fn(
        (
            NROWS, NCOLS,
        ),
        |idx| c64 {
            re: x_range
                * (idx.0 as f64 / ((NROWS - 1) as f64))
                + x_min,
            im: y_range
                * (idx.1 as f64 / ((NCOLS - 1) as f64))
                + y_min,
        },
    );

    let mut mandel: nd::Array2<u8> = nd::Array2::zeros((
        NROWS, NCOLS,
    ));
    let mandel = vec![0usize; NROWS * NCOLS];

    // nd::Zip::from(&points)
    //     .and(&mut mandel)
    //     .into_par_iter()
    //     .with_min_len(100)
    //     .for_each(
    //         |(c, m)| {
    //             let mut z = *c;
    //             for _ in 0..zn_limit {
    //                 z = z * z + c;
    //                 if z.norm() > 2.0 {
    //                     *m = 0;
    //                     return;
    //                 }
    //             }
    //             *m = 255;
    //         },
    //     );

    // initialize CUDA, this will pick the first available
    // device and will make a CUDA context from it.
    // We don't need the context for anything but it must be
    // kept alive.
    let _ctx = cust::quick_init()?;

    // Make the CUDA module, modules just house the GPU code
    // for the kernels we created. they can be made from
    // PTX code, cubins, or fatbins.
    let module = Module::from_ptx(
        PTX,
        &[],
    )?;

    // make a CUDA stream to issue calls to. You can think
    // of this as an OS thread but for dispatching
    // GPU calls.
    let stream = Stream::new(
        StreamFlags::NON_BLOCKING,
        None,
    )?;

    let mandel_gpu = mandel
        .as_slice()
        // .unwrap()
        .as_dbuf()?;

    let threads = Vec2::broadcast(16usize);
    let block_size: BlockSize = threads.into();
    let grid_size: GridSize =
        (Vec2::from_slice(&[NROWS, NCOLS]) / threads)
            .into();
    println!(
        "{:?}\n{:?}\n",
        block_size, grid_size
    );

    println!("{}", mandel_gpu.len());

    unsafe {
        launch!(
            module.mandelbrot<<<grid_size, block_size, 0, stream>>>(
                mandel_gpu.as_device_ptr()
                // 0u8
            )
        )?;
    }

    stream.synchronize()?;

    // let image = array_to_image(
    //     mandel
    //         .t()
    //         .as_standard_layout()
    //         .to_owned(),
    // );
    // image
    //     .save("out.png")
    //     .unwrap();

    Ok(())
}

fn array_to_image(arr: nd::Array2<u8>) -> GrayImage {
    assert!(arr.is_standard_layout());

    let (height, width) = arr.dim();
    let raw = arr
        .into_raw_vec_and_offset()
        .0;

    GrayImage::from_raw(width as u32, height as u32, raw)
        .expect("container should have the right size for the image dimensions")
}
