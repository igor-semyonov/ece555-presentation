#![allow(unused_imports)]
use anyhow::{Context, Result, anyhow};
use cust::function::{BlockSize, GridSize};
use cust::prelude::*;
use image::GrayImage;
use ndarray as nd;
use ndarray::parallel::prelude::*;
use num::complex::Complex32 as c32;
use std::time::Instant;
use vek::Vec2;

static PTX: &str =
    include_str!("../../../resources/mandelbrot.ptx");

const NROWS: usize = 1 << 14;
const NCOLS: usize = NROWS >> 1;
const THREADS_DIM: usize = 8;

fn main() -> Result<()> {
    let mut elapsed_times =
        std::collections::HashMap::new();

    let zn_limit: u32 = 100;
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
        |idx| c32 {
            re: x_range
                * (idx.0 as f32 / ((NROWS - 1) as f32))
                + x_min,
            im: y_range
                * (idx.1 as f32 / ((NCOLS - 1) as f32))
                + y_min,
        },
    );

    let mut out = vec![0u8; NROWS * NCOLS];
    let mut out_cpu: nd::Array2<u8> = nd::Array2::zeros((
        NROWS, NCOLS,
    ));

    let start_execution = Instant::now();
    nd::Zip::from(&points)
        .and(&mut out_cpu)
        .into_par_iter()
        .with_min_len(100)
        .for_each(
            |(c, m)| {
                let mut z = *c;
                for _ in 0..zn_limit {
                    z = z * z + c;
                    if z.re * z.re + z.im + z.im > 4.0 {
                        *m = 0;
                        return;
                    }
                }
                *m = 255;
            },
        );
    elapsed_times.insert(
        "cpu-rayon".to_string(),
        start_execution
            .elapsed()
            .as_micros() as f64
            / 1e3,
    );

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

    let points_re_gpu = points
        .iter()
        .map(|c| c.re)
        .collect::<Vec<_>>()
        .as_slice()
        .as_dbuf()?;
    let points_im_gpu = points
        .iter()
        .map(|c| c.im)
        .collect::<Vec<_>>()
        .as_slice()
        .as_dbuf()?;

    let out_gpu = out
        .as_slice()
        .as_dbuf()?;

    let threads = Vec2::broadcast(THREADS_DIM);
    let block_size: BlockSize = threads.into();
    let grid_size: GridSize =
        (Vec2::from_slice(&[NROWS, NCOLS]) / threads)
            .into();

    // the following is slightly slower, 7x cpu instead of 8x vs the square blocks aobove
    let block_size = BlockSize {
        x: 1024,
        y: 1,
        z: 1,
    };
    let grid_size = GridSize {
        x: NROWS as u32 >> 10,
        y: NCOLS as u32,
        z: 1,
    };

    println!(
        "{:?}\n{:?}\n",
        block_size, grid_size
    );

    let start_execution = Instant::now();
    unsafe {
        launch!(
            module.mandelbrot<<<grid_size, block_size, 0, stream>>>(
                NCOLS,
                zn_limit,
                points_re_gpu.as_device_ptr(),
                points_re_gpu.len(),
                points_im_gpu.as_device_ptr(),
                points_im_gpu.len(),
                out_gpu.as_device_ptr(),
            )
        )?;
    }

    stream.synchronize()?;
    elapsed_times.insert(
        "gpu".to_string(),
        start_execution
            .elapsed()
            .as_micros() as f64
            / 1e3,
    );

    out_gpu.copy_to(&mut out)?;
    let out = nd::Array2::from_shape_vec(
        (
            NROWS, NCOLS,
        ),
        out,
    )?;

    let out = nd::concatenate![nd::Axis(0), out, out_cpu];

    let image = array_to_image(
        out.t()
            .as_standard_layout()
            .to_owned(),
    );
    image
        .save("out.png")
        .unwrap();

    elapsed_times
        .iter()
        .for_each(
            |(k, v)| {
                println!(
                    "{:20.} took {:8.2}",
                    k, v
                );
            },
        );
    let cpu_time = elapsed_times
        .get("cpu-rayon")
        .unwrap();
    let gpu_time = elapsed_times
        .get("gpu")
        .unwrap();
    println!(
        "\nGPU execution time was {:.2} times faster than CPU using rayon",
        cpu_time / gpu_time
    );

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
