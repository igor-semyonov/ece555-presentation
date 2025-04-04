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
    include_str!("../../../resources/fractals.ptx");

const N_RE: usize = 1 << 13;
const N_IM: usize = N_RE >> 1;
#[allow(dead_code)]
const THREADS_DIM: usize = 16;

fn main() -> Result<()> {
    let mut elapsed_times =
        std::collections::HashMap::new();

    let zn_limit: u32 = 100;

    let re_min = -2.0;
    let re_max = 1.0;
    let im_min = -1.0;
    let im_max = 1.0;

    // let x_min = -0.5;
    // let x_max = -0.4;
    // let y_min = 0.55;
    // let y_max = 0.6;

    // let x_min = -0.465;
    // let x_max = -0.455;
    // let y_min = 0.577;
    // let y_max = 0.587;

    let re_range = re_max - re_min;
    let im_range = im_max - im_min;

    let start_execution = Instant::now();
    let points = nd::Array2::from_shape_fn(
        (
            N_RE, N_IM,
        ),
        |idx| c32 {
            re: re_range
                * (idx.0 as f32 / ((N_RE - 1) as f32))
                + re_min,
            im: im_range
                * (idx.1 as f32 / ((N_IM - 1) as f32))
                + im_min,
        },
    );
    elapsed_times.insert(
        "cpu-create-points".to_string(),
        start_execution
            .elapsed()
            .as_micros() as f64
            / 1e3,
    );

    let mut out_non_local_points = vec![0u8; N_RE * N_IM];
    let mut out = vec![0u8; N_RE * N_IM];
    let mut out64 = vec![255u8; N_RE * N_IM];
    let mut out_cpu: nd::Array2<u8> = nd::Array2::zeros((
        N_RE, N_IM,
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

    let out_gpu_non_local_points = out_non_local_points
        .as_slice()
        .as_dbuf()?;
    let out_gpu = out
        .as_slice()
        .as_dbuf()?;
    let out_gpu64 = out64
        .as_slice()
        .as_dbuf()?;
    // let out_gpu_local_points: DeviceBuffer<u8> =
    //     DeviceBuffer::zeroed(N_RE * N_IM)?;

    // for some reason I can't figure out, rectangular block
    // size breaks the local_points kernels
    // let threads = Vec2::broadcast(THREADS_DIM);
    // let block_size: BlockSize = threads.into();
    // let grid_size: GridSize =
    //     (Vec2::from_slice(&[N_IM, N_RE]) /
    // threads).into();

    let block_size = BlockSize {
        x: 1024,
        y: 1,
        z: 1,
    };
    let grid_size = GridSize {
        x: N_IM as u32 >> 10,
        y: N_RE as u32,
        z: 1,
    };

    println!(
        "{:?}\n{:?}\n",
        block_size, grid_size
    );

    let start_execution = Instant::now();
    unsafe {
        launch!(
            module.mandelbrot_non_local_points<<<grid_size, block_size, 0, stream>>>(
                zn_limit,
                points_re_gpu.as_device_ptr(),
                points_re_gpu.len(),
                points_im_gpu.as_device_ptr(),
                points_im_gpu.len(),
                out_gpu_non_local_points.as_device_ptr(),
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

    out_gpu_non_local_points
        .copy_to(&mut out_non_local_points)?;
    #[allow(unused_variables)]
    let out_non_local_points_nd =
        nd::Array2::from_shape_vec(
            (
                N_RE, N_IM,
            ),
            out_non_local_points,
        )?;

    let start_execution = Instant::now();
    unsafe {
        launch!(
            module.mandelbrot<<<grid_size, block_size, 0, stream>>>(
                N_RE,
                N_IM,
                re_min,
                re_max,
                re_range,
                im_min,
                im_max,
                im_range,
                zn_limit,
                out_gpu.as_device_ptr(),
            )
        )?;
    }

    stream.synchronize()?;
    elapsed_times.insert(
        "gpu_local_points".to_string(),
        start_execution
            .elapsed()
            .as_micros() as f64
            / 1e3,
    );

    let start_execution = Instant::now();
    unsafe {
        launch!(
            module.mandelbrot64<<<grid_size, block_size, 0, stream>>>(
                N_RE,
                N_IM,
                re_min as f64,
                re_max as f64,
                re_range as f64,
                im_min as f64,
                im_max as f64,
                im_range as f64,
                zn_limit,
                out_gpu64.as_device_ptr(),
            )
        )?;
    }

    stream.synchronize()?;
    elapsed_times.insert(
        "gpu_local_points_64".to_string(),
        start_execution
            .elapsed()
            .as_micros() as f64
            / 1e3,
    );

    out_gpu.copy_to(&mut out)?;
    #[allow(unused_variables)]
    let out_nd = nd::Array2::from_shape_vec(
        (
            N_RE, N_IM,
        ),
        out,
    )?;

    out_gpu64.copy_to(&mut out64)?;
    let out64_nd = nd::Array2::from_shape_vec(
        (
            N_RE, N_IM,
        ),
        out64,
    )?;

    // let out = nd::concatenate![
    //     nd::Axis(0),
    //     out,
    //     out_local_points
    // ];

    let image = array_to_image(
        // out_local_points.t()
        out64_nd
            .t()
            // out.t()
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
