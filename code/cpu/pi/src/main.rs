#![allow(unused_imports)]
use anyhow::{Context, Result, anyhow};
use cust::function::{BlockSize, GridSize};
use cust::memory::GpuBuffer;
use cust::prelude::*;
use gpu_rand::DefaultRand;
use image::GrayImage;
use ndarray as nd;
use ndarray::parallel::prelude::*;
use std::time::Instant;
use vek::Vec2;

static PTX: &str =
    include_str!("../../../resources/pi.ptx");

const RAND_SEED: u64 = 932174513921034;
const NREPS: usize = 1 << 24;

fn main() -> Result<()> {
    let mut elapsed_times =
        std::collections::HashMap::new();

    let start_execution = Instant::now();

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

    let mut out = vec![0u32; NREPS];
    let out_gpu = out
        .as_slice()
        .as_dbuf()?;

    let block_size = BlockSize {
        x: 1024,
        y: 1,
        z: 1,
    };
    let grid_size = GridSize {
        x: NREPS as u32 / block_size.x,
        y: 1,
        z: 1,
    };

    println!(
        "{:?}\n{:?}\n",
        block_size, grid_size
    );

    let rand_states = DefaultRand::initialize_states(
        RAND_SEED, NREPS,
    )
    .as_slice()
    .as_dbuf()?;

    let start_execution = Instant::now();
    unsafe {
        launch!(
            module.pi<<<grid_size, block_size, 0, stream>>>(
                out_gpu.as_device_ptr(),
                rand_states.as_device_ptr()
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

    let n_inside = out
        .iter()
        .sum::<u32>() as f64;
    let result_gpu = 4.0 * n_inside as f64 / NREPS as f64;
    println!(
        "{}",
        result_gpu
    );

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
    // let cpu_time = elapsed_times
    //     .get("cpu-rayon")
    //     .unwrap();
    // let gpu_time = elapsed_times
    //     .get("gpu")
    //     .unwrap();
    // println!(
    //     "\nGPU execution time was {:.2} times faster than
    // CPU using rayon",     cpu_time / gpu_time
    // );

    Ok(())
}
