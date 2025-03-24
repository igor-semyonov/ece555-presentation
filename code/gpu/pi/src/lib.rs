use cuda_std::prelude::*;
use gpu_rand::{DefaultRand, GpuRand};

#[kernel]
#[allow(
    improper_ctypes_definitions,
    clippy::missing_safety_doc
)]
pub unsafe fn pi(
    out: *mut u32,
    rand_states: *mut DefaultRand,
) {
    let idx = thread::index_1d() as usize;

    let rng = &mut *rand_states.add(idx);
    let x = rng.uniform_f32();
    let y = rng.uniform_f32();
    if x * x + y * y <= 1.0 {
        let elem = &mut *out.add(idx);
        *elem = 1;
    }
}
