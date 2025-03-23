use cuda_std::prelude::*;

#[kernel]
#[allow(improper_ctypes_definitions, clippy::missing_safety_doc)]
pub unsafe fn mandelbrot(img: &[u8]) {
    // let idx = thread::index_2d();
    // if idx < a.len() {
    //     let elem = &mut *c.add(idx);
    //     *elem = a[idx] + b[idx];
    // }
}
