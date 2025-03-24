use cuda_std::prelude::*;

#[kernel]
#[allow(
    improper_ctypes_definitions,
    clippy::missing_safety_doc
)]
pub unsafe fn mandelbrot(
    n_cols: usize,
    zn_limit: u32,
    re: &[f32],
    im: &[f32],
    out: *mut u8,
) {
    let idx = thread::index_2d();
    let idx_r = idx[0];
    let idx_c = idx[1];
    let idx = idx_c as usize + idx_r as usize * n_cols;

    let c_re = re[idx];
    let c_im = im[idx];
    let mut z_re = c_re;
    let mut z_im = c_im;
    let mut converges = 255u8;
    for _ in 0..zn_limit {
        (
            z_re, z_im,
        ) = (
            z_re * z_re - z_im * z_im + c_re,
            2.0 * z_re * z_im + c_im,
        );
        if z_re * z_re + z_im * z_im > 4.0 {
            converges = 0;
            break;
        }
    }
    let elem = &mut *out.add(idx);
    *elem = converges;
}
