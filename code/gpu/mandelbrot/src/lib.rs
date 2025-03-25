use cuda_std::prelude::*;

#[kernel]
#[allow(
    improper_ctypes_definitions,
    clippy::missing_safety_doc
)]
pub unsafe fn mandelbrot(
    zn_limit: u32,
    re: &[f32],
    im: &[f32],
    out: *mut u8,
) {
    let idx = thread::index() as usize;

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

#[kernel]
#[allow(
    improper_ctypes_definitions,
    clippy::missing_safety_doc
)]
/// Calculate the point values internally, instead of having
/// a points vector precomputed and copied to device
pub unsafe fn mandelbrot_local_points(
    n_re: usize,
    n_im: usize,
    re_min: f32,
    re_max: f32,
    re_range: f32,
    im_min: f32,
    im_max: f32,
    im_range: f32,
    zn_limit: u32,
    out: *mut u8,
) {
    let idx_linear = thread::index() as usize;
    let idx = thread::index_2d();
    let idx_re = idx[1];
    let idx_im = idx[0];

    let c_re = re_range
        * (idx_re as f32 / (n_re - 1) as f32)
        + re_min;
    let c_im = im_range
        * (idx_im as f32 / (n_im - 1) as f32)
        + im_min;

    let mut z_re = c_re;
    let mut z_im = c_im;
    for _ in 0..zn_limit {
        (
            z_re, z_im,
        ) = (
            z_re * z_re - z_im * z_im + c_re,
            2.0 * z_re * z_im + c_im,
        );
        if z_re * z_re + z_im * z_im > 4.0 {
            let elem = &mut *out.add(idx_linear);
            *elem = 255;
            break;
        }
    }
}
