use image::GrayImage;
use ndarray as nd;
use ndarray::parallel::prelude::*;
use num::complex::Complex64 as c64;

fn main() {
    let zn_limit = 100;
    let x_min = -2.0;
    let x_max = 1.0;
    let y_min = -1.0;
    let y_max = 1.0;
    let x_range = x_max - x_min;
    let y_range = y_max - y_min;
    let n_rows = 16384;
    let n_cols = 8192;

    let points = nd::Array2::from_shape_fn(
        (
            n_rows, n_cols,
        ),
        |idx| c64 {
            re: x_range
                * (idx.0 as f64 / ((n_rows - 1) as f64))
                + x_min,
            im: y_range
                * (idx.1 as f64 / ((n_cols - 1) as f64))
                + y_min,
        },
    );

    let mut mandel: nd::Array2<u8> = nd::Array2::zeros((
        n_rows, n_cols,
    ));

    nd::Zip::from(&points)
        .and(&mut mandel)
        .into_par_iter()
        .with_min_len(100)
        .for_each(
            |(c, m)| {
                let mut z = *c;
                for _ in 0..zn_limit {
                    z = z * z + c;
                    if z.norm() > 2.0 {
                        *m = 0;
                        return;
                    }
                }
                *m = 255;
            },
        );

    let mandel = mandel
        .t()
        .as_standard_layout()
        .to_owned();
    let image = array_to_image(mandel);
    image
        .save("out.png")
        .unwrap();
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
