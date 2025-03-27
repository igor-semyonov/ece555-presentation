#![allow(unused_imports)]
use bevy::diagnostic::FrameTimeDiagnosticsPlugin;
use bevy::ecs::prelude::Event;
use bevy::input::mouse::MouseWheel;
use bevy::prelude::*;
use bevy::render::{
    render_asset::RenderAssetUsages,
    render_resource::{
        Extent3d, TextureDimension, TextureFormat,
    },
};
use cust;
use iyes_perf_ui::prelude::*;

static PTX: &str =
    include_str!("../../../resources/mandelbrot.ptx");
const N_RE: usize = 1 << 11;
const N_IM: usize = N_RE >> 1;
const ZN_LIMIT: u32 = 100;

fn main() {
    App::new()
        .add_plugins(DefaultPlugins
                // .set(ImagePlugin::default_nearest())
            )
        .add_plugins(bevy::diagnostic::FrameTimeDiagnosticsPlugin::default())
        .add_plugins(bevy::diagnostic::EntityCountDiagnosticsPlugin)
        .add_plugins(bevy::diagnostic::SystemInformationDiagnosticsPlugin)
        .add_plugins(bevy::render::diagnostic::RenderDiagnosticsPlugin)
        .add_plugins(PerfUiPlugin)
        .add_systems(Update, (
                scroll_events,
                arrow_events
        ))
        .add_systems(Update, update_viewport)
        .add_event::<ViewportBoundsEvent>()
        .add_systems(Startup, setup)
        .add_systems(Startup, setup_gpu)
        .run();
}
fn setup_gpu(world: &mut World) {
    let gpu = GpuStuff::default();
    world.insert_non_send_resource(gpu);
}

#[derive(Component, Debug)]
struct ViewportBounds {
    re_min: f32,
    re_max: f32,
    im_min: f32,
    im_max: f32,
}

#[derive(Component, Debug)]
struct FrameBuffer(Vec<u8>);
impl Default for FrameBuffer {
    fn default() -> Self {
        Self(vec![0u8; N_RE * N_IM])
    }
}

#[allow(dead_code)]
#[derive(Resource)]
struct GpuStuff {
    _ctx: cust::context::Context,
    module: cust::module::Module,
    block_size: cust::function::BlockSize,
    grid_size: cust::function::GridSize,
}
impl Default for GpuStuff {
    fn default() -> Self {
        use cust::function::{BlockSize, GridSize};
        use cust::prelude::*;
        let _ctx: cust::context::Context =
            cust::quick_init().unwrap();
        let module = Module::from_ptx(
            PTX,
            &[],
        )
        .unwrap();
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
        GpuStuff {
            _ctx,
            module,
            block_size,
            grid_size,
        }
    }
}

impl Default for ViewportBounds {
    fn default() -> Self {
        Self {
            re_min: -2.0,
            re_max: 1.0,
            im_min: -1.0,
            im_max: 1.0,
        }
    }
}

#[derive(Event)]
struct ViewportBoundsEvent();

#[derive(Resource)]
struct MyProcGenImage(Handle<Image>);

fn setup(
    mut commands: Commands,
    mut images: ResMut<Assets<Image>>,
) {
    // spawn a camera to be able to see anything
    commands.spawn(Camera2d);

    // create a simple Perf UI with default settings
    // and all entries provided by the crate:
    commands.spawn(PerfUiDefaultEntries::default());
    // commands.spawn(PerfUiAllEntries::default());

    #[allow(unused_mut)]
    let mut image = Image::new_fill(
        Extent3d {
            width: N_RE as u32,
            height: N_IM as u32,
            depth_or_array_layers: 1,
        },
        TextureDimension::D2,
        &[0u8; N_RE * N_IM],
        TextureFormat::R8Unorm,
        RenderAssetUsages::MAIN_WORLD
            | RenderAssetUsages::RENDER_WORLD,
    );
    let handle = images.add(image);

    commands.spawn((
        ViewportBounds::default(),
        FrameBuffer::default(),
    ));
    commands.spawn(Sprite::from_image(handle.clone()));
    commands.insert_resource(MyProcGenImage(handle));
}

fn arrow_events(
    keys: Res<ButtonInput<KeyCode>>,
    mut ev_bounds: EventWriter<ViewportBoundsEvent>,
    mut query: Query<(&mut ViewportBounds,)>,
) {
    let mut direction = (
        0i8, 0i8,
    );
    if keys.pressed(KeyCode::ArrowRight) {
        direction.0 += 1;
    }
    if keys.pressed(KeyCode::ArrowLeft) {
        direction.0 -= 1;
    }
    if keys.pressed(KeyCode::ArrowUp) {
        direction.1 += 1;
    }
    if keys.pressed(KeyCode::ArrowDown) {
        direction.1 -= 1;
    }
    if direction.0 != 0 || direction.1 != 0 {
        let (mut bounds,) = query.single_mut();
        let re_range = bounds.re_max - bounds.re_min;
        let im_range = bounds.im_max - bounds.im_min;
        let re_delta = 0.005 * re_range * direction.0 as f32;
        let im_delta = 0.005 * im_range * direction.1 as f32;
        bounds.re_min += re_delta;
        bounds.re_max += re_delta;
        bounds.im_min += im_delta;
        bounds.im_max += im_delta;
        ev_bounds.send(ViewportBoundsEvent());
    }
}

fn scroll_events(
    mut evr_scroll: EventReader<MouseWheel>,
    mut ev_bounds: EventWriter<ViewportBoundsEvent>,
    mut query: Query<(&mut ViewportBounds,)>,
) {
    use bevy::input::mouse::MouseScrollUnit;
    for ev in evr_scroll.read() {
        let mut zoom_factor = 1.0;
        let mut zoom_nudge = 0.0;
        match ev.unit {
            MouseScrollUnit::Line => {
                // println!(
                //     "Scroll (line units): vertical: {},
                // horizontal: {}",
                //     ev.y, ev.x
                // );
                zoom_nudge -= 0.1 * ev.y;
            }
            MouseScrollUnit::Pixel => {
                // println!(
                //     "Scroll (pixel units): vertical: {},
                // horizontal: {}",
                //     ev.y, ev.x
                // );
                zoom_nudge -= 0.05 * ev.y;
            }
        }
        zoom_factor += zoom_nudge;
        let (mut bounds,) = query.single_mut();
        bounds.re_min *= zoom_factor;
        bounds.re_max *= zoom_factor;
        bounds.im_min *= zoom_factor;
        bounds.im_max *= zoom_factor;
        ev_bounds.send(ViewportBoundsEvent());
    }
}

fn update_viewport(
    mut ev_bounds: EventReader<ViewportBoundsEvent>,
    gpu: NonSend<GpuStuff>,
    my_handle: Res<MyProcGenImage>,
    mut images: ResMut<Assets<Image>>,
    mut query: Query<(
        &ViewportBounds,
        &mut FrameBuffer,
    )>,
) {
    // let (bounds, gpu_stuff, mut frame_buffer) =
    //     query.single_mut();
    let (bounds, mut frame_buffer) = query.single_mut();

    for _ in ev_bounds
        .read()
        .into_iter()
    {
        println!(
            "{:?}",
            bounds
        );
        let re_range = bounds.re_max - bounds.re_min;
        let im_range = bounds.im_max - bounds.im_min;

        use cust::prelude::*;
        let module = &gpu.module;
        let stream = Stream::new(
            StreamFlags::NON_BLOCKING,
            None,
        )
        .unwrap();
        let gpu_frame_buffer: DeviceBuffer<_> =
            cust::memory::DeviceBuffer::zeroed(N_RE * N_IM)
                .unwrap();
        let block_size = gpu.block_size;
        let grid_size = gpu.grid_size;
        unsafe {
            launch!(
                    module.mandelbrot_local_points<<<grid_size, block_size, 0, stream>>>(
                        N_RE,
                        N_IM,
                        bounds.re_min,
                        bounds.re_max,
                        re_range,
                        bounds.im_min,
                        bounds.im_max,
                        im_range,
                        ZN_LIMIT,
                        gpu_frame_buffer.as_device_ptr(),
                    )
                )
            .unwrap();
        }
        stream
            .synchronize()
            .unwrap();
        match gpu_frame_buffer.copy_to(&mut frame_buffer.0)
        {
            Ok(_) => {
                let image = images
                    .get_mut(&my_handle.0)
                    .expect("Image not found");
                // let new_image = Image::new(
                //     Extent3d {
                //         width: N_RE as u32,
                //         height: N_IM as u32,
                //         depth_or_array_layers: 1,
                //     },
                //     TextureDimension::D2,
                //     frame_buffer.0.clone(),
                //     TextureFormat::R8Unorm,
                //     RenderAssetUsages::MAIN_WORLD
                //         | RenderAssetUsages::RENDER_WORLD,
                // );

                // image.set(Box::new(new_image)).unwrap();

                use ndarray as nd;
                let frame_buffer_nd =
                    nd::Array2::from_shape_vec(
                        (
                            N_RE, N_IM,
                        ),
                        frame_buffer
                            .0
                            .clone(),
                    )
                    .unwrap();
                image
                    .data
                    .copy_from_slice(
                        frame_buffer_nd
                            .t()
                            .as_standard_layout()
                            .as_slice()
                            .unwrap()
                        // frame_buffer
                        //     .0
                        //     .as_slice(),
                    );

                // use ndarray as nd;
                // let frame_buffer_nd =
                //     nd::Array2::from_shape_vec(
                //         (
                //             N_RE, N_IM,
                //         ),
                //         frame_buffer.0.clone(),
                //     )
                //     .unwrap();
                // let image_not_bevy = array_to_image(
                //     frame_buffer_nd
                //         .t()
                //         // out.t()
                //         .as_standard_layout()
                //         .to_owned(),
                // );
                // image_not_bevy
                //     .save("out.png")
                //     .unwrap();
            }
            Err(x) => {
                eprintln!(
                    "Failed to copy device to host buffer. {}",
                    x
                );
            }
        }
    }
}
use image::GrayImage;
use ndarray as nd;
#[allow(dead_code)]
fn array_to_image(arr: nd::Array2<u8>) -> GrayImage {
    assert!(arr.is_standard_layout());

    let (height, width) = arr.dim();
    let raw = arr
        .into_raw_vec_and_offset()
        .0;

    GrayImage::from_raw(width as u32, height as u32, raw)
        .expect("container should have the right size for the image dimensions")
}
