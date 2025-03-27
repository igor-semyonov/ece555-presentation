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
const N_RE: usize = 1 << 13;
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
        .add_systems(Update, scroll_events)
        .add_systems(Update, update_viewport)
        .add_event::<ViewportBoundsEvent>()
        .add_systems(Startup, setup)
        .run();
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
#[derive(Component)]
struct GpuStuff {
    _ctx: cust::context::Context,
    module: cust::module::Module,
    stream: cust::stream::Stream,
    frame_buffer: cust::memory::DeviceBuffer<u8>,
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
        let stream = Stream::new(
            StreamFlags::NON_BLOCKING,
            None,
        )
        .unwrap();
        let frame_buffer: DeviceBuffer<_> =
            cust::memory::DeviceBuffer::zeroed(N_RE * N_IM)
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
            stream,
            frame_buffer,
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
        // GpuStuff::default(),
    ));
    commands.spawn(Sprite::from_image(handle.clone()));
    commands.insert_resource(MyProcGenImage(handle));
}

fn scroll_events(
    mut evr_scroll: EventReader<MouseWheel>,
    mut ev_bounds: EventWriter<ViewportBoundsEvent>,
    mut query: Query<(
        Entity,
        &mut ViewportBounds,
    )>,
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
                zoom_nudge += 0.1 * ev.y;
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
        let (_entity, mut bound) = query.single_mut();
        bound.re_min *= zoom_factor;
        bound.re_max *= zoom_factor;
        bound.im_min *= zoom_factor;
        bound.im_max *= zoom_factor;
        ev_bounds.send(ViewportBoundsEvent());
    }
}

fn update_viewport(
    mut _ev_bounds: EventReader<ViewportBoundsEvent>,
    my_handle: Res<MyProcGenImage>,
    mut images: ResMut<Assets<Image>>,
    mut query: Query<(
        &ViewportBounds,
        // &GpuStuff,
        &mut FrameBuffer,
    )>,
) {
    // let (bounds, gpu_stuff, mut frame_buffer) =
    //     query.single_mut();
    let (bounds, mut frame_buffer) = query.single_mut();

    let re_range = bounds.re_max - bounds.re_min;
    let im_range = bounds.im_max - bounds.im_min;

    use cust::function::{BlockSize, GridSize};
    use cust::prelude::*;
    let _ctx: cust::context::Context =
        cust::quick_init().unwrap();
    let module = Module::from_ptx(
        PTX,
        &[],
    )
    .unwrap();
    let stream = Stream::new(
        StreamFlags::NON_BLOCKING,
        None,
    )
    .unwrap();
    let gpu_frame_buffer: DeviceBuffer<_> =
        cust::memory::DeviceBuffer::zeroed(N_RE * N_IM)
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
    unsafe {
        launch!(
                module.mandelbrot_local_points<<<
        grid_size, block_size, 0, stream>>>(
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
    match gpu_frame_buffer.copy_to(&mut frame_buffer.0) {
        Ok(_) => {
            let image = images
                .get_mut(&my_handle.0)
                .expect("Image not found");
            // eprintln!(
            //     "Successfully copied device to host buffer.",
            // );
            image
                .data
                .copy_from_slice(
                    frame_buffer
                        .0
                        .as_slice(),
                );
        }
        Err(x) => {
            eprintln!(
                "Failed to copy device to host buffer. {}",
                x
            );
        }
    }
}
