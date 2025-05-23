#![allow(unused_imports)]
use bevy::diagnostic::FrameTimeDiagnosticsPlugin;
use bevy::ecs::prelude::Event;
use bevy::input::mouse::MouseMotion;
use bevy::input::mouse::MouseWheel;
use bevy::prelude::*;
use bevy::render::{
    render_asset::RenderAssetUsages,
    render_resource::{
        Extent3d, TextureDimension, TextureFormat,
    },
};
use bevy::window::PrimaryWindow;
use cust;
use iyes_perf_ui::entry::PerfUiEntry;
use iyes_perf_ui::prelude::*;

static PTX: &str =
    include_str!("../../../resources/fractals.ptx");
const N_RE: usize = 1 << 11;
const N_IM: usize = N_RE >> 1;
const ZN_LIMIT: u32 = 100;

const PANNING_SPEED: f64 = 0.003;
const DRAG_SPEED: f64 = 0.001;
const ZOOM_SPEED: f64 = 0.1;
const ZOOM_SPEED_FINE: f64 = 0.02;

fn main() {
    let mut app = App::new();
    app.add_plugins(
        DefaultPlugins.set(
            WindowPlugin {
                primary_window: Some(
                    Window {
                        title: "Zoomable Fractals".into(),
                        name: Some("zoomable_fractal.app".into()),
                        mode: bevy::window::WindowMode::BorderlessFullscreen(MonitorSelection::Current),
                        present_mode: bevy::window::PresentMode::AutoNoVsync,
                        ..Default::default()
                    },
                ),
                ..Default::default()
            },
        ),
    );
    app.add_plugins(bevy::diagnostic::FrameTimeDiagnosticsPlugin::default());
    app.add_plugins(
        bevy::diagnostic::EntityCountDiagnosticsPlugin,
    );
    app.add_plugins(bevy::diagnostic::SystemInformationDiagnosticsPlugin);
    app.add_plugins(
        bevy::render::diagnostic::RenderDiagnosticsPlugin,
    );
    app.add_plugins(PerfUiPlugin);
    app.add_perf_ui_simple_entry::<PerfUiEntryViewportBounds>();
    app.add_systems(
        Update,
        (
            scroll_events,
            arrow_events,
            my_cursor_system,
            update_viewport,
            mouse_motion,
            reset_viewport_bounds,
        ),
    );
    app.add_systems(
        PostStartup,
        show_first,
    );
    app.init_resource::<MyWorldCoords>();

    app.add_event::<ViewportBoundsEvent>();
    app.add_systems(
        Startup,
        (
            setup, setup_gpu,
        ),
    );
    app.run();
}
fn setup_gpu(world: &mut World) {
    let gpu = GpuStuff::default();
    world.insert_non_send_resource(gpu);
}

/// We will store the world position of the mouse cursor
/// here.
#[derive(Resource, Default)]
struct MyWorldCoords(Vec2);

/// Used to help identify our main camera
#[derive(Component)]
struct MainCamera;

#[derive(Component, Clone, Reflect)]
struct ViewportBounds {
    re_min: f64,
    re_max: f64,
    im_min: f64,
    im_max: f64,
}
impl std::fmt::Debug for ViewportBounds {
    fn fmt(
        &self,
        f: &mut std::fmt::Formatter<'_>,
    ) -> std::fmt::Result {
        write!(
            f,
            "Min      Max     Range\nReal{:9.2e}{:9.2e}{:10.2e}\nImag{:9.2e}{:9.2e}{:10.2e}",
            self.re_min,
            self.re_max,
            self.re_max - self.re_min,
            self.im_min,
            self.im_max,
            self.im_max - self.im_min,
        )
    }
}

use bevy::ecs::system::SystemParam;
use bevy::ecs::system::lifetimeless::SQuery;
#[derive(Default, Component, Debug, Clone)]
#[require(PerfUiRoot)]
struct PerfUiEntryViewportBounds {
    pub label: String,
    pub sort_key: i32,
}
impl PerfUiEntry for PerfUiEntryViewportBounds {
    type SystemParam = SQuery<&'static ViewportBounds>;
    type Value = ViewportBounds;

    fn label(&self) -> &str {
        if self
            .label
            .is_empty()
        {
            ""
        } else {
            &self.label
        }
    }

    fn sort_key(&self) -> i32 {
        self.sort_key
    }

    fn update_value(
        &self,
        query: &mut <Self::SystemParam as SystemParam>::Item<
            '_,
            '_,
        >,
    ) -> Option<Self::Value> {
        Some(
            query
                .get_single()
                .ok()?
                .clone(),
        )
    }
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
    commands.spawn((
        Camera2d, MainCamera,
    ));

    // create a simple Perf UI with default settings
    // and all entries provided by the crate:
    commands.spawn((
        PerfUiDefaultEntries::default(),
        PerfUiEntryViewportBounds::default(),
    ));
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

fn show_first(
    mut ev_bounds: EventWriter<ViewportBoundsEvent>,
) {
    ev_bounds.send(ViewportBoundsEvent());
}

fn arrow_events(
    keys: Res<ButtonInput<KeyCode>>,
    mut ev_bounds: EventWriter<ViewportBoundsEvent>,
    mut query: Query<&mut ViewportBounds>,
) {
    let mut direction = (
        0i8, 0i8,
    );
    if keys.pressed(KeyCode::ArrowRight)
        || keys.pressed(KeyCode::KeyS)
    {
        direction.0 += 1;
    }
    if keys.pressed(KeyCode::ArrowLeft)
        || keys.pressed(KeyCode::KeyA)
    {
        direction.0 -= 1;
    }
    if keys.pressed(KeyCode::ArrowUp)
        || keys.pressed(KeyCode::KeyW)
    {
        direction.1 -= 1;
    }
    if keys.pressed(KeyCode::ArrowDown)
        || (keys.pressed(KeyCode::KeyR)
            && !keys.pressed(KeyCode::ControlLeft))
    {
        direction.1 += 1;
    }
    if direction.0 != 0 || direction.1 != 0 {
        let mut bounds = query.single_mut();
        let re_range = bounds.re_max - bounds.re_min;
        let im_range = bounds.im_max - bounds.im_min;
        let re_delta =
            PANNING_SPEED * re_range * direction.0 as f64;
        let im_delta =
            PANNING_SPEED * im_range * direction.1 as f64;
        bounds.re_min += re_delta;
        bounds.re_max += re_delta;
        bounds.im_min += im_delta;
        bounds.im_max += im_delta;
        ev_bounds.send(ViewportBoundsEvent());
    }
}

fn scroll_events(
    mycoords: ResMut<MyWorldCoords>,
    mut ev_scroll: EventReader<MouseWheel>,
    mut ev_bounds: EventWriter<ViewportBoundsEvent>,
    mut query: Query<(&mut ViewportBounds,)>,
) {
    use bevy::input::mouse::MouseScrollUnit;
    let zoom_factor = 1.0
        + ev_scroll
            .read()
            .into_iter()
            .map(
                |ev| {
                    if ev.y > 0.0 {
                        -match ev.unit {
                            MouseScrollUnit::Line => {
                                ZOOM_SPEED
                            }
                            MouseScrollUnit::Pixel => {
                                ZOOM_SPEED_FINE
                            }
                        }
                    } else {
                        match ev.unit {
                            MouseScrollUnit::Line => {
                                ZOOM_SPEED
                            }
                            MouseScrollUnit::Pixel => {
                                ZOOM_SPEED_FINE
                            }
                        }
                    }
                },
            )
            .sum::<f64>();
    if zoom_factor != 1.0 {
        let (mut bounds,) = query.single_mut();
        let ViewportBounds {
            re_min,
            re_max,
            im_min,
            im_max,
        } = bounds.clone();

        // mouse position in viewportbounds space
        let re_m = (re_max - re_min)
            * (mycoords
                .0
                .x as f64
                / N_RE as f64)
            + (re_min + re_max) / 2.0;
        let im_m = (im_max - im_min)
            * (-mycoords
                .0
                .y as f64
                / (N_IM as f64 - 1.0))
            + (im_min + im_max) / 2.0;

        bounds.re_min =
            (re_min - re_m) * zoom_factor + re_m;
        bounds.re_max =
            (re_max - re_m) * zoom_factor + re_m;
        bounds.im_min =
            (im_min - im_m) * zoom_factor + im_m;
        bounds.im_max =
            (im_max - im_m) * zoom_factor + im_m;
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
    let (bounds, mut frame_buffer) = query.single_mut();

    if !ev_bounds.is_empty() {
        ev_bounds.clear();

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
                module.burning_ship64<<<grid_size, block_size, 0, stream>>>(
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
            ).unwrap();
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

fn my_cursor_system(
    mut mycoords: ResMut<MyWorldCoords>,
    // query to get the window (so we can read the current
    // cursor position)
    q_window: Query<&Window, With<PrimaryWindow>>,
    // query to get camera transform
    q_camera: Query<
        (
            &Camera,
            &GlobalTransform,
        ),
        With<MainCamera>,
    >,
) {
    // get the camera info and transform
    // assuming there is exactly one main camera entity, so
    // Query::single() is OK
    let (camera, camera_transform) = q_camera.single();

    // There is only one primary window, so we can similarly
    // get it from the query:
    let window = q_window.single();

    // check if the cursor is inside the window and get its
    // position then, ask bevy to convert into world
    // coordinates, and truncate to discard Z
    if let Some(world_position) = window
        .cursor_position()
        .and_then(
            |cursor| {
                camera
                    .viewport_to_world(
                        camera_transform,
                        cursor,
                    )
                    .ok()
            },
        )
        .map(
            |ray| {
                ray.origin
                    .truncate()
            },
        )
    {
        mycoords.0 = world_position;
    }
}

fn mouse_motion(
    mut evr_motion: EventReader<MouseMotion>,
    buttons: Res<ButtonInput<MouseButton>>,
    mut query: Query<&mut ViewportBounds>,
    mut ev_bounds: EventWriter<ViewportBoundsEvent>,
) {
    if buttons.pressed(MouseButton::Left) {
        let mut bounds = query.single_mut();
        let mut re_d = 0.0;
        let mut im_d = 0.0;
        for ev in evr_motion.read() {
            re_d += (bounds.re_max - bounds.re_min)
                * ev.delta
                    .x as f64
                * DRAG_SPEED;
            im_d += (bounds.im_max - bounds.im_min)
                * ev.delta
                    .y as f64
                * DRAG_SPEED;
        }
        bounds.re_min -= re_d;
        bounds.re_max -= re_d;
        bounds.im_min -= im_d;
        bounds.im_max -= im_d;
        ev_bounds.send(ViewportBoundsEvent());
    }
}

fn reset_viewport_bounds(
    keys: Res<ButtonInput<KeyCode>>,
    mut ev_bounds: EventWriter<ViewportBoundsEvent>,
    mut query: Query<&mut ViewportBounds>,
) {
    if keys.just_pressed(KeyCode::KeyR)
        && keys.pressed(KeyCode::ControlLeft)
    {
        let mut bounds = query.single_mut();
        *bounds = ViewportBounds::default();
        ev_bounds.send(ViewportBoundsEvent());
    }
}
