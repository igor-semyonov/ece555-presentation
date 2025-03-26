use bevy::diagnostic::FrameTimeDiagnosticsPlugin;
use bevy::ecs::prelude::Event;
use bevy::input::mouse::MouseWheel;
use bevy::prelude::*;
use iyes_perf_ui::prelude::*;

fn main() {
    App::new()
        .add_plugins(DefaultPlugins)
        .add_plugins(bevy::diagnostic::FrameTimeDiagnosticsPlugin::default())
        .add_plugins(bevy::diagnostic::EntityCountDiagnosticsPlugin)
        .add_plugins(bevy::diagnostic::SystemInformationDiagnosticsPlugin)
        .add_plugins(bevy::render::diagnostic::RenderDiagnosticsPlugin)
        .add_plugins(PerfUiPlugin)
        .add_systems(Startup, setup)
        .add_systems(Update, scroll_events)
        .add_systems(Update, update_viewport)
        .add_event::<ViewportBoundsEvent>()
        .run();
}

#[derive(Component, Debug)]
struct ViewportBounds {
    re_min: f32,
    re_max: f32,
    im_min: f32,
    im_max: f32,
}

impl Default for ViewportBounds {
    fn default() -> Self {
        Self {
            re_min: -1.0,
            re_max: 2.0,
            im_min: -1.0,
            im_max: 1.0,
        }
    }
}

#[derive(Event)]
struct ViewportBoundsEvent(Entity);

fn setup(mut commands: Commands) {
    // spawn a camera to be able to see anything
    commands.spawn(Camera2d);

    // create a simple Perf UI with default settings
    // and all entries provided by the crate:
    commands.spawn(PerfUiDefaultEntries::default());
    // commands.spawn(PerfUiAllEntries::default());

    commands.spawn(ViewportBounds::default());
}

fn scroll_events(
    mut evr_scroll: EventReader<MouseWheel>,
    mut ev_bounds: EventWriter<ViewportBoundsEvent>,
    mut query: Query<(Entity, &mut ViewportBounds)>,
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
                zoom_nudge -= 0.5 * ev.y;
            }
            MouseScrollUnit::Pixel => {
                // println!(
                //     "Scroll (pixel units): vertical: {},
                // horizontal: {}",
                //     ev.y, ev.x
                // );
                zoom_nudge -= 0.1 * ev.y;
            }
        }
        zoom_factor += zoom_nudge;
        let (entity, mut bound) = query.single_mut();
        bound.re_min *= zoom_factor;
        bound.re_max *= zoom_factor;
        bound.im_min *= zoom_factor;
        bound.im_max *= zoom_factor;
        ev_bounds.send(ViewportBoundsEvent(entity));
    }
}

fn update_viewport(
    mut _ev_bounds: EventReader<ViewportBoundsEvent>,
    query: Query<&ViewportBounds>,
) {
    let bounds = query.single();
    println!("{:?}", bounds);
}
