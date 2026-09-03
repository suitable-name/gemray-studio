//! Profile-guided-optimization training workload.
//!
//! Run by `scripts/pgo-build.ps1` inside an instrumented build.
//!
//! Collects a representative execution profile of everything the CPU spends its
//! time on: the spectral tracer across the material kinds that take different code
//! paths (isotropic, coloured uniaxial, biaxial, inclusion scattering, frosted
//! girdle, edge rounding), the À-Trous denoiser and tone-mapper, the meet-point
//! solver's three phases plus the verified repair search, and the external solid
//! measurement. No GPU, no database, no files written -- it only needs to
//! *execute* the hot code with realistic data.
//!
//! Deterministic and self-contained; runs in roughly 10-40 s on a 16-thread
//! desktop. Honours `GEMRAY_SIMD=scalar|avx2` so a portable (scalar) build can
//! be trained on an AVX machine -- see `simd::simd_level`.
//!
//! ```text
//! cargo run --release -p gemray --all-features --example pgo_train
//! ```

use gemray::{
    geometry::{
        GpuFacetPlane, cuts::StandardGemCuts, girdle_facet_finishes, meet_solver, stone_metrics,
    },
    optics::{
        materials::GemMaterial,
        raytracer::{
            Camera, FacetFinish, HERO_WAVELENGTH_ROTATION_STREAM, LightingPreset,
            PIXEL_JITTER_X_ROTATION_STREAM, PIXEL_JITTER_Y_ROTATION_STREAM,
            cranley_patterson_rotate, hash_u32, low_discrepancy_base2, radical_inverse_base,
            trace_spectral_ray_with_finish,
        },
    },
    renderer::{
        denoise::{AtrousDenoiser, AtrousParams, GBuffers},
        tonemap::tonemap_to_rgba,
    },
};
use glam::Vec3;
use std::{
    sync::atomic::{AtomicUsize, Ordering},
    time::Instant,
};

/// One frame's inputs, bundled so the row tracer takes one argument.
struct TraceJob<'a> {
    camera: &'a Camera,
    planes: &'a [GpuFacetPlane],
    finishes: &'a [FacetFinish],
    material: &'a GemMaterial,
    preset: LightingPreset,
    width: u32,
    height: u32,
    spp: u32,
    max_bounces: u32,
}

/// One traced row: radiance sums plus the primary-hit guide values.
struct Row {
    y: usize,
    accum: Vec<Vec3>,
    depth: Vec<f32>,
    normal: Vec<Vec3>,
    facet: Vec<i32>,
}

struct Frame {
    accum: Vec<Vec3>,
    depth: Vec<f32>,
    normal: Vec<Vec3>,
    facet: Vec<i32>,
}

/// Traces one row with the viewer's own seed/jitter construction.
fn trace_row(job: &TraceJob<'_>, y: usize) -> Row {
    let w = job.width as usize;
    let lighting = job.preset.studio(1.0, 0.85, 0.95);
    let mut row = Row {
        y,
        accum: vec![Vec3::ZERO; w],
        depth: vec![1.0e6f32; w],
        normal: vec![Vec3::ZERO; w],
        facet: vec![-1i32; w],
    };
    for x in 0..job.width {
        let pixel = y as u32 * job.width + x;
        let rot_jx = low_discrepancy_base2(hash_u32(pixel ^ PIXEL_JITTER_X_ROTATION_STREAM));
        let rot_jy = low_discrepancy_base2(hash_u32(pixel ^ PIXEL_JITTER_Y_ROTATION_STREAM));
        let rot_hero = low_discrepancy_base2(hash_u32(pixel ^ HERO_WAVELENGTH_ROTATION_STREAM));
        let mut primary = None;
        for sample in 0..job.spp {
            let seed = hash_u32(pixel.wrapping_mul(0x9e37_79b9) ^ sample.wrapping_mul(0x85eb_ca6b));
            let jx = cranley_patterson_rotate(low_discrepancy_base2(sample), rot_jx) - 0.5;
            let jy = cranley_patterson_rotate(radical_inverse_base(sample, 3), rot_jy) - 0.5;
            let hero = cranley_patterson_rotate(radical_inverse_base(sample, 5), rot_hero);
            let ray = job.camera.generate_ray(
                x as f32,
                y as f32,
                job.width as f32,
                job.height as f32,
                jx,
                jy,
            );
            row.accum[x as usize] += trace_spectral_ray_with_finish(
                ray,
                job.planes,
                job.finishes,
                job.material,
                job.max_bounces,
                lighting,
                seed,
                hero,
                Some(&mut primary),
            );
        }
        if let Some(h) = primary {
            row.depth[x as usize] = h.t;
            row.normal[x as usize] = h.normal;
            row.facet[x as usize] = h.facet_idx as i32;
        }
    }
    row
}

/// Traces every row of the frame, rows handed out through a shared counter.
fn trace_frame(job: &TraceJob<'_>) -> Frame {
    let threads = std::thread::available_parallelism().map_or(8, std::num::NonZero::get);
    let next_row = AtomicUsize::new(0);
    let rows: Vec<Row> = std::thread::scope(|s| {
        let handles: Vec<_> = (0..threads)
            .map(|_| {
                let next_row = &next_row;
                s.spawn(move || {
                    let mut out = Vec::new();
                    loop {
                        let y = next_row.fetch_add(1, Ordering::Relaxed);
                        if y >= job.height as usize {
                            break;
                        }
                        out.push(trace_row(job, y));
                    }
                    out
                })
            })
            .collect();
        handles
            .into_iter()
            .flat_map(|h| h.join().expect("trace thread panicked"))
            .collect()
    });
    let pixels = (job.width * job.height) as usize;
    let w = job.width as usize;
    let mut frame = Frame {
        accum: vec![Vec3::ZERO; pixels],
        depth: vec![1.0e6; pixels],
        normal: vec![Vec3::ZERO; pixels],
        facet: vec![-1; pixels],
    };
    for row in rows {
        let span = row.y * w..(row.y + 1) * w;
        frame.accum[span.clone()].copy_from_slice(&row.accum);
        frame.depth[span.clone()].copy_from_slice(&row.depth);
        frame.normal[span.clone()].copy_from_slice(&row.normal);
        frame.facet[span].copy_from_slice(&row.facet);
    }
    frame
}

fn training_materials() -> Vec<(&'static str, GemMaterial, bool)> {
    let by_name = |name: &str| GemMaterial::by_name(name).expect("built-in material");
    vec![
        ("Diamond", GemMaterial::diamond(), false),
        ("Zircon", by_name("Zircon"), true),
        ("Tourmaline", by_name("Tourmaline"), false),
        ("Ruby", by_name("Ruby").with_edge_rounding(0.01), false),
        ("Alexandrite", by_name("Alexandrite"), false),
        (
            "Sapphire+inclusions",
            by_name("Sapphire").with_scattering_amount(0.4),
            true,
        ),
    ]
}

fn train_tracer() -> f32 {
    let (width, height, spp) = (256u32, 192u32, 16u32);
    let cuts = [
        (
            "round brilliant",
            StandardGemCuts::standard_round_brilliant(),
        ),
        ("emerald", StandardGemCuts::emerald_cut()),
    ];
    let materials = training_materials();
    let presets = [LightingPreset::RingLights, LightingPreset::Incandescent];
    let mut denoiser = AtrousDenoiser::new();
    let mut filtered = Vec::new();
    let mut checksum = 0.0f32;
    for (cut_name, planes) in &cuts {
        let girdle = girdle_facet_finishes(planes);
        for (name, material, frosted) in &materials {
            let finishes: &[FacetFinish] = if *frosted { &girdle } else { &[] };
            for (i, preset) in presets.iter().enumerate() {
                let pose = i as f32;
                let camera =
                    Camera::new(pose.mul_add(0.3, 0.6), pose.mul_add(-0.2, 0.45), 2.4, 42.0);
                let job = TraceJob {
                    camera: &camera,
                    planes,
                    finishes,
                    material,
                    preset: *preset,
                    width,
                    height,
                    spp,
                    max_bounces: 24,
                };
                let t0 = Instant::now();
                let frame = trace_frame(&job);
                let avg: Vec<Vec3> = frame.accum.iter().map(|v| *v / spp as f32).collect();
                let g = GBuffers {
                    color: &avg,
                    depth: &frame.depth,
                    normal: &frame.normal,
                    facet_id: &frame.facet,
                    width: width as usize,
                    height: height as usize,
                    spp,
                };
                denoiser.denoise_into(&g, &AtrousParams::default(), &mut filtered);
                let rgba = tonemap_to_rgba(&filtered, 1.0);
                let sum: f32 = rgba.iter().map(|&b| f32::from(b)).sum();
                checksum = sum.mul_add(1e-6, checksum);
                println!(
                    "  tracer: {cut_name} / {name} / {preset:?}: {:.2?}",
                    t0.elapsed()
                );
            }
        }
    }
    checksum
}

const SCHEDULES: [&str; 3] = [
    "GemCad 5.0\ng 96 0.0\ny 6 y\nI 1.72\nH Train A\n\
     a -41.000000 0.64991234 92 n 1 84 76 68 60 52 44 36 28 20 12 4\n\
     a -90.000000 1.07325092 92 n 2 84 76 68 60 52 44 36 28 20 12 4\n\
     a 29.730000 0.65249790 4 n A 12 20 28 36 44 52 60 68 76 84 92\n\
     a 25.000000 0.59508784 96 n B 16 32 48 64 80\n\
     a 10.000000 0.48799664 96 n C 16 32 48 64 80\n\
     a 0.000000 0.44000000 n T\n",
    "GemCad 5.0\ng 96 0.0\ny 8 y\nI 1.54\nH Train B\n\
     a -43.000000 0.70000000 96 n P1 12 24 36 48 60 72 84\n\
     a -41.000000 0.68000000 6 n P2 18 30 42 54 66 78 90\n\
     a -90.000000 1.00000000 96 n G 12 24 36 48 60 72 84\n\
     a -90.000000 1.00000000 6 n G2 18 30 42 54 66 78 90\n\
     a 42.000000 0.72000000 96 n C1 12 24 36 48 60 72 84\n\
     a 27.000000 0.62000000 6 n C2 18 30 42 54 66 78 90\n\
     a 0.000000 0.40000000 n T\n",
    "GemCad 5.0\ng 96 0.0\ny 4 y\nI 1.62\nH Train C\n\
     a -45.000000 0.75000000 96 n 1 24 48 72\n\
     a -40.000000 0.70000000 12 n 2 36 60 84\n\
     a -90.000000 1.05000000 96 n G 24 48 72\n\
     a -90.000000 1.05000000 12 n G2 36 60 84\n\
     a 35.000000 0.70000000 96 n 3 24 48 72\n\
     a 20.000000 0.58000000 12 n 4 36 60 84\n\
     a 0.000000 0.42000000 n T\n",
];

/// Printed-proportion targets measured from the schedule's own recorded masts,
/// so the verified repair search has genuine figures to score against.
fn targets_from_recorded_masts(
    schedule: &lapidary::asc::AscSchedule,
    tiers: &[meet_solver::MeetTierInput],
) -> Option<stone_metrics::ExternalProportions> {
    let normals = meet_solver::tier_instance_normals(schedule.gear_teeth_abs(), tiers);
    let planes: Vec<(glam::DVec3, f64)> = normals
        .iter()
        .zip(&schedule.tiers)
        .flat_map(|(ns, t)| ns.iter().map(move |&n| (n, t.mast)))
        .collect();
    let m = stone_metrics::measure_solid(&planes)?;
    let w = m.width_axis;
    Some(stone_metrics::ExternalProportions {
        vol_w3: Some(m.volume / (w * w * w)),
        lw: Some(m.length_axis / w),
        cw: m.crown_height.map(|c| c / w),
        pw: m.pavilion_depth.map(|p| p / w),
        hw: Some(m.total_height / w),
    })
}

fn train_solver() -> f64 {
    let mut checksum = 0.0f64;
    for (i, text) in SCHEDULES.iter().enumerate() {
        let schedule = lapidary::asc::parse_asc(text).expect("training schedule parses");
        let mut tiers = meet_solver::meet_tier_inputs_from_asc(&schedule);
        // Anchor the first three tiers on their recorded masts, as the validation
        // probe's baseline report does.
        for j in [0usize, 1, 2] {
            tiers[j].constraint =
                meet_solver::MeetConstraint::ScaleReference(schedule.tiers[j].mast);
        }
        let t0 = Instant::now();
        for _ in 0..60 {
            let solved = meet_solver::solve_meet_points(schedule.gear_teeth_abs(), &tiers);
            checksum += solved.iter().map(|s| s.mast).sum::<f64>();
        }
        if let Some(targets) = targets_from_recorded_masts(&schedule, &tiers) {
            for _ in 0..3 {
                let (solved, report) = meet_solver::solve_meet_points_verified(
                    schedule.gear_teeth_abs(),
                    &tiers,
                    &targets,
                    &[],
                );
                checksum += solved.iter().map(|s| s.mast).sum::<f64>() + report.final_score;
            }
        }
        println!("  solver: schedule {i}: {:.2?}", t0.elapsed());
    }
    checksum
}

fn main() {
    println!("pgo_train: simd level {:?}", gemray::simd::simd_level());
    let start = Instant::now();
    let tracer_checksum = train_tracer();
    let solver_checksum = train_solver();
    println!(
        "pgo_train done in {:.1?} (checksums {tracer_checksum:.3} / {solver_checksum:.6})",
        start.elapsed()
    );
}
