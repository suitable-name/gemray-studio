//! Bit-identity and timing micro-benchmark for the two CPU hot paths: the spectral
//! tracer (a fixed ray set through a Tourmaline round brilliant, 24 bounces) and the
//! meet-point solver (repeated full solves of one mid-size design). Prints wall
//! times, the detected SIMD dispatch level, and two CHECKSUMS that must never change
//! unless a numerical change is deliberate and re-baselined:
//!
//! - tracer: `Vec3(60448.62, 85550.91, 25285.602)`
//! - solver mast checksum: `154.252885863`
//!
//! Run before and after any change to `optics::raytracer`, `simd`, or
//! `geometry::meet_solver` (machine otherwise idle, best of 3 for the timings).
//!
//! ```text
//! cargo run --profile probe -p gemray --example simd_bench
//! ```

use gemray::{
    geometry::cuts::StandardGemCuts,
    optics::{
        materials::GemMaterial,
        raytracer::{LightingPreset, Ray, hash_u32, trace_spectral_ray},
    },
};
use glam::Vec3;
use std::time::Instant;

fn main() {
    println!("simd level: {:?}", gemray::simd::simd_level());

    // ---- Tracer benchmark: fixed ray set, real material, deep bounces. ----
    let planes = StandardGemCuts::standard_round_brilliant();
    let material = GemMaterial::by_name("Tourmaline").expect("built-in material");
    let lighting = LightingPreset::RingLights.studio(1.0, 0.85, 0.95);
    let samples = 60_000u32;
    let start = Instant::now();
    let mut checksum = Vec3::ZERO;
    for s in 0..samples {
        let jitter = (hash_u32(s) as f32) / 4_294_967_295.0;
        let ray = Ray {
            origin: Vec3::new(0.0, 2.5, 0.0),
            dir: Vec3::new(
                0.30f32.mul_add(jitter, -0.15),
                -1.0,
                0.25f32.mul_add(jitter, 0.05),
            )
            .normalize(),
        };
        checksum += trace_spectral_ray(
            ray,
            &planes,
            &material,
            24,
            lighting,
            s,
            (hash_u32(s.wrapping_mul(7919)) as f32) / 4_294_967_295.0,
            None,
        );
    }
    let tracer_elapsed = start.elapsed();
    println!(
        "tracer: {samples} samples in {tracer_elapsed:.2?} ({:.1} ns/sample), checksum {checksum:?}",
        tracer_elapsed.as_nanos() as f64 / f64::from(samples)
    );

    // ---- Solver benchmark: repeated full solves of a mid-size design. ----
    let schedule = lapidary::asc::parse_asc(
        "GemCad 5.0\n\
         g 96 0.0\n\
         y 6 y\n\
         I 1.72\n\
         H Bench design\n\
         a -41.000000 0.64991234 92 n 1 84 76 68 60 52 44 36 28 20 12 4\n\
         a -90.000000 1.07325092 92 n 2 84 76 68 60 52 44 36 28 20 12 4\n\
         a 29.730000 0.65249790 4 n A 12 20 28 36 44 52 60 68 76 84 92\n\
         a 25.000000 0.59508784 96 n B 16 32 48 64 80\n\
         a 10.000000 0.48799664 96 n C 16 32 48 64 80\n\
         a 0.000000 0.44000000 n T\n",
    )
    .expect("bench schedule parses");
    let mut tiers = gemray::geometry::meet_solver::meet_tier_inputs_from_asc(&schedule);
    for i in [0usize, 1, 2] {
        tiers[i].constraint =
            gemray::geometry::meet_solver::MeetConstraint::ScaleReference(schedule.tiers[i].mast);
    }
    let solves = 40u32;
    let start = Instant::now();
    let mut mast_sum = 0.0f64;
    for _ in 0..solves {
        let solved =
            gemray::geometry::meet_solver::solve_meet_points(schedule.gear_teeth_abs(), &tiers);
        mast_sum += solved.iter().map(|s| s.mast).sum::<f64>();
    }
    let solver_elapsed = start.elapsed();
    println!(
        "solver: {solves} solves in {solver_elapsed:.2?} ({:.2} ms/solve), mast checksum {mast_sum:.9}",
        solver_elapsed.as_secs_f64() * 1000.0 / f64::from(solves)
    );
}
