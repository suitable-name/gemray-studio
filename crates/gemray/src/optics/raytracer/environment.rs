//! Environment/lighting sources a ray can sample when it misses the gemstone.
//!
//! The analytic gemological studio rig ([`LightingPreset`], [`sample_studio_environment`])
//! and the loaded-HDR-panorama alternative ([`EnvironmentSource::HdrMap`]).

use super::color::illuminant_white_balance;
use crate::renderer::env_map::EnvironmentMap;
use glam::Vec3;

/// Physical Planck Blackbody Spectral Radiance S(lambda, T) normalized to 1.0 at 560nm
#[must_use]
pub fn blackbody_spectrum(lambda_nm: f32, temp_k: f32) -> f32 {
    let t_k = temp_k.max(1000.0);
    let h_c_k = 14_388_000.0_f32; // hc / k_B in nm * K
    let exp_val = (h_c_k / (lambda_nm * t_k)).min(80.0).exp();
    let exp_560 = (h_c_k / (560.0 * t_k)).min(80.0).exp();
    let denom = (exp_val - 1.0).max(1e-6);
    let denom_560 = (exp_560 - 1.0).max(1e-6);
    let ratio = denom_560 / denom;
    ((560.0 / lambda_nm).powi(5) * ratio).clamp(0.01, 20.0)
}

/// Colour temperature and rig-intensity parameters for one named studio lighting preset.
///
/// Returned by [`LightingPreset::params`] -- the single lookup both
/// `sample_studio_environment` (which lights the traced image) and
/// `illuminant_temperature_k` (which derives the von-Kries white balance for that same
/// image) now share, so the two can no longer independently drift the way the old twin
/// `match lighting_preset: &str` blocks did. Concretely: the "D65 Daylight" preset used
/// to DISPLAY as `"D65 Daylight (5500K)"` in the UI while every numeric branch (both
/// matches' `_` fallback arm) actually used 6500K -- the label was simply wrong, and
/// nothing would have caught a *numeric* drift between the two matches either. A single
/// `LightingPreset::params()` call makes both classes of drift structurally impossible:
/// there is now exactly one place either quantity is defined.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct LightingRigParams {
    /// Blackbody colour temperature in Kelvin, fed to [`blackbody_spectrum`].
    pub temp_k: f32,
    /// Multiplier on the key-softbox and ring-emitter intensity terms (does not affect
    /// the fill light or the ambient backdrop).
    pub spot_mult: f32,
}

/// The gemological studio lighting rig presets.
///
/// Replaces the previous `&str`-keyed `match` in
/// `sample_studio_environment`/`illuminant_temperature_k` (which had to agree with each
/// other, and with the UI's own separate `[string]` options list, purely by
/// programmer-maintained convention) with a closed, exhaustively-matched set of
/// variants -- an unrecognised preset is no longer representable, so a caller can no
/// longer pass a string that silently falls through to a default nobody intended.
///
/// `Daylight` is index `0` / the [`Default`] (matches the old `_` catch-all arm's D65
/// 6500K behaviour, and is what any legacy or unrecognised persisted label -- including
/// the old, mislabelled `"D65 Daylight (5500K)"` string -- migrates to via
/// [`Self::from_label`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Default)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum LightingPreset {
    #[default]
    Daylight,
    Incandescent,
    RingLights,
    DarkSpotlight,
}

impl LightingPreset {
    /// All four presets, in the same order as their UI index / the `lighting_options`
    /// combo box list (`app.slint`).
    pub const ALL: [Self; 4] = [
        Self::Daylight,
        Self::Incandescent,
        Self::RingLights,
        Self::DarkSpotlight,
    ];

    /// This preset's colour temperature and rig-intensity multiplier -- the single
    /// source of truth both `sample_studio_environment` and `illuminant_temperature_k`
    /// read from. See the type's doc comment for why that matters.
    #[must_use]
    pub const fn params(self) -> LightingRigParams {
        match self {
            Self::Incandescent => LightingRigParams {
                temp_k: 3200.0,
                spot_mult: 1.2,
            },
            Self::RingLights => LightingRigParams {
                temp_k: 5000.0,
                spot_mult: 1.6,
            },
            Self::DarkSpotlight => LightingRigParams {
                temp_k: 6000.0,
                spot_mult: 2.4,
            },
            Self::Daylight => LightingRigParams {
                temp_k: 6500.0,
                spot_mult: 1.0,
            },
        }
    }

    /// The corrected user-facing display label: D65 daylight is 6500K, not the
    /// `"5500K"` the UI previously (and inconsistently with the actually-rendered
    /// colour) displayed.
    #[must_use]
    pub const fn label(self) -> &'static str {
        match self {
            Self::Daylight => "D65 Daylight (6500K)",
            Self::Incandescent => "Incandescent (3200K)",
            Self::RingLights => "Gem Studio Ring Lights",
            Self::DarkSpotlight => "Dramatic Dark Spotlight",
        }
    }

    /// Parses a persisted or UI-supplied label back into a preset. Falls back to
    /// [`Self::Daylight`] for anything unrecognised -- including the legacy
    /// `"D65 Daylight (5500K)"` label a settings file saved before this fix may still
    /// contain, which is exactly the preset that label already resolved to (D65
    /// 6500K), so an old settings file migrates silently rather than resetting the
    /// user's lighting choice to something else.
    #[must_use]
    pub fn from_label(label: &str) -> Self {
        match label {
            "Incandescent (3200K)" => Self::Incandescent,
            "Gem Studio Ring Lights" => Self::RingLights,
            "Dramatic Dark Spotlight" => Self::DarkSpotlight,
            _ => Self::Daylight,
        }
    }

    /// The index into [`Self::ALL`] / the UI combo box's `lighting_options` list.
    #[must_use]
    pub const fn index(self) -> i32 {
        match self {
            Self::Daylight => 0,
            Self::Incandescent => 1,
            Self::RingLights => 2,
            Self::DarkSpotlight => 3,
        }
    }

    /// Inverse of [`Self::index`]; out-of-range indices fall back to [`Self::Daylight`]
    /// (index 0), matching [`Self::from_label`]'s fallback.
    #[must_use]
    pub const fn from_index(index: i32) -> Self {
        match index {
            1 => Self::Incandescent,
            2 => Self::RingLights,
            3 => Self::DarkSpotlight,
            _ => Self::Daylight,
        }
    }

    /// Convenience constructor for the common case of tracing against the analytic
    /// studio rig: `LightingPreset::RingLights.studio(1.0, 0.85, 0.95)` reads at the
    /// call site much like the old positional `&str` argument list did.
    #[must_use]
    pub const fn studio(
        self,
        exposure: f32,
        light_yaw: f32,
        light_pitch: f32,
    ) -> EnvironmentSource<'static> {
        EnvironmentSource::Studio {
            preset: self,
            exposure,
            light_yaw,
            light_pitch,
        }
    }
}

/// Selects what `trace_spectral_ray` samples when a ray misses the gemstone.
///
/// Either the analytic studio rig (`Studio`, the default -- see
/// `sample_studio_environment`) or a loaded HDR equirectangular panorama (`HdrMap`, via
/// [`crate::renderer::env_map::EnvironmentMap`]).
///
/// The analytic rig stays available even once HDR loading is wired end to end: it is
/// genuinely useful for controlled comparisons where a real photograph would introduce
/// variables (its own exposure, white balance, capture noise) a study wants held
/// constant. No HDR file ships with this project today, and there is no UI file picker
/// yet -- `apps/diagram-gui/src/bridge/render_thread.rs`'s render-loop call site is
/// exactly where that picker's selection would plug in (it would just choose which
/// variant of this enum to build instead of always building `Studio`); building the
/// picker itself is out of scope here.
#[derive(Clone, Copy)]
pub enum EnvironmentSource<'a> {
    Studio {
        preset: LightingPreset,
        exposure: f32,
        light_yaw: f32,
        light_pitch: f32,
    },
    HdrMap(&'a EnvironmentMap),
}

/// Looks up channel `lambda_nm`'s spectral radiance, in direction `dir`, for a ray that
/// missed the gemstone and is now sampling `environment`. Pulled out of
/// `trace_spectral_ray`'s miss branch so the `HdrMap` arm doesn't grow that
/// already-oversized function -- see [`EnvironmentSource`]'s doc comment for the design.
///
/// Takes the `Studio` variant's [`StudioRig`](crate::optics::studio_rig::StudioRig)
/// pre-built (`studio_rig`) rather than reconstructing it from `light_yaw`/
/// `light_pitch` on every call -- see [`accumulate_miss_radiance`], the only caller,
/// which builds it once per ray and reuses it across all `NUM_CHANNELS` channels.
/// `studio_rig` is meaningless (and unused) for `HdrMap`.
#[inline]
pub(super) fn sample_environment_channel(
    environment: EnvironmentSource<'_>,
    dir: Vec3,
    lambda_nm: f32,
    studio_rig: Option<&crate::optics::studio_rig::StudioRig>,
) -> f32 {
    match environment {
        EnvironmentSource::Studio {
            preset, exposure, ..
        } => {
            let rig = studio_rig
                .expect("sample_environment_channel: Studio environment needs a pre-built rig");
            sample_studio_environment_with_rig(dir, lambda_nm, preset, exposure, rig)
        }
        EnvironmentSource::HdrMap(map) => map.radiance_at(dir, lambda_nm),
    }
}

/// The von-Kries white-balance scale (Bradford LMS-space, per-cone -- see
/// [`compute_illuminant_white_balance`]) [`trace_spectral_ray`] applies, via
/// [`apply_von_kries_white_balance`], to its final XYZ integration for `environment`.
/// Only the analytic studio rig has a single well-defined illuminant colour temperature
/// to neutralize against -- a loaded HDR panorama is a full spectral image with no one
/// blackbody temperature standing in for it, so this deliberately applies NO
/// white-balance correction (`Vec3::ONE`, the LMS-space identity scale) for `HdrMap`.
/// Proper HDR white-balancing (e.g. against the map's own average/dominant
/// chromaticity) is a reasonable follow-up once the HDR path has a real file and a UI
/// hook to test it against -- see [`EnvironmentSource`]'s doc comment.
#[inline]
pub(super) fn environment_white_balance(environment: EnvironmentSource<'_>) -> Vec3 {
    match environment {
        EnvironmentSource::Studio { preset, .. } => illuminant_white_balance(preset),
        EnvironmentSource::HdrMap(_) => Vec3::ONE,
    }
}

/// Evaluates high-dynamic-range gemological studio lighting at a specific continuous
/// wavelength `lambda_nm`.
///
/// Builds a fresh [`StudioRig`](crate::optics::studio_rig::StudioRig) every call, which
/// is exactly right for a single ad-hoc lookup (this function's public callers, and
/// `color::metrics::evaluate_gem_optical_metrics`) but was also -- until this fix --
/// what every one of `trace_spectral_ray`'s `NUM_CHANNELS` per-bounce environment
/// lookups did too, rebuilding the identical (`light_yaw`, `light_pitch`)-derived rig up to
/// 8 times over for values that never change within a single ray. See
/// [`sample_studio_environment_with_rig`], which factors out the rig-independent body
/// this delegates to, and its caller `accumulate_miss_radiance`, which now builds the
/// rig exactly once per ray and reuses it across all `NUM_CHANNELS` channels.
#[must_use]
pub fn sample_studio_environment(
    dir: Vec3,
    lambda_nm: f32,
    lighting_preset: LightingPreset,
    exposure: f32,
    light_yaw: f32,
    light_pitch: f32,
) -> f32 {
    // Key/fill/ring directions come from the shared `StudioRig` (see its module doc
    // for why this is not recomputed inline here) -- the SAME construction
    // `color::metrics::evaluate_gem_optical_metrics` uses to score the image this
    // function lights, so the two can never silently drift apart.
    let rig = crate::optics::studio_rig::StudioRig::new(light_yaw, light_pitch);
    sample_studio_environment_with_rig(dir, lambda_nm, lighting_preset, exposure, &rig)
}

/// The rig-independent body of [`sample_studio_environment`]: identical arithmetic, in
/// the identical order, just reading `key_dir`/`fill_dir`/`ring_dirs`/`sin_light_pitch`
/// off an already-built `rig` instead of constructing one from `(light_yaw,
/// light_pitch)` itself. A direct extraction -- see that function's doc comment for
/// why.
#[must_use]
fn sample_studio_environment_with_rig(
    dir: Vec3,
    lambda_nm: f32,
    lighting_preset: LightingPreset,
    exposure: f32,
    rig: &crate::optics::studio_rig::StudioRig,
) -> f32 {
    let d = dir.normalize();

    let LightingRigParams { temp_k, spot_mult } = lighting_preset.params();
    let spec_power = blackbody_spectrum(lambda_nm, temp_k);

    // 1. Ambient luxury studio backdrop (pure neutral dark charcoal velvet)
    let bg_val = 0.012f32.mul_add(d.y.mul_add(0.5, 0.5), 0.015).max(0.005) * exposure;
    let mut radiance = bg_val * spec_power;

    // 2. Main Key Softbox Light
    let key_dot = d.dot(rig.key_dir).max(0.0);
    if key_dot > 0.0 {
        let softbox = key_dot.powi(28) * 12.0 * spot_mult * exposure;
        radiance = softbox.mul_add(spec_power, radiance);
    }

    // 3. Fill Softbox Light (side reflector offset by 140 deg)
    let fill_dot = d.dot(rig.fill_dir).max(0.0);
    if fill_dot > 0.0 {
        let fill = fill_dot.powi(18) * 4.5 * exposure;
        radiance = fill.mul_add(spec_power, radiance);
    }

    // 4. Circular Ring Scintillation Lights (16 sparkling pinpoint sources rotating with lighting rig)
    for ring_dir in rig.ring_dirs {
        let ring_dot = d.dot(ring_dir).max(0.0);
        if ring_dot > 0.96 {
            let spark = (ring_dot - 0.96) / 0.04;
            let intensity = spark.powi(6) * 22.0 * spot_mult * exposure;
            radiance = intensity.mul_add(spec_power, radiance);
        }
    }

    radiance
}
