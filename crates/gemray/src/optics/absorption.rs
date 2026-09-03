/// A single Gaussian absorption band: a chromophore electronic transition's
/// contribution to the material's absorption coefficient, as a function of wavelength.
///
/// This is the form spectroscopic literature usually publishes gem chromophore data in
/// (a peak wavelength, a width, and a peak intensity), and it is cheap to evaluate
/// per-channel in the ray loop (see `spectral_absorption` below).
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct AbsorptionBand {
    /// Band centre wavelength, nanometers.
    pub center_nm: f32,
    /// Gaussian width (standard deviation), nanometers.
    pub width_nm: f32,
    /// Peak absorption coefficient at `center_nm`.
    pub peak: f32,
}

impl AbsorptionBand {
    #[must_use]
    pub const fn new(center_nm: f32, width_nm: f32, peak: f32) -> Self {
        Self {
            center_nm,
            width_nm,
            peak,
        }
    }

    /// This band's own contribution to the absorption coefficient at `lambda_nm`.
    #[must_use]
    pub fn evaluate(&self, lambda_nm: f32) -> f32 {
        let t = (lambda_nm - self.center_nm) / self.width_nm;
        self.peak * (-0.5 * t * t).exp()
    }
}

/// A gem material's absorption spectrum, expressed as a sum of Gaussian
/// [`AbsorptionBand`]s, one set per birefringent eigenmode (ordinary / extraordinary).
///
/// This replaces the previous representation, a single `[f32; 3]` RGB triple
/// blended against three broad Gaussian lobes fixed on the sRGB primaries (450/540/620
/// nm) in `raytracer::spectral_absorption`. Real gem colour comes from specific,
/// NARROW electronic transitions of transition-metal-ion chromophores (Cr3+, Fe2+/
/// Ti4+, etc.) -- three wide, fixed-position lobes cannot represent a transmission
/// WINDOW that sits between two absorption peaks (e.g. ruby's narrow red window plus a
/// smaller blue one, straddling its two Cr3+ bands), no matter how the RGB triple is
/// tuned, because the lobe positions themselves were never tied to any real chromophore
/// physics. See `raytracer::spectral_absorption` for the band-summing evaluator, and
/// `materials::GemMaterial::all_materials` for the specific band sets and their cited
/// spectroscopic sources.
///
/// The o-ray / e-ray split (`o_ray` / `e_ray`, already present before this task) is
/// kept as two independent band sets rather than collapsed into one, specifically so a
/// follow-up task can replace `trace_spectral_ray`'s current simple
/// `cos_c`/`sin_c`-weighted blend of the two with a proper polarization quadratic form
/// (Beer-Lambert applied along the actual dichroic axes) without having to touch this
/// type again -- each mode's Gaussian bands are already independently addressable.
///
/// Trichroism: `beta_ray` adds an OPTIONAL third independent band set for
/// genuinely biaxial materials, so a material can carry three distinct principal
/// absorption coefficients instead of the uniaxial ordinary/extraordinary pair above.
/// `None` (the default for `isotropic`/`uniaxial`, and for every existing material) is
/// the exact pre-Chapter-05 two-set representation -- `beta_ray` being absent is what
/// tells `raytracer::apply_absorption` to keep taking today's uniaxial
/// `AbsorptionTensor3::uniaxial` path rather than the new three-coefficient
/// `AbsorptionTensor3::biaxial` one, so every existing material (including Topaz, the
/// one biaxial built-in still carrying the uniaxial approximation -- Alexandrite
/// gained its own three-set data in a later pass, see its entry's comment in
/// `materials::GemMaterial::all_materials`) is completely unaffected. The naming
/// convention, pinned end to end by
/// `birefringence::absorption_tensor_tests::convention_pin_o_ray_is_perpendicular_e_ray_is_parallel_to_c_axis`
/// (for the existing two-set case) and mirrored for the new one: `o_ray` -> the
/// `n_alpha` principal coefficient, `beta_ray` -> `n_beta`, `e_ray` -> `n_gamma` -- see
/// `birefringence::AbsorptionTensor3::biaxial`'s doc comment for exactly which world
/// axis each of those lands on.
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct AbsorptionTensor {
    /// Absorption bands for the ordinary ray (uniaxial) / the `n_alpha` principal
    /// direction (biaxial, when `beta_ray` is `Some`).
    pub o_ray: Vec<AbsorptionBand>,
    /// Absorption bands for the extraordinary ray (uniaxial) / the `n_gamma` principal
    /// direction (biaxial, when `beta_ray` is `Some`).
    pub e_ray: Vec<AbsorptionBand>,
    /// The third, `n_beta`, principal direction's absorption bands, for a genuinely
    /// biaxial (trichroic) material. `None` for every uniaxial/isotropic material, and
    /// for a biaxial material still using the two-set ordinary/extraordinary
    /// approximation (Topaz, since Alexandrite's later trichroic data pass) -- see
    /// [`Self::biaxial`].
    pub beta_ray: Option<Vec<AbsorptionBand>>,
    pub is_pleochroic: bool,
}

impl AbsorptionTensor {
    /// A non-pleochroic material: the same band set applies to both eigenmodes (this
    /// is also what an isotropic, cubic-system material -- which has no o-ray/e-ray
    /// distinction at all -- uses, since `trace_spectral_ray` only takes the
    /// anisotropic code path when `crystal_system != Cubic`).
    ///
    /// An empty `Vec` (used by the three colourless built-ins: Diamond, Moissanite,
    /// Cubic Zirconia) means zero absorption at every wavelength -- `spectral_absorption`
    /// sums an empty iterator to `0.0`, identical to the pre-Task-B behaviour for those
    /// three materials (which previously used `[0.0, 0.0, 0.0]`).
    #[must_use]
    pub fn isotropic(bands: Vec<AbsorptionBand>) -> Self {
        Self {
            o_ray: bands.clone(),
            e_ray: bands,
            beta_ray: None,
            is_pleochroic: false,
        }
    }

    #[must_use]
    pub const fn uniaxial(o: Vec<AbsorptionBand>, e: Vec<AbsorptionBand>) -> Self {
        Self {
            o_ray: o,
            e_ray: e,
            beta_ray: None,
            is_pleochroic: true,
        }
    }

    /// A genuinely biaxial (trichroic) material: three independent band sets, one per
    /// principal direction. `alpha`/`beta`/`gamma` map to `o_ray`/`beta_ray`/`e_ray`
    /// respectively -- see [`Self`]'s doc comment and
    /// `birefringence::AbsorptionTensor3::biaxial` for the world-axis convention this
    /// feeds.
    ///
    /// Degenerate case: calling this with `alpha` and `beta` equal (as `Vec<AbsorptionBand>`
    /// values, i.e. the same bands) produces a tensor that evaluates IDENTICALLY to
    /// calling [`Self::uniaxial`] with `(alpha, gamma)` at every wavelength, since
    /// `beta_ray` then holds the same band data as `o_ray` -- there is no special-cased collapse here;
    /// the equivalence falls out of `AbsorptionTensor3::biaxial` and `::uniaxial`
    /// sharing the same axis construction (see that type's doc comment and
    /// `birefringence_reduction_tests` for the pinned proof at the tensor-evaluation
    /// level).
    #[must_use]
    pub const fn biaxial(
        alpha: Vec<AbsorptionBand>,
        beta: Vec<AbsorptionBand>,
        gamma: Vec<AbsorptionBand>,
    ) -> Self {
        Self {
            o_ray: alpha,
            e_ray: gamma,
            beta_ray: Some(beta),
            is_pleochroic: true,
        }
    }
}

/// Builds a 3-band absorption set from an `[R, G, B]` peak-coefficient triple.
///
/// Uses the SAME band centres/widths (620/540/450 nm, widths 55/45/45 nm) that
/// `raytracer::spectral_absorption`'s pre-Task-B three-lobe model used.
///
/// Used ONLY for the built-in gem species this task's brief did not supply cited
/// chromophore spectroscopy for: Spinel, Zircon, Alexandrite, Topaz, Tourmaline and
/// Tanzanite (see each one's doc comment in `materials::GemMaterial::all_materials`).
/// Rather than fabricate absorption-band wavelengths for those six species without a
/// source, this keeps their existing (already-tuned-for-plausible-appearance, if
/// uncited) RGB peak values and the old model's band positions, just re-expressed in
/// the new `Vec<AbsorptionBand>` shape -- an explicitly-labelled placeholder, not a
/// physical claim about where those species' real chromophores absorb. A follow-up
/// task should source real chromophore data for them the way this task did for Ruby
/// (Cr3+), Emerald (Cr3+) and Sapphire (Fe2+-Ti4+ IVCT).
///
/// NOTE: this is NOT numerically identical to the old model's output. The old
/// `spectral_absorption` computed a NORMALIZED weighted blend of the three `rgb`
/// values (dividing by the local sum of the three Gaussian weights, so e.g. `rgb[0]`
/// alone was recovered exactly at 620 nm); this instead SUMS the three bands as
/// independent physical contributions (consistent with how the real chromophore bands
/// above are summed, and with how real absorption spectra actually combine), so
/// overlapping bands compound rather than blend. This intentionally changes the
/// absorption model for every material; only the three colourless built-ins (Diamond,
/// Moissanite, Cubic Zirconia, via an empty `Vec`) are required to render bit-identically
/// to before.
#[must_use]
pub fn legacy_rgb_bands(rgb: [f32; 3]) -> Vec<AbsorptionBand> {
    vec![
        AbsorptionBand::new(620.0, 55.0, rgb[0]),
        AbsorptionBand::new(540.0, 45.0, rgb[1]),
        AbsorptionBand::new(450.0, 45.0, rgb[2]),
    ]
}
