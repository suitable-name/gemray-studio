use super::{
    absorption::{AbsorptionBand, AbsorptionTensor, legacy_rgb_bands},
    birefringence::BiaxialIndicatrix,
    dispersion::DispersionModel,
};
use glam::Vec3;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum CrystalSystem {
    Cubic,
    Tetragonal,
    Hexagonal,
    Trigonal,
    Orthorhombic,
    Monoclinic,
    Triclinic,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum OpticalCharacter {
    Isotropic,
    UniaxialPositive,
    UniaxialNegative,
    BiaxialPositive,
    BiaxialNegative,
}

#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct GemMaterial {
    pub name: String,
    pub crystal_system: CrystalSystem,
    pub optical_character: OpticalCharacter,
    pub dispersion: DispersionModel,
    pub birefringence_delta: f32, // Difference between n_e and n_o (or max-min)
    pub absorption: AbsorptionTensor,
    /// Optical (crystallographic) c-axis direction, in crystal/model space. Uniaxial
    /// birefringence (`effective_extraordinary_index`, `extraordinary_poynting_dir`)
    /// is evaluated against this axis. Defaults to `Vec3::Y` for every material below
    /// and for `new_custom`, matching the previously-hardcoded value in
    /// `trace_spectral_ray`, so existing behaviour is unchanged. For a biaxial material
    /// (`biaxial_delta_beta_alpha.is_some()`) this doubles as the `n_gamma` principal
    /// axis -- see `biaxial_indicatrix`.
    pub c_axis: Vec3,
    /// `n_beta - n_alpha` at the sodium D line, for the three biaxial
    /// (orthorhombic) built-ins -- Alexandrite, Topaz, Tanzanite. `None` for every
    /// isotropic/uniaxial material (everything else): those are completely unaffected
    /// and keep using the existing `c_axis` + `birefringence_delta` uniaxial machinery.
    ///
    /// This crate's existing convention already reads `birefringence_delta` as
    /// `n_gamma - n_alpha` for these three entries (its own doc comment above says "or
    /// max-min" for exactly this reason, and the Alexandrite entry's comment derives it
    /// that way directly from primary Sellmeier data). Combined with that field and the
    /// base `dispersion` curve -- read as `n_beta(lambda)`, the MIDDLE principal index,
    /// for a biaxial material (see `biaxial_indicatrix`; this is not a new convention,
    /// the Alexandrite entry's comment already notes its preserved `n_d` numerically
    /// IS `n_beta(opt)`) -- this one extra scalar places all three principal indices
    /// `n_alpha <= n_beta <= n_gamma` via the achromatic-delta approximation the
    /// existing uniaxial `n_e = n_o + birefringence_delta` already makes (i.e. the
    /// SPREAD between principal indices is treated as wavelength-independent, only the
    /// base curve itself disperses).
    pub biaxial_delta_beta_alpha: Option<f32>,
    /// Inclusion/subsurface scattering: the homogeneous Henyey-Greenstein
    /// scattering coefficient (`sigma_s`, a PHYSICAL, LINEAR coefficient in inverse model
    /// units of path length -- no perceptual/logarithmic remapping happens here; that
    /// belongs in a future UI slider layered on top, not in this field) modeling silk,
    /// rutile needles, and clouds as a single averaged-out volumetric density -- the
    /// "cheap 80%" version of inclusion scattering, not a discrete-particle simulation.
    ///
    /// `0.0` (every built-in material's OWN stored value, and `new_custom`'s) means no
    /// scattering medium at all: `raytracer::apply_absorption`'s exact pre-existing
    /// deterministic Beer-Lambert path is taken unconditionally whenever this is `<=
    /// 0.0`, so every existing material/scene renders bit-identically to before this
    /// field existed. See `raytracer::maybe_scatter_or_extinguish`'s doc comment for the
    /// estimator this field feeds once nonzero (extinction `sigma_t = sigma_a +
    /// sigma_s`, free-path distance sampling, single-scattering albedo `sigma_s /
    /// sigma_t`). Use [`Self::with_scattering`] (or [`Self::with_recommended_scattering`]
    /// for a per-species starting point -- see that method's doc comment for why its
    /// numbers are plausible aesthetic choices, not measurements) to opt a material into
    /// a nonzero value.
    ///
    /// # Useful range
    ///
    /// The built-in cuts (`geometry::StandardGemCuts`) have a girdle radius of order 1
    /// model unit, so a typical internal chord length is roughly 0.5-2 units and the
    /// mean free path between scattering events is `1/sigma_s`. `sigma_s` in roughly
    /// `0.05` (barely perceptible haze) to `3.0` (milky/heavily included) covers the
    /// visually meaningful range for this geometry scale; a linear UI control spanning
    /// much wider than that would spend most of its travel doing nothing visible.
    pub scattering_sigma_s: f32,
    /// The Henyey-Greenstein phase function's asymmetry parameter `g` in `(-1, 1)`: `0`
    /// is isotropic scattering, positive values forward-scatter (light continues mostly
    /// in its original direction -- silk/rutile needle inclusions are usually
    /// forward-scattering), negative values back-scatter. Meaningless (never read) while
    /// `scattering_sigma_s <= 0.0`; defaults to `0.0` alongside it for every built-in
    /// material. See [`Self::DEFAULT_SCATTERING_G`] for a sensible default when a caller
    /// only wants to control the amount ([`Self::with_scattering_amount`]).
    pub scattering_g: f32,
    /// Facet edge rounding: the micron-scale rounding radius
    /// real meet-point edges have, in the SAME world units as `scattering_sigma_s`'s own
    /// useful-range doc comment (girdle radius of order 1 model unit) -- so a value like
    /// `0.01` models an edge rounded over about 1% of the stone's own scale, comfortably
    /// in the "throws a soft glint, does not visibly bevel the facet" range. `0.0`
    /// (every built-in material) disables the effect entirely:
    /// `raytracer::shading_normal_near_edge` returns the flat facet normal completely
    /// unperturbed, bit-identically, whenever this is `<= 0.0`. See
    /// [`Self::with_edge_rounding`] to opt a material in.
    pub edge_rounding_radius: f32,
    /// Model units to absorption-length units: every interior path length is
    /// multiplied by this before Beer-Lambert absorption and inclusion scattering.
    /// `1.0` (every built-in material below, and `new_custom`'s default) is exactly
    /// today's behaviour -- multiplying a path length by exactly `1.0` is an IEEE
    /// 754 no-op, so every existing render stays bit-identical. See
    /// [`Self::with_absorption_path_scale`] to opt a material into a different
    /// physical size (e.g. a larger or smaller real-world stone rendered at the
    /// same ~1-model-unit girdle radius as every built-in cut).
    pub absorption_path_scale: f32,
}

impl GemMaterial {
    #[must_use]
    pub fn all_materials() -> Vec<Self> {
        let mut materials = Self::built_in_materials_diamond_through_emerald();
        materials.extend(Self::built_in_materials_zircon_through_topaz());
        materials.extend(Self::built_in_materials_spinel_through_tourmaline());
        materials.extend(Self::built_in_materials_tanzanite_through_cubic_zirconia());
        materials
    }

    /// First quarter of the built-in material table (Diamond through Emerald). Split
    /// out of `all_materials` purely to keep each part under clippy's function-length
    /// lint -- this is plain data, not logic, so the split point carries no
    /// significance.
    fn built_in_materials_diamond_through_emerald() -> Vec<Self> {
        vec![
            // Diamond (C)
            //
            // Source: R. Peter, Z. Phys. 15, 358 (1923) -- the standard 2-term Sellmeier
            // fit for diamond, as tabulated by refractiveindex.info ("Diamond: n
            // (Peter 1923)"), valid 0.226-0.760 um:
            //   n^2 - 1 = 4.3356*lambda^2/(lambda^2-0.1060^2) + 0.3306*lambda^2/(lambda^2-0.1750^2)
            // The previous entry here kept only the FIRST term (b1=4.3356,
            // c1=0.011236) and silently dropped the second (b2=0.3306, c2=0.030625),
            // which understated both n_d (2.341 vs the accepted ~2.417-2.424) and
            // Delta n(F-C) (0.02139 vs the ~29% larger two-term value below) by a wide
            // margin -- not a rounding error, a missing term. Represented here via
            // Sellmeier3's 3-pole form with the 3rd pole zeroed out (b=0, c=1.0 mu m^2,
            // safely outside the visible-range l^2 domain so 0/denominator never
            // arises) since DispersionModel has no native 2-term Sellmeier variant.
            // Verified (n(486.1), n(589.3), n(656.3) via this exact formula):
            //   n_d = 2.41726, Delta n(F-C) = 0.02564, Abbe V_d = 55.27
            // -- matches the commonly-cited literature figures n_D ~ 2.417, V_d ~ 55.3.
            // See `materials::tests::builtin_material_abbe_numbers_match_published_values`.
            Self {
                name: "Diamond".to_string(),
                crystal_system: CrystalSystem::Cubic,
                optical_character: OpticalCharacter::Isotropic,
                dispersion: DispersionModel::Sellmeier3 {
                    b: [4.3356, 0.3306, 0.0],
                    c: [0.011_236, 0.030_625, 1.0],
                },
                birefringence_delta: 0.0,
                // Colourless (no chromophore): empty band set, zero absorption at every
                // wavelength -- must render identically to before.
                absorption: AbsorptionTensor::isotropic(vec![]),
                c_axis: Vec3::Y,
                biaxial_delta_beta_alpha: None,
                scattering_sigma_s: 0.0,
                scattering_g: 0.0,
                edge_rounding_radius: 0.0,
                absorption_path_scale: 1.0,
            },
            // Sapphire (Al2O3)
            //
            // Source: I.H. Malitson & M.J. Dodge, J. Opt. Soc. Am. 62, 1405A (1972),
            // ordinary-ray Sellmeier fit for sapphire, as tabulated by
            // refractiveindex.info ("Al2O3: Malitson-o"), valid 0.2-5.0 um:
            //   n^2-1 = 1.4313493 l^2/(l^2-0.0726631^2) + 0.65054713 l^2/(l^2-0.1193242^2)
            //           + 5.3414021 l^2/(l^2-18.028251^2)
            // Re-verified (this pass): coefficients match Malitson & Dodge to 5-6
            // significant figures -- n_d = 1.76808, Delta n(F-C) = 0.01063,
            // Abbe V_d = 72.27, matching the standard literature figure
            // V_d(sapphire) ~ 72. GEMSTONE_RENDERING_BLUEPRINT.md's "0.0112" Delta n
            // figure for this row is not reproducible from any primary optical-constants
            // source; it is consistent with a rounded gemological (B-G Fraunhofer-line)
            // dispersion figure (corundum is widely tabulated at B-G ~0.018, and
            // 0.018 * ~0.58 (see the birefringence-ratio derivation methodology used for
            // the un-sourced species below) = 0.0104, close to the document's 0.0112) --
            // not the classical F-C interval computed here. Per this task's source
            // priority (primary optical measurement outranks the document), the
            // Malitson & Dodge o-ray figure stands as-is.
            //
            // birefringence_delta CORRECTED this pass: re-derived directly from
            // Malitson & Dodge's own companion e-ray Sellmeier fit (same 1972 paper,
            // refractiveindex.info "Al2O3: Malitson-e",
            // https://refractiveindex.info/?shelf=main&book=Al2O3&page=Malitson-e,
            // raw coefficients at
            // github.com/polyanskiy/refractiveindex.info-database/blob/master/database/data/main/Al2O3/nk/Malitson-e.yml):
            //   n_e^2 = 1 + 1.5039759*l^2/(l^2-0.0740288^2) + 0.55069141*l^2/(l^2-0.1216529^2)
            //           + 6.5927379*l^2/(l^2-20.072248^2)
            // giving n_e(D) = 1.760002 against this entry's n_o(D) = 1.768106, i.e.
            // n_e - n_o = -0.008104 at the sodium D line -- both indices from the SAME
            // primary paper, not from the document. Previous value (-0.0082, copied from
            // the document's rounded n_o/n_e pair) was already correct to within 0.0001;
            // replaced with this directly-primary-sourced figure for precision.
            // See `materials::tests::builtin_material_abbe_numbers_match_published_values`.
            Self {
                name: "Sapphire".to_string(),
                crystal_system: CrystalSystem::Trigonal,
                optical_character: OpticalCharacter::UniaxialNegative,
                dispersion: DispersionModel::Sellmeier3 {
                    b: [1.431_349, 0.650_547, 5.341_402],
                    c: [0.00528, 0.01424, 325.015],
                },
                birefringence_delta: -0.0081,
                // Chromophore source: blue sapphire's colour comes from a broad
                // Fe2+-Ti4+ intervalence charge-transfer (IVCT) band centred ~580nm
                // (yellow), absorbing yellow-orange-red and transmitting blue -- see
                // e.g. "Fe-Ti Charge Transfer: The Mechanism Behind Sapphire's Blue"
                // (skyjems.ca gemology encyclopedia), corroborated by academic
                // literature on Fe/Ti oxidation states in blue sapphire. Modelled as a
                // single broad Gaussian (width tuned wide, 90nm, to match the band's
                // published breadth); amplitude (peak=3.0) tuned for plausible
                // saturation at a typical internal path length in this renderer's gem
                // geometry (unitless facet-polyhedron scale, characteristic internal
                // path ~0.5-2 units).
                //
                // PLEOCHROISM populated this pass, replacing the isotropic placeholder
                // above with a genuine uniaxial `o_ray`/`e_ray` split -- sapphire's
                // biggest win, a real ~70nm band-CENTRE shift between the two rays, not
                // just an amplitude ratio. Source: A.J. Emmett, M. Dubinsky, R.
                // Hughes & M. Scarratt, "The Colors of Sapphires," Gems & Gemology
                // 56(1), Spring 2020: "For E-perp-c the [Fe2+-Ti4+ IVCT] band peaks at
                // 580 nm, while for E-parallel-c the peak is at 700 nm" (their notation:
                // E-perp-c is the o-ray/omega spectrum, E-parallel-c is the e-ray/
                // epsilon spectrum -- this crate's convention throughout). Same paper
                // reports the E-perp-c peak cross-section as 1.94e-18 cm^2 +/-25%,
                // consistent with this entry's already-tuned o-ray amplitude (peak=3.0,
                // UNCHANGED from the pre-existing isotropic entry above, per this task's
                // continuity principle -- the o-ray dominates the face-up/c_axis=Y hero
                // shot, so keeping it identical means the change is almost purely
                // additive for that view).
                //
                // e_ray band CENTRE (700nm) is the cited, non-tuned value. e_ray WIDTH
                // (95nm, vs the o-ray's 90nm) is tuned -- Emmett et al.'s Fig. 10 shows
                // the E-parallel-c lobe as slightly broader than the E-perp-c one, no
                // numeric width is given in text. e_ray AMPLITUDE (peak=2.1, a ~0.7x
                // ratio to the o-ray's 3.0) is FIGURE-READ, not stated in the paper's
                // text: eyeballed off the relative peak heights of the two curves in
                // Emmett et al.'s Fig. 10 (E-parallel-c visibly the smaller of the two
                // peaks there) -- flagged as the lowest-confidence number in this entry,
                // a candidate for refinement if a numeric E-parallel-c cross-section is
                // ever sourced.
                absorption: AbsorptionTensor::uniaxial(
                    vec![AbsorptionBand::new(580.0, 90.0, 3.0)], // o-ray (E-perp-c)
                    vec![AbsorptionBand::new(700.0, 95.0, 2.1)], // e-ray (E-parallel-c), IVCT at 700nm
                ),
                c_axis: Vec3::Y,
                biaxial_delta_beta_alpha: None,
                scattering_sigma_s: 0.0,
                scattering_g: 0.0,
                edge_rounding_radius: 0.0,
                absorption_path_scale: 1.0,
            },
            // Ruby (Al2O3:Cr) -- same host lattice as Sapphire above (trace Cr3+
            // substitution at the sub-1% level does not measurably shift the host
            // Al2O3 dispersion), so it shares the identical Malitson & Dodge (1972)
            // Sellmeier fit. See the Sapphire entry's comment for the source and the
            // verified n_d / Delta n(F-C) / Abbe numbers (unchanged here), and for the
            // birefringence_delta re-derivation from Malitson & Dodge's own e-ray fit
            // (-0.0081, corrected this pass from the document-rounded -0.0082).
            Self {
                name: "Ruby".to_string(),
                crystal_system: CrystalSystem::Trigonal,
                optical_character: OpticalCharacter::UniaxialNegative,
                dispersion: DispersionModel::Sellmeier3 {
                    b: [1.431_349, 0.650_547, 5.341_402],
                    c: [0.00528, 0.01424, 325.015],
                },
                birefringence_delta: -0.0081,
                // Chromophore source: ruby's Cr3+ absorbs in two bands -- a
                // violet band near 410nm and a yellow-green band near 550nm (the two
                // spin-allowed Cr3+ d-d transitions, ~4A2->4T1 and ~4A2->4T2), leaving
                // BOTH a wide open red transmission window (>~620nm) and a narrower,
                // smaller blue transmission window around ~470-490nm between the two
                // absorption peaks. Sources: GIA "Application of UV-Vis-NIR
                // Spectroscopy to Gemology" (Winter 2024 Gems & Gemology) and multiple
                // Cr:Al2O3 absorption-spectroscopy papers cite peaks at ~410-413nm and
                // ~550nm with minimal absorption beyond ~650nm. This double-window
                // structure -- unrepresentable by three broad fixed lobes (see
                // `AbsorptionTensor`'s doc comment) -- is exactly what makes ruby shift
                // redder under incandescent light (little tungsten output where the
                // small blue window sits) relative to daylight: see
                // `raytracer_tests::ruby_shifts_redder_under_incandescent_than_d65`.
                // Widths (30nm violet, 45nm yellow-green -- the violet Cr3+ transition
                // is the sharper of the two) and peak amplitudes (3.0, 2.5) tuned for a
                // clear narrow-window structure and plausible saturation at a typical
                // internal path length in this renderer's gem geometry; band POSITIONS
                // are the cited, non-tuned values.
                //
                // PLEOCHROISM populated this pass, replacing the isotropic placeholder
                // above with a genuine uniaxial `o_ray`/`e_ray` split. Source: J.A.
                // Mandarino, American Mineralogist 44, 961 (1959), Table 5, ruby
                // pleochroic absorption data: k_omega (o-ray) maxes near 560nm,
                // k_epsilon (e-ray) maxes near 550nm; raw peak ratios omega:epsilon
                // range 1.37 (pink stones) to 2.25 (deep red stones), baseline-corrected
                // (chromophore-only) ratios estimated at ~2.5-2.9, and the two rays
                // reported as near-equal around 440nm.
                //
                // o-ray yellow-green band centre nudged 550nm -> 556nm this pass (near-
                // equal, per this task's continuity principle -- the o-ray still
                // dominates the face-up/c_axis=Y hero shot) to sit closer to
                // Mandarino's cited k_omega peak (~560nm); the violet band (410nm) is
                // unchanged. e_ray band CENTRES are shifted blueward of the o-ray's,
                // reflecting Mandarino's k_epsilon peak sitting blueward of k_omega's
                // (400nm vs 410nm violet band, 550nm vs 556nm yellow-green band); e_ray
                // WIDTHS are kept equal to the o-ray's own (30nm/45nm) -- no separate
                // width figure is given in Table 5.
                //
                // e_ray AMPLITUDE RATIO: 1.8x per band (o-ray peak / e-ray peak = 3.0/
                // 2.9 for the violet band, 2.5/1.4 for the yellow-green band -- the
                // violet band is deliberately kept near-equal between the two rays,
                // matching Mandarino's "near-equal at 440nm" observation, while the
                // yellow-green band carries the bulk of the dichroic ratio). 1.8x is
                // MID-RANGE within Mandarino's cited 1.37-2.9 span (raw-to-
                // baseline-corrected) -- document this as a tuned, not directly
                // measured, single-number simplification of a ratio that in the source
                // varies by band and by specimen colour depth (pink vs red).
                absorption: AbsorptionTensor::uniaxial(
                    vec![
                        AbsorptionBand::new(410.0, 30.0, 3.0), // o-ray (E-perp-c)
                        AbsorptionBand::new(556.0, 45.0, 2.5),
                    ],
                    vec![
                        AbsorptionBand::new(400.0, 30.0, 2.9), // e-ray (E-parallel-c)
                        AbsorptionBand::new(550.0, 45.0, 1.4),
                    ],
                ),
                c_axis: Vec3::Y,
                biaxial_delta_beta_alpha: None,
                scattering_sigma_s: 0.0,
                scattering_g: 0.0,
                edge_rounding_radius: 0.0,
                absorption_path_scale: 1.0,
            },
            // Emerald (Beryl, Be3Al2(SiO3)6:Cr/V)
            //
            // No primary-literature Sellmeier fit for beryl was found (confirmed absent
            // from github.com/polyanskiy/refractiveindex.info-database -- unlike the
            // laser/technical crystals above, gem species are rarely characterized to
            // that precision), so this stays a Cauchy (visible-range-only) fit,
            // LOWER CONFIDENCE than the Sellmeier-sourced entries above.
            //
            // n_d PRESERVED at 1.5791 (within the published beryl n_o range, and inside
            // the International Gem Society's cited emerald range 1.57-1.60 --
            // gemsociety.org/article/emerald-jewelry-and-gemstone-information/).
            //
            // DISPERSION SHAPE re-derived this pass: two independent gemological
            // sources -- IGS (same URL, "dispersion .014") and
            // GEMSTONE_RENDERING_BLUEPRINT.md's own Emerald row (Delta n = 0.0142) --
            // agree closely on Beryl's standard "dispersion" figure of ~0.0141. Per
            // this task's brief, that figure is the Fraunhofer B-G interval, not F-C.
            // Converted via a B-G -> F-C ratio derived THIS pass directly from physics,
            // not from cross-comparing gemological tables: computed exactly (via
            // `cargo run -p gemray --example probe_dispersion`, deleted before
            // finishing) from the 8 genuine primary Sellmeier fits available in this
            // file/task (Diamond, Sapphire/Ruby, Quartz, Spinel, Cubic Zirconia,
            // Alexandrite-Walling, Moissanite-Wang/Singh) by evaluating each curve at
            // the actual B/G/F/C Fraunhofer wavelengths: ratio range 0.569-0.587,
            // mean 0.579. Target: Delta n(F-C) = 0.0141*0.579 = 0.00816 -> solved
            // (A, B) exactly reproduce n_d=1.5791 and Delta n(F-C)=0.00816, giving
            // Abbe V_d = 70.93 (previous pass's estimate, 69.99, used a less rigorous
            // ratio of 0.591 obtained by cross-comparing two gemological tables rather
            // than computing it from primary dispersion curves; the two are close, a
            // ~1.3% difference, well within this estimate's uncertainty). LOWER
            // CONFIDENCE than the Sellmeier-sourced entries -- flagged for human
            // cross-check.
            Self {
                name: "Emerald".to_string(),
                crystal_system: CrystalSystem::Hexagonal,
                optical_character: OpticalCharacter::UniaxialNegative,
                dispersion: DispersionModel::Cauchy {
                    a: 1.566_794,
                    b: 0.004_273,
                    c: 0.0,
                },
                birefringence_delta: -0.0060,
                // Chromophore source: emerald's Cr3+ (some localities: V3+ too)
                // absorbs in a blue-violet band near 430nm and a red-orange band near
                // 600nm, leaving a transmission peak near 510nm (verified numerically
                // from these exact band parameters: alpha(lambda) is minimized at
                // ~506nm) -- this narrow green window between two red/blue absorptions
                // is what makes emerald green rather than some other hue. Sources: this
                // task's brief (430/600nm); corroborated by GIA gemological references
                // reporting emerald's "main absorption bands at approximately 620 and
                // 430nm" (the red-side band's exact centre is reported anywhere from
                // ~600-620nm across localities/V-content -- the brief's 600nm figure is
                // used here). Widths (30nm violet, 40nm red) and peak amplitudes (2.8,
                // 2.6) tuned for a clear narrow green window and plausible saturation
                // at a typical internal path length in this renderer's gem geometry;
                // band POSITIONS are the cited, non-tuned values.
                absorption: AbsorptionTensor::isotropic(vec![
                    AbsorptionBand::new(430.0, 30.0, 2.8),
                    AbsorptionBand::new(600.0, 40.0, 2.6),
                ]),
                c_axis: Vec3::Y,
                biaxial_delta_beta_alpha: None,
                scattering_sigma_s: 0.0,
                scattering_g: 0.0,
                edge_rounding_radius: 0.0,
                absorption_path_scale: 1.0,
            },
        ]
    }

    /// Second quarter of the built-in material table (Zircon through Topaz). Split out
    /// of `built_in_materials_diamond_through_emerald` purely to keep each part under
    /// clippy's function-length lint (the biaxial `biaxial_delta_beta_alpha` fields and
    /// their sourcing comments pushed the combined function over the threshold) -- this
    /// is plain data, not logic, so the split point carries no
    /// significance.
    fn built_in_materials_zircon_through_topaz() -> Vec<Self> {
        vec![
            // Zircon (High Zircon, ZrSiO4)
            //
            // *** n_d CHANGED THIS PASS BY -0.0319 (1.956878 -> 1.925), MATERIALLY
            // ABOVE THE 0.002 CRITICAL-ANGLE-SIGNIFICANCE THRESHOLD. Flagged prominently
            // per this task's coordination note: this WILL shift zircon's critical
            // angle and therefore its brilliance/windowing/extinction behaviour. ***
            //
            // Per this task's corrected source priority (trusted primary/reputable
            // sources outrank GEMSTONE_RENDERING_BLUEPRINT.md), the previous pass's
            // approach here was backwards: it kept an unsourced, unexplained n_d
            // (1.956878 -- traceable to neither the document, which itself gives
            // n_o=1.9250, nor any external reference found) and only refit the
            // dispersion shape to hit the document's V_d=28.0. Re-sourced from scratch:
            //
            // n_o = 1.925, corroborated by THREE independent sources:
            //   1) Mineral Data Publishing's "Handbook of Mineralogy" (2001), zircon
            //      entry: omega = 1.925-1.961, epsilon = 1.980-2.015
            //      (handbookofmineralogy.org/pdfs/zircon.pdf) -- 1.925 is the low end
            //      of the authoritative mineralogical range.
            //   2) International Gem Society: "High zircon: n_o = 1.920-1.940 (often
            //      1.925); n_e = 1.970-2.010 (often 1.984)"
            //      (gemsociety.org/article/zircon-jewelry-and-gemstone-information/).
            //   3) GEMSTONE_RENDERING_BLUEPRINT.md's own row (n_o=1.9250, n_e=1.9840),
            //      itself the widely-reproduced gem-trade reference pairing -- used here
            //      only as corroboration, per this task's source-priority rules, not as
            //      the primary basis.
            // The 1.925/1.984 pairing is also the standard GIA reference point for
            // gem-quality (heat-treated) "starlite"/colourless high zircon specifically.
            // birefringence_delta = n_e - n_o = 1.984 - 1.925 = +0.0590 -- unchanged
            // from the previous entry, already correctly sourced.
            //
            // No primary Sellmeier/Cauchy fit for zircon exists in the optics
            // literature (confirmed absent from
            // github.com/polyanskiy/refractiveindex.info-database), so this stays a
            // LOWER-CONFIDENCE 2-parameter Cauchy fallback. Dispersion shape: zircon's
            // "0.039" dispersion figure is near-universal across gemological references
            // (GIA, IGS, gemstonebuzz.com/education/dispersion, and independently
            // confirmed as explicitly the Fraunhofer B-G interval by
            // https://skyjems.ca/pages/encyclopedia-fire and
            // https://www.gemrockauctions.com/learn/technical-information-on-gemstones/gemstone-dispersion)
            // -- NOT the classical F-C interval, and NOT reproducible from the
            // document's own V_d=28.0 either way (28.0 implies Delta n(F-C)=0.0330,
            // which matches neither 0.039 taken as F-C nor 0.039 converted from B-G --
            // the document's zircon V_d is itself unverifiable against its own row and
            // is not used here). Converted via the same physically-derived B-G->F-C
            // ratio described in the Emerald entry's comment (0.579, computed directly
            // from 8 primary Sellmeier curves in this file, range 0.569-0.587): Delta
            // n(F-C) = 0.039*0.579 = 0.02258. Two data points (preserved... here,
            // newly-sourced n_o=1.925, and this converted Delta n) exactly determine a
            // 2-parameter Cauchy fit:
            //   A = n_d - B/lambda_d^2,  B = (n_d-1)/(V_d * (1/lambda_F^2 - 1/lambda_C^2))
            // giving A=1.890963, B=0.011820 -> verified n_d=1.925 (CHANGED, see above),
            // Delta n(F-C)=0.02258, Abbe V_d=40.96 -- CONTRADICTS the document's V_d=28.0
            // by +46%. LOWER CONFIDENCE than a primary-literature entry -- flagged for
            // human cross-check -- but both inputs (n_o and the B-G dispersion figure)
            // are independently well-corroborated gemological references, converted via
            // a ratio derived from real physics rather than fit to the document.
            Self {
                name: "Zircon".to_string(),
                crystal_system: CrystalSystem::Tetragonal,
                optical_character: OpticalCharacter::UniaxialPositive,
                dispersion: DispersionModel::Cauchy {
                    a: 1.890_963,
                    b: 0.011_820,
                    c: 0.0,
                },
                birefringence_delta: 0.0590,
                // No cited chromophore spectroscopy for zircon -- kept as the
                // old three-lobe RGB triple, re-expressed via `legacy_rgb_bands` (see
                // its doc comment). Candidate for a follow-up task to source real data.
                absorption: AbsorptionTensor::isotropic(legacy_rgb_bands([0.2, 0.6, 1.8])),
                c_axis: Vec3::Y,
                biaxial_delta_beta_alpha: None,
                scattering_sigma_s: 0.0,
                scattering_g: 0.0,
                edge_rounding_radius: 0.0,
                absorption_path_scale: 1.0,
            },
            // Alexandrite (Chrysoberyl, BeAl2O4:Cr)
            //
            // *** SOURCE-PRIORITY REVERSAL FROM THE PREVIOUS PASS ***. This task's
            // brief explicitly names this entry as one of two known cases where a
            // primary source disagrees with the document and the previous pass wrongly
            // deferred to the document. Corrected here by using the genuine primary
            // Sellmeier fit directly instead of Cauchy-fitting to the document's V_d.
            //
            // Source: J. Walling et al., "Tunable alexandrite lasers," IEEE J. Quantum
            // Electron. 16, 1302 (1980), n(alpha)-direction Sellmeier, as tabulated by
            // refractiveindex.info ("BeAl2O4: Walling-alpha",
            // https://refractiveindex.info/?shelf=main&book=BeAl2O4&page=Walling-alpha,
            // raw coefficients confirmed at
            // github.com/polyanskiy/refractiveindex.info-database/blob/master/database/data/main/BeAl2O4/nk/Walling-alpha.yml),
            // valid 0.25-2.6 um:
            //   n^2 = 1.78522 + 1.21202*l^2/(l^2-0.01262) - 0.01681*l^2
            // Represented exactly (verified <1e-6 residual across 400-780nm via
            // `cargo run -p gemray --example probe_dispersion`, deleted before
            // finishing) as Sellmeier3 with the "-0.01681*l^2" linear term encoded as a
            // distant pole (b/c = 0.01681, c=1000 um^2, far outside the visible l^2
            // domain of 0.16-0.61 um^2) and the leading constant folded into a c=0
            // pole, same technique as Quartz's leading-constant pole above:
            //   pole0 (constant): b=0.78522, c=0
            //   pole1 (real UV pole, exact): b=1.21202, c=0.01262
            //   pole2 (linear-term approximation): b=16.81, c=1000
            // Verified: n_d = 1.742730 (matches this entry's previously-preserved n_d
            // to <0.0001 -- unchanged), Delta n(F-C) = 0.010051, Abbe V_d = 73.90.
            //
            // This CONTRADICTS GEMSTONE_RENDERING_BLUEPRINT.md section 2.2's table,
            // which gives Alexandrite Abbe V_d = 68.0 (a -8.7% mismatch) -- per this
            // task's corrected source priority, the primary Sellmeier measurement wins;
            // the document's figure is not used. (The document's own "Fire Delta
            // n(F-C)=0.0152" cell for this row is separately inconsistent with its own
            // n_alpha..n_gamma=1.745-1.754 and its own V_d=68.0, another instance of the
            // Delta-n column's B-G/F-C mixing.)
            //
            // birefringence_delta CORRECTED this pass: Walling 1980 also tabulates beta
            // and gamma Sellmeier fits (Walling-beta.yml, Walling-gamma.yml, same repo).
            // Important subtlety found while sourcing this: Walling's own alpha/beta/
            // gamma FILE LABELS are the paper's crystallographic axis labels and do NOT
            // sort by index magnitude the way the mineralogical optical-indicatrix
            // convention (n_alpha <= n_beta <= n_gamma) requires -- at the D line the
            // three Walling-labelled curves evaluate to alpha=1.742730,
            // beta=1.748360, gamma=1.740779, i.e. numerically gamma < alpha < beta.
            // Sorting by magnitude gives the true optical indicatrix: n_alpha(opt) =
            // 1.740779 (Walling's "gamma" file), n_beta(opt) = 1.742730 (Walling's
            // "alpha" file -- this is the value used as n_d above, so the base index
            // used here already IS the correct middle/beta index by coincidence of
            // which file was picked), n_gamma(opt) = 1.748360 (Walling's "beta" file).
            // True total birefringence = n_gamma(opt) - n_alpha(opt) = 0.007581,
            // replacing the document-derived 0.0090 (n_gamma-n_alpha = 1.754-1.745 from
            // its rounded, mislabelled-consistent table). A ~16% correction, sourced
            // directly from the same primary paper as n_d for the first time.
            Self {
                name: "Alexandrite".to_string(),
                crystal_system: CrystalSystem::Orthorhombic,
                optical_character: OpticalCharacter::BiaxialPositive,
                dispersion: DispersionModel::Sellmeier3 {
                    b: [0.785_22, 1.212_02, 16.81],
                    c: [0.0, 0.012_62, 1000.0],
                },
                birefringence_delta: 0.0076,
                // TRICHROISM populated this pass, replacing the previous pass's cited
                // ISOTROPIC Cr3+ pair ((580, 40, 2.6), (415, 32, 2.4)) whose comment
                // explicitly deferred pleochroism as "per-axis band AMPLITUDES were not
                // machine-extractable from the scans available". Resolved by rendering
                // Farrell & Newnham's Table 1 and Figs. 3-4 as page images and reading
                // them directly (the same figure-read provenance tier the Sapphire
                // entry's E-parallel-c amplitude already uses).
                //
                // Source: E.F. Farrell & R.E. Newnham, "Crystal-field spectra of
                // chrysoberyl, alexandrite, peridot, and sinhalite," American
                // Mineralogist 50, 1972 (1965), minsocam.org/ammin/AM50/AM50_1972.pdf.
                // Polarized absorption at 77K, space group Pnma (a=9.40, b=5.48,
                // c=4.43 A -- the SAME crystallographic axis convention as this entry's
                // dispersion source, Walling 1980), pleochroic colours yellow||a,
                // green||b, red||c. Two Cr3+ band systems: 4A2->4T2 at 560-595nm and
                // 4A2->4T1 at 410-440nm, straddling Neuhaus's 580/415nm red-green
                // critical values -- that straddle is the ILLUMINANT-dependent
                // daylight-green / incandescent-red colour change (kept from the
                // previous entry and pinned by `raytracer_tests::
                // alexandrite_shifts_redder_under_incandescent_than_d65`); the per-axis
                // split below adds DIRECTION-dependence on top, and alexandrite's
                // apparent colour genuinely depends on both at once (a c-axis-dominated
                // view leans red even in daylight; a b-axis-dominated view stays
                // green), which is exactly what makes the stone distinctive.
                //
                // AXIS MAPPING (same convention as the Tanzanite entry below): this
                // crate's `AbsorptionTensor::biaxial(alpha, beta, gamma)` slots are the
                // n_alpha/n_beta/n_gamma principal directions. Per this entry's own
                // Walling-derived indicatrix sort above (n_alpha(opt)=1.740779 is
                // Walling's E||c curve, n_beta(opt)=1.742730 is E||a, n_gamma(opt)=
                // 1.748360 is E||b), the crystallographic axes land as: alpha = crystal
                // c (RED), beta = crystal a (YELLOW), gamma = crystal b (GREEN, along
                // `c_axis` below per `biaxial_indicatrix`) -- so the absorption
                // principal frame is the SAME frame the index data already uses,
                // bit-identically (`AbsorptionTensor3::biaxial` and
                // `BiaxialIndicatrix::from_gamma_axis` share one
                // `stable_orthonormal_basis(c_axis)` construction).
                //
                // Band POSITIONS (CITED -- F&N Table 1, natural alexandrite): E||c:
                // 4T2 at 565nm (600nm shoulder not modelled), 4T1 the unresolved
                // 410/430nm pair (420nm midpoint used); E||a: 4T2 at 560nm, 4T1 at
                // 422nm; E||b: 4T2 at 595nm (605/620nm vibrational shoulders not
                // modelled), 4T1 the 410/430/440nm group (430nm central peak used).
                // WIDTHS (40nm, 32nm): TUNED, carried over from the previous isotropic
                // entry (77K bands sharpen relative to room temperature, so the
                // published low-temperature widths are not used directly).
                //
                // AMPLITUDE RATIOS (FIGURE-READ, F&N Figs. 3-4; net peak height above
                // each figure's uncorrected scatter baseline): 4T2 from Fig. 4
                // (natural, all three axes; baseline ~12 cm^-1): E||a 6.5, E||b 31,
                // E||c 19 cm^-1 -> the 0.65 : 3.1 : 1.9 split below (x0.1 tuned
                // scale). 4T1 E||a:E||b from Fig. 3 (synthetic, Cr-only, baseline
                // ~14 cm^-1): 25 : 10.5 cm^-1 ~= 2.4 : 1 (Fig. 4's 4T1 region is
                // Fe3+-contaminated -- the natural specimen is Al1.96(Cr,Fe)0.04BeO4
                // -- and its E||a peak runs off the plotted axis, "an intense
                // absorption ... dominates the a spectrum" per the text, so the
                // Cr-only synthetic figure is the cleaner ratio source). 4T1 E||c: NOT
                // separately measurable (F&N: "absorption measurements for the
                // c-polarization direction proved impracticable" on the synthetic
                // platelets); estimated from Fig. 4's Fe-contaminated natural c:b peak
                // ratio (net ~41:33 ~= 1.24) applied to the synthetic E||b value --
                // the one LOWER-CONFIDENCE amplitude in this entry, flagged rather
                // than hidden. Overall per-band SCALES (x0.1 for 4T2, x0.14 for 4T1)
                // are TUNED for plausible saturation at this renderer's typical
                // internal path length (same approach as every other entry), keeping
                // each band's direction-average near the previous isotropic entry's
                // 2.6/2.4 so overall saturation is comparable.
                absorption: AbsorptionTensor::biaxial(
                    vec![
                        // alpha -- crystal c-axis, RED: the ruby-like direction. 4T2
                        // at 565nm sits BELOW the 580nm critical value (Neuhaus), so
                        // the deep-red window past ~640nm stays open; moderate 4T1
                        // blocks blue-violet.
                        AbsorptionBand::new(565.0, 40.0, 1.9),
                        AbsorptionBand::new(420.0, 32.0, 1.8),
                    ],
                    vec![
                        // beta -- crystal a-axis, YELLOW: dominated by the strongest
                        // 4T1 of the three directions ("an intense absorption in the
                        // dark blue (0.42u) dominates the a spectrum, giving
                        // transmitted light its complementary color, yellow" -- F&N),
                        // while its 4T2 is by far the weakest, leaving yellow through
                        // red almost fully open.
                        AbsorptionBand::new(560.0, 40.0, 0.65),
                        AbsorptionBand::new(422.0, 32.0, 3.5),
                    ],
                    vec![
                        // gamma -- crystal b-axis, GREEN (along `c_axis` below): the
                        // strongest 4T2, at 595nm ABOVE the critical value, absorbing
                        // "both red and yellow" (F&N) so "only green radiation (0.5u)
                        // is transmitted"; the weakest 4T1 leaves the green window's
                        // blue edge comparatively open.
                        AbsorptionBand::new(595.0, 40.0, 3.1),
                        AbsorptionBand::new(430.0, 32.0, 1.5),
                    ],
                ),
                c_axis: Vec3::Y,
                // n_beta - n_alpha at the D line, TIGHT confidence -- both
                // numbers come directly from the SAME primary source already cited
                // above for this entry's birefringence_delta (J. Walling et al. 1980,
                // the alpha/beta/gamma Sellmeier trio, refractiveindex.info
                // BeAl2O4/Walling-{alpha,beta,gamma}.yml), sorted by magnitude into the
                // true optical indicatrix per that comment's derivation:
                // n_alpha(opt) = 1.740779, n_beta(opt) = 1.742730 (== this entry's n_d
                // above), n_gamma(opt) = 1.748360. delta_beta_alpha = 1.742730 -
                // 1.740779 = 0.001951 -- no independent sourcing needed here, it falls
                // straight out of the same three fits already used for n_d and
                // birefringence_delta (which cross-checks: n_gamma - n_alpha =
                // 0.007581, matching this entry's birefringence_delta = 0.0076 to
                // within the field's own rounding).
                biaxial_delta_beta_alpha: Some(0.001_951),
                scattering_sigma_s: 0.0,
                scattering_g: 0.0,
                edge_rounding_radius: 0.0,
                absorption_path_scale: 1.0,
            },
            // Topaz (Al2SiO4(F,OH)2)
            //
            // No primary Sellmeier/Cauchy fit for topaz was found: confirmed absent
            // from the refractiveindex.info database (no Al2SiO4(F,OH)2 / topaz entry
            // in github.com/polyanskiy/refractiveindex.info-database as of this
            // search), and no dedicated visible-range topaz dispersion paper was found
            // via general search either. LOWER-CONFIDENCE 2-parameter Cauchy fallback.
            //
            // n_d PRESERVED at 1.627178 (matches International Gem Society's cited
            // topaz range "1.61-1.638" -- gemsociety.org/article/topaz-buying-guide/ --
            // and close to n_gamma from GEMSTONE_RENDERING_BLUEPRINT.md's row,
            // n_gamma=1.627, used as corroboration only per this task's source
            // priority).
            //
            // DISPERSION SHAPE re-derived this pass: IGS's cited topaz dispersion
            // (same URL, "dispersion .014") and the document's own row (Delta n=0.0142)
            // agree closely (~0.0141 average) on the standard gemological "dispersion"
            // figure -- per this task's brief that is the Fraunhofer B-G interval, not
            // F-C (the document's own V_d=64.0 does not follow from either taking
            // 0.0142 as F-C with n_gamma=1.627, which implies V_d~44, or from
            // converting it B-G->F-C, which implies V_d~77 -- neither reproduces the
            // document's own V_d=64.0, so that figure is not independently
            // verifiable and is not used here). Converted via the same
            // physically-derived B-G->F-C ratio as Emerald's comment above (0.579):
            // Delta n(F-C) = 0.0141*0.579 = 0.00816. Two data points (preserved n_d,
            // converted Delta n) exactly determine a 2-parameter Cauchy fit (same
            // method as Zircon's comment): A=1.614872, B=0.004273 -> verified
            // n_d=1.627176 (unchanged), Delta n(F-C)=0.008163, Abbe V_d=76.83 --
            // CONTRADICTS the document's V_d=64.0 by +20%. LOWER CONFIDENCE than a
            // primary-literature entry -- flagged for human cross-check.
            Self {
                name: "Topaz".to_string(),
                crystal_system: CrystalSystem::Orthorhombic,
                optical_character: OpticalCharacter::BiaxialPositive,
                dispersion: DispersionModel::Cauchy {
                    a: 1.614_872,
                    b: 0.004_273,
                    c: 0.0,
                },
                birefringence_delta: 0.0080,
                // No cited chromophore spectroscopy for topaz -- kept as the
                // old three-lobe RGB triple via `legacy_rgb_bands` (see its doc
                // comment). Candidate for a follow-up task to source real data.
                absorption: AbsorptionTensor::isotropic(legacy_rgb_bands([1.2, 0.4, 0.1])),
                c_axis: Vec3::Y,
                // n_beta - n_alpha at the D line, LOWER CONFIDENCE (derived,
                // not directly measured for a single specimen -- consistent with this
                // entry's own n_d/dispersion confidence tier). No primary 3-index
                // topaz measurement was found (same absence as this entry's dispersion
                // fit above). Source: The Gemology Project's topaz entry
                // (gemologyproject.com/wiki/index.php?title=Topaz), citing ranges
                // n_alpha=1.606-1.634, n_beta=1.609-1.637, n_gamma=1.616-1.644 across
                // topaz's fluorine/hydroxyl compositional range. Taking each range's
                // midpoint (n_alpha=1.620, n_beta=1.623, n_gamma=1.630) gives the
                // fractional position of beta between alpha and gamma: (1.623-1.620) /
                // (1.630-1.620) = 0.30. Applied to THIS entry's own preserved
                // birefringence_delta (0.0080, i.e. n_gamma-n_alpha) rather than the
                // source's own absolute numbers (keeping n_d=1.627178 as n_beta,
                // unchanged): delta_beta_alpha = 0.30 * 0.0080 = 0.0024. Implied
                // n_alpha=1.624778, n_gamma=1.632778, both inside the cited ranges.
                biaxial_delta_beta_alpha: Some(0.0024),
                scattering_sigma_s: 0.0,
                scattering_g: 0.0,
                edge_rounding_radius: 0.0,
                absorption_path_scale: 1.0,
            },
        ]
    }

    /// Third quarter of the built-in material table (Spinel through Tourmaline). See
    /// `built_in_materials_diamond_through_emerald` for why the table is split into
    /// several functions; the trichroic `beta_ray` data added for Tanzanite (see
    /// `built_in_materials_tanzanite_through_cubic_zirconia`) is what pushed this
    /// former "second half" over clippy's function-length lint and required the split
    /// down to quarters.
    fn built_in_materials_spinel_through_tourmaline() -> Vec<Self> {
        vec![
            // Spinel (MgAl2O4)
            //
            // Source: W.J. Tropf & M.E. Thomas, "Magnesium Aluminum Oxide (Spinel),"
            // Handbook of Optical Constants of Solids III (1991), as tabulated by
            // refractiveindex.info ("MgAl2O4: Tropf"), valid 0.35-5.5 um:
            //   n^2-1 = 1.8938*l^2/(l^2-0.09942^2) + 3.0755*l^2/(l^2-15.826^2)
            // Encoded via Sellmeier3 with the unused 3rd pole zeroed (b=0, c=1.0 um^2,
            // outside the visible-range l^2 domain).
            // Previous entry was a hand-tuned Cauchy fit (n_d=1.7295,
            // Delta n(F-C)=0.01803, Abbe=40.46) with no cited source. Verified with the
            // real Sellmeier fit above: n_d = 1.71610, Delta n(F-C) = 0.01181,
            // Abbe V_d = 60.63. Cross-check: gemological (B-G) dispersion table lists
            // Spinel = 0.020; 0.020*0.591 (see Emerald entry for this conversion
            // factor) = 0.0118, matching this Sellmeier-derived value almost exactly --
            // strong independent confirmation of both the fit and the conversion factor.
            Self {
                name: "Spinel".to_string(),
                crystal_system: CrystalSystem::Cubic,
                optical_character: OpticalCharacter::Isotropic,
                dispersion: DispersionModel::Sellmeier3 {
                    b: [1.8938, 3.0755, 0.0],
                    c: [0.009_884, 250.462, 1.0],
                },
                birefringence_delta: 0.0,
                // No cited chromophore spectroscopy for spinel -- kept as the
                // old three-lobe RGB triple via `legacy_rgb_bands` (see its doc
                // comment). Candidate for a follow-up task to source real data.
                absorption: AbsorptionTensor::isotropic(legacy_rgb_bands([0.4, 2.2, 1.6])),
                c_axis: Vec3::Y,
                biaxial_delta_beta_alpha: None,
                scattering_sigma_s: 0.0,
                scattering_g: 0.0,
                edge_rounding_radius: 0.0,
                absorption_path_scale: 1.0,
            },
            // Quartz (Rock Crystal / Amethyst / Citrine, alpha-SiO2)
            //
            // Source: G. Ghosh, "Dispersion-equation coefficients for the refractive
            // index and birefringence of calcite and quartz crystals," Opt. Commun.
            // 163, 95-102 (1999), ordinary-ray fit, as tabulated by refractiveindex.info
            // ("SiO2: Ghosh-o"), valid 0.198-2.0531 um:
            //   n^2-1 = 0.28604141 + 1.07044083*l^2/(l^2-0.0100585997) + 1.10202242*l^2/(l^2-100)
            // Independently verified against a manufacturer (pmoptics.com) quartz
            // crystal spec sheet: this formula gives n_o(630nm)=1.54270 (spec:
            // 1.542737) and n_o(1550nm)=1.52770 (spec: 1.527606), both matching to
            // <1e-4. Encoded via Sellmeier3: the leading 0.28604141 constant uses pole 1
            // with c=0.0 (l^2/(l^2-0)=1 for any l>0, so that term reduces to the bare
            // constant); the other two are the genuine poles.
            // The previous entry used a single-pole Sellmeier1 whose
            // constant (1.286_041_4) and pole (0.010_704_408) do not match ANY term of
            // the real 3-term formula above -- it silently dropped the 100 um^2 pole and
            // conflated the leading constant with the first real pole, giving n_d=1.5254
            // (vs the accepted n_o(D) ~ 1.5442-1.5443) and Delta n(F-C)=0.00925.
            // Verified with the real Ghosh fit: n_d = 1.54421, Delta n(F-C) = 0.00781,
            // Abbe V_d = 69.65. NOTE: this is LOWER than the previous entry's
            // Delta n(F-C), and lower than the ~0.013 figure this task's own brief cited
            // as "published" -- that 0.013 figure matches the standard gemological (B-G
            // Fraunhofer line) dispersion table's "Quartz = 0.013" entry almost exactly,
            // not the classical F-C interval requested by this task (0.013*0.591 =
            // 0.00768, matching this Ghosh-derived 0.00781 to within 2%). Flagged for
            // human cross-check since it contradicts the task brief's stated target, but
            // independently verified against a real manufacturer spec sheet above.
            Self {
                name: "Quartz".to_string(),
                crystal_system: CrystalSystem::Trigonal,
                optical_character: OpticalCharacter::UniaxialPositive,
                dispersion: DispersionModel::Sellmeier3 {
                    b: [0.286_041_4, 1.070_440_8, 1.102_022_4],
                    c: [0.0, 0.010_058_6, 100.0],
                },
                birefringence_delta: 0.0091,
                // No cited chromophore spectroscopy for quartz's minor tint
                // varieties (this entry covers rock crystal/amethyst/citrine
                // generically) -- kept as the old three-lobe RGB triple via
                // `legacy_rgb_bands` (see its doc comment). Candidate for a follow-up
                // task to source real data (e.g. amethyst's Fe4+ band near 550nm).
                absorption: AbsorptionTensor::isotropic(legacy_rgb_bands([0.8, 1.8, 0.6])),
                c_axis: Vec3::Y,
                biaxial_delta_beta_alpha: None,
                scattering_sigma_s: 0.0,
                scattering_g: 0.0,
                edge_rounding_radius: 0.0,
                absorption_path_scale: 1.0,
            },
            // Tourmaline (Elbaite, complex borosilicate)
            //
            // No primary Sellmeier/Cauchy fit for tourmaline was found: confirmed
            // absent from the refractiveindex.info database (no tourmaline/elbaite
            // entry in github.com/polyanskiy/refractiveindex.info-database as of this
            // search), and tourmaline's compositional variability (a large solid-
            // solution family, not a fixed stoichiometric crystal) makes a single
            // universal Sellmeier fit unlikely to exist in the literature at all.
            // LOWER-CONFIDENCE 2-parameter Cauchy fallback.
            //
            // n_d PRESERVED at 1.639405, within International Gem Society's cited
            // elbaite range (n_o = 1.619-1.655, n_e = 1.603-1.634 --
            // gemsociety.org/article/tourmaline-jewelry-and-gemstone-information/) and
            // close to the document's n_o=1.6440 (corroboration only).
            //
            // DISPERSION SHAPE re-derived this pass: IGS's cited tourmaline dispersion
            // (same URL, "dispersion .017") and the document's own row (Delta
            // n=0.0172) agree closely (~0.0171 average). Per this task's brief that
            // figure is the Fraunhofer B-G interval, not F-C (the document's own
            // V_d=55.0 does not follow from its own Delta n taken either as F-C, which
            // implies V_d~37, or converted B-G->F-C, which implies V_d~65 -- neither
            // matches, so the document's V_d is not independently verifiable and is not
            // used). Converted via the same physically-derived B-G->F-C ratio as
            // Emerald's comment above (0.579): Delta n(F-C) = 0.0171*0.579 = 0.00990.
            // Two data points exactly determine a 2-parameter Cauchy fit (same method
            // as Zircon's comment): A=1.624481, B=0.005183 -> verified n_d=1.639406
            // (unchanged), Delta n(F-C)=0.009902, Abbe V_d=64.58 -- CONTRADICTS the
            // document's V_d=55.0 by +17%. LOWER CONFIDENCE than a primary-literature
            // entry -- flagged for human cross-check.
            //
            // birefringence_delta refined from -0.0200 to -0.0210, matching the more
            // precise IGS/document representative pairing (n_o~1.644, n_e~1.623) widely
            // reproduced for "typical" (green) elbaite -- IGS's own range only bounds
            // it (n_e-n_o spans roughly -0.016 to -0.021 across the elbaite range), so
            // this remains an estimate, not a single-crystal measurement.
            Self {
                name: "Tourmaline".to_string(),
                crystal_system: CrystalSystem::Trigonal,
                optical_character: OpticalCharacter::UniaxialNegative,
                dispersion: DispersionModel::Cauchy {
                    a: 1.624_481,
                    b: 0.005_183,
                    c: 0.0,
                },
                birefringence_delta: -0.0210,
                // PLEOCHROISM populated this pass, replacing the uncited
                // `legacy_rgb_bands` placeholder with a genuine uniaxial `o_ray`/`e_ray`
                // split -- tourmaline (elbaite) is famous among gemologists precisely
                // for its strong dichroism ("the dark ray" along c vs the lighter ray
                // across it), which the old isotropic placeholder could never represent
                // at all.
                //
                // Band POSITIONS: R.P. Mattson & G.R. Rossman, Phys. Chem. Minerals 14,
                // 163 (1987), Fe-bearing tourmaline polarized absorption spectra, and
                // the Caltech Mineral Spectroscopy Server's dravite entry (sample GRR
                // 787): intense E-perp-c ("the dark ray") absorptions at 730nm and
                // 1120nm (the 1120nm band is outside this renderer's 380-780nm channel
                // range and so has no representable contribution here), plus a Fe2+-Ti4+
                // intervalence charge-transfer (IVCT) band near 430nm present in both
                // rays. Centres here (720nm, 430nm) sit at the cited positions rounded
                // to the nearest 10nm; widths (55nm, 45nm) are tuned to give each band a
                // plausible visible-range footprint (no numeric width is given by
                // either source).
                //
                // DICHROIC RATIO: Mattson & Rossman report that at high Fe content the
                // E-perp-c (o-ray) intensity is enhanced MORE THAN 10x over E-parallel-c
                // (e-ray) -- i.e. real stones span roughly 1.1x (pale, low-Fe material)
                // to >10x (dark, high-Fe material) depending on iron content and path
                // length. The 3x ratio used here (o-ray peaks 3.0/1.8 vs e-ray peaks
                // 1.0/0.6, same ratio for both bands) is a DOCUMENTED MID-RANGE CHOICE,
                // not a measurement of a specific specimen: it sits well inside the
                // cited 1.1x-10x span, giving a visibly, strongly dichroic stone (real
                // tourmaline is famous for being nearly opaque down the c-axis in dark
                // material) without going all the way to the >10x extreme, which would
                // make the e-ray direction render as near-black even at short path
                // lengths. o-ray peak AMPLITUDES (3.0, 1.8) are NOT carried over from
                // the pre-existing `legacy_rgb_bands([1.5, 0.3, 1.8])` placeholder --
                // that placeholder used entirely different band positions (620/540/450
                // nm) and a normalized blend rather than a summed model (see
                // `legacy_rgb_bands`'s doc comment), so its peak values are not directly
                // comparable at the new band positions. 3.0/1.8 are freshly tuned for
                // plausible saturation at this renderer's typical internal path length,
                // matching the tuning approach used for Ruby/Emerald/Sapphire above.
                absorption: AbsorptionTensor::uniaxial(
                    vec![
                        AbsorptionBand::new(720.0, 55.0, 3.0), // o-ray (E-perp-c), "the dark ray"
                        AbsorptionBand::new(430.0, 45.0, 1.8),
                    ],
                    vec![
                        AbsorptionBand::new(720.0, 55.0, 1.0), // e-ray (E-parallel-c)
                        AbsorptionBand::new(430.0, 45.0, 0.6),
                    ],
                ),
                // c_axis set INTO the table plane (Vec3::X) rather than the Vec3::Y
                // every other built-in defaults to -- a deliberate, approved cut-
                // orientation override, not an oversight. Real tourmaline cutters
                // orient the table perpendicular to the c-axis specifically BECAUSE
                // face-up down the closed (e-ray/dark-ray) axis is tourmaline's WORST
                // viewing direction (see the dichroic-ratio comment above): with
                // c_axis=Y (the every-other-material default), the face-up hero shot
                // would look straight down the dark ray, backwards from how the stone
                // is actually cut and worn. See
                // `raytracer_tests::test_gem_materials_default_c_axis_to_y` for the
                // regression this changes, and
                // `raytracer_tests::tourmaline_face_up_is_brighter_with_c_axis_in_table_plane`
                // for the orientation-sign test this enables.
                c_axis: Vec3::X,
                biaxial_delta_beta_alpha: None,
                scattering_sigma_s: 0.0,
                scattering_g: 0.0,
                edge_rounding_radius: 0.0,
                absorption_path_scale: 1.0,
            },
        ]
    }

    /// Fourth quarter of the built-in material table (Tanzanite through Cubic
    /// Zirconia). See `built_in_materials_diamond_through_emerald` for why the table is
    /// split into several functions.
    fn built_in_materials_tanzanite_through_cubic_zirconia() -> Vec<Self> {
        vec![
            // Tanzanite (unheated, trichroic Zoisite, Ca2Al3(SiO4)3(OH):V -- see the
            // CHOICE note below for why this models the unheated stone rather than the
            // heat-treated blue-violet commercial gem)
            //
            // No primary Sellmeier/Cauchy fit for zoisite/tanzanite was found:
            // confirmed absent from the refractiveindex.info database (no zoisite
            // entry in github.com/polyanskiy/refractiveindex.info-database as of this
            // search), and no dedicated visible-range zoisite dispersion paper was
            // found via general search either. LOWER-CONFIDENCE 2-parameter Cauchy
            // fallback.
            //
            // n_d PRESERVED at 1.700858, within International Gem Society's cited
            // tanzanite range "1.691-1.70" (gemsociety.org/article/tanzanite-facet-
            // faceting-information/) and close to the document's n_beta/n_gamma range
            // (1.697-1.704, corroboration only).
            //
            // DISPERSION SHAPE re-derived this pass: IGS's cited tanzanite dispersion
            // (same URL, "dispersion .030") and the document's own row (Delta
            // n=0.0302) agree closely (~0.0301 average). Per this task's brief that
            // figure is the Fraunhofer B-G interval, not F-C (the document's own
            // V_d=45.0 does not follow from its own Delta n taken either as F-C, which
            // implies V_d~23, or converted B-G->F-C, which implies V_d~40 -- the
            // converted figure is much closer to the document's than in the other
            // un-sourced entries in this file, but still not an exact match, so the
            // document's V_d is not treated as independently verified). Converted via
            // the same physically-derived B-G->F-C ratio as Emerald's comment above
            // (0.579): Delta n(F-C) = 0.0301*0.579 = 0.01743. Two data points exactly
            // determine a 2-parameter Cauchy fit (same method as Zircon's comment):
            // A=1.674589, B=0.009123 -> verified n_d=1.700859 (unchanged), Delta
            // n(F-C)=0.017428, Abbe V_d=40.21 -- CONTRADICTS the document's V_d=45.0 by
            // -11% (tanzanite is estimated MORE dispersive than the document claims).
            // LOWER CONFIDENCE than a primary-literature entry -- flagged for human
            // cross-check.
            Self {
                name: "Tanzanite".to_string(),
                crystal_system: CrystalSystem::Orthorhombic,
                optical_character: OpticalCharacter::BiaxialPositive,
                dispersion: DispersionModel::Cauchy {
                    a: 1.674_589,
                    b: 0.009_123,
                    c: 0.0,
                },
                birefringence_delta: 0.0130,
                // PLEOCHROISM: this upgrades the previous pass's
                // two-band-set (`o_ray`/`e_ray`) APPROXIMATION to a genuine three-set
                // `AbsorptionTensor::biaxial` (`o_ray`=alpha/`beta_ray`=beta/`e_ray`
                // =gamma), because tanzanite's trichroism is now representable end to
                // end (`birefringence::AbsorptionTensor3::biaxial`,
                // `raytracer::apply_absorption`). This entry deliberately models
                // UNHEATED tanzanite -- the genuinely trichroic mineral, not the
                // heat-treated commercial gem (see the CHOICE note below) -- so it is
                // this task's own showcase for the extension, not just a data update.
                //
                // Band POSITIONS: C.S. Hurlbut, American Mineralogist 54, 702 (1969),
                // citing B.W. Anderson (1968)'s polarized absorption data for blue
                // zoisite (tanzanite): bands at 595nm, 528nm and 455nm. Hurlbut reports
                // tanzanite as trichroic: X-axis red, Y-axis blue, Z-axis yellow-green.
                // Corroborated by K. Schwarzinger, T. Ulatowski & G.R. Rossman's
                // V3+-in-zoisite work, which places V3+ absorption at 530nm and 600nm
                // IN ALL THREE polarization directions (consistent with the 528/595nm
                // positions used here, rounded to Hurlbut's own reported values), and
                // additionally identifies the 455nm band as Fe2+-Ti4+ intervalence
                // charge transfer (IVCT), present ONLY in the gamma ray (unlike the two
                // V3+ bands) and destroyed by heat treatment above 550C -- exactly the
                // mechanism by which heating collapses tanzanite's trichroism down to
                // the commercial blue-violet dichroic look (see the CHOICE note below).
                //
                // AXIS MAPPING: standard optical-mineralogy convention (e.g. Nesse,
                // *Introduction to Optical Mineralogy*) labels a biaxial crystal's X/Y/Z
                // vibration directions as the n_alpha/n_beta/n_gamma directions
                // respectively -- i.e. Hurlbut's X=red IS the alpha direction, Y=blue IS
                // beta, Z=yellow-green IS gamma. That maps directly onto this crate's
                // own `AbsorptionTensor3` alpha/beta/gamma convention with no relabelling:
                // `o_ray` (alpha) -> red, `beta_ray` (beta) -> blue, `e_ray` (gamma,
                // which sits along `c_axis` below per `biaxial_indicatrix`) -> the
                // 455nm-bearing yellow-green ray. This RESOLVES the previous pass's
                // second caveat (which flagged that `AbsorptionTensor3::uniaxial` forced
                // the physically-distinct alpha ray onto the gamma/`c_axis` slot) --
                // that mismatch no longer exists because alpha, beta and gamma each now
                // get their own independent coefficient and their own real axis, not a
                // forced two-way split.
                //
                // CHOICE (per this task's brief): this entry models UNHEATED tanzanite,
                // not the heat-treated stone sold as "tanzanite" almost universally.
                // Deliberate -- the whole point of this extension is that a genuinely trichroic
                // material can now be represented at all, and unheated tanzanite is the
                // textbook example (see the task brief this entry was written against);
                // modelling the heated
                // (effectively dichroic) form instead would leave the new three-set path
                // exercised only by synthetic-coefficient unit tests, not by any real
                // built-in's data. The heated form's 455nm-band loss could be modelled
                // later by a hypothetical "Tanzanite (heated)" entry omitting the third
                // `AbsorptionBand` below; not added here since only one tanzanite
                // built-in exists.
                //
                // AMPLITUDES are tuned for plausible saturation at this renderer's
                // typical internal path length, same as every other uncited-amplitude
                // entry in this file (see `legacy_rgb_bands`'s doc comment) -- band
                // POSITIONS (595nm, 528nm, 455nm) are the cited, non-tuned values, and
                // the QUALITATIVE amplitude pattern (beta strongest on both V3+ bands so
                // red and green-yellow are both blocked leaving blue; alpha weakest so
                // red stays open; gamma carries the 455nm band exclusively, blocking
                // blue to leave yellow-green) follows directly from the three colours
                // Hurlbut reports, even though the precise per-axis amplitude split
                // itself is not in the cited sources.
                absorption: AbsorptionTensor::biaxial(
                    vec![
                        // alpha (o_ray) -- X-axis, red: both V3+ bands present but
                        // weak, leaving the red end of the spectrum comparatively open.
                        AbsorptionBand::new(595.0, 45.0, 1.3),
                        AbsorptionBand::new(528.0, 35.0, 1.8),
                    ],
                    vec![
                        // beta (beta_ray) -- Y-axis, blue: both V3+ bands strong,
                        // absorbing red (595nm) and green-yellow (528nm) alike and
                        // leaving a blue transmission window.
                        AbsorptionBand::new(595.0, 45.0, 3.2),
                        AbsorptionBand::new(528.0, 35.0, 2.8),
                    ],
                    vec![
                        // gamma (e_ray) -- Z-axis, yellow-green: V3+ absorption weaker
                        // than beta's (opening a window between the two V3+ bands),
                        // PLUS the gamma-ray-exclusive 455nm Fe2+-Ti4+ IVCT band, which
                        // blocks blue and completes the yellow-green appearance.
                        AbsorptionBand::new(595.0, 45.0, 1.6),
                        AbsorptionBand::new(528.0, 35.0, 1.0),
                        AbsorptionBand::new(455.0, 40.0, 2.6),
                    ],
                ),
                c_axis: Vec3::Y,
                // n_beta - n_alpha at the D line, LOWER CONFIDENCE (derived,
                // not directly measured for a single specimen -- consistent with this
                // entry's own n_d/dispersion confidence tier). No primary 3-index
                // tanzanite measurement was found (same absence as this entry's
                // dispersion fit above). Source: The Gemology Project's tanzanite entry
                // (gemologyproject.com/wiki/index.php?title=Tanzanite), citing ranges
                // n_alpha=1.685-1.696, n_beta=1.693-1.702, n_gamma=1.700-1.707. Taking
                // each range's midpoint (n_alpha=1.6905, n_beta=1.6975,
                // n_gamma=1.7035) gives the fractional position of beta between alpha
                // and gamma: (1.6975-1.6905) / (1.7035-1.6905) = 0.54. Applied to THIS
                // entry's own preserved birefringence_delta (0.0130, i.e.
                // n_gamma-n_alpha) rather than the source's own absolute numbers
                // (keeping n_d=1.700858 as n_beta, unchanged): delta_beta_alpha =
                // 0.54 * 0.0130 = 0.0070. Implied n_alpha=1.693858, n_gamma=1.706858,
                // both inside (n_gamma at the edge of) the cited ranges.
                biaxial_delta_beta_alpha: Some(0.0070),
                scattering_sigma_s: 0.0,
                scattering_g: 0.0,
                edge_rounding_radius: 0.0,
                absorption_path_scale: 1.0,
            },
            // Moissanite (Synthetic SiC, 6H polytype)
            //
            // *** SOURCE-PRIORITY REVERSAL FROM THE PREVIOUS PASS ***. This task's
            // brief explicitly names this entry as the second known case where a
            // primary source disagrees with the document and the previous pass wrongly
            // deferred to the document. Corrected here by using a genuine primary
            // Sellmeier fit directly instead of Cauchy-fitting to the document's V_d.
            //
            // Source: S. Wang et al., "4H-SiC: A new nonlinear material for
            // mid-infrared lasers," Laser Photonics Rev. 7, 831 (2013), 6H-SiC
            // ordinary-ray fit, as tabulated by refractiveindex.info ("SiC:
            // Wang-6H-o", raw coefficients confirmed at
            // github.com/polyanskiy/refractiveindex.info-database/blob/master/database/data/main/SiC/nk/Wang-6H-o.yml),
            // valid 0.4358-2.325 um -- 6H is the specific SiC polytype gem moissanite
            // is predominantly cut from, and this is a dedicated 6H measurement (not a
            // 4H one despite the paper's title):
            //   n^2 = 6.57232 + 0.1401/(l^2-0.03178) - 0.02153*l^2
            // Represented exactly (verified <1e-6 residual across 400-780nm via
            // `cargo run -p gemray --example probe_dispersion`, deleted before
            // finishing) as Sellmeier3 using the same distant-pole / folded-constant
            // technique as Alexandrite's comment above (the "0.1401/(l^2-0.03178)"
            // term is itself expanded via the pole identity b*l^2/(l^2-c) - b =
            // A/(l^2-c) for b=A/c, so it becomes an exact pole rather than an
            // approximation):
            //   pole0 (constant): b=1.163887, c=0
            //   pole1 (real UV pole, exact): b=4.408433, c=0.03178
            //   pole2 (linear-term approximation): b=21.53, c=1000
            // Verified: n_d = 2.647434, Delta n(F-C) = 0.063515, Abbe V_d = 25.94 --
            // independently corroborated by a second, older primary source: S. Singh,
            // J.R. Potopowicz, L.G. Van Uitert & S.H. Wemple, "Nonlinear optical
            // properties of hexagonal silicon carbide," Appl. Phys. Lett. 19, 53
            // (1971), alpha-SiC (6H) ordinary-ray Sellmeier (n^2-1 =
            // 5.5394*l^2/(l^2-0.026945), SiC/nk/Singh-o.yml), which gives n_d=2.646763
            // (agreeing with Wang to <0.03%) and Abbe V_d=25.53 (agreeing with Wang's
            // 25.94 to within 1.6%). This was the primary source used for n_d by the
            // PREVIOUS pass, which was correct about n_d but then discarded its own
            // implied dispersion shape in favour of the document.
            //
            // This CONTRADICTS GEMSTONE_RENDERING_BLUEPRINT.md section 2.2's table,
            // which gives Moissanite Abbe V_d = 21.5 (a ~19-20% mismatch against both
            // primary sources) -- per this task's corrected source priority, the
            // primary Sellmeier measurement wins; the document's figure is not used.
            //
            // birefringence_delta CORRECTED this pass: re-derived from Wang 2013's own
            // companion 6H-SiC extraordinary-ray fit (SiC/nk/Wang-6H-e.yml,
            // coefficients 6.7452 0.15352 0 0.03597 1 0 0 0 1 -0.02249 2 in
            // refractiveindex.info "formula 4" notation, i.e. n_e^2 = 6.7452 +
            // 0.15352/(l^2-0.03597) - 0.02249*l^2), giving n_e(D) = 2.688966 against
            // this entry's n_o(D) = 2.647434, i.e. n_e - n_o = +0.041532 at the sodium D
            // line -- both indices from the SAME primary paper. Singh 1971's e-ray fit
            // (SiC/nk/Singh-e.yml) corroborates with n_e-n_o = +0.045766, bracketing the
            // document's 0.0430 from above and below. Replaced the document-derived
            // 0.0430 with the Wang-paired value, 0.0415, for same-source consistency
            // with the n_o above (a ~3.5% change).
            Self {
                name: "Synthetic Moissanite".to_string(),
                crystal_system: CrystalSystem::Hexagonal,
                optical_character: OpticalCharacter::UniaxialPositive,
                dispersion: DispersionModel::Sellmeier3 {
                    b: [1.163_887, 4.408_433, 21.53],
                    c: [0.0, 0.031_78, 1000.0],
                },
                birefringence_delta: 0.0415,
                // Colourless (no chromophore): empty band set, zero absorption at
                // every wavelength -- must render identically to before.
                absorption: AbsorptionTensor::isotropic(vec![]),
                c_axis: Vec3::Y,
                biaxial_delta_beta_alpha: None,
                scattering_sigma_s: 0.0,
                scattering_g: 0.0,
                edge_rounding_radius: 0.0,
                absorption_path_scale: 1.0,
            },
            // Cubic Zirconia (ZrO2, 12 mol% Y2O3-stabilized)
            //
            // Source: D.L. Wood & K. Nassau, "Refractive index of cubic zirconia
            // stabilized with yttria," Appl. Opt. 21, 2978-2981 (1982), as tabulated by
            // refractiveindex.info ("ZrO2: Wood"), valid 0.361-5.135 um:
            //   n^2-1 = 1.347091*l^2/(l^2-0.062543^2) + 2.117788*l^2/(l^2-0.166739^2)
            //           + 9.452943*l^2/(l^2-24.320570^2)
            // Independently verified against the paper's own directly-quoted figures:
            // N_D = 2.15847 (this fit: n_d = 2.15846) and |N_C-N_F| = 0.03455 (this fit:
            // Delta n(F-C) = 0.03456) -- both match to within 1e-5.
            // Previous entry was a hand-tuned Cauchy fit (n_d=2.1595,
            // Delta n(F-C)=0.05076, Abbe=22.84) with no cited source, overstating
            // dispersion by ~47%. Verified with the real Wood & Nassau fit:
            // Abbe V_d = 33.52.
            Self {
                name: "Cubic Zirconia".to_string(),
                crystal_system: CrystalSystem::Cubic,
                optical_character: OpticalCharacter::Isotropic,
                dispersion: DispersionModel::Sellmeier3 {
                    b: [1.347_091, 2.117_788, 9.452_943],
                    c: [0.003_912, 0.027_802, 591.489],
                },
                birefringence_delta: 0.0,
                // Colourless (no chromophore): empty band set, zero absorption at
                // every wavelength -- must render identically to before.
                absorption: AbsorptionTensor::isotropic(vec![]),
                c_axis: Vec3::Y,
                biaxial_delta_beta_alpha: None,
                scattering_sigma_s: 0.0,
                scattering_g: 0.0,
                edge_rounding_radius: 0.0,
                absorption_path_scale: 1.0,
            },
        ]
    }

    /// Creates a custom gemstone material with specified physical optical properties.
    ///
    /// # `dispersion_delta` convention: Fraunhofer F-C, not B-G
    ///
    /// `dispersion_delta` is interpreted as the Fraunhofer **F-C** interval,
    /// `n(486.1nm) - n(656.3nm)`, matching every built-in material in
    /// [`Self::all_materials`] (whose Cauchy/Sellmeier fits were, per their own sourcing
    /// comments, deliberately normalised to F-C -- gemological tables usually publish
    /// the wider Fraunhofer **B-G** interval, `n(430.8nm) - n(686.7nm)`, instead, and
    /// every one of those comments records converting B-G -> F-C before fitting).
    /// `new_custom` picks the same convention so a caller mixing a custom material into
    /// a scene with built-ins gets a consistent, comparable dispersion figure -- a
    /// caller with a genuine B-G figure in hand should convert it first (multiply by
    /// the `k_bg` ratio derived below, or just use the physics: F-C and B-G differ by
    /// roughly a factor of 1.71 for typical gemstone Cauchy curves, per the worked
    /// conversions throughout `all_materials`' sourcing comments).
    ///
    /// Solved in closed form from the single-term Cauchy fit this constructor always
    /// builds (`n(lambda) = a + b / lambda_um^2`, `c = 0`): the F-C delta this produces
    /// is `b` times a fixed geometric factor (the difference of `1 / lambda^2` at F and
    /// C), so dividing the requested delta by that SAME factor for `b` reproduces the
    /// requested F-C delta exactly (up to floating-point rounding) rather than only
    /// approximately. Computed here from the wavelengths directly (not a hand-rounded
    /// literal) so the reciprocal-square arithmetic is exact regardless of how many
    /// digits get typed into a comment; evaluates to a factor of ~0.5235 -- previously
    /// this used a flat `0.347` multiplier with no derivation given, which measurably
    /// under-delivered the requested F-C delta (only ~66% of it) while over-delivering
    /// a B-G-interpreted delta (~113% of it) -- i.e. it matched neither convention.
    ///
    /// See this module's own test
    /// `new_custom_dispersion_delta_measures_exactly_at_f_and_c` for the regression
    /// pinning this.
    #[must_use]
    pub fn new_custom(
        name: &str,
        mean_ri: f32,
        dispersion_delta: f32,
        birefringence_delta: f32,
        absorption_rgb: [f32; 3],
    ) -> Self {
        // Cauchy dispersion model fit: n(lambda) = A + B / lambda^2
        // where lambda is in um, lambda_D = 0.5893 um (sodium D line).
        const LAMBDA_D_UM: f32 = 0.5893;
        // Fraunhofer F (486.1nm) and C (656.3nm) lines, matching every other F-C
        // measurement in this crate (see `color::metrics::evaluate_gem_optical_metrics`
        // and `all_materials`' own verification comments, which evaluate at these exact
        // two wavelengths).
        const LAMBDA_F_UM: f32 = 0.4861;
        const LAMBDA_C_UM: f32 = 0.6563;
        let lambda_d_sq = LAMBDA_D_UM * LAMBDA_D_UM;
        let k_fc = 1.0 / (1.0 / (LAMBDA_F_UM * LAMBDA_F_UM) - 1.0 / (LAMBDA_C_UM * LAMBDA_C_UM));
        let b = (dispersion_delta * k_fc).max(0.0);
        let a = (mean_ri - b / lambda_d_sq).max(1.0);

        Self {
            name: name.to_string(),
            crystal_system: if birefringence_delta.abs() > 1e-4 {
                CrystalSystem::Trigonal
            } else {
                CrystalSystem::Cubic
            },
            optical_character: if birefringence_delta > 1e-4 {
                OpticalCharacter::UniaxialPositive
            } else if birefringence_delta < -1e-4 {
                OpticalCharacter::UniaxialNegative
            } else {
                OpticalCharacter::Isotropic
            },
            dispersion: DispersionModel::Cauchy { a, b, c: 0.0 },
            birefringence_delta,
            // `new_custom`'s public signature keeps accepting a plain
            // `[R, G, B]` triple (callers, including existing tests, pass one) --
            // internally converted to the new band-set representation via
            // `legacy_rgb_bands` (same three-lobe shape the whole codebase used before;
            // see that function's doc comment).
            absorption: AbsorptionTensor::isotropic(legacy_rgb_bands(absorption_rgb)),
            // No caller currently supplies a c-axis for a custom material, so
            // default to Vec3::Y -- the same value `trace_spectral_ray` previously had
            // hard-coded for every material, so this keeps `new_custom` behaviour
            // unchanged rather than adding a new parameter every call site would need
            // to be updated for.
            c_axis: Vec3::Y,
            // No caller currently supplies biaxial principal-index data
            // for a custom material -- `new_custom` remains a uniaxial/isotropic
            // constructor (see `optical_character`/`crystal_system` above, which never
            // produce Biaxial* for this constructor either).
            biaxial_delta_beta_alpha: None,
            scattering_sigma_s: 0.0,
            scattering_g: 0.0,
            edge_rounding_radius: 0.0,
            absorption_path_scale: 1.0,
        }
    }

    /// Inclusion/subsurface scattering: opts an existing material into a
    /// homogeneous Henyey-Greenstein scattering medium (silk/rutile/cloud inclusions),
    /// leaving every other field -- crucially including `absorption` -- untouched. A
    /// pure builder-style setter (`GemMaterial::ruby().with_scattering(0.4, 0.3)`),
    /// added so scenes/tests can opt individual materials in without a breaking change
    /// to `new_custom`'s or any built-in constructor's signature. See
    /// `scattering_sigma_s`/`scattering_g`'s own doc comments for what each parameter
    /// means; `sigma_s <= 0.0` (the default every built-in keeps) disables the feature
    /// entirely, reproducing today's exact deterministic Beer-Lambert path.
    #[must_use]
    pub const fn with_scattering(mut self, sigma_s: f32, g: f32) -> Self {
        self.scattering_sigma_s = sigma_s;
        self.scattering_g = g;
        self
    }

    /// A sensible default Henyey-Greenstein asymmetry for a caller who only wants to
    /// dial the AMOUNT of scattering ([`Self::with_scattering_amount`]) without thinking
    /// about anisotropy separately: mild forward scattering, physically typical for
    /// small needle-like/particulate inclusions (silk, rutile) at visible wavelengths --
    /// not a measured value for any specific species, just a reasonable "character"
    /// default.
    pub const DEFAULT_SCATTERING_G: f32 = 0.4;

    /// [`Self::with_scattering`] with [`Self::DEFAULT_SCATTERING_G`] for `g`, for a
    /// caller who wants a single "how much inclusion haze" knob. Two independent
    /// physical parameters genuinely exist here -- `sigma_s` (amount) and `g`
    /// (character: forward-scattering silk reads very differently from near-isotropic
    /// cloud) -- so this is a convenience on top of [`Self::with_scattering`], not a
    /// replacement for it; a caller who cares about the distinction should call
    /// `with_scattering` directly with an explicit `g`.
    #[must_use]
    pub const fn with_scattering_amount(self, sigma_s: f32) -> Self {
        self.with_scattering(sigma_s, Self::DEFAULT_SCATTERING_G)
    }

    /// A plausible per-species `(sigma_s, g)` starting point for "this species is
    /// typically included/hazy" -- e.g. Emerald's proverbial *jardin* -- keyed by this
    /// material's own `name`.
    ///
    /// # These are aesthetic choices, not measurements
    ///
    /// Unlike this material's Sellmeier dispersion coefficients or its pleochroic
    /// absorption bands (both cited to specific spectroscopic sources in
    /// [`Self::all_materials`]), there is no published "typical `sigma_s`" for any gem
    /// species -- inclusion density varies enormously by individual specimen, locality,
    /// and treatment, and clarity is conventionally assessed by eye/loupe grading, not a
    /// volumetric scattering coefficient. These numbers were chosen to LOOK plausible
    /// (documented per species below in descriptive terms -- "typically included",
    /// "usually eye-clean" -- deliberately NOT any standardized clarity-grade vocabulary
    /// like GIA's VVS/VS/SI scale, which grades visible-inclusion appearance under 10x
    /// magnification, a different thing this coefficient does not claim to reproduce),
    /// not derived from any citable source. Every built-in material's OWN
    /// `scattering_sigma_s` still stays exactly `0.0` (see that field's doc comment) --
    /// this method has no effect until a caller explicitly opts in via
    /// `material.with_recommended_scattering()`.
    #[must_use]
    pub fn recommended_scattering(&self) -> (f32, f32) {
        match self.name.as_str() {
            // Typically visibly included (the proverbial "jardin", French for garden --
            // multi-phase fluid inclusions and growth-tube silk are part of how an
            // untreated natural emerald is expected to look, not a flaw to hide).
            "Emerald" => (0.6, 0.35),
            // Silk (fine rutile needles) is common in natural corundum; heat treatment
            // (the overwhelming majority of the commercial supply) dissolves much of it,
            // so this is a moderate, not extreme, default.
            "Ruby" => (0.3, 0.5),
            "Sapphire" => (0.25, 0.5),
            // Natural rutilated quartz is famous for coarse, strongly forward-scattering
            // needles; ordinary rock crystal is usually clean. This default sits toward
            // the light-haze end since "Quartz" here is the generic rock-crystal entry.
            "Quartz" => (0.15, 0.3),
            // Elbaite tourmaline commonly carries visible needle/fingerprint inclusions.
            "Tourmaline" => (0.25, 0.4),
            // Faceted gem-quality diamond is typically eye-clean; a small isotropic
            // haze (cloud inclusions) rather than a strong directional character.
            "Diamond" => (0.02, 0.0),
            // Typically eye-clean once faceted (heat-treated zircon in particular).
            "Zircon" | "Alexandrite" | "Topaz" | "Spinel" | "Tanzanite" => (0.02, 0.2),
            // Lab-grown, essentially inclusion-free by construction.
            "Moissanite" | "Cubic Zirconia" => (0.0, 0.0),
            _ => (0.0, Self::DEFAULT_SCATTERING_G),
        }
    }

    /// [`Self::with_scattering`] using [`Self::recommended_scattering`]'s per-species
    /// `(sigma_s, g)` pair -- see that method's doc comment for why these are aesthetic
    /// defaults, not measurements. `GemMaterial::emerald().with_recommended_scattering()`
    /// reads at the call site the way a caller who wants "this species' typical included
    /// look" would expect.
    #[must_use]
    pub fn with_recommended_scattering(self) -> Self {
        let (sigma_s, g) = self.recommended_scattering();
        self.with_scattering(sigma_s, g)
    }

    /// Facet edge rounding: opts a material into a nonzero
    /// meet-edge rounding radius -- see [`Self::edge_rounding_radius`]'s doc comment for
    /// units/range. `radius <= 0.0` (the default every built-in keeps) reproduces
    /// today's perfectly sharp, measure-zero edges exactly.
    #[must_use]
    pub const fn with_edge_rounding(mut self, radius: f32) -> Self {
        self.edge_rounding_radius = radius;
        self
    }

    /// Model-units-to-absorption-length-units scale: opts a material into a
    /// physical size other than the implicit "girdle radius ~1 model unit" every
    /// built-in cut renders at -- see [`Self::absorption_path_scale`]'s doc comment.
    /// `scale == 1.0` (the default every built-in keeps) reproduces today's
    /// behaviour exactly (a multiply by exactly `1.0` is an IEEE 754 no-op).
    #[must_use]
    pub const fn with_absorption_path_scale(mut self, scale: f32) -> Self {
        self.absorption_path_scale = scale;
        self
    }

    /// Looks a material up by name, tolerating extra surrounding words in the query
    /// (e.g. a diagram title like "Fine Blue Sapphire").
    ///
    /// An exact match always wins outright. The substring fallback then prefers the
    /// LONGEST matching material name: "Zircon" is a substring of "Cubic Zirconia" and
    /// is listed earlier, so a naive first-match search silently returned Zircon —
    /// a completely different stone (`n_d` 1.92 vs 2.15) — for `by_name("Cubic Zirconia")`.
    #[must_use]
    pub fn by_name(name: &str) -> Option<Self> {
        let all = Self::all_materials();
        if let Some(m) = all.iter().find(|m| m.name.eq_ignore_ascii_case(name)) {
            return Some(m.clone());
        }
        let needle = name.to_lowercase();
        all.into_iter()
            .filter(|m| needle.contains(&m.name.to_lowercase()))
            .max_by_key(|m| m.name.len())
    }

    /// Convenience accessor for the built-in Diamond material.
    ///
    /// # Panics
    ///
    /// Panics if `"Diamond"` is ever removed from the built-in list returned by
    /// [`all_materials`](Self::all_materials) -- this is an internal consistency
    /// invariant of this module (every name used by a convenience accessor below must
    /// have a matching entry there), not something a caller can trigger.
    #[must_use]
    pub fn diamond() -> Self {
        Self::by_name("Diamond").expect("\"Diamond\" must be present in all_materials()")
    }

    /// Convenience accessor for the built-in Ruby material.
    ///
    /// # Panics
    ///
    /// Panics if `"Ruby"` is ever removed from the built-in list returned by
    /// [`all_materials`](Self::all_materials) -- see [`diamond`](Self::diamond) for why.
    #[must_use]
    pub fn ruby() -> Self {
        Self::by_name("Ruby").expect("\"Ruby\" must be present in all_materials()")
    }

    /// Convenience accessor for the built-in Sapphire material.
    ///
    /// # Panics
    ///
    /// Panics if `"Sapphire"` is ever removed from the built-in list returned by
    /// [`all_materials`](Self::all_materials) -- see [`diamond`](Self::diamond) for why.
    #[must_use]
    pub fn sapphire() -> Self {
        Self::by_name("Sapphire").expect("\"Sapphire\" must be present in all_materials()")
    }

    /// Convenience accessor for the built-in Emerald material.
    ///
    /// # Panics
    ///
    /// Panics if `"Emerald"` is ever removed from the built-in list returned by
    /// [`all_materials`](Self::all_materials) -- see [`diamond`](Self::diamond) for why.
    #[must_use]
    pub fn emerald() -> Self {
        Self::by_name("Emerald").expect("\"Emerald\" must be present in all_materials()")
    }

    /// Builds this material's full [`BiaxialIndicatrix`] (three principal
    /// indices `n_alpha` <= `n_beta` <= `n_gamma` plus their orthonormal axis frame) at
    /// `lambda_nm`, for the three biaxial (orthorhombic) built-ins. `None` for every
    /// other material -- see `biaxial_delta_beta_alpha`'s doc comment for the field
    /// this reads and the convention it documents (base `dispersion` curve == `n_beta`,
    /// `birefringence_delta` == `n_gamma` - `n_alpha`, `c_axis` doubles as the `n_gamma`
    /// principal axis).
    #[must_use]
    pub fn biaxial_indicatrix(&self, lambda_nm: f32) -> Option<BiaxialIndicatrix> {
        let delta_beta_alpha = self.biaxial_delta_beta_alpha?;
        let n_beta = self.dispersion.evaluate(lambda_nm);
        let n_alpha = n_beta - delta_beta_alpha;
        let n_gamma = n_alpha + self.birefringence_delta;
        Some(BiaxialIndicatrix::from_gamma_axis(
            n_alpha,
            n_beta,
            n_gamma,
            self.c_axis,
        ))
    }

    /// Phase 3 GPU routing predicate: whether this material's optics are supported by
    /// the GPU backend (`renderer::shaders::spectral_transport.wgsl`'s `transport_main`).
    ///
    /// A pure function of the material alone. Isotropic and uniaxial birefringent
    /// materials (the `theta_c` fixed-point iteration, the 50/50 ordinary/extraordinary
    /// eigenmode split, `extraordinary_poynting_dir` walk-off) have always been ported
    /// and GPU-capable.
    ///
    /// 2026-09-02: the genuinely BIAXIAL restriction (`biaxial_delta_beta_alpha.is_some()`
    /// -- Alexandrite, Topaz, Tanzanite among the built-ins) is LIFTED. It existed
    /// because `BiaxialIndicatrix::eigen_polarizations`/`mode_poynting_dir`'s CPU
    /// eigenvector construction was numerically ill-conditioned near mode degeneracy
    /// (the "cleared-denominator" formula's pre-normalization magnitude collapsed
    /// toward zero whenever the two mode indices were close -- the common case across a
    /// wide swath of directions for any weakly birefringent gem, not just near the two
    /// true optic axes), so a 1-ULP CPU/GPU input difference got amplified through
    /// `normalize` into a materially different unit vector: `eigen_polarizations`/
    /// `mode_poynting_dir` measured up to ~3.5M ULP against the GPU port even though
    /// that port was a faithful, already-complete translation. Fixed by reformulating
    /// the eigenvector construction (`BiaxialIndicatrix::eigenvector_world`: the
    /// symmetric "transverse impermeability" matrix `Gamma = P.B.P`, its null vector
    /// extracted via a smooth, sign-aligned sum of row-pair cross products rather than a
    /// hard largest-of-three selection) AND `wave_indices`' own discriminant
    /// (`BiaxialIndicatrix::precise_root_near`: an algebraically-exact reformulation
    /// that replaces the `B^2-4C` cancellation with Sterbenz-exact differences of the
    /// principal `1/n^2` values) -- see both functions' doc comments in
    /// `optics::birefringence` for the full derivations. Ported op-for-op to
    /// `transport_physics.wgsl`, then FULLY reverified: every phase of
    /// `examples/gpu_equivalence_harness` passes, including `eigen_polarizations` and
    /// `mode_poynting_dir` at genuine 0 ULP (an `abs_floor` in `transport_check.rs`
    /// covers the last few ULP of ordinary cross-platform `f32` rounding noise on
    /// components that are themselves near a direction-dependent zero-crossing -- see
    /// `BIAXIAL_EIGEN_POLARIZATION_ABS_FLOOR`'s own doc comment) and the Tier-3
    /// statistical image comparison for Alexandrite, Topaz and Tanzanite.
    ///
    /// Deliberately checks only material data, never anything about the machine running
    /// the render (GPU availability, VRAM, load): the whole point of a routing predicate
    /// living in this crate's hashed source (`lib::BUILD_ID` covers `src/**/*.rs`) is
    /// that a viewer and a remote render worker -- each independently deciding whether to
    /// hand a given scene to the GPU backend -- can never disagree about what runs where,
    /// which requires the decision to depend on nothing but the scene (here: the
    /// material) itself. A caller that assembles a full scene (material + geometry +
    /// environment) should call this once per distinct material the scene uses and
    /// require all of them to return `true` before routing that scene's render to the
    /// GPU backend.
    #[must_use]
    pub const fn gpu_supported(&self) -> bool {
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// "Zircon" is a substring of "Cubic Zirconia" and is listed earlier in
    /// `all_materials()`, so a first-match search silently returned the wrong stone.
    #[test]
    fn by_name_prefers_the_longest_match_not_the_first() {
        let cz = GemMaterial::by_name("Cubic Zirconia").expect("Cubic Zirconia must resolve");
        assert_eq!(cz.name, "Cubic Zirconia", "must not fall through to Zircon");

        let zircon = GemMaterial::by_name("Zircon").expect("Zircon must resolve");
        assert_eq!(
            zircon.name, "Zircon",
            "an exact name must still resolve to itself"
        );

        assert!(
            (cz.dispersion.evaluate(589.3) - zircon.dispersion.evaluate(589.3)).abs() > 0.1,
            "test premise: the two stones must have clearly different refractive indices"
        );
    }

    /// Every built-in material must resolve to itself by its own exact name.
    #[test]
    fn by_name_round_trips_every_builtin_material() {
        for m in GemMaterial::all_materials() {
            let found = GemMaterial::by_name(&m.name)
                .unwrap_or_else(|| panic!("{} must resolve by its own name", m.name));
            assert_eq!(
                found.name, m.name,
                "{} resolved to the wrong material",
                m.name
            );
        }
    }

    /// The substring fallback is what lets a diagram title carry extra words.
    #[test]
    fn by_name_still_matches_a_name_embedded_in_a_longer_string() {
        let m = GemMaterial::by_name("Fine Blue Sapphire, Ceylon").expect("should match Sapphire");
        assert_eq!(m.name, "Sapphire");
    }

    /// Verifies every built-in material's computed Abbe number
    /// `V_d = (n_d - 1) / (n_F - n_C)` against a trusted-source reference value, so a
    /// future mistyped coefficient (Sellmeier or Cauchy) gets caught instead of
    /// silently shipping -- see each material's own doc comment in `all_materials` for
    /// the primary/reputable source and the derivation of its target Abbe number.
    ///
    /// Source priority for these targets (re-verified against trusted online sources,
    /// which now OUTRANK `GEMSTONE_RENDERING_BLUEPRINT.md` -- the document is used only
    /// as a tiebreaker/corroboration, never as the primary authority, because it is a
    /// research compilation whose own "Fire Delta n(F-C)" column demonstrably mixes
    /// Fraunhofer B-G and F-C conventions row-to-row, e.g. Quartz lists 0.013, the
    /// well-known B-G figure, in a column labelled F-C, contradicting that same row's
    /// own `V_d`):
    ///   1. Primary optical measurements (Sellmeier/Cauchy fits from refractiveindex.info
    ///      and the papers it cites).
    ///   2. Reputable gemological references (GIA, Mindat, Handbook of Mineralogy,
    ///      International Gem Society), corroborated by 2+ independent sources where
    ///      possible, with their "dispersion" figures explicitly treated as Fraunhofer
    ///      B-G (never plugged directly into an F-C slot).
    ///   3. The document, as a tiebreaker only.
    ///
    /// Two confidence tiers, reflected in the tolerance:
    ///   - TIGHT (0.5): Diamond, Sapphire, Ruby, Quartz, Spinel, Cubic Zirconia,
    ///     Alexandrite, Synthetic Moissanite -- dispersion coefficients transcribed
    ///     directly from a primary optical-constants paper (Peter 1923; Malitson &
    ///     Dodge 1972; Ghosh 1999; Tropf & Thomas 1991; Wood & Nassau 1982; Walling
    ///     1980; Wang 2013 corroborated by Singh 1971), target computed from that exact
    ///     published formula. Alexandrite and Moissanite CONTRADICT the document here
    ///     (68.0 -> 73.90 and 21.5 -> 25.94 respectively) -- per this task's source
    ///     priority the primary measurement wins; see each entry's comment in
    ///     `all_materials` for the two-source corroboration.
    ///   - LOWER CONFIDENCE (2.5-4.0, roughly 5-12% relative): Zircon, Topaz,
    ///     Tourmaline, Tanzanite, Emerald -- no primary Sellmeier/Cauchy fit exists in
    ///     the optics literature for these species (confirmed absent from
    ///     refractiveindex.info's database), so `n_d` is taken from 2+ corroborating
    ///     reputable gemological sources and the dispersion shape is a 2-parameter
    ///     Cauchy fit solved from that `n_d` and a Delta n(F-C) obtained by converting
    ///     each species' well-corroborated gemological B-G "dispersion" figure via a
    ///     B-G->F-C ratio (0.579, range 0.569-0.587) computed THIS pass directly from
    ///     the 8 genuine primary Sellmeier curves above -- not from cross-comparing
    ///     gemological tables against each other, as the previous pass's 0.591 factor
    ///     was. All five CONTRADICT the document (Zircon 28.0 -> 40.96, Topaz 64.0 ->
    ///     76.83, Tourmaline 55.0 -> 64.58, Tanzanite 45.0 -> 40.21, Emerald's estimate
    ///     shifts modestly from 69.99 to 70.93 with the refined ratio). Zircon also had
    ///     an unrelated, unsourced `n_d` error corrected this pass (1.956878 -> 1.925,
    ///     see its entry's comment) -- flagged prominently as a critical-angle-relevant
    ///     change.
    #[test]
    fn builtin_material_abbe_numbers_match_published_values() {
        // (name, sourced Abbe number V_d, tolerance)
        let expected: &[(&str, f32, f32)] = &[
            // -- tight tier: primary Sellmeier source for both n_d and V_d --
            ("Diamond", 55.27, 0.5),
            ("Sapphire", 72.27, 0.5),
            ("Ruby", 72.27, 0.5),
            ("Quartz", 69.65, 0.5),
            ("Spinel", 60.63, 0.5),
            ("Cubic Zirconia", 33.52, 0.5),
            ("Alexandrite", 73.90, 0.5),
            ("Synthetic Moissanite", 25.94, 0.5),
            // -- lower-confidence tier: no primary fit exists; n_d from corroborated
            // gemological sources, dispersion shape from a gemological B-G figure
            // converted via the physically-derived 0.579 ratio -- see each entry's
            // comment in `all_materials` --
            ("Zircon", 40.96, 3.0),
            ("Topaz", 76.83, 4.0),
            ("Tourmaline", 64.58, 3.5),
            ("Tanzanite", 40.21, 3.0),
            ("Emerald", 70.93, 3.5),
        ];

        for &(name, published_abbe, tol) in expected {
            let material = GemMaterial::by_name(name)
                .unwrap_or_else(|| panic!("{name} must be a built-in material"));
            let n_f = material.dispersion.evaluate(486.1);
            let n_d = material.dispersion.evaluate(589.3);
            let n_c = material.dispersion.evaluate(656.3);
            let dn = n_f - n_c;
            assert!(
                dn > 0.0,
                "{name}: Delta n(F-C) must be positive (normal dispersion), got {dn}"
            );
            let computed_abbe = (n_d - 1.0) / dn;
            assert!(
                (computed_abbe - published_abbe).abs() <= tol,
                "{name}: computed Abbe V_d={computed_abbe:.2} (n_d={n_d:.5}, n_F={n_f:.5}, n_C={n_c:.5}) \
                 does not match published V_d={published_abbe:.2} within tolerance {tol} -- check the \
                 dispersion coefficients against the source cited in all_materials()"
            );
        }
    }

    /// Companion to `builtin_material_abbe_numbers_match_published_values`: pins down
    /// `n_d` at the sodium D line (589.3nm) for every material this pass re-sourced, so
    /// a future coefficient edit that silently shifts `n_d` gets caught. `n_d` drives
    /// the critical angle and therefore brilliance/windowing/extinction, so any
    /// intentional change here is significant downstream and must be deliberate, not
    /// an accident of refitting the dispersion shape.
    ///
    /// For five of the six materials this task's predecessor touched (Synthetic
    /// Moissanite, Topaz, Tourmaline, Tanzanite, Alexandrite), `n_d` is unchanged from
    /// before -- only the dispersion shape (`V_d`) moved, per
    /// `builtin_material_abbe_numbers_match_published_values`.
    ///
    /// Zircon is the exception, and the one material in this file whose `n_d` this
    /// pass DID change materially: from 1.956878 to 1.925, a -0.0319 shift, far above
    /// the ~0.002 threshold at which critical-angle-driven behaviour (brilliance,
    /// windowing, extinction) is expected to move. The old value was never actually
    /// sourced from anywhere (not the document, which gives `n_o=1.9250`, nor any
    /// external reference found); 1.925 is corroborated by the Handbook of Mineralogy,
    /// International Gem Society, and the document itself. See Zircon's comment in
    /// `all_materials` for the full derivation. This is flagged prominently in this
    /// task's report as a coordination point for re-stabilising any test elsewhere in
    /// the workspace that hardcodes zircon's brilliance/windowing/pose behaviour.
    #[test]
    fn builtin_material_n_d_matches_sourced_values() {
        // (name, sourced n_d, tolerance)
        let expected_n_d: &[(&str, f32, f32)] = &[
            ("Zircon", 1.925, 1e-4),
            ("Synthetic Moissanite", 2.647_434, 1e-4),
            ("Topaz", 1.627_178, 1e-4),
            ("Tourmaline", 1.639_405, 1e-4),
            ("Tanzanite", 1.700_858, 1e-4),
            ("Alexandrite", 1.742_73, 1e-4),
        ];

        for &(name, expected, tol) in expected_n_d {
            let material = GemMaterial::by_name(name)
                .unwrap_or_else(|| panic!("{name} must be a built-in material"));
            let n_d = material.dispersion.evaluate(589.3);
            assert!(
                (n_d - expected).abs() < tol,
                "{name}: n_d={n_d:.6} does not match the sourced value {expected:.6} -- check the \
                 dispersion coefficients against the source cited in all_materials(); if this is a \
                 deliberate re-sourcing, update this test's expected value and flag the n_d change \
                 in the task report if it moves by more than ~0.002 (critical-angle significance)"
            );
        }
    }

    /// Only the three orthorhombic (biaxial) built-ins -- Alexandrite,
    /// Topaz, Tanzanite -- should carry biaxial principal-index data; every other
    /// material (isotropic or uniaxial) must resolve to `None`, keeping them on the
    /// existing uniaxial code path unchanged.
    #[test]
    fn only_biaxial_materials_expose_a_biaxial_indicatrix() {
        let biaxial_names = ["Alexandrite", "Topaz", "Tanzanite"];
        for material in GemMaterial::all_materials() {
            let indicatrix = material.biaxial_indicatrix(589.3);
            if biaxial_names.contains(&material.name.as_str()) {
                assert!(
                    indicatrix.is_some(),
                    "{} should expose a biaxial indicatrix",
                    material.name
                );
            } else {
                assert!(
                    indicatrix.is_none(),
                    "{} should NOT expose a biaxial indicatrix (uniaxial/isotropic)",
                    material.name
                );
            }
        }
    }

    /// Phase 3 GPU routing test, re-baselined 2026-09-02 when the biaxial restriction
    /// was lifted (see `gpu_supported`'s own doc comment for the eigenvector-
    /// conditioning fix and full re-verification that made this safe): `gpu_supported`
    /// must now report `true` for EVERY built-in material, biaxial ones (Alexandrite,
    /// Topaz, Tanzanite -- the same set `only_biaxial_materials_expose_a_biaxial_indicatrix`
    /// above pins for `biaxial_indicatrix`) included. Named to mirror the pre-fix test
    /// this replaced (`gpu_supported_is_false_only_for_biaxial_materials`), which pinned
    /// the opposite verdict for exactly the same reason: keeping a single test that
    /// tracks the routing predicate's CURRENT contract, rather than layering an
    /// exception list on top, so a future regression shows up here directly.
    #[test]
    fn gpu_supported_is_true_for_every_built_in_material() {
        let biaxial_names = ["Alexandrite", "Topaz", "Tanzanite"];
        for material in GemMaterial::all_materials() {
            assert!(
                material.gpu_supported(),
                "{} must be reported gpu_supported() == true",
                material.name
            );
            if biaxial_names.contains(&material.name.as_str()) {
                assert!(
                    material.biaxial_delta_beta_alpha.is_some(),
                    "{} must actually carry biaxial data for this test to be meaningful",
                    material.name
                );
            }
        }
    }

    /// Re-baselined 2026-09-02 alongside the test above: `gpu_supported` no longer
    /// depends on `biaxial_delta_beta_alpha` at all (nor on anything else) -- it is
    /// `true` unconditionally now that the biaxial eigenvector conditioning fix let
    /// Alexandrite/Topaz/Tanzanite pass the same GPU-equivalence bar every other
    /// material already had to. Kept as a dedicated test (rather than folded into the
    /// one above) specifically because the OLD behavior here was itself a documented,
    /// load-bearing contract point -- pinning that it is GONE, not merely untested, is
    /// the useful signal a future regression (e.g. reintroducing a biaxial-only gate)
    /// should trip.
    #[test]
    fn gpu_supported_no_longer_depends_on_biaxial_delta_beta_alpha() {
        let mut zircon = GemMaterial::by_name("Zircon").expect("Zircon must be a built-in");
        assert!(
            zircon.gpu_supported(),
            "Zircon (uniaxial) must be GPU-supported"
        );

        zircon.birefringence_delta *= -3.0;
        zircon.c_axis = Vec3::X;
        assert!(
            zircon.gpu_supported(),
            "changing unrelated uniaxial fields must not affect gpu_supported()"
        );

        zircon.biaxial_delta_beta_alpha = Some(0.0);
        assert!(
            zircon.gpu_supported(),
            "biaxial_delta_beta_alpha no longer gates GPU support as of 2026-09-02"
        );

        zircon.biaxial_delta_beta_alpha = None;
        assert!(zircon.gpu_supported(), "and stays supported once cleared");
    }

    /// Each biaxial built-in's indicatrix must order its three principal indices
    /// `n_alpha` <= `n_beta` <= `n_gamma`; must place `n_beta` exactly at the
    /// material's own base dispersion curve (the documented convention); and must
    /// place `n_gamma` minus `n_alpha` exactly at the material's own
    /// `birefringence_delta` (its pre-existing "or max-min" meaning, unchanged) --
    /// pinning the `biaxial_indicatrix` wiring itself, independent of whether the
    /// underlying `biaxial_delta_beta_alpha` numbers are later refined.
    #[test]
    fn biaxial_indicatrix_is_internally_consistent_with_existing_fields() {
        for name in ["Alexandrite", "Topaz", "Tanzanite"] {
            let material = GemMaterial::by_name(name)
                .unwrap_or_else(|| panic!("{name} must be a built-in material"));
            let n_d = material.dispersion.evaluate(589.3);
            let indicatrix = material
                .biaxial_indicatrix(589.3)
                .unwrap_or_else(|| panic!("{name} must expose a biaxial indicatrix"));

            assert!(
                indicatrix.n_alpha <= indicatrix.n_beta && indicatrix.n_beta <= indicatrix.n_gamma,
                "{name}: principal indices must be ordered n_alpha<=n_beta<=n_gamma, got ({}, {}, {})",
                indicatrix.n_alpha,
                indicatrix.n_beta,
                indicatrix.n_gamma
            );
            assert!(
                (indicatrix.n_beta - n_d).abs() < 1e-5,
                "{name}: n_beta ({}) must equal the base dispersion curve's n_d ({n_d})",
                indicatrix.n_beta
            );
            assert!(
                (indicatrix.n_gamma - indicatrix.n_alpha - material.birefringence_delta).abs()
                    < 1e-5,
                "{name}: n_gamma - n_alpha ({}) must equal birefringence_delta ({})",
                indicatrix.n_gamma - indicatrix.n_alpha,
                material.birefringence_delta
            );
        }
    }

    /// `new_custom`'s `dispersion_delta` must be interpreted as the Fraunhofer
    /// F-C interval -- i.e. `n(486.1nm) - n(656.3nm)` on the constructed material must
    /// equal the requested `dispersion_delta`, not 66% or 113% of it (the old flat
    /// `0.347` multiplier's actual behaviour). Checked across several representative
    /// deltas, including one large enough
    /// that a wrong conversion factor would be very obvious.
    #[test]
    fn new_custom_dispersion_delta_measures_exactly_at_f_and_c() {
        for dispersion_delta in [0.005f32, 0.010, 0.02564, 0.05] {
            let material =
                GemMaterial::new_custom("F-C probe", 1.6, dispersion_delta, 0.0, [0.0, 0.0, 0.0]);
            let n_f = material.dispersion.evaluate(486.1);
            let n_c = material.dispersion.evaluate(656.3);
            let measured = n_f - n_c;
            assert!(
                (measured - dispersion_delta).abs() < 1e-4,
                "requested F-C delta {dispersion_delta} but measured {measured} \
                 (n_F={n_f}, n_C={n_c})"
            );
        }
    }

    /// Trichroism convention pin for Alexandrite's own shipped data (the same
    /// discipline `birefringence::biaxial_reduction_tests` applies with synthetic
    /// coefficients, applied here to the real Farrell & Newnham-derived entry):
    /// each principal direction's band set must carry the qualitative amplitude
    /// pattern the cited pleochroic colours dictate, and feeding the shipped band
    /// sums through the REAL `AbsorptionTensor3::biaxial` constructor must land each
    /// one on its own world axis (alpha -> +X, beta -> Z, gamma -> +Y for
    /// `c_axis = Vec3::Y`, per `stable_orthonormal_basis`'s pinned construction) --
    /// so a future swap of the alpha/beta/gamma argument order, or of the axis
    /// convention underneath, fails loudly instead of silently recolouring the stone.
    #[test]
    fn alexandrite_trichroic_band_sets_follow_the_cited_pleochroic_pattern() {
        use super::super::birefringence::AbsorptionTensor3;

        let alex =
            GemMaterial::by_name("Alexandrite").expect("Alexandrite must be a built-in material");
        let bands_alpha = &alex.absorption.o_ray;
        let bands_gamma = &alex.absorption.e_ray;
        let bands_beta = alex
            .absorption
            .beta_ray
            .as_deref()
            .expect("Alexandrite must carry a third (beta) trichroic band set");
        assert!(
            alex.absorption.is_pleochroic,
            "Alexandrite's trichroic tensor must be flagged pleochroic"
        );

        let sum_at = |bands: &[AbsorptionBand], nm: f32| -> f32 {
            bands.iter().map(|b| b.evaluate(nm)).sum()
        };

        // 4T2 system (yellow-to-red region, evaluated at gamma's cited 595nm centre):
        // gamma (crystal b, GREEN -- absorbs "both red and yellow") strongest, alpha
        // (crystal c, RED) intermediate, beta (crystal a, YELLOW -- red/yellow left
        // open) weakest, per the figure-read Fig. 4 amplitudes in the entry's comment.
        let (t2_alpha, t2_beta, t2_gamma) = (
            sum_at(bands_alpha, 595.0),
            sum_at(bands_beta, 595.0),
            sum_at(bands_gamma, 595.0),
        );
        assert!(
            t2_gamma > t2_alpha && t2_alpha > t2_beta,
            "4T2 amplitude order must be gamma(green) > alpha(red) > beta(yellow), got \
             gamma={t2_gamma:.3}, alpha={t2_alpha:.3}, beta={t2_beta:.3}"
        );

        // 4T1 system (blue-violet, evaluated at beta's cited 422nm centre): beta
        // (crystal a) strongest -- F&N: the 0.42u absorption "dominates the a
        // spectrum" -- with alpha and gamma both clearly weaker.
        let (t1_alpha, t1_beta, t1_gamma) = (
            sum_at(bands_alpha, 422.0),
            sum_at(bands_beta, 422.0),
            sum_at(bands_gamma, 422.0),
        );
        assert!(
            t1_beta > t1_alpha && t1_beta > t1_gamma,
            "4T1 amplitude must peak on beta(yellow, crystal a), got alpha={t1_alpha:.3}, \
             beta={t1_beta:.3}, gamma={t1_gamma:.3}"
        );

        // Convention pin through the real constructor: with c_axis = +Y, the alpha
        // set's sum must appear along +X, the beta set's along Z, and the gamma set's
        // along +Y (the `c_axis`/n_gamma direction) -- see
        // `AbsorptionTensor3::biaxial`'s doc comment.
        assert_eq!(
            alex.c_axis,
            Vec3::Y,
            "test premise: Alexandrite c_axis is +Y"
        );
        let tensor = AbsorptionTensor3::biaxial(t2_alpha, t2_beta, t2_gamma, alex.c_axis);
        for (axis, expected, label) in [
            (Vec3::X, t2_alpha, "alpha on +X"),
            (Vec3::Z, t2_beta, "beta on Z"),
            (Vec3::Y, t2_gamma, "gamma on +Y (c_axis)"),
        ] {
            let measured = tensor.quadratic_form(axis);
            assert!(
                (measured - expected).abs() < 1e-5,
                "{label}: quadratic_form along {axis:?} must return that principal set's \
                 band sum ({expected:.4}), got {measured:.4}"
            );
        }
    }

    /// The mean refractive index (`mean_ri`) must be preserved exactly at the sodium D
    /// line regardless of `dispersion_delta` -- the F-C conversion factor changes how
    /// steeply `n(lambda)` varies away from D, not its value AT D (the Cauchy `a` term
    /// is solved to compensate exactly, per `new_custom`'s own formula).
    #[test]
    fn new_custom_preserves_mean_ri_at_sodium_d_line_regardless_of_dispersion_delta() {
        for dispersion_delta in [0.0f32, 0.01, 0.03] {
            let material = GemMaterial::new_custom(
                "D-line probe",
                1.72,
                dispersion_delta,
                0.0,
                [0.0, 0.0, 0.0],
            );
            let n_d = material.dispersion.evaluate(589.3);
            assert!(
                (n_d - 1.72).abs() < 1e-4,
                "dispersion_delta={dispersion_delta}: n_d should stay 1.72, got {n_d}"
            );
        }
    }
}
