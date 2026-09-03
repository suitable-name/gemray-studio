use super::polarization::{StokesVector, electric_field_direction};
use glam::{Mat3, Vec3};

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BirefringenceParams {
    pub delta_n: f32,
    pub c_axis: [f32; 3],
}

impl BirefringenceParams {
    #[must_use]
    pub fn new(delta_n: f32, c_axis: Vec3) -> Self {
        let norm_c = c_axis.normalize_or_zero();
        Self {
            delta_n,
            c_axis: [norm_c.x, norm_c.y, norm_c.z],
        }
    }

    #[must_use]
    pub const fn c_axis_vec3(&self) -> Vec3 {
        Vec3::from_array(self.c_axis)
    }

    /// Evaluates direction-dependent extraordinary refractive index `n_e(theta)`.
    #[must_use]
    pub fn effective_extraordinary_index(n_o: f32, n_e: f32, theta: f32) -> f32 {
        if (n_o - n_e).abs() < 1e-5 {
            return n_o;
        }
        let sin_t = theta.sin();
        let cos_t = theta.cos();
        let denom_sq = (n_e * cos_t).mul_add(n_e * cos_t, (n_o * sin_t).powi(2));
        if denom_sq <= 1e-8 {
            return n_o;
        }
        (n_o * n_e) / denom_sq.sqrt()
    }

    /// Walk-off angle estimation for extraordinary ray Poynting vector.
    #[must_use]
    pub fn walk_off_angle(n_o: f32, n_e: f32, theta: f32) -> f32 {
        if (n_o - n_e).abs() < 1e-5 {
            return 0.0;
        }
        let n_o2 = n_o * n_o;
        let n_e2 = n_e * n_e;
        let sin_t = theta.sin();
        let cos_t = theta.cos();
        let tan_rho = ((n_o2 - n_e2) * sin_t * cos_t)
            / (n_e2 * cos_t)
                .mul_add(cos_t, n_o2 * sin_t * sin_t)
                .max(1e-6);
        tan_rho.atan()
    }

    /// Calculates the deviated Poynting energy direction for the extraordinary ray.
    ///
    /// The optic axis is a DIRECTOR, not a directed vector -- a crystal's `c_axis` and
    /// `-c_axis` describe the same physical axis, so this must satisfy `S(c) == S(-c)`
    /// for every `wave_normal`. It didn't: `theta` (and therefore `delta`, the walk-off
    /// magnitude) was computed from the UNSIGNED `|cos_theta|`, but `c_proj` -- the
    /// in-plane direction the wave normal tilts toward -- was built from the SIGNED
    /// `cos_theta`, which flips sign under `c_axis -> -c_axis` while `delta` does not.
    /// The two branches (`wave_normal.dot(c_axis)` positive vs. negative) therefore
    /// tilted by `+delta` and `-delta` respectively relative to the correct answer --
    /// only one of them (the negative-dot branch) happened to land on the physically
    /// correct side. Folding `c_axis` onto the wave normal's own hemisphere first (via
    /// `sign`) and negating the tilt on that branch to compensate makes both branches
    /// agree, matching the always-correct general `poynting_direction`/biaxial
    /// `mode_poynting_dir` construction this uniaxial fast path approximates. See
    /// `birefringence_reduction_tests::extraordinary_poynting_dir_is_axis_direction_symmetric`.
    #[must_use]
    pub fn extraordinary_poynting_dir(wave_normal: Vec3, c_axis: Vec3, n_o: f32, n_e: f32) -> Vec3 {
        let cos_theta = wave_normal.dot(c_axis).clamp(-1.0, 1.0);
        let theta = cos_theta.abs().acos();
        let delta = Self::walk_off_angle(n_o, n_e, theta);

        if delta.abs() < 1e-5 {
            return wave_normal;
        }

        // Walk-off occurs in the plane containing wave normal and c-axis
        let c_proj = (c_axis - cos_theta * wave_normal).normalize_or_zero();
        if c_proj.length_squared() < 1e-6 {
            return wave_normal;
        }
        let sign = if cos_theta >= 0.0 { 1.0 } else { -1.0 };

        (wave_normal * delta.cos() - sign * c_proj * delta.sin()).normalize()
    }

    /// The ordinary eigenmode's electric-field direction for a wave travelling along
    /// `wave_normal` in a uniaxial crystal with optical axis `c_axis`: perpendicular to
    /// the plane containing the wave normal and the c-axis, so it always has zero
    /// component along `c_axis` (purely "ordinary") regardless of propagation angle.
    /// Degenerates to an arbitrary direction perpendicular to `wave_normal` when
    /// `wave_normal` is parallel to `c_axis` (propagation along the optic axis, where
    /// the two eigenmodes are physically degenerate anyway -- no birefringence there).
    #[must_use]
    pub fn ordinary_eigen_polarization(wave_normal: Vec3, c_axis: Vec3) -> Vec3 {
        let cross = wave_normal.cross(c_axis);
        if cross.length_squared() > 1e-8 {
            cross.normalize()
        } else {
            stable_orthonormal_basis(wave_normal.normalize_or_zero()).0
        }
    }

    /// The extraordinary eigenmode's electric-field direction: perpendicular to both
    /// the wave normal and the ordinary eigenmode above, i.e. lying in the plane
    /// containing the wave normal and the c-axis (the plane the walk-off displacement
    /// itself occurs in -- see `extraordinary_poynting_dir`).
    #[must_use]
    pub fn extraordinary_eigen_polarization(wave_normal: Vec3, c_axis: Vec3) -> Vec3 {
        let o_hat = Self::ordinary_eigen_polarization(wave_normal, c_axis);
        wave_normal.cross(o_hat).normalize_or_zero()
    }
}

/// A stable (branch-minimal) orthonormal basis `(t, b)` perpendicular to unit vector
/// `n`. Used both to fill in a uniaxial [`AbsorptionTensor3`]'s two degenerate
/// principal axes (their specific directions don't matter there -- only that they're
/// orthonormal to each other and to `n`) and as a last-resort fallback when a
/// direction-dependent construction elsewhere degenerates.
fn stable_orthonormal_basis(n: Vec3) -> (Vec3, Vec3) {
    let a = if n.x.abs() > 0.9 { Vec3::Y } else { Vec3::X };
    let t = (a - n * n.dot(a)).normalize_or_zero();
    let b = n.cross(t);
    (t, b)
}

/// Deterministic sign convention for an eigenvector (an eigenvector is only defined up
/// to an overall sign): flips `v` so that its largest-magnitude component is positive.
/// Ties (component magnitudes exactly equal) resolve in x, then y, then z priority --
/// the same `>=`/`>=` comparison structure on both the CPU (`f32`) and GPU (WGSL
/// `f32`) sides, so both agree bit-for-bit on which component decides the sign. Used by
/// `BiaxialIndicatrix::eigenvector_world`.
fn canonicalize_eigenvector_sign(v: Vec3) -> Vec3 {
    let ax = v.x.abs();
    let ay = v.y.abs();
    let az = v.z.abs();
    let largest = if ax >= ay && ax >= az {
        v.x
    } else if ay >= az {
        v.y
    } else {
        v.z
    };
    if largest < 0.0 { -v } else { v }
}

/// `a.cross(b)`, computed via explicit `mul_add` (fused multiply-subtract) rather than
/// plain `*`/`-`, so it rounds identically to the WGSL mirror's `fma`-based
/// `cross_fma` in `transport_physics.wgsl` -- see `BiaxialIndicatrix::eigenvector_world`'s
/// doc comment for why this specific cross product needs that guarantee.
fn cross_fma(a: Vec3, b: Vec3) -> Vec3 {
    Vec3::new(
        a.y.mul_add(b.z, -(a.z * b.y)),
        a.z.mul_add(b.x, -(a.x * b.z)),
        a.x.mul_add(b.y, -(a.y * b.x)),
    )
}

/// `a.dot(b)`, computed via explicit `mul_add` for the same CPU/GPU rounding-parity
/// reason as `cross_fma` above -- used by `BiaxialIndicatrix::eigenvector_world` to
/// determine the ROBUST sign relating two (anti)parallel cross products.
fn dot_fma(a: Vec3, b: Vec3) -> f32 {
    a.x.mul_add(b.x, a.y.mul_add(b.y, a.z * b.z))
}

/// Symmetric second-rank absorption tensor **A**, diagonal in the crystal's principal
/// axes.
///
/// Represented as three principal coefficients plus that orthonormal axis
/// frame (`axes`) rather than a dense 3x3 matrix -- cheaper to evaluate, and the
/// natural form gem chromophore data already takes (an ordinary/extraordinary pair via
/// [`Self::uniaxial`], or three independent coefficients for a fully biaxial tensor).
///
/// Absorption in an anisotropic crystal depends on the light's ELECTRIC-FIELD
/// polarization direction, not its propagation direction -- this is pleochroism, the
/// same tanzanite reading blue, violet or brownish depending on which polarization you
/// view it through. See [`Self::quadratic_form`].
#[derive(Debug, Clone, Copy)]
pub struct AbsorptionTensor3 {
    /// Principal absorption coefficients, one per column of `axes`, same order.
    pub alpha: Vec3,
    /// Orthonormal principal-axis frame in world space; columns are the three
    /// principal directions the tensor is diagonal in.
    pub axes: Mat3,
}

impl AbsorptionTensor3 {
    /// Builds the uniaxial special case directly from a material's existing
    /// ordinary/extraordinary absorption coefficients and single optical c-axis: two
    /// degenerate principal coefficients (`alpha_o`) in the plane perpendicular to
    /// `c_axis`, and one (`alpha_e`) along it. The in-plane pair of principal axes is
    /// arbitrary -- the tensor is isotropic within that plane, so their specific
    /// directions don't affect `quadratic_form`'s result -- and filled in via
    /// `stable_orthonormal_basis`.
    #[must_use]
    pub fn uniaxial(alpha_o: f32, alpha_e: f32, c_axis: Vec3) -> Self {
        let c = c_axis.normalize_or_zero();
        let (a1, a2) = stable_orthonormal_basis(c);
        Self {
            alpha: Vec3::new(alpha_o, alpha_o, alpha_e),
            axes: Mat3::from_cols(a1, a2, c),
        }
    }

    /// Trichroism: the genuinely biaxial case -- three INDEPENDENT
    /// principal absorption coefficients rather than `uniaxial`'s degenerate
    /// `(alpha_o, alpha_o, alpha_e)` pair.
    ///
    /// Critically, this builds its axis frame via the exact SAME
    /// `stable_orthonormal_basis(gamma_axis)` call, on the exact same input, that
    /// `BiaxialIndicatrix::from_gamma_axis` uses to build the material's INDEX
    /// principal frame -- not a merely-similar reconstruction. That is deliberate and
    /// load-bearing: `BiaxialIndicatrix::eigen_polarizations` (the eigenmode directions
    /// `effective_pleochroic_alpha` evaluates this tensor's `quadratic_form` against)
    /// is expressed in that SAME index frame, so if this tensor's absorption frame ever
    /// drifted from it -- even by a direction that's merely close, not identical --
    /// absorption would be silently scored against the wrong axes for every biaxial
    /// material with a third band set. Pinned by
    /// `biaxial_reduction_tests::absorption_frame_is_bit_identical_to_index_frame`.
    ///
    /// With `gamma_axis` alone determining the frame (no separate alpha/beta axis
    /// inputs, same as `BiaxialIndicatrix::from_gamma_axis`), `alpha` lands on the first
    /// `stable_orthonormal_basis` output, `beta` on the second, and `gamma` on
    /// `gamma_axis` itself -- e.g. for `gamma_axis = +Y`, `alpha` -> `+X`, `beta` ->
    /// `-Z`, `gamma` -> `+Y` (see `stable_orthonormal_basis`'s own construction).
    ///
    /// Degenerate case: `alpha == beta` reproduces `uniaxial(alpha, gamma, gamma_axis)`
    /// with a BIT-IDENTICAL result (same `Vec3::new` arguments in the same order, same
    /// `stable_orthonormal_basis` call on the same input -- not merely numerically
    /// close). See `biaxial_reduction_tests::biaxial_tensor_alpha_beta_degenerate_matches_uniaxial_bit_exact`.
    #[must_use]
    pub fn biaxial(alpha: f32, beta: f32, gamma: f32, gamma_axis: Vec3) -> Self {
        let g = gamma_axis.normalize_or_zero();
        let (a1, a2) = stable_orthonormal_basis(g);
        Self {
            alpha: Vec3::new(alpha, beta, gamma),
            axes: Mat3::from_cols(a1, a2, g),
        }
    }

    /// Evaluates the quadratic form `alpha = e_hat . A . e_hat` for a unit electric-
    /// field polarization direction `e_hat`. Since `A` is diagonal in `axes`, this is
    /// just the alpha-weighted sum of `e_hat`'s squared components along each
    /// principal axis -- no dense 3x3 matrix multiply needed.
    #[must_use]
    pub fn quadratic_form(&self, e_hat: Vec3) -> f32 {
        let local = self.axes.transpose() * e_hat;
        self.alpha.dot(local * local)
    }
}

/// Effective pleochroic (polarization-dependent) absorption coefficient.
///
/// Combines the light's actual electric-field direction (for the polarized fraction)
/// with an unweighted average over the two eigenmodes (for the unpolarized fraction)
/// -- physically, unpolarized light couples equally into both eigenmodes, so its
/// absorption is their average regardless of the crystal's own principal axes.
///
/// `degree_of_polarization` in `[0, 1]` (see `StokesVector::degree_of_polarization`)
/// linearly interpolates between the two limits. An isotropic tensor (`alpha_o ==
/// alpha_e`, or any tensor with all three principal coefficients equal) returns the
/// same value regardless of `e_hat`/`eigenmode_a`/`eigenmode_b`/`degree_of_polarization`,
/// so isotropic materials are automatically azimuth-independent with no special case
/// needed here.
#[must_use]
pub fn effective_pleochroic_alpha(
    tensor: &AbsorptionTensor3,
    e_hat: Vec3,
    eigenmode_a: Vec3,
    eigenmode_b: Vec3,
    degree_of_polarization: f32,
) -> f32 {
    let p = degree_of_polarization.clamp(0.0, 1.0);
    let alpha_polarized = tensor.quadratic_form(e_hat);
    let alpha_unpolarized = f32::midpoint(
        tensor.quadratic_form(eigenmode_a),
        tensor.quadratic_form(eigenmode_b),
    );
    p.mul_add(alpha_polarized - alpha_unpolarized, alpha_unpolarized)
}

/// All-in-one pleochroic Beer-Lambert coefficient for one spectral channel.
///
/// Derives the channel's own electric-field direction from its own `stokes` vector
/// (see `electric_field_direction`), builds this channel's `AbsorptionTensor3` from
/// `alpha_o`/`alpha_e` (and, for a genuinely biaxial material with a third band set,
/// `alpha_beta`), and combines with the two eigenmode directions via
/// `effective_pleochroic_alpha`. Pulled out as the single call site
/// `trace_spectral_ray`'s per-channel absorption loop needs, so that loop body stays
/// one line regardless of how much machinery sits behind it.
///
/// `alpha_beta` is `Some` only when the caller has BOTH confirmed the
/// material is genuinely biaxial (its eigenmode directions -- `eigenmode_a`/
/// `eigenmode_b` -- came from `BiaxialIndicatrix::eigen_polarizations`, not the
/// uniaxial ordinary/extraordinary approximation) AND the material carries a third
/// band set (`AbsorptionTensor::beta_ray.is_some()`); see `raytracer::apply_absorption`.
/// `None` (every material before this task, and every uniaxial/two-set material after
/// it) takes exactly the pre-existing `AbsorptionTensor3::uniaxial` path.
#[must_use]
#[expect(
    clippy::too_many_arguments,
    reason = "argument order deliberately mirrors transport_physics.wgsl's \
              pleochroic_channel_alpha/pleochroic_channel_alpha_biaxial (alpha_o, \
              alpha_e, [alpha_beta,] c_axis, s_axis, propagation_dir, eigen_a, eigen_b, \
              stokes) one-for-one; bundling these into a context struct here would break \
              that correspondence and make it harder to check the CPU and GPU paths \
              against each other, which is the whole point of this function existing \
              as a direct WGSL mirror"
)]
pub fn pleochroic_channel_alpha(
    alpha_o: f32,
    alpha_e: f32,
    alpha_beta: Option<f32>,
    c_axis: Vec3,
    s_axis: Vec3,
    propagation_dir: Vec3,
    eigenmode_a: Vec3,
    eigenmode_b: Vec3,
    stokes: &StokesVector,
) -> f32 {
    let tensor = alpha_beta.map_or_else(
        || AbsorptionTensor3::uniaxial(alpha_o, alpha_e, c_axis),
        |beta| AbsorptionTensor3::biaxial(alpha_o, beta, alpha_e, c_axis),
    );
    let e_hat = electric_field_direction(stokes, s_axis, propagation_dir);
    effective_pleochroic_alpha(
        &tensor,
        e_hat,
        eigenmode_a,
        eigenmode_b,
        stokes.degree_of_polarization(),
    )
}

/// A biaxial crystal's optical indicatrix.
///
/// Three principal refractive indices `n_alpha <= n_beta <= n_gamma` and the
/// orthonormal frame (world/model space) they are defined along (`axes` columns 0, 1,
/// 2 correspond to alpha, beta, gamma respectively). Uniaxial crystals are the special
/// case `n_alpha == n_beta` (positive) or `n_beta == n_gamma` (negative); isotropic is
/// all three equal.
#[derive(Debug, Clone, Copy)]
pub struct BiaxialIndicatrix {
    pub n_alpha: f32,
    pub n_beta: f32,
    pub n_gamma: f32,
    pub axes: Mat3,
}

impl BiaxialIndicatrix {
    #[must_use]
    pub const fn new(n_alpha: f32, n_beta: f32, n_gamma: f32, axes: Mat3) -> Self {
        Self {
            n_alpha,
            n_beta,
            n_gamma,
            axes,
        }
    }

    /// Convenience constructor for the uniaxial special case (`n_alpha == n_beta ==
    /// n_o`, `n_gamma == n_e`), used by tests to pin the biaxial equation's reduction
    /// to the existing uniaxial formula (`BirefringenceParams::effective_extraordinary_index`).
    #[must_use]
    pub fn uniaxial(n_o: f32, n_e: f32, c_axis: Vec3) -> Self {
        let c = c_axis.normalize_or_zero();
        let (a1, a2) = stable_orthonormal_basis(c);
        Self::new(n_o, n_o, n_e, Mat3::from_cols(a1, a2, c))
    }

    /// Builds a genuinely biaxial indicatrix from the three principal indices plus a
    /// single reference axis (`gamma_axis`, taken as the `n_gamma` principal direction):
    /// the other two principal axes are an arbitrary orthonormal completion of it, via
    /// `stable_orthonormal_basis` -- the same placeholder-orientation convention this
    /// renderer already uses for every uniaxial material's `c_axis` (it does not model
    /// a gem's crystallographic orientation relative to its cut). See
    /// `materials::GemMaterial::biaxial_indicatrix` for how the built-in Alexandrite,
    /// Topaz and Tanzanite entries feed this.
    #[must_use]
    pub fn from_gamma_axis(n_alpha: f32, n_beta: f32, n_gamma: f32, gamma_axis: Vec3) -> Self {
        let g = gamma_axis.normalize_or_zero();
        let (a1, a2) = stable_orthonormal_basis(g);
        Self::new(n_alpha, n_beta, n_gamma, Mat3::from_cols(a1, a2, g))
    }

    /// `1/n_i^2` for each principal index, in axis order -- the natural variable
    /// Fresnel's equation of the wave normals is quadratic in.
    fn b_coeffs(&self) -> [f32; 3] {
        [
            1.0 / (self.n_alpha * self.n_alpha),
            1.0 / (self.n_beta * self.n_beta),
            1.0 / (self.n_gamma * self.n_gamma),
        ]
    }

    /// Fresnel's equation of the wave normals (Born & Wolf, *Principles of Optics*,
    /// Sec. 14.2; Hecht, *Optics*, the "normal ellipsoid" construction), giving the two
    /// refractive indices allowed for a wave whose unit wave-normal direction is
    /// `wave_normal` (world space). Closed-form: with `(a, b, g)` the direction
    /// cosines of `wave_normal` in the principal frame and `b_i = 1/n_i^2`, the
    /// equation
    ///   a^2 (x-b_beta)(x-b_gamma) + b^2 (x-b_alpha)(x-b_gamma) + g^2 (x-b_alpha)(x-b_beta) = 0
    /// (x = 1/n^2) expands to a plain quadratic `x^2 - B*x + C = 0` (the `x^2`
    /// coefficient is `a^2+b^2+g^2 = 1`), solved directly via the quadratic formula --
    /// no iteration. Returns `(n_slow, n_fast)` with `n_slow >= n_fast` (`n_slow` is
    /// the higher-index, lower-phase-velocity mode).
    ///
    /// Reduces EXACTLY to the uniaxial formula when `n_alpha == n_beta`: substituting
    /// `b_alpha = b_beta = b_o` collapses the equation to `x = b_o` (the ordinary
    /// index, for every direction) and, via the roots' product `x_a * x_b = C`, the
    /// other root to `sin^2(theta)*b_e + cos^2(theta)*b_o` -- algebraically identical
    /// to `BirefringenceParams::effective_extraordinary_index`'s `1/n_eff^2 =
    /// cos^2(theta)/n_o^2 + sin^2(theta)/n_e^2`. See
    /// `birefringence_reduction_tests::biaxial_wave_indices_reduce_to_uniaxial_formula`.
    ///
    /// Near/exact degeneracy between two (or three) principal indices is detected
    /// UP FRONT, via `indices_are_degenerate`, and short-circuited to the exact
    /// closed forms below rather than run through the general quadratic: as that
    /// discriminant `big_b^2 - 4*big_c` approaches zero it becomes a subtraction of
    /// two nearly-equal f32 values, and the resulting catastrophic cancellation
    /// (amplified further by the subsequent `sqrt`) is exactly what
    /// `indices_are_degenerate`'s doc comment derives the ~3.5e-4 relative error
    /// bound from. Isotropic materials (all three equal) skip the solve entirely;
    /// uniaxial ones (the common case in this crate's material catalogue -- most
    /// gems are uniaxial, i.e. `n_alpha == n_beta` or `n_beta == n_gamma`) reuse
    /// `BirefringenceParams::effective_extraordinary_index`, the exact
    /// ordinary/extraordinary formulation, instead of duplicating its algebra.
    #[must_use]
    pub fn wave_indices(&self, wave_normal: Vec3) -> (f32, f32) {
        let k = wave_normal.normalize_or_zero();

        let alpha_beta_degenerate = Self::indices_are_degenerate(self.n_alpha, self.n_beta);
        let beta_gamma_degenerate = Self::indices_are_degenerate(self.n_beta, self.n_gamma);

        if alpha_beta_degenerate && beta_gamma_degenerate {
            // Isotropic: all three principal indices agree, so every direction gives
            // exactly that index for both roots -- no quadratic (and no sqrt) needed.
            let n = (self.n_alpha + self.n_beta + self.n_gamma) / 3.0;
            return (n, n);
        }
        if alpha_beta_degenerate {
            // Positive-uniaxial degeneracy: n_alpha == n_beta == n_o, n_gamma == n_e,
            // optic axis along principal axis 2 (gamma).
            return Self::uniaxial_wave_indices(
                k,
                self.axes.z_axis,
                f32::midpoint(self.n_alpha, self.n_beta),
                self.n_gamma,
            );
        }
        if beta_gamma_degenerate {
            // Negative-uniaxial degeneracy: n_beta == n_gamma == n_o, n_alpha == n_e,
            // optic axis along principal axis 0 (alpha).
            return Self::uniaxial_wave_indices(
                k,
                self.axes.x_axis,
                f32::midpoint(self.n_beta, self.n_gamma),
                self.n_alpha,
            );
        }

        // Genuinely biaxial (no two principal indices numerically degenerate): the
        // general quadratic solve is safe here since the discriminant stays well
        // clear of zero.
        let local = self.axes.transpose() * k;
        let (a2, b2, g2) = (local.x * local.x, local.y * local.y, local.z * local.z);
        let [b1, b2c, b3] = self.b_coeffs();

        let big_b = a2.mul_add(b2c + b3, b2.mul_add(b1 + b3, g2 * (b1 + b2c)));
        let big_c = a2.mul_add(b2c * b3, b2.mul_add(b1 * b3, g2 * b1 * b2c));
        let disc = big_b.mul_add(big_b, -4.0 * big_c).max(0.0).sqrt();

        let x_lo = 0.5 * (big_b - disc); // smaller x -> larger n (slow ray)
        let x_hi = f32::midpoint(big_b, disc); // larger x -> smaller n (fast ray)

        let n_slow = 1.0 / x_lo.max(1e-12).sqrt();
        let n_fast = 1.0 / x_hi.max(1e-12).sqrt();
        (n_slow, n_fast)
    }

    /// Closed-form `wave_indices` for a uniaxial degeneracy: `n_o` is the
    /// direction-independent ordinary index, `n_e` the extraordinary principal
    /// index, and `c_axis` the optic axis (the principal axis of the non-degenerate
    /// pair). Reuses `BirefringenceParams::effective_extraordinary_index` -- the
    /// same exact ordinary/extraordinary formula already used elsewhere in this
    /// crate -- rather than duplicating its algebra, and returns `(n_slow, n_fast)`
    /// with `n_slow >= n_fast` regardless of whether this is a positive
    /// (`n_e > n_o`) or negative (`n_e < n_o`) uniaxial material.
    fn uniaxial_wave_indices(k_hat: Vec3, c_axis: Vec3, n_o: f32, n_e: f32) -> (f32, f32) {
        let theta = k_hat.dot(c_axis).clamp(-1.0, 1.0).abs().acos();
        let n_eff = BirefringenceParams::effective_extraordinary_index(n_o, n_e, theta);
        (n_o.max(n_eff), n_o.min(n_eff))
    }

    /// True when two principal refractive indices are close enough that the general
    /// biaxial quadratic in `wave_indices` would lose numerical precision solving
    /// for them, and its exact closed-form short-circuit should be used instead.
    ///
    /// As the discriminant `big_b^2 - 4*big_c` in that quadratic approaches zero
    /// (which happens exactly when two principal indices coincide), evaluating it
    /// becomes a subtraction of two nearly-equal `f32` values -- catastrophic
    /// cancellation, whose relative error scales as `f32::EPSILON` divided by how
    /// small the discriminant is, and that error is then halved in exponent (i.e.
    /// square-rooted) by the subsequent `sqrt` in the quadratic formula. So the
    /// smallest relative discriminant error the general solve can resolve before
    /// its `sqrt` starts returning noise is on the order of `sqrt(f32::EPSILON)` ~=
    /// 3.45e-4 -- which is the same order as the ~1.6e-4 absolute error (~8.3e-5
    /// relative, for indices around 1.9) actually observed for the exactly-isotropic
    /// case before this fix. Below that threshold, in other words, the general
    /// solve is never more trustworthy than the closed forms, so indices within it
    /// (scaled by their own magnitude, since the tolerance is relative) are treated
    /// as degenerate.
    fn indices_are_degenerate(a: f32, b: f32) -> bool {
        let scale = a.abs().max(b.abs()).max(1.0);
        (a - b).abs() <= f32::EPSILON.sqrt() * scale
    }

    /// The two eigenmodes' electric-DISPLACEMENT (D) direction at `wave_normal`, in
    /// the same (slow, fast) order as `wave_indices`.
    ///
    /// For a root `x` of the Fresnel equation above, the D-eigenvector in the
    /// principal frame is proportional to `(a/(x-b_alpha), b/(x-b_beta),
    /// g/(x-b_gamma))` (Born & Wolf, same section). Evaluated here (2026-09-02) via
    /// the symmetric "transverse impermeability" matrix construction in
    /// `eigenvector_world` -- see that method's doc comment for the numerical-
    /// conditioning argument for why this replaced the previous "cleared-denominator"
    /// polynomial form (`v_i = local_i * prod_{j!=i}(x-b_j)`).
    ///
    /// Uses D (not E) as the pleochroism axis fed to `AbsorptionTensor3::quadratic_form`
    /// -- the two coincide exactly along a principal axis and are close elsewhere,
    /// the same order of approximation this renderer already makes treating gem
    /// dichroism as diagonal in a fixed crystal frame. `d_to_e_direction` below
    /// recovers the (different) E direction needed for the walk-off/Poynting
    /// calculation.
    #[must_use]
    pub fn eigen_polarizations(&self, wave_normal: Vec3) -> (Vec3, Vec3) {
        let k = wave_normal.normalize_or_zero();
        let local = self.axes.transpose() * k;
        let [b1, b2, b3] = self.b_coeffs();
        let (n_slow, n_fast) = self.wave_indices(wave_normal);
        let x_slow = 1.0 / (n_slow * n_slow);
        let x_fast = 1.0 / (n_fast * n_fast);

        let d_slow = self.eigenvector_world(local, [b1, b2, b3], x_slow, k);
        let d_fast = self.eigenvector_world(local, [b1, b2, b3], x_fast, k);
        (d_slow, d_fast)
    }

    /// Shared body of `eigen_polarizations` for a single root `x`.
    ///
    /// Builds the eigenvector via the symmetric "transverse impermeability" matrix
    /// `Gamma = P . B . P` (`P = I - k k^T`, the projector onto the plane perpendicular
    /// to the wave normal; `B = diag(b_alpha, b_beta, b_gamma)`), expressed in the
    /// PRINCIPAL-AXIS frame (`local` is `k_hat`'s direction cosines in that frame,
    /// i.e. `k` below). `Gamma` is symmetric, `k` is always exactly its zero-eigenvalue
    /// eigenvector (`Gamma` maps every vector into `P`'s range, i.e. the plane
    /// perpendicular to `k`, so `Gamma k = 0` identically), and its other two
    /// eigenvalues are exactly `x_slow`/`x_fast` -- the standard "index ellipsoid
    /// restricted to the transverse plane" formulation (Yariv & Yeh, *Optical Waves in
    /// Crystals*), algebraically equivalent to the `D_i (x-b_i) = -k_i (k . E)`
    /// relation the previous "cleared-denominator" construction
    /// (`v_i = local_i * prod_{j!=i}(x-b_j)`) also implemented, via `E_j = b_j D_j` and
    /// eliminating `E`.
    ///
    /// Reformulated (2026-09-02) for numerical conditioning: the cleared-denominator
    /// form's pre-normalization magnitude collapses toward zero whenever `x` sits
    /// close to a principal `b_i` -- which, since the two roots interlace the three
    /// `b_i` (`b_alpha <= x_fast <= b_beta <= x_slow <= b_gamma` or similar ordering),
    /// happens across a WIDE swath of directions for any weakly birefringent biaxial
    /// gem (the common case -- real gemstones' whole birefringence, `n_gamma-n_alpha`,
    /// is itself only a few thousandths). A near-zero pre-normalization vector means a
    /// 1-ULP difference in `x` (e.g. CPU vs. GPU evaluation order) gets amplified
    /// through `normalize` into a materially different unit vector -- the root cause
    /// diagnosed for the GPU port's `eigen_polarizations`/`mode_poynting_dir` Tier-2
    /// mismatches (up to ~3.5M ULP before this fix).
    ///
    /// Given `M = Gamma - x*I` (singular by construction, since `x` is one of
    /// `Gamma`'s eigenvalues), the standard robust technique for a rank-<=2 symmetric
    /// 3x3 matrix's null space is the cross product of whichever pair of `M`'s three
    /// rows has the largest cross-product magnitude (at least one of the three pairs
    /// must be well clear of parallel whenever `M` has rank exactly 2, regardless of
    /// how close `x` sits to any individual `b_i` -- unlike the cleared-denominator
    /// form, nothing here divides by, or scales with, `(x-b_i)` alone). An EARLIER
    /// version of this function picked that single largest pair via a hard `argmax`
    /// branch and, empirically, still failed the GPU port's Tier-2 comparison by up to
    /// ~640K ULP: `eigen_polarizations` calls `wave_indices` (above) to get `x` itself,
    /// and for a weakly birefringent gem's near-degenerate directions `wave_indices`'
    /// own CPU/GPU agreement, while within its documented ULP budget, is not always
    /// EXACTLY 0 ULP -- so `x` (and therefore every `l01`/`l02`/`l12` derived from it)
    /// can differ between CPU and GPU by a few ULP. A `>=` comparison between three
    /// nearly-tied magnitudes has no tolerance for that: a few-ULP disagreement flips
    /// which pair "wins", and the two candidate directions are NOT close to each other
    /// in this near-degenerate regime (unlike a smooth `normalize`, a discrete branch
    /// has no "close enough").
    ///
    /// Fixed (still 2026-09-02) by replacing the argmax with a smooth, branch-free
    /// combination that is EXACTLY equivalent in idealized arithmetic: for a true
    /// rank-2 `M`, all three row pairs' cross products are exactly parallel or
    /// antiparallel (row2 is a linear combination of row0/row1 whenever `M v = 0`, so
    /// `cross(row0,row2)` and `cross(row1,row2)` reduce to scalar multiples of
    /// `cross(row0,row1)`). So rather than SELECTING the largest, this sign-aligns
    /// `c02`/`c12` to `c01` via the ROBUST sign of their dot products (a dot product
    /// between two vectors that are exactly (anti)parallel has an unambiguous sign --
    /// no near-tie -- except when the pair itself is small, in which case a wrong sign
    /// there costs little since that term barely contributes to the sum anyway) and
    /// sums all three: `c01 + sign(c01.c02)*c02 + sign(c01.c12)*c12`. This is always a
    /// POSITIVE multiple of the true null direction (each aligned term adds, never
    /// cancels), so it reduces to the old argmax's answer up to overall sign whenever
    /// one pair dominates, but stays CONTINUOUS (not branch-flippy) as `x` is perturbed
    /// by a few ULP -- exactly the property needed here. Falls back to
    /// `stable_orthonormal_basis` on the wave normal only at TRUE degeneracy (`x` a
    /// REPEATED eigenvalue of `Gamma`, i.e. `M` has rank <=1 -- the uniaxial
    /// ORDINARY-ray case, where the eigenspace is genuinely 2-D and any perpendicular
    /// direction is an equally valid answer), mirroring the previous construction's own
    /// degenerate fallback and its `1e-12` threshold.
    ///
    /// Sign convention (an eigenvector is only defined up to an overall sign): the
    /// returned vector's largest-magnitude component is made positive, deterministically
    /// -- see `canonicalize_eigenvector_sign`. The WGSL port (`biaxial_eigenvector_world`
    /// in `transport_physics.wgsl`) mirrors every operation here op-for-op, including
    /// this sign policy, so CPU and GPU agree on which of the two valid signs to return.
    ///
    /// One more wrinkle (still 2026-09-02): even with a smooth, branch-free
    /// null-vector extraction, `eigen_polarizations` STILL failed Tier-2 by up to
    /// double-digit percent for the weakest-birefringence built-ins -- traced to `x`
    /// itself (from `wave_indices` above), not this method. `wave_indices`' general-
    /// quadratic branch computes both roots from a shared `disc = sqrt(B^2 - 4C)`; for
    /// a real gem's small overall birefringence, `B^2` and `4C` are close (mode
    /// separation is small almost everywhere, not just near the two true optic axes --
    /// see `wave_indices`' own doc comment), so `disc` -- computed as the difference of
    /// two large, nearly-equal quantities -- resolves only to roughly `sqrt(f32::EPSILON)`
    /// RELATIVE to `B`'s own scale (per `indices_are_degenerate`'s doc comment).
    /// Eigenvector PERTURBATION theory says a near-degenerate eigenvector's sensitivity
    /// to its own eigenvalue scales as `1/gap` (`gap` = the mode separation itself) --
    /// so a single ULP of upstream noise (in `local`, or in how `fma` happens to round
    /// on a given GPU driver) that would be invisible for a well-separated direction
    /// gets divided by that already-small gap and shows up amplified in the eigenvector
    /// components precisely where the gap is smallest.
    ///
    /// Fixing that without touching `wave_indices`'s own (separately ULP-verified)
    /// public return value: `precise_root_near` below recomputes the SAME quadratic's
    /// discriminant via an algebraically-EXACT reformulation (verified by
    /// `discriminant_reformulation_matches_naive_b2_minus_4c` below) that replaces the
    /// `B^2 - 4C` subtraction with a combination of `b_alpha - b_beta` and
    /// `b_gamma - b_alpha` -- DIRECT differences of the three principal `1/n^2` values,
    /// each computed via a single Sterbenz-exact subtraction (the two operands are
    /// always within a factor of 2, since realistic gem indices cluster in a narrow
    /// range) rather than inherited from a subtraction of two ~1 magnitude sums. Since
    /// those differences are themselves already the "gap-scale" quantities the old
    /// formula was fighting to resolve out of much larger numbers, this recovers
    /// several more digits of precision for exactly the near-degenerate directions
    /// that matter here -- see that function's own doc comment for the derivation.
    fn eigenvector_world(&self, local: Vec3, b: [f32; 3], x: f32, k_hat: Vec3) -> Vec3 {
        let x = Self::precise_root_near(local, b, x);

        let s = b[0].mul_add(
            local.x * local.x,
            b[1].mul_add(local.y * local.y, b[2] * local.z * local.z),
        );

        let m00 = (local.x * local.x).mul_add(2.0f32.mul_add(-b[0], s), b[0] - x);
        let m11 = (local.y * local.y).mul_add(2.0f32.mul_add(-b[1], s), b[1] - x);
        let m22 = (local.z * local.z).mul_add(2.0f32.mul_add(-b[2], s), b[2] - x);
        let m01 = local.x * local.y * (s - b[0] - b[1]);
        let m02 = local.x * local.z * (s - b[0] - b[2]);
        let m12 = local.y * local.z * (s - b[1] - b[2]);

        let row0 = Vec3::new(m00, m01, m02);
        let row1 = Vec3::new(m01, m11, m12);
        let row2 = Vec3::new(m02, m12, m22);

        // Explicit `mul_add` cross products (rather than `Vec3::cross`) for the same
        // reason `wave_indices`' own discriminant uses `mul_add` rather than plain
        // `*`/`-`: a plain multiply-then-subtract chain is free to be auto-contracted
        // into a fused multiply-add by one compiler (CPU LLVM or the GPU shader
        // compiler) and not the other, which would round differently and silently
        // break CPU/GPU bit-parity.
        let c01 = cross_fma(row0, row1);
        let c02 = cross_fma(row0, row2);
        let c12 = cross_fma(row1, row2);

        // Sign-align c02/c12 to c01 and sum -- see this method's doc comment for why
        // this replaced a hard "largest of three" selection.
        let sign02 = if dot_fma(c01, c02) < 0.0 { -1.0 } else { 1.0 };
        let sign12 = if dot_fma(c01, c12) < 0.0 { -1.0 } else { 1.0 };
        let v_local = c01 + c02 * sign02 + c12 * sign12;

        let v_world = self.axes * v_local;
        if v_world.length_squared() > 1e-12 {
            canonicalize_eigenvector_sign(v_world).normalize()
        } else {
            stable_orthonormal_basis(k_hat).0
        }
    }

    /// Recomputes the general biaxial Fresnel quadratic's discriminant via an
    /// algebraically-exact reformulation, then returns whichever of its two roots
    /// (`x_lo`, `x_hi`) is nearest the `x` this was handed (see `eigenvector_world`'s
    /// doc comment for why: the input `x`, from `wave_indices`, is already accurate
    /// enough -- gap-relative error a few percent at worst, always far less than 100%
    /// -- to unambiguously identify WHICH of the two roots it was meant to be, even
    /// though it is not accurate enough for the eigenvector built from it).
    ///
    /// Derivation: writing `a = local.x^2`, `bb = local.y^2`, `cc = local.z^2` (so
    /// `a+bb+cc = 1`) and `p,q,r = b[0],b[1],b[2]`, the naive discriminant is
    /// `disc^2 = B^2 - 4C` with `B = a(q+r)+bb(p+r)+cc(p+q)`, `C = aqr+bb*pr+cc*pq` --
    /// exactly what `wave_indices` computes. Expanding and eliminating `cc = 1-a-bb`
    /// gives, EXACTLY (verified symbolically and pinned by
    /// `discriminant_reformulation_matches_naive_b2_minus_4c` below):
    ///   `disc^2 = (a+cc)^2 * X^2  +  2*(a - bb*cc) * X*Y  +  (a+bb)^2 * Y^2`
    /// where `X = p-q` (`b_alpha - b_beta`) and `Y = r-p` (`b_gamma - b_alpha`) are
    /// DIRECT differences of the principal `1/n^2` values rather than the original
    /// formula's difference of two ~1-magnitude sums `B^2` and `4C`. Since any two
    /// principal indices of a realistic gem material sit within a factor of 2 of each
    /// other (indices cluster in roughly the 1.4-2.5 range, so `1/n^2` does too), `X`
    /// and `Y` are each computed via a Sterbenz-exact subtraction -- no rounding error
    /// at all beyond what `p`, `q`, `r` already carried from their own `1/n^2`. That
    /// makes this reformulation's cancellation benign (the three terms above are each
    /// already `O(disc^2)`, not `O(B^2)` collapsing down to `O(disc^2)`), where the
    /// naive `B^2-4C` loses essentially all its precision resolving a `disc` far
    /// smaller than either operand.
    fn precise_root_near(local: Vec3, b: [f32; 3], x: f32) -> f32 {
        let a = local.x * local.x;
        let bb = local.y * local.y;
        let cc = local.z * local.z;
        let big_b = a.mul_add(b[1] + b[2], bb.mul_add(b[0] + b[2], cc * (b[0] + b[1])));

        let xdiff = b[0] - b[1]; // p - q, Sterbenz-exact
        let ydiff = b[2] - b[0]; // r - p, Sterbenz-exact

        let a_plus_c = a + cc;
        let a_plus_bb = a + bb;
        let two_a_minus_bc = 2.0 * bb.mul_add(-cc, a);

        let disc_sq = (a_plus_c * a_plus_c).mul_add(
            xdiff * xdiff,
            (a_plus_bb * a_plus_bb).mul_add(ydiff * ydiff, two_a_minus_bc * xdiff * ydiff),
        );
        let disc = disc_sq.max(0.0).sqrt();

        let x_lo = 0.5 * (big_b - disc);
        let x_hi = f32::midpoint(big_b, disc);

        if (x - x_lo).abs() <= (x - x_hi).abs() {
            x_lo
        } else {
            x_hi
        }
    }

    /// Converts a D-field eigen-direction to the corresponding E-field direction via
    /// `E = eps^-1 . D`, diagonal in the principal frame (`eps_i = n_i^2`). Needed
    /// because the Poynting/walk-off direction depends on E, not D: D is always
    /// exactly perpendicular to the wave normal by construction; E is what tilts away
    /// from that perpendicular by the walk-off angle (see `poynting_direction`).
    #[must_use]
    pub fn d_to_e_direction(&self, d_hat: Vec3) -> Vec3 {
        let d_local = self.axes.transpose() * d_hat;
        let e_local = Vec3::new(
            d_local.x / (self.n_alpha * self.n_alpha),
            d_local.y / (self.n_beta * self.n_beta),
            d_local.z / (self.n_gamma * self.n_gamma),
        );
        (self.axes * e_local).normalize_or_zero()
    }

    /// Fixed-point resolution of the refracted WAVE-NORMAL direction for
    /// one eigenmode (`want_slow`: the `n_slow` root if true, `n_fast` if false) at an
    /// air->crystal entry into a biaxial material.
    ///
    /// Generalizes `trace_spectral_ray`'s existing uniaxial `theta_c` iteration (see
    /// that call site's doc comment) from a single angle-to-c-axis to a full 3D
    /// direction: a biaxial mode's index depends on the FULL wave-normal direction (not
    /// just the angle to one fixed axis, since there is no single optic axis), which
    /// itself depends on the index via Snell's law -- the same circularity, resolved
    /// with the same two-iteration fixed point seeded from an isotropic first guess
    /// `n_seed` (in practice the material's own base dispersion curve value, i.e.
    /// `n_beta` -- exactly correct for neither root individually, but a reasonable
    /// starting point, mirroring how the uniaxial iteration seeds from `n_o` even
    /// though that is only exactly correct for the ordinary root).
    ///
    /// Unlike the uniaxial case, NEITHER root has a direction-independent index to fall
    /// back on without iterating -- there is no "ordinary ray" in a biaxial crystal --
    /// so this is called once per mode, not just for the extraordinary-like one.
    ///
    /// Returns `(n, wave_normal)`: the converged index for the requested root and the
    /// wave-normal direction it was evaluated at.
    #[must_use]
    pub fn resolve_entry_mode(
        &self,
        incident_dir: Vec3,
        normal: Vec3,
        cos_i: f32,
        n_seed: f32,
        want_slow: bool,
    ) -> (f32, Vec3) {
        let mut n_guess = n_seed;
        let mut wave_dir = incident_dir;
        for _ in 0..2 {
            let eta_guess = 1.0 / n_guess;
            let sin2_t_guess = eta_guess * eta_guess * cos_i.mul_add(-cos_i, 1.0);
            if sin2_t_guess > 1.0 {
                break;
            }
            let cos_t_guess = (1.0 - sin2_t_guess).max(0.0).sqrt();
            wave_dir = (eta_guess * incident_dir + eta_guess.mul_add(cos_i, -cos_t_guess) * normal)
                .normalize();
            let (n_slow, n_fast) = self.wave_indices(wave_dir);
            n_guess = if want_slow { n_slow } else { n_fast };
        }
        (n_guess, wave_dir)
    }

    /// The Poynting (walk-off) energy-propagation direction for one
    /// eigenmode at `wave_normal`, generalizing
    /// `BirefringenceParams::extraordinary_poynting_dir` (uniaxial-only, and only ever
    /// applied to the extraordinary mode -- the ordinary ray never walks off) to the
    /// biaxial case where BOTH eigenmodes walk off, since neither is ever exactly
    /// perpendicular to the D-field in general.
    ///
    /// Builds the requested mode's D-field eigenvector via `eigen_polarizations`,
    /// converts it to the corresponding E-field direction via `d_to_e_direction`, and
    /// feeds both into the general `poynting_direction` (S = E x H) formula above.
    #[must_use]
    pub fn mode_poynting_dir(&self, wave_normal: Vec3, want_slow: bool) -> Vec3 {
        let (d_slow, d_fast) = self.eigen_polarizations(wave_normal);
        let d_hat = if want_slow { d_slow } else { d_fast };
        let e_hat = self.d_to_e_direction(d_hat);
        poynting_direction(wave_normal, e_hat)
    }
}

/// The Poynting (ray/walk-off) energy-propagation direction for an eigenmode.
///
/// Takes the wave normal `wave_normal` and the world-space electric-FIELD direction
/// `e_field_hat` (see `BiaxialIndicatrix::d_to_e_direction`) -- the general
/// (uniaxial-or-biaxial) counterpart of `extraordinary_poynting_dir` above, "the ray
/// direction is the gradient of the index surface" in its equivalent Poynting-vector
/// form: `S = E x H`, and for a plane wave `H` is parallel to `k_hat x E` (from
/// Maxwell's `curl E = -dB/dt`), so
///   S ~ E x (`k_hat` x E) = `k_hat`*(E.E) - E*(E.`k_hat`)
/// which for unit `E` reduces to the component of `k_hat` perpendicular to
/// `e_field_hat` -- i.e. `k_hat` tilted just enough to become exactly perpendicular to
/// E (S is always perpendicular to E, the same way D is always perpendicular to
/// `k_hat`). Falls back to `k_hat` itself (no walk-off) in the degenerate case where
/// `e_field_hat` is parallel to `k_hat` (should not occur for a genuine eigenmode, but
/// guarded rather than risking a zero-length normalize).
#[must_use]
pub fn poynting_direction(wave_normal: Vec3, e_field_hat: Vec3) -> Vec3 {
    let k_hat = wave_normal.normalize_or_zero();
    let e_hat = e_field_hat.normalize_or_zero();
    let perp = k_hat - e_hat * e_hat.dot(k_hat);
    let len2 = perp.length_squared();
    if len2 > 1e-10 {
        perp / len2.sqrt()
    } else {
        k_hat
    }
}

#[cfg(test)]
mod absorption_tensor_tests {
    use super::*;
    use crate::optics::polarization::StokesVector;

    /// An isotropic tensor (`alpha_o` == `alpha_e`) must return the same coefficient
    /// regardless of the electric-field direction.
    #[test]
    fn isotropic_tensor_is_azimuth_independent() {
        let tensor = AbsorptionTensor3::uniaxial(1.5, 1.5, Vec3::Y);
        let dirs = [
            Vec3::X,
            Vec3::Y,
            Vec3::Z,
            Vec3::new(1.0, 1.0, 1.0).normalize(),
            Vec3::new(0.3, -0.7, 0.2).normalize(),
        ];
        for d in dirs {
            let a = tensor.quadratic_form(d);
            assert!(
                (a - 1.5).abs() < 1e-5,
                "isotropic tensor must be azimuth-independent, got {a} for {d:?}"
            );
        }
    }

    /// A uniaxial tensor evaluated exactly along the c-axis must give `alpha_e`, and
    /// exactly perpendicular to it must give `alpha_o`.
    #[test]
    fn uniaxial_tensor_matches_principal_coefficients_on_axis() {
        let c_axis = Vec3::Y;
        let tensor = AbsorptionTensor3::uniaxial(2.0, 5.0, c_axis);
        assert!((tensor.quadratic_form(c_axis) - 5.0).abs() < 1e-5);
        assert!((tensor.quadratic_form(Vec3::X) - 2.0).abs() < 1e-5);
        assert!((tensor.quadratic_form(Vec3::Z) - 2.0).abs() < 1e-5);
    }

    /// The decisive test: send linearly polarized light through a strongly
    /// dichroic uniaxial material along a FIXED propagation direction and rotate the
    /// polarization azimuth through 180 degrees. Absorption must vary smoothly and
    /// return to its starting value -- a clean sinusoid in `2*psi` -- and must clear a
    /// wide range (not be constant, which is what the old propagation-angle-only
    /// blend would give for a fixed direction).
    #[test]
    fn rotating_polarization_azimuth_traces_a_clean_sinusoid() {
        let c_axis = Vec3::new(0.2, 0.6, 0.77).normalize(); // deliberately off-axis
        let propagation_dir = Vec3::new(0.9, 0.1, 0.3).normalize(); // fixed direction, oblique to c_axis
        let alpha_o = 0.5f32;
        let alpha_e = 6.0f32; // strongly dichroic
        let tensor = AbsorptionTensor3::uniaxial(alpha_o, alpha_e, c_axis);

        // Build a fixed (s_axis, p_axis) frame perpendicular to propagation_dir, as
        // trace_spectral_ray would from a plane-of-incidence normal.
        let s_axis = propagation_dir.cross(Vec3::Y).normalize();

        let samples = 37;
        let mut values = Vec::with_capacity(samples);
        for i in 0..samples {
            let psi = (i as f32) / (samples as f32 - 1.0) * std::f32::consts::PI; // sweep 0..=180deg
            let stokes = StokesVector::new(1.0, (2.0 * psi).cos(), (2.0 * psi).sin(), 0.0);
            let e_hat = electric_field_direction(&stokes, s_axis, propagation_dir);
            let eigen_a = BirefringenceParams::ordinary_eigen_polarization(propagation_dir, c_axis);
            let eigen_b =
                BirefringenceParams::extraordinary_eigen_polarization(propagation_dir, c_axis);
            let alpha = effective_pleochroic_alpha(&tensor, e_hat, eigen_a, eigen_b, 1.0);
            values.push(alpha);
        }

        // 1. Must return (close) to its starting value after a full 180 degree sweep.
        let start = values[0];
        let end = values[samples - 1];
        assert!(
            (start - end).abs() < 1e-3,
            "azimuth sweep must return to its starting value (start={start}, end={end})"
        );

        // 2. Must actually vary -- not the old azimuth-blind behaviour.
        let min = values.iter().copied().fold(f32::INFINITY, f32::min);
        let max = values.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        assert!(
            max - min > 1.0,
            "absorption must vary substantially as azimuth rotates for a strongly dichroic material (min={min}, max={max})"
        );

        // 3. Every value must stay within the [alpha_o, alpha_e] envelope implied by
        // the quadratic form (a physical sanity bound).
        for &a in &values {
            assert!(
                a >= alpha_o - 1e-3 && a <= alpha_e + 1e-3,
                "alpha={a} out of the physical [alpha_o, alpha_e]=[{alpha_o},{alpha_e}] envelope"
            );
        }

        // 4. Smoothness: no adjacent-sample jump should be large relative to the total
        // span (catches a sign flip / discontinuity bug, as opposed to a genuine smooth
        // sinusoid).
        let span = max - min;
        for w in values.windows(2) {
            assert!(
                (w[1] - w[0]).abs() < 0.35 * span,
                "adjacent samples should vary smoothly, not jump (got delta={})",
                (w[1] - w[0]).abs()
            );
        }
    }

    /// Fully unpolarized light must be azimuth-independent (it has no well-defined
    /// azimuth) and must sit near the two-eigenmode average, not at either pure
    /// extreme -- the "graceful degradation" requirement.
    #[test]
    fn unpolarized_light_degrades_gracefully_and_is_azimuth_independent() {
        let c_axis = Vec3::Y;
        let propagation_dir = Vec3::new(0.6, 0.3, 0.74).normalize();
        let alpha_o = 1.0f32;
        let alpha_e = 4.0f32;
        let tensor = AbsorptionTensor3::uniaxial(alpha_o, alpha_e, c_axis);
        let eigen_a = BirefringenceParams::ordinary_eigen_polarization(propagation_dir, c_axis);
        let eigen_b =
            BirefringenceParams::extraordinary_eigen_polarization(propagation_dir, c_axis);

        let unpolarized = StokesVector::unpolarized(1.0);
        let s_axis = propagation_dir.cross(Vec3::Z).normalize();
        let e_hat = electric_field_direction(&unpolarized, s_axis, propagation_dir);
        let alpha_unpolarized = effective_pleochroic_alpha(
            &tensor,
            e_hat,
            eigen_a,
            eigen_b,
            unpolarized.degree_of_polarization(),
        );

        // Must sit strictly between the two pure principal coefficients (a genuine
        // blend, not collapsed to either extreme).
        assert!(
            alpha_unpolarized > alpha_o && alpha_unpolarized < alpha_e,
            "unpolarized absorption {alpha_unpolarized} should sit strictly between alpha_o={alpha_o} and alpha_e={alpha_e}"
        );

        // Must not depend on which s_axis/frame was used to build e_hat -- with p=0
        // the polarized term is weighted out entirely.
        let s_axis_2 = propagation_dir.cross(Vec3::X).normalize_or_zero();
        let e_hat_2 = electric_field_direction(&unpolarized, s_axis_2, propagation_dir);
        let alpha_unpolarized_2 = effective_pleochroic_alpha(
            &tensor,
            e_hat_2,
            eigen_a,
            eigen_b,
            unpolarized.degree_of_polarization(),
        );
        assert!(
            (alpha_unpolarized - alpha_unpolarized_2).abs() < 1e-5,
            "unpolarized result must not depend on the reference frame used to build e_hat"
        );
    }

    /// Pleochroism data pass: pins the `o_ray`/`e_ray` <-> `AbsorptionTensor3` naming
    /// convention using REAL Ruby material band data (every other test in this module
    /// uses synthetic `alpha_o`/`alpha_e` scalars) -- get this convention backwards and
    /// every populated built-in's dichroism would render mirrored (e.g. Sapphire's
    /// ~70nm band-centre shift would show up in exactly the wrong viewing direction).
    ///
    /// `quadratic_form` evaluated exactly perpendicular to `c_axis` (E-perp-c) must
    /// equal the o-ray band sum, and evaluated exactly parallel to `c_axis`
    /// (E-parallel-c) must equal the e-ray band sum, at Ruby's own real band centre
    /// 556nm -- the o-ray yellow-green Cr3+ band peak (see the Ruby entry's comment in
    /// `materials::GemMaterial::all_materials`), chosen specifically because the two
    /// rays differ substantially there (o-ray peak 2.5 vs. the e-ray's off-peak
    /// contribution), making this discriminating rather than a coincidental match.
    #[test]
    fn convention_pin_o_ray_is_perpendicular_e_ray_is_parallel_to_c_axis() {
        use crate::optics::{materials::GemMaterial, raytracer::spectral_absorption};

        const LAMBDA_NM: f32 = 556.0;
        let ruby = GemMaterial::by_name("Ruby").expect("Ruby must be a built-in material");
        let alpha_o = spectral_absorption(&ruby.absorption.o_ray, LAMBDA_NM);
        let alpha_e = spectral_absorption(&ruby.absorption.e_ray, LAMBDA_NM);
        assert!(
            alpha_o > 0.0 && alpha_e > 0.0,
            "test premise: both rays must contribute at 556nm (o={alpha_o}, e={alpha_e})"
        );
        assert!(
            (alpha_o - alpha_e).abs() > 0.1,
            "test premise: o-ray and e-ray must actually differ at 556nm for this test to be \
             discriminating (o={alpha_o}, e={alpha_e})"
        );

        let tensor = AbsorptionTensor3::uniaxial(alpha_o, alpha_e, Vec3::Y);
        let a_perp = tensor.quadratic_form(Vec3::X);
        let a_parallel = tensor.quadratic_form(Vec3::Y);

        assert!(
            (a_perp - alpha_o).abs() < 1e-4,
            "E-perp-c quadratic_form ({a_perp}) must equal the o-ray band sum ({alpha_o}) -- \
             o_ray must map to the PERPENDICULAR-to-c principal coefficient"
        );
        assert!(
            (a_parallel - alpha_e).abs() < 1e-4,
            "E-parallel-c quadratic_form ({a_parallel}) must equal the e-ray band sum \
             ({alpha_e}) -- e_ray must map to the PARALLEL-to-c principal coefficient"
        );
    }

    /// Polarized probe, extending the synthetic-coefficient tests above (which used
    /// hand-picked `alpha_o`/`alpha_e` scalars) to run against REAL material data end to
    /// end through `pleochroic_channel_alpha` -- the exact function
    /// `raytracer::apply_absorption` calls per spectral channel per bounce. For a
    /// propagation direction exactly perpendicular to `c_axis`, feeds a FULLY polarized
    /// Stokes vector (`degree_of_polarization` == 1) first with E exactly parallel to
    /// `c_axis`, then exactly perpendicular to it, and checks the returned effective
    /// coefficient equals the respective band sum exactly -- at full polarization the
    /// unpolarized eigenmode-average term in `effective_pleochroic_alpha` is weighted
    /// out entirely, so this isolates the quadratic-form term alone, run through the
    /// SAME `electric_field_direction`/azimuth machinery a real bounce uses (unlike the
    /// convention-pin test above, which calls `quadratic_form` directly).
    #[test]
    fn polarized_probe_matches_band_sums_for_real_tourmaline_data() {
        use crate::optics::{materials::GemMaterial, raytracer::spectral_absorption};

        const LAMBDA_NM: f32 = 430.0; // Fe2+-Ti4+ IVCT band, strongly dichroic at this wavelength
        let tourmaline =
            GemMaterial::by_name("Tourmaline").expect("Tourmaline must be a built-in material");
        let c_axis = tourmaline.c_axis; // Vec3::X, the documented cut-orientation override
        let alpha_o = spectral_absorption(&tourmaline.absorption.o_ray, LAMBDA_NM);
        let alpha_e = spectral_absorption(&tourmaline.absorption.e_ray, LAMBDA_NM);
        assert!(
            alpha_o > alpha_e * 2.0,
            "test premise: Tourmaline's o-ray must be substantially stronger than its e-ray at \
             430nm (o={alpha_o}, e={alpha_e})"
        );

        // Propagation exactly perpendicular to c_axis, with a stable orthonormal frame
        // built the same way `apply_absorption` builds its eigenmodes.
        let propagation_dir = if c_axis.x.abs() > 0.9 {
            Vec3::Y
        } else {
            Vec3::X
        };
        assert!(
            propagation_dir.dot(c_axis).abs() < 1e-6,
            "test premise: propagation_dir must be exactly perpendicular to c_axis"
        );
        let eigen_a = BirefringenceParams::ordinary_eigen_polarization(propagation_dir, c_axis);
        let eigen_b =
            BirefringenceParams::extraordinary_eigen_polarization(propagation_dir, c_axis);
        let s_axis = propagation_dir.cross(c_axis).normalize();

        // E exactly parallel to c_axis: azimuth chosen so `electric_field_direction`
        // recovers `c_axis` itself (psi = 90 deg from s_axis, i.e. Q=-1, U=0).
        let stokes_parallel = StokesVector::new(1.0, -1.0, 0.0, 0.0);
        let e_parallel = electric_field_direction(&stokes_parallel, s_axis, propagation_dir);
        assert!(
            (e_parallel.dot(c_axis).abs() - 1.0).abs() < 1e-4,
            "test premise: this Stokes vector's field direction must be parallel to c_axis, \
             got {e_parallel:?} vs c_axis {c_axis:?}"
        );
        let alpha_measured_parallel = pleochroic_channel_alpha(
            alpha_o,
            alpha_e,
            None,
            c_axis,
            s_axis,
            propagation_dir,
            eigen_a,
            eigen_b,
            &stokes_parallel,
        );
        assert!(
            (alpha_measured_parallel - alpha_e).abs() < 1e-4,
            "E-parallel-c fully-polarized probe ({alpha_measured_parallel}) must equal the \
             e-ray band sum ({alpha_e})"
        );

        // E exactly perpendicular to c_axis (along s_axis itself: psi = 0, Q=+1, U=0).
        let stokes_perp = StokesVector::new(1.0, 1.0, 0.0, 0.0);
        let e_perp = electric_field_direction(&stokes_perp, s_axis, propagation_dir);
        assert!(
            e_perp.dot(c_axis).abs() < 1e-4,
            "test premise: this Stokes vector's field direction must be perpendicular to \
             c_axis, got {e_perp:?} vs c_axis {c_axis:?}"
        );
        let alpha_measured_perp = pleochroic_channel_alpha(
            alpha_o,
            alpha_e,
            None,
            c_axis,
            s_axis,
            propagation_dir,
            eigen_a,
            eigen_b,
            &stokes_perp,
        );
        assert!(
            (alpha_measured_perp - alpha_o).abs() < 1e-4,
            "E-perp-c fully-polarized probe ({alpha_measured_perp}) must equal the o-ray band \
             sum ({alpha_o})"
        );
    }
}

#[cfg(test)]
mod biaxial_reduction_tests {
    use super::*;

    /// The decisive test: setting `n_alpha == n_beta` must reproduce the
    /// existing uniaxial code path's index -- `BirefringenceParams::effective_extraordinary_index`
    /// -- to within float tolerance, across many propagation directions and a
    /// deliberately off-axis (non-axis-aligned) c-axis. This pins the biaxial Fresnel
    /// equation's reduction and would catch most algebra errors in `wave_indices`.
    #[test]
    fn biaxial_wave_indices_reduce_to_uniaxial_formula() {
        let n_o = 1.65f32;
        let n_e = 1.72f32; // positive uniaxial
        let c_axis = Vec3::new(0.35, 0.82, -0.45).normalize(); // deliberately off-axis
        let indicatrix = BiaxialIndicatrix::uniaxial(n_o, n_e, c_axis);

        let directions = [
            Vec3::X,
            Vec3::Y,
            Vec3::Z,
            c_axis, // exactly along the optic axis
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::new(0.2, 0.9, 0.1).normalize(),
            Vec3::new(-0.5, 0.3, 0.8).normalize(),
            Vec3::new(0.7, -0.6, 0.2).normalize(),
            Vec3::new(0.1, 0.1, 0.99).normalize(),
            Vec3::new(-0.9, -0.3, 0.2).normalize(),
        ];

        for wave_normal in directions {
            let (n_slow, n_fast) = indicatrix.wave_indices(wave_normal);
            let cos_theta = wave_normal.normalize().dot(c_axis).clamp(-1.0, 1.0).abs();
            let theta = cos_theta.acos();
            let n_eff = BirefringenceParams::effective_extraordinary_index(n_o, n_e, theta);

            let expected_lo = n_o.min(n_eff);
            let expected_hi = n_o.max(n_eff);

            assert!(
                (n_fast - expected_lo).abs() < 1e-3,
                "n_fast should match min(n_o, n_eff) at direction {wave_normal:?} (theta={theta}): got {n_fast}, expected {expected_lo}"
            );
            assert!(
                (n_slow - expected_hi).abs() < 1e-3,
                "n_slow should match max(n_o, n_eff) at direction {wave_normal:?} (theta={theta}): got {n_slow}, expected {expected_hi}"
            );
        }
    }

    /// Isotropic special case (all three principal indices equal): both roots must
    /// equal that single index for every direction.
    #[test]
    fn biaxial_wave_indices_reduce_to_isotropic_when_all_equal() {
        let n = 1.9f32;
        let indicatrix = BiaxialIndicatrix::new(n, n, n, Mat3::IDENTITY);
        for wave_normal in [
            Vec3::X,
            Vec3::Y,
            Vec3::Z,
            Vec3::new(0.3, 0.5, 0.8).normalize(),
        ] {
            let (n_slow, n_fast) = indicatrix.wave_indices(wave_normal);
            assert!(
                (n_slow - n).abs() < 1e-4,
                "isotropic n_slow should equal {n}, got {n_slow}"
            );
            assert!(
                (n_fast - n).abs() < 1e-4,
                "isotropic n_fast should equal {n}, got {n_fast}"
            );
        }
    }

    /// For a genuinely biaxial indicatrix (three distinct principal indices), the two
    /// eigen D-directions returned by `eigen_polarizations` must (a) be finite, (b) be
    /// unit length, (c) be mutually perpendicular, and (d) each be perpendicular to
    /// the wave normal -- D is always transverse, that's the defining property the
    /// Fresnel equation is built from.
    #[test]
    fn biaxial_eigen_polarizations_are_orthonormal_and_transverse() {
        let axes = Mat3::from_cols(Vec3::X, Vec3::Y, Vec3::Z);
        let indicatrix = BiaxialIndicatrix::new(1.60, 1.65, 1.75, axes);

        for wave_normal in [
            Vec3::new(0.3, 0.5, 0.8).normalize(),
            Vec3::new(0.9, 0.1, -0.2).normalize(),
            Vec3::new(-0.4, 0.6, 0.5).normalize(),
        ] {
            let (d_slow, d_fast) = indicatrix.eigen_polarizations(wave_normal);
            assert!(
                d_slow.is_finite() && d_fast.is_finite(),
                "eigenvectors must be finite (d_slow={d_slow:?}, d_fast={d_fast:?})"
            );
            assert!(
                (d_slow.length() - 1.0).abs() < 1e-3,
                "d_slow must be unit length, got {}",
                d_slow.length()
            );
            assert!(
                (d_fast.length() - 1.0).abs() < 1e-3,
                "d_fast must be unit length, got {}",
                d_fast.length()
            );
            assert!(
                d_slow.dot(d_fast).abs() < 1e-2,
                "eigenvectors must be mutually perpendicular, got dot={}",
                d_slow.dot(d_fast)
            );
            assert!(
                d_slow.dot(wave_normal).abs() < 1e-2,
                "d_slow must be transverse to the wave normal, got dot={}",
                d_slow.dot(wave_normal)
            );
            assert!(
                d_fast.dot(wave_normal).abs() < 1e-2,
                "d_fast must be transverse to the wave normal, got dot={}",
                d_fast.dot(wave_normal)
            );
        }
    }

    /// Reformulation-quality pin (2026-09-02): the well-conditioned
    /// `eigenvector_world` (cross-product-of-rows of the "transverse impermeability"
    /// matrix `Gamma - x*I`) must satisfy the eigen-equation at least as tightly as the
    /// previous cleared-denominator construction did, across a DENSE direction sweep
    /// (not just the handful of spot-checks above) for all three real biaxial
    /// built-ins (Alexandrite, Topaz, Tanzanite -- exactly the materials whose small
    /// real-world birefringence made the old construction ill-conditioned).
    ///
    /// The residual checked is the textbook relation the eigenvector must satisfy
    /// independent of ANY particular matrix construction (so this is a genuine
    /// correctness check, not a tautological re-derivation of `eigenvector_world`'s own
    /// algebra): in the principal-axis frame, `D_i (x - b_i) = -k_i (k . E)` with
    /// `E_j = b_j D_j` (see `eigenvector_world`'s doc comment). Also pins unit length
    /// and continuous-sign (adjacent samples along the sweep must have `|dot| ~= 1`,
    /// i.e. the eigen-LINE direction -- which is only defined up to sign -- varies
    /// smoothly; an isolated sign flip at the deterministic sign convention's own
    /// crossover is fine and is exactly why this checks `abs(dot)`, not `dot`, against
    /// its floor).
    #[test]
    fn biaxial_eigenvectors_satisfy_eigen_equation_across_dense_sweep() {
        use crate::optics::materials::GemMaterial;

        const RESIDUAL_BOUND: f32 = 2e-4;

        for name in ["Alexandrite", "Topaz", "Tanzanite"] {
            let material = GemMaterial::by_name(name)
                .unwrap_or_else(|| panic!("{name} must be a built-in material"));
            let indicatrix = material
                .biaxial_indicatrix(589.3)
                .unwrap_or_else(|| panic!("{name} must expose a biaxial indicatrix"));
            let b = indicatrix.b_coeffs();

            let samples = 181;
            let mut prev_slow: Option<Vec3> = None;
            let mut prev_fast: Option<Vec3> = None;
            for i in 0..samples {
                // Dense sweep covering a full great-circle arc, deliberately not
                // aligned to any principal axis, so it crosses many directions where
                // the two mode indices sit close together (the common case for these
                // materials' small birefringence).
                let theta = (i as f32) / (samples as f32 - 1.0) * std::f32::consts::PI;
                let wave_normal = (theta.cos() * Vec3::new(0.4, 0.6, 0.69).normalize()
                    + theta.sin() * Vec3::new(0.8, -0.3, 0.2).normalize())
                .normalize();
                let k = wave_normal; // already unit
                let local = indicatrix.axes.transpose() * k;

                let (n_slow, n_fast) = indicatrix.wave_indices(wave_normal);
                let (d_slow, d_fast) = indicatrix.eigen_polarizations(wave_normal);

                for (n, d_hat, prev) in [
                    (n_slow, d_slow, &mut prev_slow),
                    (n_fast, d_fast, &mut prev_fast),
                ] {
                    assert!(
                        (d_hat.length() - 1.0).abs() < 1e-4,
                        "{name}: eigenvector must be unit length at theta={theta}, got {}",
                        d_hat.length()
                    );

                    let x = 1.0 / (n * n);
                    let d_local = indicatrix.axes.transpose() * d_hat;
                    let e_dot_k = (local.z * b[2]).mul_add(
                        d_local.z,
                        (local.y * b[1]).mul_add(d_local.y, local.x * b[0] * d_local.x),
                    );
                    let residual = Vec3::new(
                        d_local.x.mul_add(x - b[0], local.x * e_dot_k),
                        d_local.y.mul_add(x - b[1], local.y * e_dot_k),
                        d_local.z.mul_add(x - b[2], local.z * e_dot_k),
                    );
                    assert!(
                        residual.length() < RESIDUAL_BOUND,
                        "{name}: eigen-equation residual too large at theta={theta} \
                         (wave_normal={wave_normal:?}): |residual|={} >= {RESIDUAL_BOUND}",
                        residual.length()
                    );

                    if let Some(p) = *prev {
                        let cos = p.dot(d_hat).clamp(-1.0, 1.0).abs();
                        assert!(
                            cos > 0.98,
                            "{name}: eigenvector direction should vary continuously \
                             (up to sign) between adjacent sweep samples at theta={theta}, \
                             got |dot|={cos}"
                        );
                    }
                    *prev = Some(d_hat);
                }
            }
        }
    }

    /// Pins `precise_root_near`'s discriminant reformulation (see its doc comment for
    /// the derivation) against the NAIVE `B^2 - 4C` formula `wave_indices` itself uses,
    /// in `f64` (so this test's own arithmetic doesn't share whatever `f32` rounding
    /// is under scrutiny) -- a dense sweep of direction-cosine triples and principal
    /// `1/n^2` triples, asserting the two formulas' discriminants agree to within a
    /// tight `f64` tolerance. This is an ALGEBRAIC identity (verified symbolically
    /// during development, not just spot-checked), so any real disagreement here would
    /// mean a transcription bug in the reformulation, not mere floating-point noise.
    #[test]
    fn discriminant_reformulation_matches_naive_b2_minus_4c() {
        fn naive_disc_sq(a: f64, bb: f64, cc: f64, p: f64, q: f64, r: f64) -> f64 {
            let big_b = cc.mul_add(p + q, bb.mul_add(p + r, a * (q + r)));
            let big_c = (cc * p).mul_add(q, (bb * p).mul_add(r, a * q * r));
            4.0f64.mul_add(-big_c, big_b * big_b)
        }

        fn reformulated_disc_sq(a: f64, bb: f64, cc: f64, p: f64, q: f64, r: f64) -> f64 {
            let xdiff = p - q;
            let ydiff = r - p;
            let a_plus_c = a + cc;
            let a_plus_bb = a + bb;
            let two_a_minus_bc = 2.0 * bb.mul_add(-cc, a);
            (a_plus_bb * a_plus_bb).mul_add(
                ydiff * ydiff,
                (two_a_minus_bc * xdiff).mul_add(ydiff, (a_plus_c * a_plus_c) * (xdiff * xdiff)),
            )
        }

        let cosine_triples: [(f64, f64, f64); 6] = [
            (0.046_512, 0.011_628, 0.941_860),
            (1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0),
            (0.7, 0.2, 0.1),
            (0.02, 0.9, 0.08),
            (0.5, 0.0, 0.5),
            (0.9999, 0.00005, 0.00005),
        ];
        let index_triples = [
            (
                1.0 / 1.60_f64.powi(2),
                1.0 / 1.65_f64.powi(2),
                1.0 / 1.75_f64.powi(2),
            ),
            (
                1.0 / 1.740_778_4_f64.powi(2),
                1.0 / 1.742_729_4_f64.powi(2),
                1.0 / 1.748_378_4_f64.powi(2),
            ),
            (
                1.0 / 1.4_f64.powi(2),
                1.0 / 2.4_f64.powi(2),
                1.0 / 2.49_f64.powi(2),
            ),
        ];

        for &(a, bb, cc) in &cosine_triples {
            assert!(
                (a + bb + cc - 1.0).abs() < 1e-9,
                "test premise: direction cosines squared must sum to 1"
            );
            for &(p, q, r) in &index_triples {
                let naive = naive_disc_sq(a, bb, cc, p, q, r);
                let reformulated = reformulated_disc_sq(a, bb, cc, p, q, r);
                let scale = naive.abs().max(reformulated.abs()).max(1e-12);
                assert!(
                    (naive - reformulated).abs() / scale < 1e-9,
                    "discriminant reformulation disagrees with naive B^2-4C: \
                     cosines=({a},{bb},{cc}) indices=({p},{q},{r}) naive={naive:e} \
                     reformulated={reformulated:e}"
                );
            }
        }
    }

    /// `poynting_direction` must reduce to zero walk-off (S == `k_hat`) whenever the
    /// electric field is already perpendicular to the wave normal (the isotropic /
    /// ordinary-ray case), and must be finite and unit length in general.
    #[test]
    fn poynting_direction_has_no_walk_off_when_e_is_transverse() {
        let k_hat = Vec3::new(0.3, 0.6, 0.74).normalize();
        let e_hat = k_hat.cross(Vec3::Y).normalize(); // exactly transverse to k_hat
        let s = poynting_direction(k_hat, e_hat);
        assert!(
            (s - k_hat).length() < 1e-4,
            "S should equal k_hat when E is exactly transverse, got {s:?}"
        );
    }

    /// `poynting_direction` must always return a finite unit vector, including for an
    /// E direction with a genuine (non-degenerate) component along `k_hat` (the walk-off
    /// case).
    #[test]
    fn poynting_direction_is_finite_unit_vector_with_walk_off() {
        let k_hat = Vec3::new(0.0, 0.0, 1.0);
        let e_hat = Vec3::new(0.1, 0.0, 0.99).normalize(); // tilted toward k_hat
        let s = poynting_direction(k_hat, e_hat);
        assert!(s.is_finite(), "S must be finite, got {s:?}");
        assert!(
            (s.length() - 1.0).abs() < 1e-4,
            "S must be unit length, got {}",
            s.length()
        );
        assert!(
            s.dot(e_hat).abs() < 1e-3,
            "S must be perpendicular to E, got dot={}",
            s.dot(e_hat)
        );
    }

    /// Trichroism -- THE decisive frame-identity test. `AbsorptionTensor3::
    /// biaxial`'s doc comment claims its axis frame is bit-identical to
    /// `BiaxialIndicatrix::from_gamma_axis`'s, not merely close: both call the exact
    /// same `stable_orthonormal_basis` free function on the exact same `gamma_axis`
    /// input. This pins that claim directly by comparing the two `Mat3`s column by
    /// column, bit for bit (`to_bits()`), across several non-axis-aligned `gamma_axis`
    /// directions -- getting this subtly wrong (e.g. re-deriving the completion via a
    /// different but numerically-similar construction) would silently score absorption
    /// against axes that drift from the ones `eigen_polarizations` computed its
    /// eigenmodes in, exactly the failure mode this task's brief calls out.
    #[test]
    fn absorption_frame_is_bit_identical_to_index_frame() {
        let gamma_axes = [
            Vec3::Y,
            Vec3::X,
            Vec3::Z,
            Vec3::new(0.35, 0.82, -0.45).normalize(),
            Vec3::new(-0.6, 0.1, 0.79).normalize(),
        ];
        for gamma_axis in gamma_axes {
            let tensor = AbsorptionTensor3::biaxial(1.0, 2.0, 3.0, gamma_axis);
            let indicatrix = BiaxialIndicatrix::from_gamma_axis(1.6, 1.65, 1.75, gamma_axis);

            let pairs = [
                ("x_axis", tensor.axes.x_axis, indicatrix.axes.x_axis),
                ("y_axis", tensor.axes.y_axis, indicatrix.axes.y_axis),
                ("z_axis", tensor.axes.z_axis, indicatrix.axes.z_axis),
            ];
            for (label, t_col, i_col) in pairs {
                assert_eq!(
                    t_col.x.to_bits(),
                    i_col.x.to_bits(),
                    "gamma_axis={gamma_axis:?} {label}: x component bits differ (tensor={t_col:?}, indicatrix={i_col:?})"
                );
                assert_eq!(
                    t_col.y.to_bits(),
                    i_col.y.to_bits(),
                    "gamma_axis={gamma_axis:?} {label}: y component bits differ (tensor={t_col:?}, indicatrix={i_col:?})"
                );
                assert_eq!(
                    t_col.z.to_bits(),
                    i_col.z.to_bits(),
                    "gamma_axis={gamma_axis:?} {label}: z component bits differ (tensor={t_col:?}, indicatrix={i_col:?})"
                );
            }
        }
    }

    /// Pins the brief's own worked example: with `gamma_axis = +Y`, the deterministic
    /// `stable_orthonormal_basis` completion must place `alpha` on `+X`, `beta` on
    /// `-Z`, and `gamma` on `+Y` itself.
    #[test]
    fn absorption_frame_matches_worked_example_for_plus_y_gamma_axis() {
        let tensor = AbsorptionTensor3::biaxial(1.0, 2.0, 3.0, Vec3::Y);
        assert!(
            (tensor.axes.x_axis - Vec3::X).length() < 1e-6,
            "alpha axis should be +X, got {:?}",
            tensor.axes.x_axis
        );
        assert!(
            (tensor.axes.y_axis - (-Vec3::Z)).length() < 1e-6,
            "beta axis should be -Z, got {:?}",
            tensor.axes.y_axis
        );
        assert!(
            (tensor.axes.z_axis - Vec3::Y).length() < 1e-6,
            "gamma axis should be +Y, got {:?}",
            tensor.axes.z_axis
        );
    }

    /// Three-coefficient test: for a tensor with three genuinely distinct principal
    /// coefficients, `quadratic_form` evaluated exactly along each principal axis must
    /// return exactly that axis's own coefficient (not some blend with the other two).
    #[test]
    fn biaxial_tensor_quadratic_form_matches_each_principal_coefficient_on_axis() {
        let gamma_axis = Vec3::new(0.2, 0.6, 0.77).normalize();
        let (alpha, beta, gamma) = (1.5f32, 4.0f32, 9.0f32);
        let tensor = AbsorptionTensor3::biaxial(alpha, beta, gamma, gamma_axis);

        let alpha_axis = tensor.axes.x_axis;
        let beta_axis = tensor.axes.y_axis;

        assert!(
            (tensor.quadratic_form(alpha_axis) - alpha).abs() < 1e-4,
            "E along the alpha axis should give alpha ({alpha}), got {}",
            tensor.quadratic_form(alpha_axis)
        );
        assert!(
            (tensor.quadratic_form(beta_axis) - beta).abs() < 1e-4,
            "E along the beta axis should give beta ({beta}), got {}",
            tensor.quadratic_form(beta_axis)
        );
        assert!(
            (tensor.quadratic_form(gamma_axis) - gamma).abs() < 1e-4,
            "E along the gamma axis should give gamma ({gamma}), got {}",
            tensor.quadratic_form(gamma_axis)
        );
    }

    /// Degeneracy test: `AbsorptionTensor3::biaxial(a, a, b, axis)` (alpha == beta) must
    /// reduce to `AbsorptionTensor3::uniaxial(a, b, axis)` BIT-IDENTICALLY -- not just
    /// numerically close -- since both share the exact same `Vec3::new`/`Mat3::from_cols`
    /// construction over the exact same `stable_orthonormal_basis(axis)` call. Checked
    /// via `to_bits()` on every component of both `alpha` and `axes`.
    #[test]
    fn biaxial_tensor_alpha_beta_degenerate_matches_uniaxial_bit_exact() {
        let axis = Vec3::new(0.35, 0.82, -0.45).normalize();
        let (a, b) = (2.5f32, 7.25f32);

        let biaxial = AbsorptionTensor3::biaxial(a, a, b, axis);
        let uniaxial = AbsorptionTensor3::uniaxial(a, b, axis);

        assert_eq!(biaxial.alpha.x.to_bits(), uniaxial.alpha.x.to_bits());
        assert_eq!(biaxial.alpha.y.to_bits(), uniaxial.alpha.y.to_bits());
        assert_eq!(biaxial.alpha.z.to_bits(), uniaxial.alpha.z.to_bits());
        let axis_pairs = [
            ("x_axis", biaxial.axes.x_axis, uniaxial.axes.x_axis),
            ("y_axis", biaxial.axes.y_axis, uniaxial.axes.y_axis),
            ("z_axis", biaxial.axes.z_axis, uniaxial.axes.z_axis),
        ];
        for (label, b_col, u_col) in axis_pairs {
            assert_eq!(b_col.x.to_bits(), u_col.x.to_bits(), "{label} x");
            assert_eq!(b_col.y.to_bits(), u_col.y.to_bits(), "{label} y");
            assert_eq!(b_col.z.to_bits(), u_col.z.to_bits(), "{label} z");
        }

        // And, as a corollary, quadratic_form itself must agree bit-for-bit for any
        // direction (not just special ones), since it's computed purely from `alpha`
        // and `axes`, which are now proven identical above.
        for d in [
            Vec3::X,
            Vec3::Y,
            Vec3::Z,
            Vec3::new(0.4, -0.5, 0.77).normalize(),
        ] {
            assert_eq!(
                biaxial.quadratic_form(d).to_bits(),
                uniaxial.quadratic_form(d).to_bits(),
                "quadratic_form must agree bit-exactly at d={d:?}"
            );
        }
    }
}

#[cfg(test)]
mod walk_off_symmetry_tests {
    use super::*;

    /// The optic axis is a director, not a directed vector -- `c_axis` and
    /// `-c_axis` describe the identical physical crystal, so
    /// `extraordinary_poynting_dir` must return the same Poynting direction for both.
    /// Swept across many incidence angles, deliberately crossing both sides of the
    /// previously-broken `wave_normal.dot(c_axis) >= 0` branch, for calcite-like
    /// ordinary/extraordinary indices and an off-axis `c_axis`.
    #[test]
    fn extraordinary_poynting_dir_is_axis_direction_symmetric() {
        let n_o = 1.658f32;
        let n_e = 1.486f32;
        let c_axis = Vec3::new(0.2, 0.9, -0.35).normalize(); // deliberately off-axis
        let perp = {
            let p = c_axis.cross(Vec3::X);
            if p.length_squared() < 1e-6 {
                c_axis.cross(Vec3::Y).normalize()
            } else {
                p.normalize()
            }
        };

        for i in 0..37 {
            let theta = (i as f32) / 36.0 * std::f32::consts::PI; // sweep 0..=180 deg
            let wave_normal = (theta.cos() * c_axis + theta.sin() * perp).normalize();

            let s_pos =
                BirefringenceParams::extraordinary_poynting_dir(wave_normal, c_axis, n_o, n_e);
            let s_neg =
                BirefringenceParams::extraordinary_poynting_dir(wave_normal, -c_axis, n_o, n_e);

            assert!(
                (s_pos - s_neg).length() < 1e-4,
                "S(c) must equal S(-c) at theta={} deg (got s_pos={s_pos:?}, s_neg={s_neg:?})",
                theta.to_degrees()
            );
        }
    }

    /// Pins the exact measured regression from the physics review: a wave normal at 45
    /// degrees from the optic axis in a calcite-like negative uniaxial material
    /// (`n_o`=1.658, `n_e`=1.486) must walk off to 51.23 degrees from the axis -- not 38.77
    /// degrees (the pre-fix answer on the `wave_normal.dot(c_axis) > 0` branch, off by
    /// exactly `2*delta` -- and this must hold whichever of `c_axis`/`-c_axis` is passed
    /// in, since both must agree per the symmetry test above.
    #[test]
    fn extraordinary_poynting_dir_matches_analytic_walk_off_both_hemispheres() {
        let n_o = 1.658f32;
        let n_e = 1.486f32;
        let c_axis = Vec3::Y;
        let theta = 45.0f32.to_radians();
        let wave_normal = Vec3::new(theta.sin(), theta.cos(), 0.0); // 45 deg from +Y

        for c in [c_axis, -c_axis] {
            let s = BirefringenceParams::extraordinary_poynting_dir(wave_normal, c, n_o, n_e);
            let angle_from_axis = s.dot(c_axis).clamp(-1.0, 1.0).abs().acos().to_degrees();
            assert!(
                (angle_from_axis - 51.23).abs() < 0.05,
                "expected walk-off to land at 51.23 deg from the axis for c={c:?}, got {angle_from_axis}"
            );
        }
    }
}
