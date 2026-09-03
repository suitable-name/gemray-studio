/// A user-defined custom gemstone material, exactly as stored in the
/// `custom_gem_materials` table.
///
/// This is a plain data row owned by `diagram-catalog` -- it deliberately does
/// NOT depend on `gemray`'s richer `GemMaterial` type (with its dispersion
/// model, crystal system, etc.), so that this crate stays independently
/// publishable. Callers that need a `GemMaterial` (e.g. a renderer) convert
/// this row via `GemMaterial::new_custom(..)` at the call site, then apply
/// `crystal_system`/`optical_character`/`biaxial_delta_beta_alpha` on top (see
/// those fields' own doc comments).
#[derive(Debug, Clone, PartialEq)]
pub struct CustomMaterialRow {
    pub name: String,
    pub refractive_index: f32,
    pub dispersion: f32,
    pub birefringence: f32,
    pub absorption_rgb: [f32; 3],
    /// `gemray::optics::materials::CrystalSystem`'s variant name as plain text (e.g.
    /// `"Trigonal"`), or `None`. Stored as a string rather than the enum itself so this
    /// crate still doesn't need to depend on `gemray` (see the struct doc comment) --
    /// the boundary that knows both types (`apps/diagram-gui`) is responsible for
    /// parsing it back.
    ///
    /// `None` covers two cases that must be handled identically: a row saved before
    /// this field existed, and a row whose author never overrode the crystal system.
    /// Either way the caller must fall back to exactly what
    /// `GemMaterial::new_custom(..)` infers from `birefringence` alone, so a material
    /// saved before this field existed keeps rendering bit-identically.
    pub crystal_system: Option<String>,
    /// `gemray::optics::materials::OpticalCharacter`'s variant name as plain text
    /// (e.g. `"UniaxialNegative"`), or `None`. Same string-not-enum reasoning and same
    /// "fall back to `new_custom`'s own inference" contract as `crystal_system` above.
    pub optical_character: Option<String>,
    /// `gemray::optics::materials::GemMaterial::biaxial_delta_beta_alpha` --
    /// `n_beta - n_alpha` at the sodium D line -- carried straight through as `f32`
    /// since it's already a plain number, not an enum needing a crate boundary. `None`
    /// for every isotropic/uniaxial material (the overwhelming majority) and for any
    /// row saved before biaxial authoring existed; only meaningful when
    /// `optical_character` is one of the two biaxial variants.
    pub biaxial_delta_beta_alpha: Option<f32>,
}
