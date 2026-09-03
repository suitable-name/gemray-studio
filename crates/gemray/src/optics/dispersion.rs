#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum DispersionModel {
    Sellmeier1 { b1: f32, c1: f32 },
    Sellmeier3 { b: [f32; 3], c: [f32; 3] },
    Cauchy { a: f32, b: f32, c: f32 },
}

impl DispersionModel {
    /// Evaluates the refractive index at a given wavelength (in nanometers).
    #[must_use]
    pub fn evaluate(&self, lambda_nm: f32) -> f32 {
        let lambda_um = lambda_nm * 1e-3;
        let l2 = lambda_um * lambda_um;

        match self {
            Self::Sellmeier1 { b1, c1 } => {
                let n2 = 1.0 + (b1 * l2) / (l2 - c1);
                n2.max(1.0).sqrt()
            }
            Self::Sellmeier3 { b, c } => {
                let mut n2 = 1.0;
                n2 += (b[0] * l2) / (l2 - c[0]);
                n2 += (b[1] * l2) / (l2 - c[1]);
                n2 += (b[2] * l2) / (l2 - c[2]);
                n2.max(1.0).sqrt()
            }
            Self::Cauchy {
                a,
                b: b_coeff,
                c: c_coeff,
            } => {
                let l4 = l2 * l2;
                a + (b_coeff / l2) + (c_coeff / l4)
            }
        }
    }
}
