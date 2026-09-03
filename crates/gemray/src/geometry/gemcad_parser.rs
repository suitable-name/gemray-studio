use super::plane::GpuFacetPlane;
use glam::Vec3;
use std::f32::consts::PI;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FacetTierType {
    Crown,
    Pavilion,
    Girdle,
}

#[derive(Debug, Clone)]
pub struct FacetTier {
    pub tier_type: FacetTierType,
    pub angle_deg: f32,    // 0 to 90
    pub indices: Vec<u32>, // Gear indices
    pub offset: f32,       // negative depth (d)
}

#[derive(Debug, Clone)]
pub struct CuttingSchedule {
    pub gear_teeth: u32,
    pub tiers: Vec<FacetTier>,
}

impl CuttingSchedule {
    #[must_use]
    pub const fn new(gear_teeth: u32) -> Self {
        Self {
            gear_teeth,
            tiers: Vec::new(),
        }
    }

    pub fn add_tier(
        &mut self,
        tier_type: FacetTierType,
        angle_deg: f32,
        indices: Vec<u32>,
        offset: f32,
    ) {
        self.tiers.push(FacetTier {
            tier_type,
            angle_deg,
            indices,
            offset,
        });
    }

    #[must_use]
    pub fn into_planes(&self) -> Vec<GpuFacetPlane> {
        let mut planes = Vec::new();

        for tier in &self.tiers {
            let theta = tier.angle_deg.to_radians();
            let sin_theta = theta.sin();
            let cos_theta = theta.cos();

            for &g in &tier.indices {
                let phi = 2.0 * PI * (g as f32) / (self.gear_teeth as f32);
                let sin_phi = phi.sin();
                let cos_phi = phi.cos();

                let n = match tier.tier_type {
                    FacetTierType::Crown => {
                        Vec3::new(sin_theta * cos_phi, cos_theta, sin_theta * sin_phi)
                    }
                    FacetTierType::Pavilion => {
                        Vec3::new(sin_theta * cos_phi, -cos_theta, sin_theta * sin_phi)
                    }
                    FacetTierType::Girdle => Vec3::new(cos_phi, 0.0, sin_phi),
                };

                planes.push(GpuFacetPlane::new(n, tier.offset));
            }
        }
        planes
    }
}

/// Simplified fallback parser for `GemCAD` `.asc` cutting-schedule files.
///
/// # Errors
///
/// This parser is lenient by design: a line it cannot make sense of (unknown
/// keyword, unparseable angle, tier with no valid index) is silently skipped rather
/// than rejected, so it currently never returns `Err` -- an unparseable or empty
/// input simply produces a `CuttingSchedule` with fewer (possibly zero) tiers. It
/// returns `Result` to match the shape of the other parsers in this crate and to
/// leave room for real validation (e.g. rejecting a file with zero recognized tiers)
/// without an API break.
pub fn parse_gemcad_asc(content: &str) -> Result<CuttingSchedule, String> {
    // This is a simplified fallback parser. A production parser would read exact GemCAD .asc keywords.
    // Assuming format:
    // GEAR 96
    // PAVILION
    // 45.00  3 15 27 39 ...   d=-1.5
    // CROWN
    // 42.00  3 15 ...         d=-1.0
    //
    let mut schedule = CuttingSchedule::new(96);
    let mut current_type = FacetTierType::Pavilion;
    let mut tier_counter = 1;

    for line in content.lines() {
        let line = line.trim();
        if line.is_empty() || line.starts_with(';') {
            continue;
        }

        if line.starts_with("GEAR") {
            if let Some(num) = line.split_whitespace().nth(1) {
                schedule.gear_teeth = num.parse().unwrap_or(96);
            }
            continue;
        }

        if line.starts_with("PAVILION") {
            current_type = FacetTierType::Pavilion;
            continue;
        }

        if line.starts_with("CROWN") {
            current_type = FacetTierType::Crown;
            continue;
        }

        if line.starts_with("GIRDLE") {
            current_type = FacetTierType::Girdle;
            continue;
        }

        // Parse tier line
        let parts: Vec<&str> = line.split_whitespace().collect();
        if parts.len() < 2 {
            continue;
        }

        if let Ok(angle) = parts[0].parse::<f32>() {
            let mut indices = Vec::new();
            let mut offset = (tier_counter as f32).mul_add(-0.1, -1.0); // arbitrary fallback offset

            for &p in &parts[1..] {
                if let Some(rest) = p.strip_prefix("d=") {
                    if let Ok(d) = rest.parse::<f32>() {
                        offset = d;
                        if offset > 0.0 {
                            offset = -offset;
                        }
                    }
                } else if let Ok(idx) = p.parse::<u32>() {
                    indices.push(idx);
                }
            }

            if !indices.is_empty() {
                schedule.add_tier(current_type, angle, indices, offset);
                tier_counter += 1;
            }
        }
    }

    Ok(schedule)
}
