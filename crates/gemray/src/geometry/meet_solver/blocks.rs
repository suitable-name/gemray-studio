//! Tier/block classification: which of crown, pavilion or girdle each tier
//! belongs to, and the crown/pavilion side convention ([`tier_sides`]) that
//! classification and the normal-vector construction in `candidates` both
//! build on. Kept as one small unit because it is the shared classification
//! every other submodule (anchors, name resolution, the solve pipeline)
//! depends on.

use super::MeetTierInput;

/// Which block a tier belongs to.
///
/// Classified the same way [`solve_meet_points`](super::solve_meet_points)'s
/// arrangement does: by the sign/magnitude of the tier's first-instance normal
/// `y`-component, which is a pure function of `angle_deg` and crown/pavilion side
/// (identical across every index instance of one tier, so this never needs the
/// actual per-instance azimuth).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Block {
    Crown,
    Pavilion,
    Girdle,
}

/// Classifies every tier's [`Block`] in one pass, honoring the same unsigned-zero
/// crown/pavilion inheritance [`tier_sides`] does.
#[must_use]
pub fn classify_blocks(tiers: &[MeetTierInput]) -> Vec<Block> {
    let sides = tier_sides(tiers);
    tiers
        .iter()
        .zip(&sides)
        .map(|(t, &crown)| {
            let theta = t.angle_deg.abs().to_radians();
            let y = if crown { theta.cos() } else { -theta.cos() };
            if y.abs() <= 1e-6 {
                Block::Girdle
            } else if y > 0.0 {
                Block::Crown
            } else {
                Block::Pavilion
            }
        })
        .collect()
}

/// Crown/pavilion side per tier, honoring the unsigned-zero inheritance rule.
pub(super) fn tier_sides(tiers: &[MeetTierInput]) -> Vec<bool> {
    let mut sides = Vec::with_capacity(tiers.len());
    let mut last_crown = true;
    for tier in tiers {
        let crown = if tier.angle_deg == 0.0 {
            if tier.angle_deg.is_sign_negative() {
                false
            } else {
                last_crown
            }
        } else {
            tier.angle_deg > 0.0
        };
        last_crown = crown;
        sides.push(crown);
    }
    sides
}

#[cfg(test)]
mod tests {
    use super::{super::MeetConstraint, *};

    #[test]
    fn classify_blocks_matches_angle_sign_and_girdle_threshold() {
        let tiers = vec![
            MeetTierInput {
                angle_deg: 45.0,
                indices: vec![],
                constraint: MeetConstraint::MeetExisting,
                names: vec![],
            },
            MeetTierInput {
                angle_deg: -45.0,
                indices: vec![],
                constraint: MeetConstraint::MeetExisting,
                names: vec![],
            },
            MeetTierInput {
                angle_deg: 90.0,
                indices: vec![],
                constraint: MeetConstraint::MeetExisting,
                names: vec![],
            },
            MeetTierInput {
                angle_deg: -90.0,
                indices: vec![],
                constraint: MeetConstraint::MeetExisting,
                names: vec![],
            },
        ];
        let blocks = classify_blocks(&tiers);
        assert_eq!(
            blocks,
            vec![Block::Crown, Block::Pavilion, Block::Girdle, Block::Girdle]
        );
    }
}
