use crate::train::prelude::*;

pub fn teacher_variant_dim(variant: VisionTeacherVariant) -> usize {
    match variant {
        VisionTeacherVariant::Vits => 384,
        VisionTeacherVariant::Vitb => 768,
        VisionTeacherVariant::Vitl => 1024,
        VisionTeacherVariant::Vitg => 1536,
    }
}
