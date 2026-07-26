#[derive(Debug, Clone)]
pub struct SplitMix64 {
    state: u64,
}

impl SplitMix64 {
    pub fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    pub fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut value = self.state;
        value = (value ^ (value >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        value = (value ^ (value >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        value ^ (value >> 31)
    }

    pub fn next_u8(&mut self) -> u8 {
        (self.next_u64() >> 56) as u8
    }

    pub fn next_bool(&mut self) -> bool {
        (self.next_u64() >> 63) != 0
    }

    pub fn next_f32(&mut self) -> f32 {
        let bits = (self.next_u64() >> 40) as u32;
        bits as f32 / (1u32 << 24) as f32
    }

    pub fn next_usize(&mut self, upper_exclusive: usize) -> usize {
        if upper_exclusive <= 1 {
            return 0;
        }
        let bound = upper_exclusive as u128;
        let zone = ((u64::MAX as u128 + 1) / bound) * bound;
        loop {
            let value = self.next_u64() as u128;
            if value < zone {
                return (value % bound) as usize;
            }
        }
    }

    pub fn range_usize(&mut self, min: usize, max: usize) -> usize {
        if min >= max {
            return min;
        }
        min + self.next_usize(max - min + 1)
    }
}

pub fn mix_seed(seed: u64, parts: impl IntoIterator<Item = u64>) -> u64 {
    let mut rng = SplitMix64::new(seed);
    for part in parts {
        rng.state ^= part.wrapping_add(0xD1B5_4A32_D192_ED03);
        let _ = rng.next_u64();
    }
    rng.next_u64()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn splitmix64_is_deterministic() {
        let mut left = SplitMix64::new(7);
        let mut right = SplitMix64::new(7);
        for _ in 0..32 {
            assert_eq!(left.next_u64(), right.next_u64());
        }
    }

    #[test]
    fn mixed_seed_uses_parts() {
        assert_eq!(mix_seed(1, [2, 3]), mix_seed(1, [2, 3]));
        assert_ne!(mix_seed(1, [2, 3]), mix_seed(1, [3, 2]));
    }

    #[test]
    fn bounded_sampler_stays_in_range_and_covers_buckets() {
        let mut rng = SplitMix64::new(19);
        let mut counts = [0usize; 7];
        for _ in 0..7000 {
            let value = rng.next_usize(counts.len());
            assert!(value < counts.len());
            counts[value] += 1;
        }

        for count in counts {
            assert!(
                (800..=1200).contains(&count),
                "bounded sampler bucket outside coarse balance range: {count}"
            );
        }
    }
}
