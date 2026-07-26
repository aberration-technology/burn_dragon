use crate::ruliad::rng::SplitMix64;

pub fn random_state(width: usize, rng: &mut SplitMix64) -> Vec<u8> {
    (0..width).map(|_| u8::from(rng.next_bool())).collect()
}

pub fn step(rule: u8, state: &[u8]) -> Vec<u8> {
    let width = state.len();
    if width == 0 {
        return Vec::new();
    }
    (0..width)
        .map(|index| {
            let left = state[(index + width - 1) % width] & 1;
            let center = state[index] & 1;
            let right = state[(index + 1) % width] & 1;
            let neighborhood = (left << 2) | (center << 1) | right;
            (rule >> neighborhood) & 1
        })
        .collect()
}

pub fn trace(rule: u8, initial: &[u8], steps: usize) -> Vec<Vec<u8>> {
    let mut frames = Vec::with_capacity(steps.saturating_add(1));
    frames.push(initial.to_vec());
    for _ in 0..steps {
        let next = step(rule, &frames[frames.len() - 1]);
        frames.push(next);
    }
    frames
}

pub fn complement_state(state: &[u8]) -> Vec<u8> {
    state.iter().map(|value| 1 - (value & 1)).collect()
}

pub fn complement_rule(rule: u8) -> u8 {
    let mut target = 0u8;
    for neighborhood in 0..8 {
        let source_neighborhood = 7 - neighborhood;
        let source_bit = (rule >> source_neighborhood) & 1;
        target |= (1 - source_bit) << neighborhood;
    }
    target
}

pub fn states_equal(left: &[Vec<u8>], right: &[Vec<u8>]) -> bool {
    left.len() == right.len()
        && left
            .iter()
            .zip(right)
            .all(|(left_frame, right_frame)| left_frame == right_frame)
}

pub fn format_state(state: &[u8]) -> String {
    state
        .iter()
        .map(|value| if (value & 1) == 0 { '0' } else { '1' })
        .collect()
}

pub fn parse_state(value: &str) -> Vec<u8> {
    value
        .bytes()
        .filter_map(|byte| match byte {
            b'0' => Some(0),
            b'1' => Some(1),
            _ => None,
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rule_30_complement_is_rule_135() {
        assert_eq!(complement_rule(30), 135);
    }

    #[test]
    fn trace_recomputes_expected_length() {
        let frames = trace(30, &parse_state("00101000"), 4);
        assert_eq!(frames.len(), 5);
        assert_eq!(frames[0], parse_state("00101000"));
    }
}
