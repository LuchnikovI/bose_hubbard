#[inline(always)]
pub(super) fn hash_state<const N: usize>(
    state: &[u8; N],
) -> usize
{
    let mut hash_value = 1;
    for elem in state {
        hash_value <<= elem + 1;
        hash_value |= 1;
    }
    hash_value >> 1
}

#[allow(dead_code)]
#[inline(always)]
pub(super) fn unhash<const N: usize>(
    mut hash: usize
) -> [u8; N]
{
    let mut state = [0; N];
    for particles_number in state.iter_mut().rev() {
        while (hash & 1) == 0 {
            *particles_number += 1;
            hash >>= 1;
        }
        hash >>= 1;
    }
    state
}

#[cfg(test)]
mod tests {
    use super::{hash_state, unhash};

    #[test]
    fn test_hash_state()
    {
        let hash_val = hash_state(&[]);
        assert_eq!(hash_val, 0);
        let hash_val = hash_state(&[0]);
        assert_eq!(hash_val, 1);
        let hash_val = hash_state(&[3]);
        assert_eq!(hash_val, 0b1000);
        let hash_val = hash_state(&[1, 2, 3]);
        assert_eq!(hash_val, 0b101001000);
        let hash_val = hash_state(&[4, 2, 1, 3]);
        assert_eq!(hash_val, 0b10000100101000);
    }

    #[test]
    fn test_unhash()
    {
        let hash = 0b0;
        let state = unhash::<0>(hash);
        assert_eq!([0u8; 0], state);
        let hash = 0b1;
        let state = unhash::<1>(hash);
        assert_eq!([0], state);
        let hash = 0b101001000;
        let state = unhash::<3>(hash);
        assert_eq!([1, 2, 3], state);
        let hash = 0b10000100101000;
        let state = unhash::<4>(hash);
        assert_eq!([4, 2, 1, 3], state);
    }
}