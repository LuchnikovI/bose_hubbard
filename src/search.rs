use crate::{basis_hashing::hash_state, basis_iterator::BasisIterator};

#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) struct ValuePositionMap<const N: usize>(Vec<(usize, usize)>);

impl<const N: usize> ValuePositionMap<N> {

    pub(super) fn new(particles_number: u8, constraints: &[u8]) -> Self
    {
        let mut value_pos_map = BasisIterator::<N>::new(particles_number, constraints)
            .map(|x| hash_state(&x))
            .enumerate()
            .collect::<Vec<_>>();
        value_pos_map.sort_unstable_by(|lhs, rhs| {
            lhs.1.cmp(&rhs.1)
        });
        ValuePositionMap(value_pos_map)
    }

    #[inline(always)]
    pub(super) fn get_pos(&self, hash_val: usize) -> usize
    {
        let pos = self.0.binary_search_by(|x| { x.1.cmp(&hash_val) }).unwrap();
        unsafe { self.0.get_unchecked(pos).0 }
    }
}

#[cfg(test)]
mod tests {
    use crate::{basis_hashing::hash_state, basis_iterator::BasisIterator};

    use super::ValuePositionMap;

    fn _test_search<const N: usize>(particles_number: u8)
    {
        let constraints = [particles_number; N];
        let map = ValuePositionMap::<N>::new(particles_number, &constraints);
        let iter = BasisIterator::<N>::new(particles_number, &constraints)
            .map(|x| hash_state(&x));
        for (i, hash) in iter.enumerate() {
            assert_eq!(i, map.get_pos(hash));
        }
    }

    #[test]
    fn test_search()
    {
        _test_search::<10>(6);
        _test_search::<6>(10);
        _test_search::<10>(10);
        _test_search::<2>(11);
        _test_search::<11>(0);
        _test_search::<11>(1);
        _test_search::<11>(2);
    }
}