//! A fixed-size heap structure with manually provided stateful comparison function.
#![doc(html_root_url = "https://docs.rs/fixed_heap")]
#![crate_name = "fixed_heap"]
#![warn(
    missing_debug_implementations,
    trivial_casts,
    trivial_numeric_casts,
    unused_lifetimes,
    unused_import_braces,
    clippy::shadow_unrelated
)]
#![deny(missing_docs, unaligned_references, unsafe_op_in_unsafe_fn)]
#![cfg_attr(
    all(nightly, feature = "unstable"),
    feature(maybe_uninit_uninit_array, slice_swap_unchecked)
)]

use std::{
    fmt::{Debug, Formatter, Result},
    iter::FusedIterator,
    mem::{self, ManuallyDrop, MaybeUninit},
    slice::{Iter, IterMut},
};

/// A fixed-size heap structure with manually provided stateful comparison function.
pub struct FixedHeap<T, const N: usize> {
    high: usize,
    data: [MaybeUninit<T>; N],
}

impl<T, const N: usize> FixedHeap<T, N> {
    /// Creates a new empty `FixedHeap`.
    ///
    /// Passing in a `comparer` and `state` is deferred so as to not hold a reference
    /// preventing mutation of other parts of `state` that do not affect the heap order.
    ///
    /// # Examples
    ///
    /// ```
    /// # use fixed_heap::*;
    /// let mut heap: FixedHeap<i32, 16> = FixedHeap::new();
    /// ```
    pub fn new() -> Self {
        Self {
            high: 0,
            #[cfg(all(nightly, feature = "unstable"))]
            data: MaybeUninit::uninit_array(),
            #[cfg(not(all(nightly, feature = "unstable")))]
            data: unsafe { MaybeUninit::uninit().assume_init() },
        }
    }

    /// Copies `slice` into the backing storage, ignoring heap properties.
    ///
    /// Caution: this will not preserve the heap property of the structure
    ///
    /// # Examples
    ///
    /// ```
    /// # use fixed_heap::*;
    /// let mut heap: FixedHeap<i32, 16> = FixedHeap::new();
    /// let array = [4; 12];
    /// heap.copy_from_slice(&array[2..8]);
    /// assert_eq!(heap.len(), 6);
    /// ```
    pub fn copy_from_slice(&mut self, slice: &[T])
    where
        T: Copy,
    {
        assert!(slice.len() <= N);
        self.high = slice.len();
        self.as_slice_mut().copy_from_slice(slice);
    }

    /// Returns a reference to the highest priority element.
    ///
    /// # Returns
    ///
    /// `None` if there are no elements in the heap.
    ///
    /// `Some(elem)` if there was an element. `elem` is higher priority than all other elements.
    ///
    /// # Examples
    ///
    /// ```
    /// # use fixed_heap::*;
    /// let mut heap: FixedHeap<i32, 16> = FixedHeap::new();
    /// assert_eq!(heap.add_last(1), None);
    /// assert_eq!(heap.peek(), Some(&1));
    /// ```
    #[inline(always)]
    pub fn peek(&self) -> Option<&T> {
        self.peek_at(0)
    }

    /// Returns a reference to the element at `index`.
    ///
    /// # Returns
    ///
    /// `None` if there is no element at `index`.
    ///
    /// `Some(elem)` if there was an element.
    ///
    /// # Examples
    ///
    /// ```
    /// # use fixed_heap::*;
    /// let mut heap: FixedHeap<i32, 16> = FixedHeap::new();
    /// assert_eq!(heap.add_last(1), None);
    /// assert_eq!(heap.add_last(2), None);
    /// assert_eq!(heap.peek_at(1), Some(&2));
    /// ```
    #[inline(always)]
    pub fn peek_at(&self, index: usize) -> Option<&T> {
        // # Safety
        // If `index` is below `high` then `data[index]` is initialized.
        if index < self.high {
            Some(unsafe { self.data.get_unchecked(index).assume_init_ref() })
        } else {
            None
        }
    }

    /// Tries to add `value` to the end, ignoring heap properties.
    ///
    /// Caution: this will not preserve the heap property of the structure
    ///
    /// # Returns
    ///
    /// `None` if there was spare capacity to accommodate `value`
    ///
    /// `Some(value)` if there was no spare capacity
    ///
    /// # Time Complexity
    ///
    /// O(1)
    ///
    /// # Examples
    ///
    /// ```
    /// # use fixed_heap::*;
    /// let mut heap: FixedHeap<i32, 1> = FixedHeap::new();
    /// assert_eq!(heap.add_last(1), None);
    /// assert_eq!(heap.add_last(2), Some(2));
    /// ```
    pub fn add_last(&mut self, value: T) -> Option<T> {
        if self.high == N {
            // Not enough space to add it
            Some(value)
        } else {
            // There's enough space to add it
            // # Safety
            // `high` is guaranteed to be a valid index here because it is less than `N`
            unsafe {
                *self.data.get_unchecked_mut(self.high) = MaybeUninit::new(value);
            }
            self.high += 1;
            None
        }
    }

    /// Removes and returns the element at `index`, ignoring heap properties.
    /// Use `pop_at` instead to preserve heap properties.
    ///
    /// Caution: this will not preserve the heap property of the structure
    ///
    /// # Returns
    ///
    /// `None` if there's no element at `index`.
    ///
    /// `Some(elem)` if there was an element at `index`.
    ///
    /// # Time Complexity
    ///
    /// O(1)
    ///
    /// # Examples
    ///
    /// ```
    /// # use fixed_heap::*;
    /// let mut heap: FixedHeap<i32, 16> = FixedHeap::new();
    /// assert_eq!(heap.swap_remove(0), None); // []
    /// assert_eq!(heap.add_last(1), None); // [1]
    /// assert_eq!(heap.swap_remove(1), None); // [1]
    /// assert_eq!(heap.add_last(2), None); // [1, 2]
    /// assert_eq!(heap.swap_remove(0), Some(1)); // [2]
    /// assert_eq!(heap.swap_remove(0), Some(2)); // []
    /// ```
    pub fn swap_remove(&mut self, index: usize) -> Option<T> {
        if index < self.high {
            self.high -= 1;

            // # Safety
            // `data[index]` holds an initialized value because `index` is less than `high`
            // We just copy it because we're about to overwrite it anyways
            let removed_node = unsafe { self.data.get_unchecked(index).assume_init_read() };
            // We can also just copy the last element because the last index will now be treated as uninit
            unsafe {
                *self.data.get_unchecked_mut(index) =
                    MaybeUninit::new(self.data.get_unchecked(self.high).assume_init_read());
            }

            Some(removed_node)
        } else {
            None
        }
    }

    /// Tries to push `value` onto the heap, calling `comparer` with `state` to determine ordering.
    ///
    /// `comparer` should return true if its first argument is strictly higher priority than the second.
    /// It is technically permitted to return true when given elements of equal priority,
    /// although it is recommended to return false in those cases to avoid swaps for performance reasons.
    /// A possible use case for returning true when equal is to have newly added elements take priority over older ones.
    ///
    /// Use `state` to pass in another datastructure in order to sort keys by associated values.
    ///
    /// The same comparer should always be used for `push` and `pop`, and `state` should be stable.
    ///
    /// If `comparer` judges that a particular element is higher priority than another one,
    /// it is expected that that remains true for as long as those elements are in this heap.
    ///
    /// # Returns
    ///
    /// `None` if there was spare capacity to accommodate `value`
    ///
    /// `Some(elem)` if the lowest priority element `elem` had to be evicted to accommodate `value`.
    /// `elem` may be `value` if all the elements already present were higher priority than `value`.
    ///
    /// # Time Complexity
    ///
    /// If there was spare capacity, average time complexity O(1) and worst case O(log N)
    ///
    /// If the heap was full, average time complexity O(log N) and worst case O(N)
    /// It is recommended to avoid letting the heap reach capacity.
    ///
    /// # Examples
    ///
    /// ```
    /// # use fixed_heap::*;
    /// let mut heap: FixedHeap<i32, 16> = FixedHeap::new();
    /// let comparer = |a: &i32, b: &i32, _: &()| a > b;
    /// heap.push(1, &comparer, &());
    /// ```
    ///
    /// With keys into another struct:
    /// ```
    /// # use fixed_heap::*;
    /// let mut heap: FixedHeap<usize, 16> = FixedHeap::new();
    /// let comparer = |a: &usize, b: &usize, state: &[i32; 4]| state[*a] > state[*b];
    /// let state = [1, 3, 1, 2];
    /// heap.push(0, &comparer, &state);
    /// ```
    pub fn push<S, F>(&mut self, value: T, comparer: &F, state: &S) -> Option<T>
    where
        F: Fn(&T, &T, &S) -> bool,
    {
        let mut result = None;
        let mut node_index = self.high;
        if let Some(value) = self.add_last(value) {
            // There was no space for the value. Let's try to evict something.
            if N == 0 {
                // Trivial special case to avoid invalid array access
                return Some(value);
            } else if self.high == N {
                // Slow path, replaces smallest element. Avoid if possible.
                // # Safety
                // All indexes are initialized because the heap is full
                let mut smallest_index = N >> 1;
                for index in N >> 1..N {
                    let node = unsafe { self.data.get_unchecked(index).assume_init_ref() };
                    let smallest =
                        unsafe { self.data.get_unchecked(smallest_index).assume_init_ref() };
                    if comparer(smallest, node, state) {
                        smallest_index = index;
                    }
                }
                let smallest = unsafe { self.data.get_unchecked(smallest_index).assume_init_ref() };
                if comparer(&value, smallest, state) {
                    let replaced = mem::replace(
                        unsafe { self.data.get_unchecked_mut(smallest_index) },
                        MaybeUninit::new(value),
                    );
                    node_index = smallest_index;
                    result = Some(unsafe { replaced.assume_init() });
                } else {
                    return Some(value);
                }
            }
        }

        while node_index != 0 {
            let parent_index = (node_index - 1) >> 1;
            // # Safety
            // These indices are initialized because they are in `0..high`
            let node = unsafe { self.data.get_unchecked(node_index).assume_init_ref() };
            let parent = unsafe { self.data.get_unchecked(parent_index).assume_init_ref() };
            if !comparer(node, parent, state) {
                break;
            }

            #[cfg(all(nightly, feature = "unstable"))]
            unsafe {
                self.data.swap_unchecked(node_index, parent_index);
            }
            #[cfg(not(all(nightly, feature = "unstable")))]
            self.data.swap(node_index, parent_index);

            node_index = parent_index;
        }

        result
    }

    /// Removes and returns the highest priority element, calling `comparer` with `state` to determine ordering.
    ///
    /// `comparer` should return true if its first argument is strictly higher priority than the second.
    /// It is technically permitted to return true when given elements of equal priority,
    /// although it is recommended to return false in those cases to avoid swaps for performance reasons.
    /// A possible use case for returning true when equal is to have newly added elements take priority over older ones.
    ///
    /// Use `state` to pass in another datastructure in order to sort keys by associated values.
    ///
    /// The same comparer should always be used for `push` and `pop`, and `state` should be stable.
    ///
    /// If `comparer` judges that a particular element is higher priority than another one,
    /// it is expected that that remains true for as long as those elements are in this heap.
    ///
    /// # Returns
    ///
    /// `None` if there are no elements in the heap.
    ///
    /// `Some(elem)` if there was an element. `elem` is higher priority than all remaining elements.
    ///
    /// # Time Complexity
    ///
    /// Average time complexity O(1) and worst case O(log N)
    ///
    /// # Examples
    ///
    /// ```
    /// # use fixed_heap::*;
    /// let mut heap: FixedHeap<i32, 16> = FixedHeap::new();
    /// let comparer = |a: &i32, b: &i32, _: &()| a > b;
    /// heap.push(1, &comparer, &());
    /// assert_eq!(heap.pop(&comparer, &()), Some(1));
    /// ```
    ///
    /// With keys into another struct:
    /// ```
    /// # use fixed_heap::*;
    /// let mut heap: FixedHeap<usize, 16> = FixedHeap::new();
    /// let comparer = |a: &usize, b: &usize, state: &[i32; 4]| state[*a] > state[*b];
    /// let state = [1, 3, 1, 2];
    /// heap.push(1, &comparer, &state);
    /// assert_eq!(heap.pop(&comparer, &state), Some(1));
    /// ```
    pub fn pop<S, F>(&mut self, comparer: &F, state: &S) -> Option<T>
    where
        F: Fn(&T, &T, &S) -> bool,
    {
        self.pop_at(0, comparer, state)
    }

    /// Removes and returns the element at index,
    /// preserving the heap property by calling `comparer` with `state` to determine ordering.
    ///
    /// `comparer` should return true if its first argument is strictly higher priority than the second.
    /// It is technically permitted to return true when given elements of equal priority,
    /// although it is recommended to return false in those cases to avoid swaps for performance reasons.
    /// A possible use case for returning true when equal is to have newly added elements take priority over older ones.
    ///
    /// Use `state` to pass in another datastructure in order to sort keys by associated values.
    ///
    /// The same comparer should always be used for `push` and `pop`, and `state` should be stable.
    ///
    /// If `comparer` judges that a particular element is higher priority than another one,
    /// it is expected that that remains true for as long as those elements are in this heap.
    ///
    /// # Returns
    ///
    /// `None` if there is no element at `index`.
    ///
    /// `Some(elem)` if there was an element.
    ///
    /// # Time Complexity
    ///
    /// Average time complexity O(1) and worst case O(log N)
    ///
    /// # Examples
    ///
    /// ```
    /// # use fixed_heap::*;
    /// let mut heap: FixedHeap<i32, 16> = FixedHeap::new();
    /// let comparer = |a: &i32, b: &i32, _: &()| a > b;
    /// heap.push(1, &comparer, &());
    /// heap.push(2, &comparer, &());
    /// assert_eq!(heap.pop_at(1, &comparer, &()), Some(1));
    /// ```
    ///
    /// With keys into another struct:
    /// ```
    /// # use fixed_heap::*;
    /// let mut heap: FixedHeap<usize, 16> = FixedHeap::new();
    /// let comparer = |a: &usize, b: &usize, state: &[i32; 4]| state[*a] > state[*b];
    /// let state = [1, 3, 1, 2];
    /// heap.push(1, &comparer, &state);
    /// heap.push(2, &comparer, &state);
    /// assert_eq!(heap.pop_at(1, &comparer, &state), Some(2));
    /// ```
    pub fn pop_at<S, F>(&mut self, index: usize, comparer: &F, state: &S) -> Option<T>
    where
        F: Fn(&T, &T, &S) -> bool,
    {
        if let Some(removed_node) = self.swap_remove(index) {
            let mut node_index: usize = index;
            loop {
                let lchild_index = (node_index << 1) + 1;
                let rchild_index = (node_index << 1) + 2;
                // # Safety
                // These indices are initialized because they are in `0..high`
                let node = unsafe { self.data.get_unchecked(node_index).assume_init_ref() };
                // Determine which child to sift upwards by comparing
                let swap = if rchild_index < self.high {
                    let lchild = unsafe { self.data.get_unchecked(lchild_index).assume_init_ref() };
                    let rchild = unsafe { self.data.get_unchecked(rchild_index).assume_init_ref() };
                    match comparer(lchild, rchild, state) {
                        true => (comparer(lchild, node, state), lchild_index),
                        false => (comparer(rchild, node, state), rchild_index),
                    }
                } else if lchild_index < self.high {
                    let lchild = unsafe { self.data.get_unchecked(lchild_index).assume_init_ref() };
                    (comparer(lchild, node, state), lchild_index)
                } else {
                    (false, 0)
                };
                // Sift upwards if the `compared_index` is higher priority
                if let (true, compared_index) = swap {
                    #[cfg(all(nightly, feature = "unstable"))]
                    unsafe {
                        self.data.swap_unchecked(node_index, compared_index);
                    }
                    #[cfg(not(all(nightly, feature = "unstable")))]
                    self.data.swap(node_index, compared_index);

                    node_index = compared_index;
                } else {
                    break;
                }
            }

            Some(removed_node)
        } else {
            None
        }
    }

    /// Provides immutable access to the backing array of the heap.
    #[inline(always)]
    pub fn as_slice(&self) -> &[T] {
        unsafe { std::slice::from_raw_parts(self.data.as_ptr() as *const T, self.high) }
    }

    /// Provides mutable access to the backing array of the heap.
    /// Caution: you must preserve the heap property of the structure
    #[inline(always)]
    pub fn as_slice_mut(&mut self) -> &mut [T] {
        unsafe { std::slice::from_raw_parts_mut(self.data.as_mut_ptr() as *mut T, self.high) }
    }

    /// Provides mutable iteration of the heap's elements.
    /// NOTE: The elements are NOT in the order they'd be popped in!
    #[inline(always)]
    pub fn iter(&self) -> Iter<T> {
        self.as_slice().iter()
    }

    /// Provides mutable iteration of the heap's elements.
    /// NOTE: The elements are NOT in the order they'd be popped in!
    /// Caution: you must preserve the heap property of the structure
    #[inline(always)]
    pub fn iter_mut(&mut self) -> IterMut<T> {
        self.as_slice_mut().iter_mut()
    }

    /// Returns the number of elements.
    ///
    /// # Examples
    ///
    /// ```
    /// # use fixed_heap::*;
    /// let mut heap: FixedHeap<i32, 16> = FixedHeap::new();
    /// let comparer = |a: &i32, b: &i32, _: &()| a > b;
    /// heap.push(1, &comparer, &());
    /// assert_eq!(heap.len(), 1);
    /// ```
    #[inline(always)]
    pub fn len(&self) -> usize {
        self.high
    }

    /// Returns true if there are no elements.
    ///
    /// # Examples
    ///
    /// ```
    /// # use fixed_heap::*;
    /// let mut heap: FixedHeap<i32, 16> = FixedHeap::new();
    /// let comparer = |a: &i32, b: &i32, _: &()| a > b;
    /// assert!(heap.is_empty());
    /// heap.push(1, &comparer, &());
    /// assert!(!heap.is_empty());
    /// ```
    #[inline(always)]
    pub fn is_empty(&self) -> bool {
        self.high == 0
    }

    /// Returns true if there is no free space left.
    ///
    /// # Examples
    ///
    /// ```
    /// # use fixed_heap::*;
    /// let mut heap: FixedHeap<i32, 1> = FixedHeap::new();
    /// let comparer = |a: &i32, b: &i32, _: &()| a > b;
    /// assert!(!heap.is_full());
    /// heap.push(1, &comparer, &());
    /// assert!(heap.is_full());
    /// ```
    #[inline(always)]
    pub fn is_full(&self) -> bool {
        self.high == N
    }
}

unsafe impl<T: Sync, const N: usize> Sync for FixedHeap<T, N> {}
unsafe impl<T: Send, const N: usize> Send for FixedHeap<T, N> {}

impl<T, const N: usize> Drop for FixedHeap<T, N> {
    #[inline(always)]
    fn drop(&mut self) {
        for i in 0..self.high {
            unsafe { self.data.get_unchecked_mut(i).assume_init_drop() };
        }
    }
}

impl<T, const N: usize> Debug for FixedHeap<T, N>
where
    T: Debug,
{
    fn fmt(&self, f: &mut Formatter) -> Result {
        f.debug_struct("FixedHeap")
            .field("high", &self.high)
            .field("data", &self.as_slice())
            .finish()
    }
}

impl<T, const N: usize> Default for FixedHeap<T, N> {
    fn default() -> Self {
        Self::new()
    }
}

impl<'a, T: 'a, const N: usize> IntoIterator for &'a FixedHeap<T, N> {
    type Item = &'a T;
    type IntoIter = Iter<'a, T>;

    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

impl<'a, T: 'a, const N: usize> IntoIterator for &'a mut FixedHeap<T, N> {
    type Item = &'a mut T;
    type IntoIter = IterMut<'a, T>;

    fn into_iter(self) -> Self::IntoIter {
        self.iter_mut()
    }
}

impl<T, const N: usize> IntoIterator for FixedHeap<T, N> {
    type Item = T;
    type IntoIter = IntoIter<T, N>;

    fn into_iter(self) -> Self::IntoIter {
        IntoIter {
            next: 0,
            heap: ManuallyDrop::new(self),
        }
    }
}

/// Ownership transferring iterator
#[derive(Debug)]
pub struct IntoIter<T, const N: usize> {
    next: usize,
    heap: ManuallyDrop<FixedHeap<T, N>>,
}

impl<T, const N: usize> Iterator for IntoIter<T, N> {
    type Item = T;

    fn next(&mut self) -> Option<T> {
        if self.next < self.heap.high {
            let index = self.next;
            self.next += 1;
            // # Safety
            // We can hand over this value without modifying the array,
            // because we manually drop only the remainder of uniterated elements
            Some(unsafe { self.heap.data.get_unchecked(index).assume_init_read() })
        } else {
            None
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let size = self.heap.high - self.next;
        (size, Some(size))
    }
}

impl<T, const N: usize> Drop for IntoIter<T, N> {
    #[inline(always)]
    fn drop(&mut self) {
        for i in self.next..self.heap.high {
            // # Safety
            // We manually drop only the remainder of uniterated elements
            unsafe { self.heap.data[i].assume_init_drop() };
        }
    }
}

impl<T, const N: usize> ExactSizeIterator for IntoIter<T, N> {}
impl<T, const N: usize> FusedIterator for IntoIter<T, N> {}

#[cfg(test)]
mod test {
    use crate::*;
    use rand::{rngs::ThreadRng, Rng};
    use std::cell::RefCell;

    #[test]
    fn test_push_peek_pop() {
        let mut heap: FixedHeap<i32, 16> = FixedHeap::new();
        let comparer = |a: &i32, b: &i32, _: &()| a > b;
        assert_eq!(None, heap.peek());
        assert_eq!(heap.push(1, &comparer, &()), None);
        assert_eq!(Some(&1), heap.peek());
        assert_eq!(heap.push(3, &comparer, &()), None);
        assert_eq!(Some(&3), heap.peek());
        assert_eq!(heap.push(2, &comparer, &()), None);
        assert_eq!(Some(&3), heap.peek());

        assert_eq!(Some(3), heap.pop(&comparer, &()));
        assert_eq!(Some(&2), heap.peek());
        assert_eq!(Some(2), heap.pop(&comparer, &()));
        assert_eq!(Some(&1), heap.peek());
        assert_eq!(Some(1), heap.pop(&comparer, &()));
        assert_eq!(None, heap.peek());
        assert_eq!(None, heap.pop(&comparer, &()));
    }

    #[test]
    fn test_add_last_swap_remove() {
        let mut heap: FixedHeap<i32, 4> = FixedHeap::new();
        assert_eq!(heap.add_last(1), None);
        assert_eq!(heap.add_last(2), None);
        assert_eq!(heap.add_last(4), None);
        assert_eq!(heap.add_last(3), None);
        assert_eq!(heap.add_last(5), Some(5));

        assert_eq!(Some(1), heap.swap_remove(0));
        assert_eq!(Some(3), heap.swap_remove(0));
        assert_eq!(Some(4), heap.swap_remove(0));
        assert_eq!(Some(2), heap.swap_remove(0));
        assert_eq!(None, heap.swap_remove(0));
    }

    #[test]
    fn test_push_full() {
        let mut heap: FixedHeap<i32, 4> = FixedHeap::new();
        let comparer = |a: &i32, b: &i32, _: &()| a > b;
        assert_eq!(heap.push(1, &comparer, &()), None);
        assert_eq!(heap.push(2, &comparer, &()), None);
        assert_eq!(heap.push(4, &comparer, &()), None);
        assert_eq!(heap.push(3, &comparer, &()), None);
        assert_eq!(heap.push(5, &comparer, &()), Some(1));

        assert_eq!(Some(5), heap.pop(&comparer, &()));
        assert_eq!(Some(4), heap.pop(&comparer, &()));
        assert_eq!(Some(3), heap.pop(&comparer, &()));
        assert_eq!(Some(2), heap.pop(&comparer, &()));
        assert_eq!(None, heap.pop(&comparer, &()));
    }

    #[test]
    fn test_push_pop_equal() {
        let mut heap: FixedHeap<i32, 4> = FixedHeap::new();
        let comparer = |a: &i32, b: &i32, _: &()| a > b;
        assert_eq!(heap.push(7, &comparer, &()), None);
        assert_eq!(heap.push(7, &comparer, &()), None);
        assert_eq!(heap.push(7, &comparer, &()), None);

        assert_eq!(Some(7), heap.pop(&comparer, &()));
        assert_eq!(Some(7), heap.pop(&comparer, &()));
        assert_eq!(Some(7), heap.pop(&comparer, &()));
        assert_eq!(None, heap.pop(&comparer, &()));
    }

    #[test]
    fn test_keys() {
        let mut heap: FixedHeap<usize, 4> = FixedHeap::new();
        fn comparer(a: &usize, b: &usize, state: &[i32; 4]) -> bool {
            state[*a] > state[*b]
        }
        let state = [1, 3, 1, 2];
        assert_eq!(heap.push(0, &comparer, &state), None);
        assert_eq!(heap.push(1, &comparer, &state), None);
        assert_eq!(heap.push(3, &comparer, &state), None);

        assert_eq!(Some(1), heap.pop(&comparer, &state));
        assert_eq!(Some(3), heap.pop(&comparer, &state));
        assert_eq!(Some(0), heap.pop(&comparer, &state));
        assert_eq!(None, heap.pop(&comparer, &state));
    }

    #[test]
    fn test_as_slice() {
        let mut heap: FixedHeap<i32, 16> = FixedHeap::new();
        let comparer = |a: &i32, b: &i32, _: &()| a > b;
        assert_eq!(heap.push(7, &comparer, &()), None);
        assert_eq!(heap.push(9, &comparer, &()), None);
        assert_eq!(heap.push(2, &comparer, &()), None);
        assert_eq!(heap.push(5, &comparer, &()), None);
        assert_eq!(heap.push(8, &comparer, &()), None);
        assert_eq!(heap.push(8, &comparer, &()), None);
        assert_eq!(heap.push(3, &comparer, &()), None);

        let slice = heap.as_slice();
        assert_eq!(7, slice.len());
        assert_eq!(9, slice[0]);
        assert_eq!(8, slice[1]);
        assert_eq!(8, slice[2]);
        assert_eq!(5, slice[3]);
        assert_eq!(7, slice[4]);
        assert_eq!(2, slice[5]);
        assert_eq!(3, slice[6]);
    }

    #[test]
    fn test_debug() {
        let mut heap: FixedHeap<i32, 16> = FixedHeap::new();
        let comparer = |a: &i32, b: &i32, _: &()| a > b;
        assert_eq!(heap.push(7, &comparer, &()), None);
        assert_eq!(heap.push(9, &comparer, &()), None);
        assert_eq!(heap.push(2, &comparer, &()), None);
        assert_eq!(heap.push(5, &comparer, &()), None);
        assert_eq!(heap.push(8, &comparer, &()), None);
        assert_eq!(heap.push(8, &comparer, &()), None);
        assert_eq!(heap.push(3, &comparer, &()), None);
        assert_eq!(
            format!("{:?}", heap),
            "FixedHeap { high: 7, data: [9, 8, 8, 5, 7, 2, 3] }"
        );
    }

    #[test]
    fn test_drop() {
        let drops = RefCell::new(0usize);
        {
            let comparer = |_: &_, _: &_, _: &()| false;
            let mut list: FixedHeap<DropCounted, 16> = FixedHeap::new();
            for _ in 0..11 {
                list.push(DropCounted(&drops), &comparer, &());
            }
            assert_eq!(*drops.borrow(), 0);

            // Pop and drop a few
            for _ in 0..4 {
                list.pop(&comparer, &());
            }
            assert_eq!(*drops.borrow(), 4);

            // Move it into a consuming iterator
            let mut iter = list.into_iter();
            assert_eq!(*drops.borrow(), 4);

            // Consume the first item
            iter.next();
            assert_eq!(*drops.borrow(), 5);

            // Let the rest drop
        }
        assert_eq!(*drops.borrow(), 11);
    }

    #[derive(Clone)]
    struct DropCounted<'a>(&'a RefCell<usize>);

    impl<'a> Drop for DropCounted<'a> {
        fn drop(&mut self) {
            *self.0.borrow_mut() += 1;
        }
    }

    #[test]
    fn test_fuzz_gt_partial() {
        let gt = |a: &i8, b: &i8, _: &()| a > b;
        fuzz::<_, 8, 7>(10, &gt);
        fuzz::<_, 9, 7>(10, &gt);
        fuzz::<_, 10, 7>(10, &gt);
        fuzz::<_, 11, 7>(10, &gt);
        fuzz::<_, 12, 7>(10, &gt);
        fuzz::<_, 13, 7>(10, &gt);
        fuzz::<_, 14, 7>(10, &gt);
        fuzz::<_, 15, 7>(10, &gt);
        fuzz::<_, 16, 7>(10, &gt);
    }

    #[test]
    fn test_fuzz_gt_full() {
        let gt = |a: &i8, b: &i8, _: &()| a > b;
        fuzz::<_, 0, 4>(10, &gt);
        fuzz::<_, 1, 4>(10, &gt);
        fuzz::<_, 2, 4>(10, &gt);
        fuzz::<_, 3, 4>(10, &gt);
        fuzz::<_, 8, 16>(10, &gt);
        fuzz::<_, 9, 16>(10, &gt);
        fuzz::<_, 10, 16>(10, &gt);
        fuzz::<_, 11, 16>(10, &gt);
        fuzz::<_, 12, 16>(10, &gt);
        fuzz::<_, 13, 16>(10, &gt);
        fuzz::<_, 14, 16>(10, &gt);
        fuzz::<_, 15, 16>(10, &gt);
    }

    #[test]
    fn test_fuzz_ge_partial() {
        let ge = |a: &i8, b: &i8, _: &()| a >= b;
        fuzz::<_, 8, 7>(10, &ge);
        fuzz::<_, 9, 7>(10, &ge);
        fuzz::<_, 10, 7>(10, &ge);
        fuzz::<_, 11, 7>(10, &ge);
        fuzz::<_, 12, 7>(10, &ge);
        fuzz::<_, 13, 7>(10, &ge);
        fuzz::<_, 14, 7>(10, &ge);
        fuzz::<_, 15, 7>(10, &ge);
        fuzz::<_, 16, 7>(10, &ge);
    }

    #[test]
    fn test_fuzz_ge_full() {
        let ge = |a: &i8, b: &i8, _: &()| a >= b;
        fuzz::<_, 0, 4>(10, &ge);
        fuzz::<_, 1, 4>(10, &ge);
        fuzz::<_, 2, 4>(10, &ge);
        fuzz::<_, 3, 4>(10, &ge);
        fuzz::<_, 8, 16>(10, &ge);
        fuzz::<_, 9, 16>(10, &ge);
        fuzz::<_, 10, 16>(10, &ge);
        fuzz::<_, 11, 16>(10, &ge);
        fuzz::<_, 12, 16>(10, &ge);
        fuzz::<_, 13, 16>(10, &ge);
        fuzz::<_, 14, 16>(10, &ge);
        fuzz::<_, 15, 16>(10, &ge);
    }

    fn fuzz<F, const N: usize, const M: usize>(iters: usize, comparer: &F)
    where
        F: Fn(&i8, &i8, &()) -> bool,
    {
        for _ in 0..iters {
            let mut heap: FixedHeap<i8, N> = FixedHeap::new();
            let mut array = [0i8; M];
            rand::thread_rng().fill(&mut array[..]);
            for element in array {
                heap.push(element, comparer, &());
            }
            array.sort_by(|a, b| b.cmp(a));
            for &element in array.iter().take(N) {
                assert_eq!(Some(element), heap.pop(comparer, &()));
            }
            assert_eq!(None, heap.pop(comparer, &()));
        }
    }

    #[test]
    fn test_fuzz_true() {
        let comparer = |_: &usize, _: &usize, _: &()| true;
        fuzz_state::<_, _, 0, 4>(&comparer, &());
        fuzz_state::<_, _, 1, 4>(&comparer, &());
        fuzz_state::<_, _, 2, 4>(&comparer, &());
        fuzz_state::<_, _, 3, 4>(&comparer, &());
        fuzz_state::<_, _, 8, 16>(&comparer, &());
        fuzz_state::<_, _, 9, 16>(&comparer, &());
        fuzz_state::<_, _, 10, 16>(&comparer, &());
        fuzz_state::<_, _, 11, 16>(&comparer, &());
        fuzz_state::<_, _, 12, 16>(&comparer, &());
        fuzz_state::<_, _, 13, 16>(&comparer, &());
        fuzz_state::<_, _, 14, 16>(&comparer, &());
        fuzz_state::<_, _, 15, 16>(&comparer, &());
        fuzz_state::<_, _, 16, 16>(&comparer, &());
        fuzz_state::<_, _, 17, 16>(&comparer, &());
    }

    #[test]
    fn test_fuzz_false() {
        let comparer = |_: &usize, _: &usize, _: &()| false;
        fuzz_state::<_, _, 0, 4>(&comparer, &());
        fuzz_state::<_, _, 1, 4>(&comparer, &());
        fuzz_state::<_, _, 2, 4>(&comparer, &());
        fuzz_state::<_, _, 3, 4>(&comparer, &());
        fuzz_state::<_, _, 8, 16>(&comparer, &());
        fuzz_state::<_, _, 9, 16>(&comparer, &());
        fuzz_state::<_, _, 10, 16>(&comparer, &());
        fuzz_state::<_, _, 11, 16>(&comparer, &());
        fuzz_state::<_, _, 12, 16>(&comparer, &());
        fuzz_state::<_, _, 13, 16>(&comparer, &());
        fuzz_state::<_, _, 14, 16>(&comparer, &());
        fuzz_state::<_, _, 15, 16>(&comparer, &());
        fuzz_state::<_, _, 16, 16>(&comparer, &());
        fuzz_state::<_, _, 17, 16>(&comparer, &());
    }

    #[test]
    fn test_fuzz() {
        let rng = RefCell::new(rand::thread_rng());
        fn comparer(_: &usize, _: &usize, rng: &RefCell<ThreadRng>) -> bool {
            rng.borrow_mut().gen()
        }
        fuzz_state::<_, _, 0, 4>(&comparer, &rng);
        fuzz_state::<_, _, 1, 4>(&comparer, &rng);
        fuzz_state::<_, _, 2, 4>(&comparer, &rng);
        fuzz_state::<_, _, 3, 4>(&comparer, &rng);
        fuzz_state::<_, _, 8, 16>(&comparer, &rng);
        fuzz_state::<_, _, 9, 16>(&comparer, &rng);
        fuzz_state::<_, _, 10, 16>(&comparer, &rng);
        fuzz_state::<_, _, 11, 16>(&comparer, &rng);
        fuzz_state::<_, _, 12, 16>(&comparer, &rng);
        fuzz_state::<_, _, 13, 16>(&comparer, &rng);
        fuzz_state::<_, _, 14, 16>(&comparer, &rng);
        fuzz_state::<_, _, 15, 16>(&comparer, &rng);
        fuzz_state::<_, _, 16, 16>(&comparer, &rng);
        fuzz_state::<_, _, 17, 16>(&comparer, &rng);
    }

    fn fuzz_state<S, F, const N: usize, const M: usize>(
        comparer: &F,
        state: &S,
    ) -> [Option<usize>; M]
    where
        F: Fn(&usize, &usize, &S) -> bool,
    {
        let mut heap: FixedHeap<usize, N> = FixedHeap::new();
        for i in 0..M {
            heap.push(i, comparer, state);
        }
        let mut result = [None; M];
        for item in result.iter_mut() {
            *item = heap.pop(comparer, state);
        }
        if N < M {
            for &item in result.iter().skip(N) {
                assert_eq!(None, item);
            }
        } else {
            assert_eq!(None, heap.pop(comparer, state));
        }
        result
    }
}
