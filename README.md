# fixed_heap

A Rust data-structure like `BinaryHeap` except fixed-size, backed by an array,
with manually provided comparison functions which may depend on external
data-structures. This permits having a heap of keys ordered by the values
associated with them in another data-structure. Please consult
[**the documentation**](https://docs.rs/fixed_heap) for more information.

The minimum required Rust version for `fixed_heap` is 1.60.0. To start using
`fixed_heap` add the following to your `Cargo.toml`:

```toml
[dependencies]
fixed_heap = "0.2"
```

For additional performance, enable `unstable` feature on nightly with

```toml
[dependencies]
fixed_heap = { version = "0.2", features = ["unstable"] }
```

## Example

A short example:

```rust
use fixed_heap::FixedHeap;
let mut heap: FixedHeap<i32, 16> = FixedHeap::new();
let comparer = |a: &i32, b: &i32, _: &()| a > b;
heap.pop(&comparer, &());
```

With keys into another struct:
```rust
use fixed_heap::FixedHeap;
let mut heap: FixedHeap<usize, 16> = FixedHeap::new();
let comparer = |a: &usize, b: &usize, state: &[i32; 4]| state[*a] > state[*b];
let state = [1, 3, 1, 2];
heap.push(0, &comparer, &state);
```

## Safety

This crate uses unsafe code for performance.
It has been extensively fuzz tested with miri to ensure it behaves correctly.

## License

`fixed_heap` is dual-licensed under either:

* MIT License ([LICENSE-MIT](LICENSE-MIT) or [http://opensource.org/licenses/MIT](http://opensource.org/licenses/MIT))
* Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE) or [http://www.apache.org/licenses/LICENSE-2.0](http://www.apache.org/licenses/LICENSE-2.0))

at your option.

### Your contributions

Unless you explicitly state otherwise,
any contribution intentionally submitted for inclusion in the work by you,
as defined in the Apache-2.0 license,
shall be dual licensed as above,
without any additional terms or conditions.