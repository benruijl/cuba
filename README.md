# Rust Cuba interface

This library provides safe access to the Cuba integration library.

## Example

Below we show an example of an integration of a test function
with user data (this can be of any type):

```rust
extern crate cuba;
use cuba::{CubaIntegrator, CubaVerbosity};

#[derive(Debug)]
struct UserData {
    f1: f64,
    f2: f64,
}

#[inline(always)]
fn integrand(
    x: &[f64],
    f: &mut [f64],
    user_data: &mut UserData,
    nvec: usize,
    _core: i32,
    _weight: &[f64],
    _iter: usize,
) -> Result<(), &'static str> {
    for i in 0..nvec {
        f[i * 2] = (x[i * 2] * x[i * 2]).sin() * user_data.f1;
        f[i * 2 + 1] = (x[i * 2 + 1] * x[i * 2 + 1]).cos() * user_data.f2;
    }

    Ok(())
}

fn main() {
    let mut ci = CubaIntegrator::new();
    ci.set_mineval(10)
        .set_maxeval(10000000)
        .set_epsrel(0.0001)
        .set_seed(0) // use quasi-random numbers
        .set_cores(2, 1000);

    let data = UserData { f1: 5., f2: 7. };
    let r = ci.vegas(2, 2, 4, CubaVerbosity::Progress, 0, integrand, data);

    println!("{:#?}", r);
}
```
