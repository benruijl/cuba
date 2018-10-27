extern crate cuba;
use cuba::{CubaIntegrator, CubaVerbosity};

#[derive(Debug)]
struct TestUserData {
    f1: f64,
    f2: f64,
}

#[inline(always)]
fn test_integrand(x: &[f64], f: &mut [f64], user_data: &mut TestUserData, _nvec: usize, _core: i32) -> i32 {
    f[0] = (x[0] * x[1]).sin() * user_data.f1;
    f[1] = (x[1] * x[1]).cos() * user_data.f2;
    0
}

fn main() {
    let mut ci = CubaIntegrator::new(test_integrand);
    ci.set_mineval(10).set_maxeval(10000);

    let r = ci.vegas(
        2,
        2,
        CubaVerbosity::Progress,
        TestUserData { f1: 5., f2: 7. },
    );
    println!("{:#?}", r);
}
