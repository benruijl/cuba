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
    _nvec: usize,
    _core: i32,
) -> Result<(), &'static str> {
    f[0] = (x[0] * x[1]).sin() * user_data.f1;
    f[1] = (x[1] * x[1]).cos() * user_data.f2;
    Ok(())
}

fn main() {
    let mut ci = CubaIntegrator::new(integrand);
    ci.set_mineval(10)
        .set_maxeval(10000000)
        .set_epsrel(0.0001)
        .set_cores(2, 1000);

    let data = UserData { f1: 5., f2: 7. };

    let peaks = vec![0.5, 0.5, 0.25, 0.57];
    let r = ci.divonne(2, 2, 1, &peaks, CubaVerbosity::Progress, data);

    println!("{:#?}", r);
}
