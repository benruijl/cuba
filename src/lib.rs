//! Rust binding for the Cuba integrator.
//!
//! # Usage
//! Create a `CubaIntegrator` and supply it with a function of the form
//!
//! ```
//! fn test_integrand(x: &[f64], f: &mut [f64], user_data: &mut T) -> i32 {
//! }
//! ```
//! where `T` can be any type. If you don't want to provide user data,
//! simply make `T` a `usize` and provide any number.
//!
//! # Example
//!
//! ```
//! extern crate cuba;
//! use cuba::{CubaIntegrator, CubaVerbosity};
//!
//! #[derive(Debug)]
//! struct TestUserData {
//!     f1: f64,
//!     f2: f64,
//! }
//!
//! #[inline(always)]
//! fn test_integrand(x: &[f64], f: &mut [f64], user_data: &mut TestUserData) -> i32 {
//!     f[0] = (x[0] * x[1]).sin() * user_data.f1;
//!     f[1] = (x[1] * x[1]).cos() * user_data.f2;
//!     0
//! }
//!
//! fn main() {
//!     let mut ci = CubaIntegrator::new(test_integrand);
//!     ci.set_mineval(10).set_maxeval(10000);
//!
//!     let r = ci.vegas(
//!         2,
//!         2,
//!         CubaVerbosity::Progress,
//!         TestUserData { f1: 5., f2: 7. },
//!     );
//!     println!("{:#?}", r);
//! }
//! ```
extern crate libc;
use libc::{c_char, c_double, c_int, c_void};
use std::ptr;
use std::slice;

/// Logging level.
///
/// `Silent` does not print any output, `Progress` prints 'reasonable' information on the
/// progress of the integration, `Input` also echoes the input parameters,
/// and `Subregions` further prints the subregion results.
pub enum CubaVerbosity {
    Silent = 0,
    Progress = 1,
    Input = 2,
    Subregions = 3,
}

macro_rules! gen_setter {
    ($setr:ident, $r:ident, $t: ty) => {
        pub fn $setr(&mut self, $r: $t) -> &mut Self {
            self.$r = $r;
            self
        }
    };
}

#[link(name = "cuba")]
extern "C" {
    fn cubacores(n: c_int, p: c_int);

    fn Vegas(
        ndim: c_int,
        ncomp: c_int,
        integrand: Option<IntegrandC>,
        userdata: *mut c_void,
        nvec: c_int,
        epsrel: c_double,
        epsabs: c_double,
        flags: c_int,
        seed: c_int,
        mineval: c_int,
        maxeval: c_int,
        nstart: c_int,
        nincrease: c_int,
        nbatch: c_int,
        gridno: c_int,
        statefile: *const c_char,
        spin: *mut c_void,
        neval: *mut c_int,
        fail: *mut c_int,
        integral: *mut c_double,
        error: *mut c_double,
        prob: *mut c_double,
    );
}

type IntegrandC = extern "C" fn(
    ndim: *const c_int,
    x: *const c_double,
    ncomp: *const c_int,
    f: *mut c_double,
    userdata: *mut c_void,
    nvec: *const c_int,
    core: *const c_int,
) -> c_int;

/// Integrand evaluation function.
///
/// The dimensions of random input variables `x` and output `f`
/// are provided to the integration routine as dimension and components respectively.
/// `T` can be any type. If you don't want to provide user data,
/// simply make `T` a `usize` and provide any number.
///
/// `core` specifies the current core that is being used. This can be used to write to
/// the user data in a thread-safe way.
///
/// The return value is ignored, unless it is -999. Then the integration will be aborted.
pub type Integrand<T> =
    fn(x: &[f64], f: &mut [f64], user_data: &mut T, nvec: usize, core: i32) -> i32;

#[repr(C)]
struct CubaUserData<T> {
    integrand: Integrand<T>,
    user_data: T,
}

/// The result of an integration with Cuba.
#[derive(Debug)]
pub struct CubaResult {
    pub neval: i32,
    pub fail: i32,
    pub result: Vec<f64>,
    pub error: Vec<f64>,
    pub prob: Vec<f64>,
}

/// A Cuba integrator. It should be created with an integrand function.
pub struct CubaIntegrator<T> {
    integrand: Integrand<T>,
    mineval: i32,
    maxeval: i32,
    nstart: i32,
    nincrease: i32,
    epsrel: f64,
    epsabs: f64,
    batch: i32,
    pseudo_random: bool,
}

impl<T> CubaIntegrator<T> {
    /// Create a new Cuba integrator. Use the `set_` functions
    /// to set integration parameters.
    pub fn new(integrand: Integrand<T>) -> CubaIntegrator<T> {
        CubaIntegrator {
            integrand,
            mineval: 0,
            maxeval: 50000,
            nstart: 1000,
            nincrease: 500,
            epsrel: 0.001,
            epsabs: 1e-12,
            batch: 1000,
            pseudo_random: false,
        }
    }

    /// Set the number of cores and the maximum number of points per core.
    /// The default is the number of idle cores for `cores` and
    /// 1000 for `max_points_per_core`.
    pub fn set_cores(&mut self, cores: usize, max_points_per_core: usize) -> &mut Self {
        unsafe {
            cubacores(cores as c_int, max_points_per_core as c_int);
        }
        self
    }

    gen_setter!(set_mineval, mineval, i32);
    gen_setter!(set_maxeval, maxeval, i32);
    gen_setter!(set_nstart, nstart, i32);
    gen_setter!(set_nincrease, nincrease, i32);
    gen_setter!(set_epsrel, epsrel, f64);
    gen_setter!(set_epsabs, epsabs, f64);
    gen_setter!(set_batch, batch, i32);
    gen_setter!(set_pseudo_random, pseudo_random, bool);

    extern "C" fn c_integrand(
        ndim: *const c_int,
        x: *const c_double,
        ncomp: *const c_int,
        f: *mut c_double,
        userdata: *mut c_void,
        nvec: *const c_int,
        core: *const c_int,
    ) -> c_int {
        unsafe {
            let k: &mut CubaUserData<T> = &mut *(userdata as *mut _);

            // call the safe integrand
            let res: i32 = (k.integrand)(
                &slice::from_raw_parts(x, *ndim as usize),
                &mut slice::from_raw_parts_mut(f, *ncomp as usize),
                &mut k.user_data,
                *nvec as usize,
                *core as i32,
            );
            res as c_int
        }
    }

    /// Integrate using the Vegas integrator.
    ///
    /// * `ndim` - dimension of the input
    /// * `ncomp` - dimension (components) of the output
    /// * `verbosity` - Verbosity level
    /// * `user_data` - User data used by the integrand function
    pub fn vegas(
        &mut self,
        ndim: usize,
        ncomp: usize,
        verbosity: CubaVerbosity,
        user_data: T,
    ) -> CubaResult {
        let mut out = CubaResult {
            neval: 0,
            fail: 0,
            result: vec![0.; ncomp],
            error: vec![0.; ncomp],
            prob: vec![0.; ncomp],
        };

        // pass the safe integrand and the user data
        let mut x = CubaUserData {
            integrand: self.integrand,
            user_data: user_data,
        };

        let user_data_ptr = &mut x as *mut _ as *mut c_void;

        unsafe {
            Vegas(
                ndim as c_int,                          // ndim
                ncomp as c_int,                         // ncomp
                Some(CubaIntegrator::<T>::c_integrand), // integrand
                user_data_ptr,                          // user data
                1,                                      // nvec
                self.epsrel,                            // epsrel
                self.epsabs,                            // epsabs
                verbosity as c_int,                     // flags
                self.pseudo_random as c_int,            // seed
                self.mineval,                           // mineval
                self.maxeval,                           // maxeval
                self.nstart,                            // nstart
                self.nincrease,                         // nincrease
                self.batch,                             // batch
                0,                                      // grid no
                ptr::null_mut(),                        // statefile
                ptr::null_mut(),                        // spin
                &mut out.neval,
                &mut out.fail,
                &mut out.result[0],
                &mut out.error[0],
                &mut out.prob[0],
            );
        }

        out
    }
}
