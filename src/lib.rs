extern crate libc;
use libc::{c_char, c_double, c_int, c_void};
use std::ptr;
use std::slice;

macro_rules! gen_setter {
    ($setr:ident, $r:ident, $t: ty) => {
        pub fn $setr(&mut self, $r: $t) -> &mut CubaIntegrator<T> {
            self.$r = $r;
            self
        }
    };
}

#[link(name = "cuba")]
extern "C" {
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
) -> c_int;

type Integrand<T> = fn(x: &[f64], f: &mut [f64], user_data: &mut T) -> i32;

#[repr(C)]
struct CubaUserData<T> {
    integrand: Integrand<T>,
    user_data: T,
}

#[derive(Debug)]
pub struct CubaResult {
    pub neval: i32,
    pub fail: i32,
    pub result: Vec<f64>,
    pub error: Vec<f64>,
    pub prob: Vec<f64>,
}

pub struct CubaIntegrator<T> {
    integrand: Integrand<T>,
    mineval: i32,
    maxeval: i32,
    nstart: i32,
    nincrease: i32,
    seed: i32,
    epsrel: f64,
    epsabs: f64,
    batch: i32,
}

impl<T> CubaIntegrator<T> {
    pub fn new(integrand: Integrand<T>) -> CubaIntegrator<T> {
        CubaIntegrator {
            integrand,
            mineval: 0,
            maxeval: 50000,
            nstart: 1000,
            nincrease: 500,
            seed: 0,
            epsrel: 0.001,
            epsabs: 1e-12,
            batch: 1000,
        }
    }

    gen_setter!(set_mineval, mineval, i32);
    gen_setter!(set_maxeval, maxeval, i32);
    gen_setter!(set_nstart, nstart, i32);
    gen_setter!(set_nincrease, nincrease, i32);
    gen_setter!(set_seed, seed, i32);
    gen_setter!(set_epsrel, epsrel, f64);
    gen_setter!(set_epsabs, epsabs, f64);
    gen_setter!(set_batch, batch, i32);

    extern "C" fn c_integrand(
        ndim: *const c_int,
        x: *const c_double,
        ncomp: *const c_int,
        f: *mut c_double,
        userdata: *mut c_void,
    ) -> c_int {
        unsafe {
            let k: &mut CubaUserData<T> = &mut *(userdata as *mut _);

            // call the safe integrand
            let res: i32 = (k.integrand)(
                &slice::from_raw_parts(x, *ndim as usize),
                &mut slice::from_raw_parts_mut(f, *ncomp as usize),
                &mut k.user_data,
            );
            res as c_int
        }
    }

    pub fn vegas(&mut self, ndim: usize, ncomp: usize, user_data: T) -> CubaResult {
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
                0,                                      // flags
                self.seed,                              // seed
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
