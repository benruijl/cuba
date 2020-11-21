//! Rust binding for the Cuba integrator.
//!
//! Cuba (http://www.feynarts.de/cuba/) is written by Thomas Hahn.
//!
//! # Usage
//! Create a `CubaIntegrator` and supply it with a function of the form
//!
//! ```
//! fn integrand(
//!     x: &[f64],
//!     f: &mut [f64],
//!     user_data: &mut UserData,
//!     nvec: usize,
//!     core: i32,
//!     weight: &[f64],
//!     iter: usize,
//! ) -> i32 {
//! }
//! ```
//! where `UserData` can be any type. If you don't want to provide user data,
//! simply make `UserData` a `usize` and provide any number.
//!
//! # Example
//!
//! ```
//! extern crate cuba;
//! use cuba::{CubaIntegrator, CubaVerbosity};
//! 
//! #[derive(Debug)]
//! struct UserData {
//!     f1: f64,
//!     f2: f64,
//! }
//! 
//! #[inline(always)]
//! fn integrand(
//!     x: &[f64],
//!     f: &mut [f64],
//!     user_data: &mut UserData,
//!     nvec: usize,
//!     _core: i32,
//!     _weight: &[f64],
//!     _iter: usize,
//! ) -> Result<(), &'static str> {
//!     for i in 0..nvec {
//!         f[i * 2] = (x[i * 2] * x[i * 2]).sin() * user_data.f1;
//!         f[i * 2 + 1] = (x[i * 2 + 1] * x[i * 2 + 1]).cos() * user_data.f2;
//!     }
//! 
//!     Ok(())
//! }
//! 
//! fn main() {
//!     let mut ci = CubaIntegrator::new();
//!     ci.set_mineval(10)
//!         .set_maxeval(10000000)
//!         .set_epsrel(0.0001)
//!         .set_seed(0) // use quasi-random numbers
//!         .set_cores(2, 1000);
//! 
//!     let data = UserData { f1: 5., f2: 7. };
//!     let r = ci.vegas(2, 2, 4, CubaVerbosity::Progress, 0, integrand, data);
//! 
//!     println!("{:#?}", r);
//! }
//! ```
extern crate libc;
use libc::{c_char, c_double, c_int, c_longlong, c_void};
use std::ffi::CString;
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

    fn llVegas(
        ndim: c_int,
        ncomp: c_int,
        integrand: Option<VegasIntegrandC>,
        userdata: *mut c_void,
        nvec: c_longlong,
        epsrel: c_double,
        epsabs: c_double,
        flags: c_int,
        seed: c_int,
        mineval: c_longlong,
        maxeval: c_longlong,
        nstart: c_longlong,
        nincrease: c_longlong,
        nbatch: c_longlong,
        gridno: c_int,
        statefile: *const c_char,
        spin: *mut c_void,
        neval: *mut c_longlong,
        fail: *mut c_int,
        integral: *mut c_double,
        error: *mut c_double,
        prob: *mut c_double,
    );

    fn llSuave(
        ndim: c_int,
        ncomp: c_int,
        integrand: Option<SuaveIntegrandC>,
        userdata: *mut c_void,
        nvec: c_longlong,
        epsrel: c_double,
        epsabs: c_double,
        flags: c_int,
        seed: c_int,
        mineval: c_longlong,
        maxeval: c_longlong,
        nnew: c_longlong,
        nmin: c_longlong,
        flatness: c_double,
        statefile: *const c_char,
        spin: *mut c_void,
        nregions: *mut c_int,
        neval: *mut c_longlong,
        fail: *mut c_int,
        integral: *mut c_double,
        error: *mut c_double,
        prob: *mut c_double,
    );

    fn llDivonne(
        ndim: c_int,
        ncomp: c_int,
        integrand: Option<DivonneIntegrandC>,
        userdata: *mut c_void,
        nvec: c_longlong,
        epsrel: c_double,
        epsabs: c_double,
        flags: c_int,
        seed: c_int,
        mineval: c_longlong,
        maxeval: c_longlong,
        key1: c_int,
        key2: c_int,
        key3: c_int,
        maxpass: c_int,
        border: c_double,
        maxchisq: c_double,
        mindeviation: c_double,
        ngiven: c_longlong,
        lxdgiven: c_int,
        xgiven: *const c_double,
        nextra: c_longlong,
        peakfinder: Option<PeakfinderC>,
        statefile: *const c_char,
        spin: *mut c_void,
        nregions: *mut c_int,
        neval: *mut c_longlong,
        fail: *mut c_int,
        integral: *mut c_double,
        error: *mut c_double,
        prob: *mut c_double,
    );

    fn llCuhre(
        ndim: c_int,
        ncomp: c_int,
        integrand: Option<CuhreIntegrandC>,
        userdata: *mut c_void,
        nvec: c_longlong,
        epsrel: c_double,
        epsabs: c_double,
        flags: c_int,
        mineval: c_longlong,
        maxeval: c_longlong,
        key: c_int,
        statefile: *const c_char,
        spin: *mut c_void,
        nregions: *mut c_int,
        neval: *mut c_longlong,
        fail: *mut c_int,
        integral: *mut c_double,
        error: *mut c_double,
        prob: *mut c_double,
    );
}

type VegasIntegrandC = extern "C" fn(
    ndim: *const c_int,
    x: *const c_double,
    ncomp: *const c_int,
    f: *mut c_double,
    userdata: *mut c_void,
    nvec: *const c_int,
    core: *const c_int,
    weight: *const c_double,
    iter: *const c_int,
) -> c_int;

type SuaveIntegrandC = extern "C" fn(
    ndim: *const c_int,
    x: *const c_double,
    ncomp: *const c_int,
    f: *mut c_double,
    userdata: *mut c_void,
    nvec: *const c_int,
    core: *const c_int,
    weight: *const c_double,
    iter: *const c_int,
) -> c_int;

type CuhreIntegrandC = extern "C" fn(
    ndim: *const c_int,
    x: *const c_double,
    ncomp: *const c_int,
    f: *mut c_double,
    userdata: *mut c_void,
    nvec: *const c_int,
    core: *const c_int,
) -> c_int;

type DivonneIntegrandC = extern "C" fn(
    ndim: *const c_int,
    x: *const c_double,
    ncomp: *const c_int,
    f: *mut c_double,
    userdata: *mut c_void,
    nvec: *const c_int,
    core: *const c_int,
    phase: *const c_int,
) -> c_int;

type PeakfinderC = extern "C" fn(
    ndim: *const c_int,
    b: *const c_double,
    n: *mut c_int,
    x: *mut c_double,
    userdata: *mut c_void,
);

/// Vegas integrand evaluation function.
///
/// The dimensions of random input variables `x` and output `f`
/// are provided to the integration routine as dimension and components respectively.
/// `T` can be any type. If you don't want to provide user data,
/// simply make `T` a `usize` and provide any number.
///
/// `core` specifies the current core that is being used. This can be used to write to
/// the user data in a thread-safe way.
///
/// On returning an error, the integration will be aborted.
pub type VegasIntegrand<T> = fn(
    x: &[f64],
    f: &mut [f64],
    user_data: &mut T,
    nvec: usize,
    core: i32,
    weight: &[f64],
    iter: usize,
) -> Result<(), &'static str>;

/// Suave integrand evaluation function.
///
/// The dimensions of random input variables `x` and output `f`
/// are provided to the integration routine as dimension and components respectively.
/// `T` can be any type. If you don't want to provide user data,
/// simply make `T` a `usize` and provide any number.
///
/// `core` specifies the current core that is being used. This can be used to write to
/// the user data in a thread-safe way.
///
/// On returning an error, the integration will be aborted.
pub type SuaveIntegrand<T> = fn(
    x: &[f64],
    f: &mut [f64],
    user_data: &mut T,
    nvec: usize,
    core: i32,
    weight: &[f64],
    iter: usize,
) -> Result<(), &'static str>;

/// Cuhre integrand evaluation function.
///
/// The dimensions of random input variables `x` and output `f`
/// are provided to the integration routine as dimension and components respectively.
/// `T` can be any type. If you don't want to provide user data,
/// simply make `T` a `usize` and provide any number.
///
/// `core` specifies the current core that is being used. This can be used to write to
/// the user data in a thread-safe way.
///
/// On returning an error, the integration will be aborted.
pub type CuhreIntegrand<T> = fn(
    x: &[f64],
    f: &mut [f64],
    user_data: &mut T,
    nvec: usize,
    core: i32,
) -> Result<(), &'static str>;

/// Divonne integrand evaluation function.
///
/// The dimensions of random input variables `x` and output `f`
/// are provided to the integration routine as dimension and components respectively.
/// `T` can be any type. If you don't want to provide user data,
/// simply make `T` a `usize` and provide any number.
///
/// `core` specifies the current core that is being used. This can be used to write to
/// the user data in a thread-safe way.
///
/// On returning an error, the integration will be aborted.
pub type DivonneIntegrand<T> = fn(
    x: &[f64],
    f: &mut [f64],
    user_data: &mut T,
    nvec: usize,
    core: i32,
    phase: usize,
) -> Result<(), &'static str>;

#[repr(C)]
struct VegasUserData<T> {
    integrand: VegasIntegrand<T>,
    user_data: T,
}

#[repr(C)]
struct CuhreUserData<T> {
    integrand: CuhreIntegrand<T>,
    user_data: T,
}

#[repr(C)]
struct SuaveUserData<T> {
    integrand: SuaveIntegrand<T>,
    user_data: T,
}

#[repr(C)]
struct DivonneUserData<T> {
    integrand: DivonneIntegrand<T>,
    user_data: T,
}

/// The result of an integration with Cuba.
#[derive(Debug)]
pub struct CubaResult {
    pub neval: i64,
    pub fail: i32,
    pub result: Vec<f64>,
    pub error: Vec<f64>,
    pub prob: Vec<f64>,
}

/// A Cuba integrator. It should be created with an integrand function.
pub struct CubaIntegrator {
    mineval: i64,
    maxeval: i64,
    nstart: i64,
    nincrease: i64,
    epsrel: f64,
    epsabs: f64,
    batch: i64,
    cores: usize,
    max_points_per_core: usize,
    seed: i32,
    use_only_last_sample: bool,
    save_state_file: String,
    keep_state_file: bool,
    reset_vegas_integrator: bool,
    // Cuhre
    key: i32,
    // Divonne
    key1: i32,
    key2: i32,
    key3: i32,
    maxpass: i32,
    border: f64,
    maxchisq: f64,
    mindeviation: f64,
}

impl CubaIntegrator {
    /// Create a new Cuba integrator. Use the `set_` functions
    /// to set integration parameters.
    pub fn new() -> CubaIntegrator {
        CubaIntegrator {
            mineval: 0,
            maxeval: 50000,
            nstart: 1000,
            nincrease: 500,
            epsrel: 0.001,
            epsabs: 1e-12,
            batch: 1000,
            cores: 1,
            max_points_per_core: 1000,
            seed: 0,
            use_only_last_sample: false,
            save_state_file: String::new(),
            keep_state_file: false,
            reset_vegas_integrator: false,
            key: 0,
            key1: 47,
            key2: 1,
            key3: 1,
            maxpass: 5,
            border: 0.,
            maxchisq: 10.,
            mindeviation: 0.25,
        }
    }

    /// Set the number of cores and the maximum number of points per core.
    /// The default is the number of idle cores for `cores` and
    /// 1000 for `max_points_per_core`.
    pub fn set_cores(&mut self, cores: usize, max_points_per_core: usize) -> &mut Self {
        self.cores = cores;
        self.max_points_per_core = max_points_per_core;
        unsafe {
            cubacores(cores as c_int, max_points_per_core as c_int);
        }
        self
    }

    gen_setter!(set_mineval, mineval, i64);
    gen_setter!(set_maxeval, maxeval, i64);
    gen_setter!(set_nstart, nstart, i64);
    gen_setter!(set_nincrease, nincrease, i64);
    gen_setter!(set_epsrel, epsrel, f64);
    gen_setter!(set_epsabs, epsabs, f64);
    gen_setter!(set_batch, batch, i64);
    gen_setter!(set_seed, seed, i32);
    gen_setter!(set_use_only_last_sample, use_only_last_sample, bool);
    gen_setter!(set_save_state_file, save_state_file, String);
    gen_setter!(set_keep_state_file, keep_state_file, bool);
    gen_setter!(set_reset_vegas_integrator, reset_vegas_integrator, bool);
    gen_setter!(set_key, key, i32);
    gen_setter!(set_key1, key1, i32);
    gen_setter!(set_key2, key2, i32);
    gen_setter!(set_key3, key3, i32);
    gen_setter!(set_maxpass, maxpass, i32);
    gen_setter!(set_border, border, f64);
    gen_setter!(set_maxchisq, maxchisq, f64);
    gen_setter!(set_mindeviation, mindeviation, f64);

    extern "C" fn c_vegas_integrand<T>(
        ndim: *const c_int,
        x: *const c_double,
        ncomp: *const c_int,
        f: *mut c_double,
        userdata: *mut c_void,
        nvec: *const c_int,
        core: *const c_int,
        weight: *const c_double,
        iter: *const c_int,
    ) -> c_int {
        unsafe {
            let k: &mut VegasUserData<T> = &mut *(userdata as *mut _);

            // call the safe integrand
            match (k.integrand)(
                &slice::from_raw_parts(x, *ndim as usize * *nvec as usize),
                &mut slice::from_raw_parts_mut(f, *ncomp as usize * *nvec as usize),
                &mut k.user_data,
                *nvec as usize,
                *core as i32,
                &slice::from_raw_parts(weight, *nvec as usize),
                *iter as usize,
            ) {
                Ok(_) => 0,
                Err(e) => {
                    println!("Error during integration: {}. Aborting.", e);
                    -999
                }
            }
        }
    }

    extern "C" fn c_suave_integrand<T>(
        ndim: *const c_int,
        x: *const c_double,
        ncomp: *const c_int,
        f: *mut c_double,
        userdata: *mut c_void,
        nvec: *const c_int,
        core: *const c_int,
        weight: *const c_double,
        iter: *const c_int,
    ) -> c_int {
        unsafe {
            let k: &mut SuaveUserData<T> = &mut *(userdata as *mut _);

            // call the safe integrand
            match (k.integrand)(
                &slice::from_raw_parts(x, *ndim as usize * *nvec as usize),
                &mut slice::from_raw_parts_mut(f, *ncomp as usize * *nvec as usize),
                &mut k.user_data,
                *nvec as usize,
                *core as i32,
                &slice::from_raw_parts(weight, *ndim as usize * *nvec as usize),
                *iter as usize,
            ) {
                Ok(_) => 0,
                Err(e) => {
                    println!("Error during integration: {}. Aborting.", e);
                    -999
                }
            }
        }
    }

    extern "C" fn c_cuhre_integrand<T>(
        ndim: *const c_int,
        x: *const c_double,
        ncomp: *const c_int,
        f: *mut c_double,
        userdata: *mut c_void,
        nvec: *const c_int,
        core: *const c_int,
    ) -> c_int {
        unsafe {
            let k: &mut CuhreUserData<T> = &mut *(userdata as *mut _);

            // call the safe integrand
            match (k.integrand)(
                &slice::from_raw_parts(x, *ndim as usize * *nvec as usize),
                &mut slice::from_raw_parts_mut(f, *ncomp as usize * *nvec as usize),
                &mut k.user_data,
                *nvec as usize,
                *core as i32,
            ) {
                Ok(_) => 0,
                Err(e) => {
                    println!("Error during integration: {}. Aborting.", e);
                    -999
                }
            }
        }
    }

    extern "C" fn c_divonne_integrand<T>(
        ndim: *const c_int,
        x: *const c_double,
        ncomp: *const c_int,
        f: *mut c_double,
        userdata: *mut c_void,
        nvec: *const c_int,
        core: *const c_int,
        phase: *const c_int,
    ) -> c_int {
        unsafe {
            let k: &mut DivonneUserData<T> = &mut *(userdata as *mut _);

            // call the safe integrand
            match (k.integrand)(
                &slice::from_raw_parts(x, *ndim as usize * *nvec as usize),
                &mut slice::from_raw_parts_mut(f, *ncomp as usize * *nvec as usize),
                &mut k.user_data,
                *nvec as usize,
                *core as i32,
                *phase as usize,
            ) {
                Ok(_) => 0,
                Err(e) => {
                    println!("Error during integration: {}. Aborting.", e);
                    -999
                }
            }
        }
    }

    extern "C" fn c_peakfinder(
        ndim: *const c_int,
        b: *const c_double,
        n: *mut c_int,
        x: *mut c_double,
        userdata: *mut c_void,
    ) {
        // TODO: call a safe peakfinder routine
    }

    /// Integrate using the Vegas integrator.
    ///
    /// * `ndim` - Dimension of the input
    /// * `ncomp` - Dimension (components) of the output
    /// * `nvec` - Number of input points given to the integrand function
    /// * `verbosity` - Verbosity level
    /// * `gridno` - Grid number between -10 and 10. If 0, no grid is stored.
    ///              If it is positive, the grid is storedin the `gridno`th slot.
    ///              With a negative number the grid is cleared.
    /// * `user_data` - User data used by the integrand function
    pub fn vegas<T>(
        &mut self,
        ndim: usize,
        ncomp: usize,
        nvec: usize,
        verbosity: CubaVerbosity,
        gridno: i32,
        integrand: VegasIntegrand<T>,
        user_data: T,
    ) -> CubaResult {
        let mut out = CubaResult {
            neval: 0,
            fail: 0,
            result: vec![0.; ncomp],
            error: vec![0.; ncomp],
            prob: vec![0.; ncomp],
        };

        assert!(
            gridno >= -10 && gridno <= 10,
            "Grid number needs to be between -10 and 10."
        );

        assert!(
            nvec <= self.batch as usize && nvec <= self.max_points_per_core,
            "nvec needs to be at most the vegas batch size or the max points per core"
        );

        // pass the safe integrand and the user data
        let mut x = VegasUserData {
            integrand: integrand,
            user_data: user_data,
        };

        let user_data_ptr = &mut x as *mut _ as *mut c_void;

        // Bits 0 and 1 set the CubaVerbosity
        let mut cubaflags = verbosity as i32;
        // Bit 2 sets whether only last sample should be used
        if self.use_only_last_sample {
            cubaflags |= 0b100;
        }
        // Bit 4 specifies whether the state file should be retained after integration
        if self.keep_state_file {
            cubaflags |= 0b10000;
        }
        // Bit 5 specifies whether the integrator state (except the grid) should be reset
        // after having loaded a state file (Vegas only)
        if self.reset_vegas_integrator {
            cubaflags |= 0b100000;
        }
        let c_str = CString::new(self.save_state_file.as_str()).expect("CString::new failed");
        unsafe {
            llVegas(
                ndim as c_int,                                // ndim
                ncomp as c_int,                               // ncomp
                Some(CubaIntegrator::c_vegas_integrand::<T>), // integrand
                user_data_ptr,                                // user data
                nvec as c_longlong,                           // nvec
                self.epsrel,                                  // epsrel
                self.epsabs,                                  // epsabs
                cubaflags as c_int,                           // flags
                self.seed,                                    // seed
                self.mineval,                                 // mineval
                self.maxeval,                                 // maxeval
                self.nstart,                                  // nstart
                self.nincrease,                               // nincrease
                self.batch,                                   // batch
                gridno,                                       // grid no
                c_str.as_ptr(),                               // statefile
                ptr::null_mut(),                              // spin
                &mut out.neval,
                &mut out.fail,
                &mut out.result[0],
                &mut out.error[0],
                &mut out.prob[0],
            );
        }

        out
    }
    /// Integrate using the Suave integrator.
    ///
    /// * `ndim` - Dimension of the input
    /// * `ncomp` - Dimension (components) of the output
    /// * `nvec` - Number of input points given to the integrand function
    /// * `nnew` - Number of new integrand evaluations in each subdivision
    /// * `nmin` - Minimum number of samples a former pass must contribute to a
    ///            subregion to be considered in that region’s compound integral value
    /// * `flatness` - This determines how prominently individual samples with a large fluctuation
    ///                figure in the total fluctuation
    /// * `verbosity` - Verbosity level
    /// * `user_data` - User data used by the integrand function
    pub fn suave<T>(
        &mut self,
        ndim: usize,
        ncomp: usize,
        nvec: usize,
        nnew: usize,
        nmin: usize,
        flatness: f64,
        verbosity: CubaVerbosity,
        integrand: SuaveIntegrand<T>,
        user_data: T,
    ) -> CubaResult {
        let mut out = CubaResult {
            neval: 0,
            fail: 0,
            result: vec![0.; ncomp],
            error: vec![0.; ncomp],
            prob: vec![0.; ncomp],
        };

        assert!(
            nvec <= self.max_points_per_core && nvec <= nnew,
            "nvec needs to be at most the max points per core and nnew"
        );

        // pass the safe integrand and the user data
        let mut x = SuaveUserData {
            integrand: integrand,
            user_data: user_data,
        };

        let user_data_ptr = &mut x as *mut _ as *mut c_void;

        // Bits 0 and 1 set the CubaVerbosity
        let mut cubaflags = verbosity as i32;
        // Bit 2 sets whether only last sample should be used
        if self.use_only_last_sample {
            cubaflags |= 0b100;
        }
        // Bit 4 specifies whether the state file should be retained after integration
        if self.keep_state_file {
            cubaflags |= 0b10000;
        }
        // Bit 5 specifies whether the integrator state (except the grid) should be reset
        // after having loaded a state file (Vegas only)
        if self.reset_vegas_integrator {
            cubaflags |= 0b100000;
        }

        let mut nregions = 0;
        let c_str = CString::new(self.save_state_file.as_str()).expect("CString::new failed");
        unsafe {
            llSuave(
                ndim as c_int,                                // ndim
                ncomp as c_int,                               // ncomp
                Some(CubaIntegrator::c_suave_integrand::<T>), // integrand
                user_data_ptr,                                // user data
                nvec as c_longlong,                           // nvec
                self.epsrel,                                  // epsrel
                self.epsabs,                                  // epsabs
                cubaflags as c_int,                           // flags
                self.seed,                                    // seed
                self.mineval,                                 // mineval
                self.maxeval,                                 // maxeval
                nnew as c_longlong,
                nmin as c_longlong,
                flatness,
                c_str.as_ptr(),
                ptr::null_mut(),
                &mut nregions,
                &mut out.neval,
                &mut out.fail,
                &mut out.result[0],
                &mut out.error[0],
                &mut out.prob[0],
            );
        }

        out
    }

    /// Integrate using the Divonne integrator.
    ///
    /// * `ndim` - Dimension of the input
    /// * `ncomp` - Dimension (components) of the output
    /// * `nvec` - Number of input points given to the integrand function
    /// * `xgiven` - A list of input points which lie close to peaks
    /// * `verbosity` - Verbosity level
    /// * `user_data` - User data used by the integrand function
    pub fn divonne<T>(
        &mut self,
        ndim: usize,
        ncomp: usize,
        nvec: usize,
        xgiven: &[f64],
        verbosity: CubaVerbosity,
        integrand: DivonneIntegrand<T>,
        user_data: T,
    ) -> CubaResult {
        let mut out = CubaResult {
            neval: 0,
            fail: 0,
            result: vec![0.; ncomp],
            error: vec![0.; ncomp],
            prob: vec![0.; ncomp],
        };

        assert!(
            nvec <= self.max_points_per_core,
            "nvec needs to be at most the max points per core"
        );

        // pass the safe integrand and the user data
        let mut x = DivonneUserData {
            integrand: integrand,
            user_data: user_data,
        };

        let user_data_ptr = &mut x as *mut _ as *mut c_void;

        // Bits 0 and 1 set the CubaVerbosity
        let mut cubaflags = verbosity as i32;
        // Bit 2 sets whether only last sample should be used
        if self.use_only_last_sample {
            cubaflags |= 0b100;
        }
        // Bit 4 specifies whether the state file should be retained after integration
        if self.keep_state_file {
            cubaflags |= 0b10000;
        }
        // Bit 5 specifies whether the integrator state (except the grid) should be reset
        // after having loaded a state file (Vegas only)
        if self.reset_vegas_integrator {
            cubaflags |= 0b100000;
        }

        let mut nregions = 0;
        let c_str = CString::new(self.save_state_file.as_str()).expect("CString::new failed");
        unsafe {
            llDivonne(
                ndim as c_int,                                  // ndim
                ncomp as c_int,                                 // ncomp
                Some(CubaIntegrator::c_divonne_integrand::<T>), // integrand
                user_data_ptr,                                  // user data
                nvec as c_longlong,                             // nvec
                self.epsrel,                                    // epsrel
                self.epsabs,                                    // epsabs
                cubaflags as c_int,                             // flags
                self.seed,                                      // seed
                self.mineval,                                   // mineval
                self.maxeval,                                   // maxeval
                self.key1,
                self.key2,
                self.key3,
                self.maxpass,
                self.border,
                self.maxchisq,
                self.mindeviation,
                (xgiven.len() / ndim) as c_longlong,
                ndim as c_int,
                if xgiven.len() == 0 {
                    ptr::null_mut()
                } else {
                    &xgiven[0]
                },
                0,
                None,
                c_str.as_ptr(),
                ptr::null_mut(),
                &mut nregions,
                &mut out.neval,
                &mut out.fail,
                &mut out.result[0],
                &mut out.error[0],
                &mut out.prob[0],
            );
        }

        out
    }

    /// Integrate using the Cuhre integrator.
    ///
    /// * `ndim` - Dimension of the input
    /// * `ncomp` - Dimension (components) of the output
    /// * `nvec` - Number of input points given to the integrand function
    /// * `verbosity` - Verbosity level
    /// * `user_data` - User data used by the integrand function
    pub fn cuhre<T>(
        &mut self,
        ndim: usize,
        ncomp: usize,
        nvec: usize,
        verbosity: CubaVerbosity,
        integrand: CuhreIntegrand<T>,
        user_data: T,
    ) -> CubaResult {
        let mut out = CubaResult {
            neval: 0,
            fail: 0,
            result: vec![0.; ncomp],
            error: vec![0.; ncomp],
            prob: vec![0.; ncomp],
        };

        assert!(nvec <= 32, "nvec needs to be at most 32");

        // pass the safe integrand and the user data
        let mut x = CuhreUserData {
            integrand: integrand,
            user_data: user_data,
        };

        let user_data_ptr = &mut x as *mut _ as *mut c_void;

        // Bits 0 and 1 set the CubaVerbosity
        let mut cubaflags = verbosity as i32;
        // Bit 2 sets whether only last sample should be used
        if self.use_only_last_sample {
            cubaflags |= 0b100;
        }
        // Bit 4 specifies whether the state file should be retained after integration
        if self.keep_state_file {
            cubaflags |= 0b10000;
        }

        let mut nregions = 0;
        let c_str = CString::new(self.save_state_file.as_str()).expect("CString::new failed");
        unsafe {
            llCuhre(
                ndim as c_int,                                // ndim
                ncomp as c_int,                               // ncomp
                Some(CubaIntegrator::c_cuhre_integrand::<T>), // integrand
                user_data_ptr,                                // user data
                nvec as c_longlong,                           // nvec
                self.epsrel,                                  // epsrel
                self.epsabs,                                  // epsabs
                cubaflags as c_int,                           // flags
                self.mineval,                                 // mineval
                self.maxeval,                                 // maxeval
                self.key,                                     // key
                c_str.as_ptr(),                               // statefile
                ptr::null_mut(),                              // spin
                &mut nregions,
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
