#![cfg_attr(not(feature = "cli"), allow(dead_code))]

mod prelude;

pub mod backend;
mod continual_backprop;
mod dynamics;
pub mod events;
pub mod neuron_scaling;
pub mod objective;
pub mod optimizer_dynamics;
pub mod phased;
pub mod profile;
#[cfg(feature = "rerun")]
pub mod rerun;
pub mod schedule;
pub mod startup_autotune;
pub mod steps;
pub mod utils;

#[allow(unused_imports)]
pub use backend::*;
#[allow(unused_imports)]
pub use continual_backprop::*;
#[allow(unused_imports)]
pub use dynamics::*;
#[allow(unused_imports)]
pub use neuron_scaling::*;
#[allow(unused_imports)]
pub use objective::*;
#[allow(unused_imports)]
pub use optimizer_dynamics::*;
#[allow(unused_imports)]
pub use phased::*;
#[allow(unused_imports)]
pub use profile::*;
#[cfg(feature = "rerun")]
#[allow(unused_imports)]
pub use rerun::*;
#[allow(unused_imports)]
pub use schedule::*;
#[allow(unused_imports)]
pub use startup_autotune::*;
#[allow(unused_imports)]
pub use steps::*;
#[allow(unused_imports)]
pub use utils::*;
