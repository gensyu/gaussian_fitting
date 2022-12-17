use mathru::algebra::linear::{Vector};
use mathru::vector;
mod fit_gausiann;
use mathru::statistics::distrib::{Distribution, Normal};

fn main() {
    let num_samples: usize = 100;
    let noise: Normal<f64> = Normal::new(0.0, 0.05);
    let mut t_vec: Vec<f64> = Vec::with_capacity(num_samples);
    // Start time
    let t_0 = 0.0f64;
    // End time
    let t_1 = 10.0f64;
    let mut y_vec: Vec<f64> = Vec::with_capacity(num_samples);
    // True function parameters

    let beta = vector![3.0; 5.0; 1.0];
    for i in 0..num_samples
    {
        let t_i: f64 = (t_1 - t_0) / (num_samples as f64) * (i as f64);
        //Add some noise
        y_vec.push(fit_gausiann::GaussianSample::model_func(t_i, &beta) + noise.random()*0.5);
        t_vec.push(t_i);
    }

    let a = fit_gausiann::fit_gausiann(t_vec.clone(), y_vec.clone());
    println!("{},{},{}\n", a.0, a.1, a.2);

    for (i, &v) in y_vec.iter().enumerate() {
        println!("{},{}", t_vec[i], v);
    }
}