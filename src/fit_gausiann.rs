use mathru::algebra::linear::{Matrix, Vector};
use mathru::optimization::{LevenbergMarquardt, Optim};
use mathru::vector;

//y = a * exp(-0.5*(x-mu)^2/sigma^2)
pub struct GaussianSample {
    x: Vector<f64>,
    y: Vector<f64>,
}

impl GaussianSample {
    pub fn new(x: Vector<f64>, y: Vector<f64>) -> GaussianSample {
        GaussianSample { x, y }
    }

    pub fn model_func(x: f64, beta: &Vector<f64>) -> f64 {
        let a: f64 = beta[0];
        let mu: f64 = beta[1];
        let sigma: f64 = beta[2];
        // fx = a * exp(-(x_mu)^2/(2*sigma^2))
        let f_x: f64 = a * (-0.5 * (x - mu).powf(2.0) / (sigma.powf(2.0))).exp();

        return f_x;
    }
}

impl Optim<f64> for GaussianSample {
    // y(x_i) - f(x_i)
    fn eval(&self, beta: &Vector<f64>) -> Vector<f64> {
        let f_x = self
            .x
            .clone()
            .apply(&|x: &f64| GaussianSample::model_func(*x, beta));
        let r: Vector<f64> = &self.y - &f_x;
        return vector![r.dotp(&r)];
    }

    fn jacobian(&self, beta: &Vector<f64>) -> Matrix<f64> {
        let (x_m, _x_n) = self.x.dim();
        let (beta_m, _beta_n) = beta.dim();

        let mut jacobian_f: Matrix<f64> = Matrix::zero(x_m, beta_m);

        let f_x = self
            .x
            .clone()
            .apply(&|x: &f64| GaussianSample::model_func(*x, beta));
        let residual: Vector<f64> = &self.y - &f_x;

        for i in 0..x_m {
            let a: f64 = beta[0];
            let mu: f64 = beta[1];
            let sigma: f64 = beta[2];

            let x_i: f64 = self.x[i];

            jacobian_f[[i, 0]] = (-0.5 * (x_i - mu).powf(2.0) / sigma.powf(2.0)).exp();
            jacobian_f[[i, 1]] = a * (x_i - mu) / (sigma.powf(2.0))
                * (-0.5 * (x_i - mu).powf(2.0) / sigma.powf(2.0)).exp();
            jacobian_f[[i, 2]] = a * (x_i - mu).powf(2.0) / (sigma.powf(3.0))
                * (-0.5 * (x_i - mu).powf(2.0) / sigma.powf(2.0)).exp();
        }

        let jacobian: Matrix<f64> = (residual.transpose() * jacobian_f * -2.0).into();
        return jacobian;
    }
}

pub fn fit_gausiann(x: Vec<f64>, y: Vec<f64>) -> (f64, f64, f64) {
    // init value is y_max and index
    let (index, a_0) = y
        .iter()
        .enumerate()
        .fold((usize::MIN, f64::MIN), |(i_a, a), (i_b, &b)| {
            if b > a {
                (i_b, b)
            } else {
                (i_a, a)
            }
        });
    let mu_0 = x[index];
    let x: Vector<f64> = Vector::new_column(x.clone());
    let y: Vector<f64> = Vector::new_column(y.clone());
    let gaussian = GaussianSample::new(x, y);
    let optim: LevenbergMarquardt<f64> = LevenbergMarquardt::new(1000, 0.3, 0.99);
    let arg = optim.minimize(&gaussian, &vector![a_0; mu_0; 1.0]).unwrap().arg();
    return (arg[0], arg[1], arg[2])
}
