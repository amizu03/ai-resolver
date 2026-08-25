use core::ops::RangeInclusive;

use num_traits::float::Float;
use rand::rngs::SmallRng;
use serde::{Deserializer, Serializer};
use serde_big_array::BigArray;

use crate::prelude::*;

trait MatrixEx {
    fn zeroed() -> Self;
    fn rand(rng: &mut SmallRng, range: RangeInclusive<f32>) -> Self;
}

impl<const R: usize, const C: usize> MatrixEx for Matrix<R, C> {
    fn zeroed() -> Self {
        unsafe { zeroed() }
    }

    fn rand(rng: &mut SmallRng, range: RangeInclusive<f32>) -> Self {
        use rand::SeedableRng;
        use rand::distributions::{Distribution, Uniform};

        // Create RNG and seed with default
        let uniform_dist = Uniform::new(0.0, 1.0);

        // Create raw matrix buffer with correct dimensions
        let mut raw = [[0.0; C]; R];

        // Initialize arbitrary matrix with random values in distribution
        raw.iter_mut().for_each(|column| {
            column
                .iter_mut()
                .for_each(|x| *x = uniform_dist.sample(rng))
        });

        // This is safe because we already know the matrix size at compile time because we have R, C
        let mut matrix = unsafe { Matrix::from_data_statically_unchecked(raw) };

        matrix
    }
}

#[derive(Copy, Clone)]
struct DenseLayer<const IN: usize, const OUT: usize> {
    weights: [[f32; IN]; OUT],
    biases: [f32; OUT],
    weight_grads: [[f32; IN]; OUT], // Gradient storage for weights
    bias_grads: [f32; OUT],         // Gradient storage for biases
}

impl<const IN: usize, const OUT: usize> DenseLayer<IN, OUT> {
    fn to_raw_data(&self) -> [Vec<f32>; 4] {
        let weights = self
            .weights
            .iter()
            .flat_map(|row| row.iter())
            .copied()
            .collect();
        let biases = self.biases.to_vec();
        let weight_grads = self
            .weight_grads
            .iter()
            .flat_map(|row| row.iter())
            .copied()
            .collect();
        let bias_grads = self.bias_grads.to_vec();
        [weights, biases, weight_grads, bias_grads]
    }

    fn from_raw_data([weights, biases, weight_grads, bias_grads]: [Vec<f32>; 4]) -> Self {
        let mut weights_arr = [[0.0; IN]; OUT];
        let mut biases_arr = [0.0; OUT];
        let mut weight_grads_arr = [[0.0; IN]; OUT];
        let mut bias_grads_arr = [0.0; OUT];

        weights.iter().enumerate().for_each(|(i, &val)| {
            let row = i / IN;
            let col = i % IN;
            weights_arr[row][col] = val;
        });

        biases.iter().enumerate().for_each(|(i, &val)| {
            biases_arr[i] = val;
        });

        weight_grads.iter().enumerate().for_each(|(i, &val)| {
            let row = i / IN;
            let col = i % IN;
            weight_grads_arr[row][col] = val;
        });

        bias_grads.iter().enumerate().for_each(|(i, &val)| {
            bias_grads_arr[i] = val;
        });

        Self {
            weights: weights_arr,
            biases: biases_arr,
            weight_grads: weight_grads_arr,
            bias_grads: bias_grads_arr,
        }
    }

    fn new() -> Self {
        use rand::SeedableRng;
        use rand::distributions::{Distribution, Uniform};
        use rand::rngs::SmallRng;

        // Create RNG and seed with default
        let mut rng = SmallRng::from_seed([0; _]);
        let uniform_dist = Uniform::new(-1.0, 1.0);

        // Create raw matrix buffer with correct dimensions
        let mut weights = [[0.0; IN]; OUT];

        weights.iter_mut().enumerate().for_each(|(i, column)| {
            let column_len = column.len() as f32;

            column.iter_mut().enumerate().for_each(|(j, x)| {
                // Random initialization
                *x = uniform_dist.sample(&mut rng)

                // Uniform initialization
                // *x = j as f32 / column_len;
            })
        });

        // Create raw matrix buffer with correct dimensions
        let mut biases = [0.0; OUT];

        Self {
            weights,
            biases,
            weight_grads: [[0.0; IN]; OUT],
            bias_grads: [0.0; OUT],
        }
    }

    fn forward(&self, input: [f32; IN]) -> [f32; OUT] {
        let mut output = [0.0; OUT];
        for i in 0..OUT {
            for j in 0..IN {
                output[i] += input[j] * self.weights[i][j];
            }
            output[i] += self.biases[i];
        }
        output
    }

    fn backward(&mut self, input: [f32; IN], grad_output: [f32; OUT]) -> [f32; IN] {
        let mut grad_input = [0.0; IN];

        // Calculate gradients for weights and biases
        for i in 0..OUT {
            self.bias_grads[i] += grad_output[i];
            for j in 0..IN {
                self.weight_grads[i][j] += input[j] * grad_output[i];
                grad_input[j] += self.weights[i][j] * grad_output[i];
            }
        }

        grad_input
    }

    fn update_weights(&mut self, learning_rate: f32) {
        for i in 0..OUT {
            self.biases[i] -= learning_rate * self.bias_grads[i];
            self.bias_grads[i] = 0.0; // Reset gradient after update

            for j in 0..IN {
                self.weights[i][j] -= learning_rate * self.weight_grads[i][j];
                self.weight_grads[i][j] = 0.0; // Reset gradient after update
            }
        }
    }
}

fn relu<const N: usize>(input: [f32; N]) -> ([f32; N], [f32; N]) {
    let mut output = [0.0; N];
    let mut grad = [0.0; N];

    for i in 0..N {
        output[i] = input[i].max(0.0);
        grad[i] = if input[i] > 0.0 { 1.0 } else { 0.0 };
    }

    (output, grad)
}

fn tanh<const N: usize>(input: [f32; N]) -> ([f32; N], [f32; N]) {
    let mut output = [0.0; N];
    let mut grad = [0.0; N];

    for i in 0..N {
        output[i] = input[i].tanh();
        grad[i] = 1.0 - (output[i] * output[i]);
    }

    (output, grad)
}

fn normalize<const N: usize>(mut input: [f32; N]) -> [f32; N] {
    let mut norm = 0.0;
    for i in &input {
        norm += i * i;
    }

    norm = norm.sqrt() + f32::EPSILON;

    for i in &mut input {
        *i /= norm;
    }

    input
}

pub fn norm_yaw(mut yaw: f32) -> f32 {
    if yaw.is_nan() || !yaw.is_finite() {
        0.0
    } else {
        while yaw > 180.0 {
            yaw -= 360.0;
        }

        while yaw < -180.0 {
            yaw += 360.0;
        }

        yaw
    }
}

pub struct BaseResolver {
    layer1: DenseLayer<12, 16>,
    layer2: DenseLayer<16, 8>,
    layer3: DenseLayer<8, 2>,
}

impl BaseResolver {
    pub fn new() -> Self {
        Self {
            layer1: DenseLayer::new(),
            layer2: DenseLayer::new(),
            layer3: DenseLayer::new(),
        }
    }

    pub fn forward(&self, input: [f32; 12]) -> [f32; 2] {
        let (x1, _) = tanh(self.layer1.forward(input));
        let (x2, _) = tanh(self.layer2.forward(x1));
        self.layer3.forward(x2)
    }

    pub fn loss_with_grads<const N: usize>(
        y_true: [[f32; 2]; N], // True angles (y1, y2) pairs
        y_pred: [[f32; 2]; N], // Predicted angles (y1, y2) pairs
        lambda: f32,           // Regularization constant
    ) -> (f32, [[f32; 2]; N], [[f32; 2]; N]) {
        let mut loss = 0.0;
        let mut mse_grads = [[0.0; 2]; N]; // Gradients for MSE
        let mut regularization_grads = [[0.0; 2]; N]; // Gradients for regularization
        let mut regularization_term = 0.0;

        for i in 0..N {
            // MSE Loss and Gradients for each (y1, y2)
            let diff_y1 = y_pred[i][0] - y_true[i][0];
            let diff_y2 = y_pred[i][1] - y_true[i][1];

            loss += diff_y1.powi(2) + diff_y2.powi(2);
            mse_grads[i][0] = 2.0 * diff_y1;
            mse_grads[i][1] = 2.0 * diff_y2;

            // Regularization Term (penalty for incorrect normalization)
            let radius = (y_pred[i][0].powi(2) + y_pred[i][1].powi(2)).sqrt();
            regularization_term += 1.0 - radius;

            // Gradients for Regularization Term
            regularization_grads[i][0] = -lambda * y_pred[i][0] / radius;
            regularization_grads[i][1] = -lambda * y_pred[i][1] / radius;
        }

        // Total Loss (MSE + Regularization Term)
        loss /= N as f32; // Average the MSE
        regularization_term /= N as f32; // Average the regularization term

        // Total Loss = MSE Loss + Regularization Term
        let total_loss = loss + lambda * regularization_term;

        // Combine MSE gradients with regularization gradients
        let mut total_grads = [[0.0; 2]; N];
        for i in 0..N {
            total_grads[i][0] = mse_grads[i][0] + regularization_grads[i][0];
            total_grads[i][1] = mse_grads[i][1] + regularization_grads[i][1];
        }

        (total_loss, total_grads, mse_grads)
    }

    pub fn train_step(&mut self, input: [f32; 12], target: [f32; 2], learning_rate: f32) -> f32 {
        // Forward pass
        let (x1, grad1) = tanh(self.layer1.forward(input));
        let (x2, grad2) = tanh(self.layer2.forward(x1));
        let output = self.layer3.forward(x2);

        // Calculate loss and initial gradient (mean squared error)
        let (total_loss, grad_output, mse_grads) = Self::loss_with_grads([target], [output], 0.1);
        let grad_output = grad_output[0];

        // Backward pass
        let grad_x2 = self.layer3.backward(x2, grad_output);
        let grad_x1 = self.layer2.backward(
            x1,
            grad_x2
                .iter()
                .zip(grad2)
                .map(|(g, r)| g * r)
                .collect::<Vec<_>>()
                .try_into()
                .unwrap(),
        );
        self.layer1.backward(
            input,
            grad_x1
                .iter()
                .zip(grad1)
                .map(|(g, r)| g * r)
                .collect::<Vec<_>>()
                .try_into()
                .unwrap(),
        );

        // Update weights
        self.layer3.update_weights(learning_rate);
        self.layer2.update_weights(learning_rate);
        self.layer1.update_weights(learning_rate);

        grad_output.iter().sum()
    }
}

