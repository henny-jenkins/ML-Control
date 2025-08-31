// A module containing an implementation of the nonlinear dynamics for an inverted
// pendulum on a cart

use nalgebra::Vector4;
use rand::prelude::*;
use rand_distr::{Normal, Distribution, WeightedIndex, Uniform};

pub struct ModelParameters(pub f32, pub f32, pub f32, pub f32);

pub fn nonlinear_dynamics(state_vec: &Vector4<f32>, ctrl_signal: &f32, params: &ModelParameters) -> Vector4<f32> {
    let pendulum_mass = params.0;
    let cart_mass = params.1;
    let pendulum_length = params.2;
    let damping = params.3;
    let g: f32 = -9.813;    // gravity

    // state definition: state_vec = [x, x_dot, theta, theta_dot]
    let (sx, cx) = state_vec[2].sin_cos();   // short-hand for sin(theta) and cos(theta)
    let x_dot = state_vec[1];
    let theta_dot = state_vec[3];

    let pm = pendulum_mass;
    let cm = cart_mass;
    let pl = pendulum_length;

    let theta_dot_sq = theta_dot * theta_dot;
    let sx_sq = sx * sx;

    let denom = pm * pl * pl * (cm + pm * sx_sq);
    
    let mut dx = Vector4::<f32>::zeros();
    dx[0] = x_dot;
    dx[2] = theta_dot;

    let common_term = pm * pl * theta_dot_sq * sx - damping * x_dot;
    
    let pm_pl = pm * pl;

    dx[1] = (pm_pl * pl / denom) * (-pm * g * cx * sx + common_term + ctrl_signal);
    dx[3] = (pm_pl / denom) * ((cm + pm) * g * sx - cx * (common_term + ctrl_signal));
    
    return dx;  // return state derivatives
}

fn rk4(initial_condition: &Vector4<f32>, t_vec: &Vec<f32>, k: &Vector4<f32>, reference_signal: &Vector4<f32>, params: &ModelParameters) -> Vec<[f32; 5]> {
    // function that runs a 4th order Runge Kutta integrator method on nonlinear pendulum
    // dynamics
    // input arguments:
    // initial_condition: Vector4<f32> = [x0, x_dot_0, theta_0, theta_dot_0]
    // t_vec: vector of time points at which to evaluate the solution of the dynamics
    // K: the 4 controller gains for the full-state feedback controller 
    // reference_signal: the reference signal to follow
    // output:
    // a vector containing the solution vector of all 4 state variables across time,
    // including the time points at which the solution was evaluated

    fn compute_ctrl_signal(gains: &Vector4<f32>, reference_signal: &Vector4<f32>, state_vec: &Vector4<f32>) -> f32 {
        // calculates the control signal given the current state of the
        // system, reference signal, and controller gains
        let mut ctrl_signal: f32 = 0f32;
        for i in 0..state_vec.len() {
            let err: f32 = reference_signal[i] - state_vec[i];
            ctrl_signal += gains[i] * err
        }
        return ctrl_signal;
    }
    // pre-allocate the simulation output
    let mut sim_out: Vec<[f32; 5]> = vec![[0.0; 5]; t_vec.len()];
    let h: f32 = t_vec[1] - t_vec[0];   // determine step size from time vector
    let mut x_new: Vector4<f32> = *initial_condition;    // init state vec as initial condition
    for i in 0..t_vec.len() {
        // core rk4 computation
        let mut ctrl_signal = compute_ctrl_signal(&k, &reference_signal, &x_new);
        let k1: Vector4<f32> = nonlinear_dynamics(&x_new, &ctrl_signal, params);
        let y1: Vector4<f32> = x_new + (k1 * (h/2f32));
        ctrl_signal = compute_ctrl_signal(&k, &reference_signal, &y1);
        let k2: Vector4<f32> = nonlinear_dynamics(&y1, &ctrl_signal, params);
        let y2: Vector4<f32> = x_new + (k2 *(h/2f32));
        ctrl_signal = compute_ctrl_signal(&k, &reference_signal, &y2);
        let k3: Vector4<f32> = nonlinear_dynamics(&y2, &ctrl_signal, params);
        let y3: Vector4<f32> = x_new + (k3 * h);
        ctrl_signal = compute_ctrl_signal(&k, &reference_signal, &y3);
        let k4: Vector4<f32> = nonlinear_dynamics(&y3, &ctrl_signal, params);
        x_new += (h/6f32) * (k1 + (2f32 * k2) + (2f32 * k3) + k4);

        // assign new state and time val to sim_out
        sim_out[i][0] = t_vec[i];
        sim_out[i][1..].copy_from_slice(x_new.as_slice());
    }
    return sim_out;
}

pub fn run_physics(initial_condition: &Vector4<f32>, sim_time: &f32, dt: &f32, k: &Vector4<f32>, reference_signal: &Vector4<f32>, params: &ModelParameters) -> Vec<[f32; 5]> {
    // function to handle the rk4 physics simulation
    fn linspace_step(start: f32, end: &f32, step: &f32) -> Vec<f32> {
        // internal method to linspace a vector
        if *step <= 0.0 || *end < start {
            return Vec::new();
        }
        let num_points = ((*end - start) / *step).ceil() as usize;
        let mut linspaced_vec = Vec::with_capacity(num_points);
        let mut val = start;

        while val < *end {
            linspaced_vec.push(val);
            val += step;
        }
        return linspaced_vec;
    }

    let time_vec: Vec<f32> = linspace_step(0f32, sim_time, dt);
    let sim_out: Vec<[f32; 5]> = rk4(initial_condition,
        &time_vec,
        k,
        reference_signal,
        params);
    return sim_out;
}

pub fn cost(reference_signal: &Vector4<f32>, simulation_output: &Vec<[f32; 5]>, weight_vec: &Vector4<f32>) -> f32 {
    // cost function to evaluate the quality of a given controller
    // compares the state variables of the simulation to the reference signal across time
    
    // define kill values and corresponding kill cost to enforce max state deviation and stabilize simulation
    let kill_x: f32 = 10f32;
    let kill_v: f32 = 10f32;
    let kill_theta: f32 = 10f32;
    let kill_theta_dot: f32 = 10f32;
    let thresholds = [kill_x, kill_v, kill_theta, kill_theta_dot];
    let kill_cost: f32 = 1000f32;

    // determine if any state variable exceeds its corresponding threshold
    let exceeds_threshold: bool = simulation_output.iter().any(|arr| {
        arr[1..].iter().zip(thresholds.iter()).any(|(&value, &thresh)| value > thresh)
    });

    if exceeds_threshold {
        // if the simulation is unstable, return the kill cost
        return kill_cost;
    }

    if simulation_output.len() < 2 {
        return 0.0; // Not enough data to calculate dt
    }

    let dt: f32 = simulation_output[1][0] - simulation_output[0][0];
    let mut err_sums = Vector4::<f32>::zeros();

    for data_point in simulation_output {
        err_sums[0] += (reference_signal[0] - data_point[1]).abs();
        err_sums[1] += (reference_signal[1] - data_point[2]).abs();
        err_sums[2] += (reference_signal[2] - data_point[3]).abs();
        err_sums[3] += (reference_signal[3] - data_point[4]).abs();
    }

    let weighted_costs = err_sums.component_mul(weight_vec);
    let total_cost: f32 = dt * weighted_costs.sum();
    
    total_cost
}

pub fn select(population: &Vec<Vector4<f32>>, cost_vals: &Vec<f32>) -> Option<(Vector4<f32>, Vector4<f32>)> {
    // function to select a pair of individuals from a population, based on cost values
    
    // invert cost values to contruct weighted distribution — need f64 because WeightedAliasIndex
    // expects this
     let cost_inverse: Vec<f64> = cost_vals
            .iter()
            .map(|&x| if x > 0.0 { 1.0 / (x as f64) } else { 0.0 }) // avoid div by 0
            .collect();

     // try to construct weighted distribution
        let dist = match WeightedIndex::new(&cost_inverse) {
            Ok(d) => d,
            Err(_) => {
                eprintln!("Warning: Invalid weight distribution, falling back to uniform random.");
                let mut rng = thread_rng();
                let idx1 = rng.gen_range(0..population.len());
                let idx2 = rng.gen_range(0..population.len());
                return Some((population[idx1], population[idx2]));
            }
        };
    let mut rng = thread_rng();

    // select two individuals from the population using weighted random selection
    let idx1 = dist.sample(&mut rng);
    let idx2 = dist.sample(&mut rng);
    Some((population[idx1], population[idx2]))
}

pub fn crossover(pair: &(Vector4<f32>, Vector4<f32>)) -> Vector4<f32> {
    // function to average a pair of individuals
    let a: Vector4<f32> = pair.0;
    let b: Vector4<f32> = pair.1;
    let child: Vector4<f32> = (a + b) / 2f32;
    return child;
}

pub fn mutate(individual: &mut Vector4<f32>, stochasticity: &f32, lsl: i32, usl: i32) {
    // function to mutate an individual in-place based on the randomness specified by the genetic algorithm
    let dist = Normal::new(0.0, *stochasticity as f64).unwrap();    // construct normal distribution
    for x in individual.iter_mut() {
        *x += dist.sample(&mut thread_rng()) as f32;
        if *x > usl as f32 { *x = usl as f32; }
        else if *x < lsl as f32 {*x = lsl as f32; }
    }
}

pub fn generate_individual(lsl: i32, usl: i32) -> Vector4<f32> {
    // function to generate a random individual from a bounded uniform distribution
    let dist = Uniform::new(lsl as f32, usl as f32);
    let mut rng = thread_rng();
    let individual = Vector4::new(
        dist.sample(&mut rng),
        dist.sample(&mut rng),
        dist.sample(&mut rng),
        dist.sample(&mut rng)
    );
    return individual;
}

pub fn generate_population(population_size: usize,
    // function to generate an population of individuals within a range
    lsl: i32,
    usl: i32) -> Vec<Vector4<f32>> {
    let mut population: Vec<Vector4<f32>> = Vec::with_capacity(population_size);
    for _i in 0..population_size {
        let individual: Vector4<f32> = generate_individual(lsl, usl);
        population.push(individual);
    }
    return population;
}

