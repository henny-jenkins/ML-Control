// A module containing an implementation of the linear and nonlinear dynamics for an inverted
// pendulum on a cart

use nalgebra::Vector4;
use rand::prelude::*;
use rand_distr::{Normal, Distribution, WeightedIndex};

pub struct ModelParameters(pub f32, pub f32, pub f32, pub f32);

pub fn nonlinear_dynamics(state_vec: &Vector4<f32>, ctrl_signal: &f32, params: &ModelParameters) -> Vector4<f32> {
    let pendulum_mass = params.0;
    let cart_mass = params.1;
    let pendulum_length = params.2;
    let damping = params.3;
    let g: f32 = -9.813;    // gravity

    // state definition: state_vec = [x, x_dot, theta, theta_dot]
    let sx: f32 = state_vec[2].sin();   // short-hand for sin(theta)
    let cx: f32 = state_vec[2].cos();   // short-hand for cos(theta)
    let denom: f32 = pendulum_mass * pendulum_length * pendulum_length * (cart_mass + pendulum_mass * (1f32 - (cx.powi(2))));   // denominator of expression

    let mut dx = Vector4::<f32>::zeros();  // allocate the output

    // define dynamics
    dx[0] = state_vec[1];
    dx[1] = (1f32 / denom) * ((-pendulum_mass.powi(2)) * pendulum_length.powi(2) * g * cx * sx + pendulum_mass * pendulum_length.powi(2) * (pendulum_mass * pendulum_length * (state_vec[3].powi(2)) * sx - damping * state_vec[1])) + pendulum_mass * pendulum_length * pendulum_length * (1f32 / denom) * ctrl_signal;
    dx[2] = state_vec[3];
    dx[3] = (1f32 / denom) * ((pendulum_mass + cart_mass) * pendulum_mass * g * pendulum_length * sx - pendulum_mass * pendulum_length * cx * (pendulum_mass * pendulum_length * (state_vec[3].powi(2)) * sx - damping * state_vec[1])) - pendulum_mass * pendulum_length * cx * (1f32 / denom) * ctrl_signal;
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
        let mut linspaced_vec = Vec::new();
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
        println!("simulation exceeds thresholds of valid state values");
        // if the simulation is unstable, return the kill cost
        return kill_cost;
    }

    // initialize some error vectors to fill
    let mut err_x = Vec::with_capacity(simulation_output.len());
    let mut err_v = Vec::with_capacity(simulation_output.len());
    let mut err_theta = Vec::with_capacity(simulation_output.len());
    let mut err_theta_dot = Vec::with_capacity(simulation_output.len());

    for array in simulation_output {
        let slice = &array[1..];    // pull out the last 4 values of each array
        // err = ref - x (do this for each state variable)
        // load the error vectors, taking absolute value to get absolute error
        err_x.push((reference_signal[0] - slice[0]).abs());
        err_v.push((reference_signal[1] - slice[1]).abs());
        err_theta.push((reference_signal[2] - slice[2]).abs());
        err_theta_dot.push((reference_signal[3] - slice[3]).abs());
    }

    // use the error vectors with the weights to calculate cost
    // sum the errors in the vector (lazy "integrate") and multiply by the weights
    let cost_x: f32 = weight_vec[0] * err_x.iter().sum::<f32>();
    let cost_v: f32 = weight_vec[1] * err_v.iter().sum::<f32>();
    let cost_theta: f32 = weight_vec[2] * err_theta.iter().sum::<f32>();
    let cost_theta_dot: f32 = weight_vec[3] * err_theta_dot.iter().sum::<f32>();

    let total_cost: f32 = cost_x + cost_v + cost_theta + cost_theta_dot;
    println!("total cost: {}", total_cost);
    return total_cost;
}

fn select(population: &[Vector4<f32>], cost_vals: &[f32]) -> (Vector4<f32>, Vector4<f32>) {
    // function to select a pair of individuals from a population, based on cost values
    
    // invert cost values to contruct weighted distribution — need f64 because WeightedAliasIndex
    // expects this
    let cost_inverse: Vec<f64> = cost_vals.iter().map(|x| 1.0 / (*x as f64)).collect();
    let dist = WeightedIndex::new(&cost_inverse).unwrap();
    let mut rng = thread_rng();

    // select two individuals from the population using weighted random selection
    let idx1 = dist.sample(&mut rng);
    let idx2 = dist.sample(&mut rng);
    let result: (Vector4<f32>, Vector4<f32>) = (population[idx1], population[idx2]);
    return result;
}

pub fn crossover(pair: &(Vector4<f32>, Vector4<f32>)) -> Vector4<f32> {
    // function to average a pair of individuals
    let a: Vector4<f32> = pair.0;
    let b: Vector4<f32> = pair.1;
    let child: Vector4<f32> = (a + b) / 2f32;
    return child;
}

pub fn mutate(individual: &Vector4<f32>, stochasticity: f32) -> Vector4<f32> {
    // function to mutate an individual based on the randomness specified by the genetic algorithm
    let dist = Normal::new(0.0, stochasticity as f64).unwrap();    // construct normal distribution
    Vector4::from_iterator(individual.iter().map(|x| x + (dist.sample(&mut thread_rng()) as f32)))
}

// pub fn evolution_stepper(prv_generation: &[Vector4<f32>],
//     prv_best_individual: Vector4<f32>,
//     prv_best_cost: f32,
//     current_gen_num: usize,
//     num_elitism: i32,
//     population_size: usize,
//     t_end: f32,
//     stochasticity: f32,
//     reference_state: Vector4<f32>,
//     initial_state: Vector4<f32>) -> () {
//     // function to evolve the population of individuals by a single generation
// }
