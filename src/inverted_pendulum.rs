// A module containing an implementation of the linear and nonlinear dynamics for an inverted
// pendulum on a cart
mod inverted_pendulum {
use nalgebra::Vector4;

    pub struct ModelParameters {
        // structure to hold model parameters for nonlinear pendulum model
        pub m: f32, // pendulum mass
        pub M: f32, // cart mass
        pub L: f32, // pendulum length
        pub d: f32, // damping coefficient
    }

    pub fn nonlinear_dynamics(state_vec: &Vector4<f32>, ctrl_signal: &f32, params: &ModelParameters) -> Vector4<f32> {
        let m = params.m;
        let M = params.M;
        let L = params.L;
        let d = params.d;
        let g: f32 = -9.813;    // gravity

        // state definition: state_vec = [x, x_dot, theta, theta_dot]
        let sx: f32 = state_vec[2].sin();   // short-hand for sin(theta)
        let cx: f32 = state_vec[2].cos();   // short-hand for cos(theta)
        let denom: f32 = m * L * L * (M + m * (1f32 - (cx.powi(2))));   // denominator of expression

        let mut dx = Vector4::<f32>::zeros();  // allocate the output
        
        // define dynamics
        dx[0] = state_vec[1];
        dx[1] = (1f32 / denom) * ((-m.powi(2)) * L.powi(2) * g * cx * sx + m * L.powi(2) * (m * L * (state_vec[3].powi(2)) * sx - d * state_vec[1])) + m * L * L * (1f32 / denom) * ctrl_signal;
        dx[2] = state_vec[3];
        dx[3] = (1f32 / denom) * ((m + M) * m * g * L * sx - m * L * cx * (m * L * (state_vec[3].powi(2)) * sx - d * state_vec[1])) - m * L * cx * (1f32 / denom) * ctrl_signal;
        return dx;  // return state derivatives
    }

    fn rk4(initial_condition: Vector4<f32>, t_vec: Vec<f32>, k: Vector4<f32>, reference_signal: Vector4<f32>, params: &ModelParameters) -> Vec<[f32; 5]> {
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
        let mut sim_out: Vec<[f32; 5]> = Vec::with_capacity(t_vec.len());
        let h: f32 = t_vec[1] - t_vec[0];   // determine step size from time vector
        let mut x_new: Vector4<f32> = initial_condition;    // init state vec as initial condition
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
}
