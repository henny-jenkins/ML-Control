use std::time::Instant;
use eframe::{egui, self};
use egui_plot::{Line, Plot, PlotBounds, PlotPoints, Points};
use egui::Color32;
mod inverted_pendulum;

fn evolve(mut prv_generation: Vec<nalgebra::Vector4<f32>>,
    mut cost_vals: Vec<f32>,
    gui_data: &MyApp) -> (Vec<nalgebra::Vector4<f32>>, Vec<f32>) {
    // function to evolve the population of individuals by a single generation
    // prv_generation is ranked in order of fitness (first element is most fit)

    // pull out GUI data
    let num_elitism: &usize = &gui_data.num_elitisim;
    let stochasticity: &f32 = &gui_data.stochasticity;
    let initial_state: &nalgebra::Vector4<f32> = &gui_data.initial_condition;
    let reference_state: &nalgebra::Vector4<f32> = &gui_data.reference_signal;
    let t_end: &f32 = &gui_data.sim_time;
    let dt: &f32 = &gui_data.dt;
    let params: &inverted_pendulum::ModelParameters = &gui_data.params;
    let lsl = gui_data.search_space_lsl;
    let usl = gui_data.search_space_usl;

    // define weight vector for cost function
    let weight_vec = nalgebra::Vector4::new(1f32, 0.01f32, 10f32, 5f32);

    // initialize the next generation as the previous generation
    // elitism already handled, so determine the rest of the population
    for i in (*num_elitism)..prv_generation.len() {
        // select parents for ith individual
        let parents: (nalgebra::Vector4<f32>, nalgebra::Vector4<f32>) = inverted_pendulum::select(&prv_generation, &cost_vals).unwrap();
        let mut child: nalgebra::Vector4<f32> = inverted_pendulum::crossover(&parents); // crossover
        inverted_pendulum::mutate(&mut child, stochasticity, lsl, usl); // mutate child
        prv_generation[i] = child; // slot in next individual
        // evaluate fitness
        let child_performance: Vec<[f32; 5]> = inverted_pendulum::run_physics(&initial_state, &t_end, &dt, &child, &reference_state, &params);
        let child_cost: f32 = inverted_pendulum::cost(&reference_state, &child_performance, &weight_vec);
        // if child_cost == 1000f32 {
        //     println!("invalid state values detected for individual: {}", child);
        // }
        cost_vals[i] = child_cost;  // slot in next individual's cost
    }

    // rank order the population in terms of cost
    let mut paired: Vec<_> = prv_generation
        .into_iter()
        .zip(cost_vals.into_iter())
        .map(|(ind, cost)| {
            let safe_cost = if cost.is_finite() { cost } else { f32::MAX };
            (ind, safe_cost)
        })
        .collect();

    paired.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
    let (next_generation, cost_vals): (Vec<_>, Vec<_>) = paired.into_iter().unzip();

    return (next_generation, cost_vals);
}

fn main() -> Result<(), eframe::Error> {
    let native_options = eframe::NativeOptions{
        viewport: egui::ViewportBuilder::default().with_inner_size([1280.0, 1024.0]),
        ..Default::default()
    };
    eframe::run_native(
        "My egui App",
        native_options,
        Box::new(|_cc| Box::new(MyApp::default())),
    )
}

#[derive(PartialEq)]
enum SimState {
    Running,
    Paused,
    Finished,
    Idle
}

struct MyApp {
    // Simulation Parameters
    sim_time: f32,
    dt: f32,
    reference_signal: nalgebra::Vector4<f32>,
    initial_condition: nalgebra::Vector4<f32>,
    start_sim_time: Option<Instant>,

    // GA Parameters
    population_size: usize,
    stochasticity: f32,
    num_elitisim: usize,
    current_generation_num: usize,
    max_generations: usize,
    search_space_lsl: i32,
    search_space_usl: i32,

    // GA state of current generation
    current_population_sorted: Option<Vec<nalgebra::Vector4<f32>>>,
    current_costs_sorted: Option<Vec<f32>>,
    best_cost: f32,
    best_individual: Option<nalgebra::Vector4<f32>>,
    elapsed_sim_time: f32,

    // Pendulum Parameters
    params: inverted_pendulum::ModelParameters,

    // plot data
    cost_points: Vec<[f64; 2]>,
    lqr_cost: f32,
    lqr_angle_points: Vec<[f64; 2]>,
    lqr_angle_vel_points: Vec<[f64; 2]>,
    lqr_pos_points: Vec<[f64; 2]>,
    lqr_vel_points: Vec<[f64; 2]>,
    angle_points: Vec<[f64; 2]>,
    angle_vel_points: Vec<[f64; 2]>,
    pos_points: Vec<[f64; 2]>,
    vel_points: Vec<[f64; 2]>,

    // other app state variables
    lqr_data_available: bool,
    sim_state: SimState,
}

impl Default for MyApp {
    fn default() -> Self {
        Self {
            sim_time: 10.0,
            dt: 0.01,
            reference_signal: nalgebra::Vector4::new(1f32, 0f32, 3.14f32, 0f32),
            initial_condition: nalgebra::Vector4::new(-1f32, 0f32, 3.15f32, 0f32),
            start_sim_time: None,
            population_size: 25,
            stochasticity: 250f32,
            num_elitisim: 5,
            current_generation_num: 0,
            max_generations: 1000,
            search_space_lsl: -2000i32,
            search_space_usl: 2000i32,
            current_population_sorted: None,
            current_costs_sorted: None,
            best_cost: f32::INFINITY,
            best_individual: None,
            elapsed_sim_time: 0f32,
            params: inverted_pendulum::ModelParameters(1f32, 5f32, 2f32, 1f32),
            cost_points: Vec::new(),
            lqr_cost: f32::INFINITY,
            lqr_angle_points: Vec::new(),
            lqr_angle_vel_points: Vec::new(),
            lqr_pos_points: Vec::new(),
            lqr_vel_points: Vec::new(),
            angle_points: Vec::new(),
            angle_vel_points: Vec::new(),
            pos_points: Vec::new(),
            vel_points: Vec::new(),
            lqr_data_available: false,
            sim_state: SimState::Idle,
        }
    }
}

impl eframe::App for MyApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        ctx.set_visuals(egui::Visuals::dark());

        // code to evolve the GA if appropriate
        if (self.sim_state == SimState::Running) && (self.current_generation_num < self.max_generations) {
            // evolve the next generation for the next frame
            if let (Some(pop), Some(costs)) = (self.current_population_sorted.take(), self.current_costs_sorted.take()) {
                let (next_gen, next_costs) = evolve(pop, costs, self);
                self.current_generation_num += 1;
                
                // update the GUI and app state
                /*
                 * elitism is always positive integer, so the best global individual & cost will always be in the
                 * current generation — the algorithm will never regress in performance
                */
                self.best_cost = next_costs[0];
                self.best_individual = Some(next_gen[0]);

                let best_simulation_output = inverted_pendulum::run_physics(&self.initial_condition,
                    &self.sim_time,
                    &self.dt,
                    &next_gen[0],
                    &self.reference_signal,
                    &self.params);

                // pull out the simulation output and assign to app state
                let mut x_pts = Vec::with_capacity(best_simulation_output.len());
                let mut v_pts = Vec::with_capacity(best_simulation_output.len());
                let mut theta_pts = Vec::with_capacity(best_simulation_output.len());
                let mut theta_dot_pts = Vec::with_capacity(best_simulation_output.len());
                for array in best_simulation_output {
                    // load the state vectors
                    let slice = &array;
                    let current_time = slice[0];
                    x_pts.push([current_time as f64, slice[1] as f64]);
                    v_pts.push([current_time as f64, slice[2] as f64]);
                    theta_pts.push([current_time as f64, slice[3] as f64]);
                    theta_dot_pts.push([current_time as f64, slice[4] as f64]);
                }
                self.pos_points = x_pts;
                self.vel_points = v_pts;
                self.angle_points = theta_pts;
                self.angle_vel_points = theta_dot_pts;

                self.cost_points.push([self.current_generation_num as f64, self.best_cost as f64]);
                self.current_population_sorted = Some(next_gen);
                self.current_costs_sorted = Some(next_costs);
                if let Some(local_instant) = self.start_sim_time.take() {
                    self.elapsed_sim_time += local_instant.elapsed().as_secs_f32();
                    self.start_sim_time = Some(Instant::now());
                }
            }
        } else if (self.sim_state == SimState::Running) && (self.current_generation_num >= self.max_generations) {
            self.sim_state = SimState::Finished;
        }

        // --- Side Panel for Configuration ---
        egui::SidePanel::left("config_panel").show(ctx, |ui| {
            ui.heading("Configuration & Controls");
            ui.separator();
            ui.collapsing("Simulation Control", |ui| {
                ui.label("Reference Signal:");
                egui::Grid::new("reference_signal_grid").num_columns(2).striped(true).show(ui, |ui| {
                    ui.label("r_x [m]:");
                    ui.add(egui::DragValue::new(&mut self.reference_signal[0]).speed(0.1));
                    ui.end_row();
                    ui.label("r_v [m/s]:");
                    ui.add(egui::DragValue::new(&mut self.reference_signal[1]).speed(0.1));
                    ui.end_row();
                    ui.label("r_theta [rad]:");
                    ui.add(egui::DragValue::new(&mut self.reference_signal[2]).speed(0.1));
                    ui.end_row();
                    ui.label("r_theta_dot [rad/s]:");
                    ui.add(egui::DragValue::new(&mut self.reference_signal[3]).speed(0.1));
                    ui.end_row();
                });

                ui.separator();

                ui.label("Initial Conditions:");
                egui::Grid::new("initial_condition_grid").num_columns(2).striped(true).show(ui, |ui| {
                    ui.label("x0_x [m]:");
                    ui.add(egui::DragValue::new(&mut self.initial_condition[0]).speed(0.1));
                    ui.end_row();
                    ui.label("x0_v [m/s]:");
                    ui.add(egui::DragValue::new(&mut self.initial_condition[1]).speed(0.1));
                    ui.end_row();
                    ui.label("x0_theta [rad]:");
                    ui.add(egui::DragValue::new(&mut self.initial_condition[2]).speed(0.1));
                    ui.end_row();
                    ui.label("x0_theta_dot [rad/s]:");
                    ui.add(egui::DragValue::new(&mut self.initial_condition[3]).speed(0.1));
                    ui.end_row();
                });
                
                ui.separator();

                // simulation time values
                ui.add(egui::DragValue::new(&mut self.sim_time).speed(0.1).prefix("simulation duration: [s]: "));
                ui.add(egui::DragValue::new(&mut self.dt).speed(0.1).prefix("simulation dt: [s]: "));
                
                ui.horizontal(|ui| {
                    if ui.button("Start GA").clicked() {
                        self.sim_state = SimState::Running;
                        self.current_generation_num = 0;
                        self.cost_points.clear();
                        self.elapsed_sim_time = 0f32;
                        self.start_sim_time = Some(Instant::now());

                        // pull out algorithm config from GUI data
                        let initial_condition = self.initial_condition;
                        let sim_time = self.sim_time;
                        let dt = self.dt;
                        let reference_signal = self.reference_signal;
                        let params = &self.params;

                        // define some other important variables for the simulation
                        let weight_vec = nalgebra::Vector4::new(1f32, 0.01f32, 10f32, 5f32);

                        // initialize the first population and cost vector
                        let mut init_generation: Vec<nalgebra::Vector4<f32>> = inverted_pendulum::generate_population(
                            self.population_size,
                            self.search_space_lsl,
                            self.search_space_usl);
                        let mut cost_vals: Vec<f32> = Vec::with_capacity(self.population_size);

                        // evaluate the first population
                        for i in 0..self.population_size {
                            // run physics for each individual
                            let individual_performance: Vec<[f32; 5]> = inverted_pendulum::run_physics(
                                &initial_condition,
                                &sim_time,
                                &dt,
                                &init_generation[i],
                                &reference_signal,
                                params);
                            // calculate cost for each individual
                            cost_vals.push(inverted_pendulum::cost(
                                &reference_signal,
                                &individual_performance,
                                &weight_vec));
                        }
                        self.current_generation_num += 1;
                        
                        // sort (ascending) the first population in terms of cost
                        let mut paired: Vec<_> = init_generation
                            .into_iter()
                            .zip(cost_vals.into_iter())
                            .map(|(ind, cost)| {
                                let safe_cost = if cost.is_finite() { cost } else { f32::MAX };
                                (ind, safe_cost)
                            })
                            .collect();

                        paired.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
                        (init_generation, cost_vals) = paired.into_iter().unzip();

                        // update the GUI and app state
                        self.cost_points.push([self.current_generation_num as f64, cost_vals[0] as f64]);
                        self.current_population_sorted = Some(init_generation);
                        self.current_costs_sorted = Some(cost_vals);
                    }

                    match self.sim_state {
                        SimState::Running => {
                            if ui.button("Pause").clicked() {
                                self.sim_state = SimState::Paused;
                            }
                        }
                        SimState::Paused => {
                            if ui.button("Resume").clicked() {
                                self.sim_state = SimState::Running;
                            }
                        }
                        SimState::Finished => {
                            egui::Window::new("Simulation Complete")
                                .collapsible(false)
                                .resizable(false)
                                .anchor(egui::Align2::CENTER_CENTER, [0.0, 0.0])
                                .show(ctx, |ui| {
                                    ui.label("The simulation has finished!");
                                    if ui.button("OK").clicked() {
                                        self.sim_state = SimState::Idle;
                                    }
                                });
                        }
                        _ => {}
                    }

                    if ui.button("Reset").clicked() { 
                        *self = Self::default();
                    }
                });
            });
            ui.separator();
            ui.collapsing("Genetic Algorithm", |ui| {
                ui.add(egui::DragValue::new(&mut self.population_size).prefix("Population Size: "));
                ui.add(egui::Slider::new(&mut self.stochasticity, 0.0..=1000f32).text("Stochasticity"));
                ui.add(egui::Slider::new(&mut self.num_elitisim, 1..=self.population_size).text("Elitism"));
                ui.add(egui::Slider::new(&mut self.max_generations, 0..=5000).text("Max Generations"));
                ui.add(egui::Slider::new(&mut self.search_space_lsl, -5000i32..=self.search_space_usl.min(5000i32)).text("Search Space Lower Bound"));
                ui.add(egui::Slider::new(&mut self.search_space_usl, self.search_space_lsl.max(-5000i32)..=5000i32).text("Search Space Upper Bound"));
            });
            ui.separator();
            ui.collapsing("Pendulum Model", |ui| {
                ui.add(egui::DragValue::new(&mut self.params.0).speed(0.1).prefix("Pole Mass [kg]: "));
                ui.add(egui::DragValue::new(&mut self.params.1).speed(0.1).prefix("Cart Mass [kg]: "));
                ui.add(egui::DragValue::new(&mut self.params.2).speed(0.1).prefix("Pole Length [m]: "));
                ui.add(egui::DragValue::new(&mut self.params.3).speed(0.1).prefix("Damping Coefficient [dim]: "));
            });
            ui.separator();
            ui.collapsing("LQR Baseline", |ui| {
                if ui.button("Calculate LQR Solution").clicked() {
                    // define simulation parameters
                    let lqr_gains = nalgebra::Vector4::new(-100f32, -183.2793, 1.6832e03, 646.6130);

                    // run the simulation
                    let sim_out: Vec<[f32; 5]> = inverted_pendulum::run_physics(&self.initial_condition, &self.sim_time, &self.dt, &lqr_gains, &self.reference_signal, &self.params);

                    // calculate the cost of the LQR simulation
                    let wt_vec = nalgebra::Vector4::new(1f32, 0.01f32, 10f32, 5f32);
                    self.lqr_cost = inverted_pendulum::cost(&self.reference_signal, &sim_out, &wt_vec);

                    // plot the simulation
                    self.lqr_pos_points.clear();
                    self.lqr_vel_points.clear();
                    self.lqr_angle_points.clear();
                    self.lqr_angle_vel_points.clear();

                    for data_point in sim_out {
                        let time = data_point[0] as f64;
                        let pos = data_point[1] as f64;
                        let vel = data_point[2] as f64;
                        let angle = data_point[3] as f64;
                        let angle_vel = data_point[4] as f64;

                        self.lqr_pos_points.push([time, pos]);
                        self.lqr_vel_points.push([time, vel]);
                        self.lqr_angle_points.push([time, angle]);
                        self.lqr_angle_vel_points.push([time, angle_vel]);
                    }
                    self.lqr_data_available = true;
                }
                ui.label("LQR Gains: [-100, -183.2793, 1.6832e03, 646.6130]");
            });
        });

        // --- Central Panel for Visualizations ---
        egui::CentralPanel::default().show(ctx, |ui| {
            egui::ScrollArea::vertical().show(ui, |ui| {
                
                // --- LIVE GA MONITORING SECTION ---
                ui.heading("Live GA Monitoring");
                ui.separator();
                ui.horizontal(|ui| {
                    ui.label(format!("Generation: {}", self.current_generation_num));
                    ui.label(format!("Best GA Cost: {}", self.best_cost));
                    ui.label(format!("LQR Cost: {}", self.lqr_cost));
                    if let Some(best_ga_controller) = &self.current_population_sorted {
                        ui.label(format!("Best controller: [{}, {}, {}, {}]", best_ga_controller[0][0], best_ga_controller[0][1], best_ga_controller[0][2], best_ga_controller[0][3],));
                    }
                    ui.label(format!("Elapsed Time: {} [s]", self.elapsed_sim_time));
                });

                ui.horizontal(|ui| {
                    // Cost Plot
                    ui.group(|ui| {

                        let cost_plot = Plot::new("cost_plot")
                            .x_axis_label("Generation")
                            .y_axis_label("Cost")
                            .width(ui.available_width() * 0.6)
                            .height(300.0)
                            .legend(egui_plot::Legend::default());
                        cost_plot.show(ui, |plot_ui| {
                            let x_min = 0.0;
                            let x_max = self.max_generations as f64;

                            // Determine y bounds
                            let (y_min, y_max) = if !self.cost_points.is_empty() {
                                let mut y_min = self.cost_points.iter().map(|p| p[1]).fold(f64::INFINITY, f64::min);
                                let mut y_max = self.cost_points.iter().map(|p| p[1]).fold(f64::NEG_INFINITY, f64::max);

                                if self.lqr_data_available {
                                    y_min = y_min.min(self.lqr_cost as f64);
                                    y_max = y_max.max(self.lqr_cost as f64);
                                }

                                (y_min, y_max)
                            } else if self.lqr_data_available {
                                // Only LQR line exists
                                let y = self.lqr_cost as f64;
                                (y - 1.0, y + 1.0) // small range around LQR cost
                            } else {
                                (0.0, 1.0) // default bounds if nothing exists
                            };

                            plot_ui.set_plot_bounds(PlotBounds::from_min_max([x_min, y_min], [x_max, y_max]));

                            // Plot GA cost
                            if !self.cost_points.is_empty() {
                                plot_ui.line(Line::new(PlotPoints::from(self.cost_points.clone())));
                            }

                            // Plot LQR cost
                            if self.lqr_data_available {
                                let lqr_line = Line::new(vec![[0.0, self.lqr_cost as f64], [x_max, self.lqr_cost as f64]])
                                    .color(Color32::GOLD)
                                    .width(2.0)
                                    .name("LQR Cost");
                                plot_ui.line(lqr_line);
                            }
                        });
                    });

                    // 2d projections for controller gains
                    ui.vertical(|ui| {
                        let proj_plot_1 = Plot::new("x_v_proj")
                            .x_axis_label("k_x")
                            .y_axis_label("k_v")
                            .show_x(true)
                            .show_y(true)
                            .height(ui.available_height() / 2f32)
                            .legend(egui_plot::Legend::default().position(egui_plot::Corner::LeftTop));

                        proj_plot_1.show(ui, |plot_ui| {
                            // pull out 1st & 2nd element from the current population
                            if self.current_population_sorted.is_some() {
                                let pts_data: Vec<[f64; 2]> = self.current_population_sorted
                                    .clone()
                                    .unwrap()
                                    .iter()
                                    .map(|x| [x[0] as f64, x[1] as f64])
                                    .collect();
                                let proj_pts_1 = Points::new(pts_data)
                                    .radius(3.0);
                                plot_ui.points(proj_pts_1.name("GA Population"));
                            }

                            let lqr_soln = Points::new([-100f64, -183f64])
                                .radius(3.0)
                                .color(Color32::GOLD)
                                .name("LQR Solution");
                            if self.lqr_data_available { plot_ui.points(lqr_soln); }
                            plot_ui.set_plot_bounds(PlotBounds::from_min_max([self.search_space_lsl as f64, self.search_space_lsl as f64], [self.search_space_usl as f64, self.search_space_usl as f64]));
                        });

                        let proj_plot_2 = Plot::new("theta_theta_dot_proj")
                            .x_axis_label("k_theta")
                            .y_axis_label("k_theta_dot")
                            .show_x(true)
                            .show_y(true)
                            .height(ui.available_height())
                            .legend(egui_plot::Legend::default().position(egui_plot::Corner::LeftTop));


                        proj_plot_2.show(ui, |plot_ui| {
                            // pull out 3rd & 4th element from the current population
                            if self.current_population_sorted.is_some() {
                                let pts_data: Vec<[f64; 2]> = self.current_population_sorted
                                    .clone()
                                    .unwrap()
                                    .iter()
                                    .map(|x| [x[2] as f64, x[3] as f64])
                                    .collect();
                                let proj_pts_2 = Points::new(pts_data)
                                    .radius(3.0);
                                plot_ui.set_plot_bounds(PlotBounds::from_min_max([self.search_space_lsl as f64, self.search_space_lsl as f64], [self.search_space_usl as f64, self.search_space_usl as f64]));
                                plot_ui.points(proj_pts_2.name("GA Population"));
                            }

                            let lqr_soln = Points::new([1683.2f64, 646.6130f64])
                                .radius(3.0)
                                .color(Color32::GOLD)
                                .name("LQR Solution");
                            if self.lqr_data_available { plot_ui.points(lqr_soln); }
                            plot_ui.set_plot_bounds(PlotBounds::from_min_max([self.search_space_lsl as f64, self.search_space_lsl as f64], [self.search_space_usl as f64, self.search_space_usl as f64]));
                        });
                    });
                });

                ui.add_space(20.0);

                // --- SIMULATION & RESULTS SECTION ---
                ui.heading("Final Controller Performance");
                ui.separator();

                let plot_height = 250.0;

                ui.horizontal(|ui| {
                    ui.group(|ui| {
                        let angle_plot = Plot::new("angle_plot")
                            .x_axis_label("Time [s]")
                            .y_axis_label("Angle [rad]")
                            .width(ui.available_width() / 2f32)
                            .height(plot_height)
                            .show_x(true)
                            .legend(egui_plot::Legend::default());
                        angle_plot.show(ui, |plot_ui| {
                            let lqr_line = Line::new(PlotPoints::from(self.lqr_angle_points.clone()))
                                .color(Color32::GOLD);
                            let ga_line = Line::new(PlotPoints::from(self.angle_points.clone()))
                                .color(Color32::RED);
                            let ref_line = Line::new(vec![[0.0, self.reference_signal[2] as f64], [self.sim_time as f64, self.reference_signal[2] as f64]])
                                .color(Color32::GREEN);
                            plot_ui.line(ref_line.name("Reference"));

                            if self.best_individual.is_some() {
                                plot_ui.line(ga_line.name("GA Solution"));
                            } else {
                                plot_ui.line(ga_line);
                            }
                            if self.lqr_data_available {
                                plot_ui.line(lqr_line.name("LQR Solution"));
                            } else {
                                plot_ui.line(lqr_line);
                            }
                        });
                    });
                    ui.group(|ui| {
                        let angle_vel_plot = Plot::new("angle_vel_plot")
                            .x_axis_label("Time [s]")
                            .y_axis_label("Ang. Vel [rad/s]")
                            .width(ui.available_width())
                            .height(plot_height)
                            .show_x(true)
                            .legend(egui_plot::Legend::default());
                        angle_vel_plot.show(ui, |plot_ui| {
                            let lqr_line = Line::new(PlotPoints::from(self.lqr_angle_vel_points.clone()))
                                .color(Color32::GOLD);
                            let ga_line = Line::new(PlotPoints::from(self.angle_vel_points.clone()))
                                .color(Color32::RED);
                            let ref_line = Line::new(vec![[0.0, self.reference_signal[3] as f64], [self.sim_time as f64, self.reference_signal[3] as f64]])
                                .color(Color32::GREEN);
                            plot_ui.line(ref_line.name("Reference"));

                            if self.best_individual.is_some() {
                                plot_ui.line(ga_line.name("GA Solution"));
                            } else {
                                plot_ui.line(ga_line);
                            }
                            if self.lqr_data_available {
                                plot_ui.line(lqr_line.name("LQR Solution"));
                            } else {
                                plot_ui.line(lqr_line);
                            }
                        });
                    });
                });
                ui.horizontal(|ui| {
                    ui.group(|ui| {
                        let pos_plot = Plot::new("pos_plot")
                            .x_axis_label("Time [s]")
                            .y_axis_label("Position [m]")
                            .width(ui.available_width() / 2f32)
                            .height(plot_height)
                            .show_x(true)
                            .legend(egui_plot::Legend::default());
                        pos_plot.show(ui, |plot_ui| {
                            let lqr_pos_line = Line::new(PlotPoints::from(self.lqr_pos_points.clone()))
                                .color(Color32::GOLD);
                            let ga_line = Line::new(PlotPoints::from(self.pos_points.clone()))
                                .color(Color32::RED);
                            let ref_line = Line::new(vec![[0.0, self.reference_signal[0] as f64], [self.sim_time as f64, self.reference_signal[0] as f64]])
                                .color(Color32::GREEN);
                            plot_ui.line(ref_line.name("Reference"));

                            if self.best_individual.is_some() {
                                plot_ui.line(ga_line.name("GA Solution"));
                            } else {
                                plot_ui.line(ga_line);
                            }
                            if self.lqr_data_available {
                                plot_ui.line(lqr_pos_line.name("LQR Solution"));
                            } else {
                                plot_ui.line(lqr_pos_line);
                            }
                        });
                    });
                    ui.group(|ui| {
                        let pos_vel_plot = Plot::new("pos_vel_plot")
                            .x_axis_label("Time [s]")
                            .y_axis_label("Velocity [m/s]")
                            .width(ui.available_width())
                            .height(plot_height)
                            .show_x(true)
                            .legend(egui_plot::Legend::default());
                        pos_vel_plot.show(ui, |plot_ui| {
                            let lqr_line_vel = Line::new(PlotPoints::from(self.lqr_vel_points.clone()))
                                .color(Color32::GOLD);
                            let ga_line = Line::new(PlotPoints::from(self.vel_points.clone()))
                                .color(Color32::RED);
                            let ref_line = Line::new(vec![[0.0, self.reference_signal[1] as f64], [self.sim_time as f64, self.reference_signal[1] as f64]])
                                .color(Color32::GREEN);
                            plot_ui.line(ref_line.name("Reference"));

                            if self.best_individual.is_some() {
                                plot_ui.line(ga_line.name("GA Solution"));
                            } else {
                                plot_ui.line(ga_line);
                            }
                            if self.lqr_data_available {
                                plot_ui.line(lqr_line_vel.name("LQR Solution"));
                            } else {
                                plot_ui.line(lqr_line_vel);
                            }
                        });
                    });
                });
            });
        });
        ctx.request_repaint();  // redraw the window even if idle
    }
}


