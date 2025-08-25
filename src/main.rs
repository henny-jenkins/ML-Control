use eframe::{egui, self};
use egui_plot::{Bar, BarChart, Line, Plot, PlotPoints};
use egui::Color32;
mod inverted_pendulum;

fn run_ga(gui_data: &mut MyApp) {
    // function to handle the evolution of a genetic algorithm
    while !gui_data.stop_simulation_flag {
        //
    }
}

fn evolve(mut prv_generation: Vec<nalgebra::Vector4<f32>>,
    mut cost_vals: Vec<f32>,
    prv_best_individual: nalgebra::Vector4<f32>,
    prv_best_cost: f32,
    gui_data: &MyApp) -> (Vec<nalgebra::Vector4<f32>>, Vec<f32>, f32, nalgebra::Vector4<f32>) {
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

    // define weight vector for cost function
    let weight_vec = nalgebra::Vector4::new(1f32, 0.01f32, 10f32, 5f32);

    // initialize the next generation as the previous generation
    // elitism already handled, so determine the rest of the population
    for i in (*num_elitism)..prv_generation.len() {
        // select parents for ith individual
        let parents: (nalgebra::Vector4<f32>, nalgebra::Vector4<f32>) = inverted_pendulum::select(&prv_generation, &cost_vals);
        let mut child: nalgebra::Vector4<f32> = inverted_pendulum::crossover(&parents); // crossover
        inverted_pendulum::mutate(&mut child, stochasticity); // mutate child
        prv_generation[i] = child; // slot in next individual
        // evaluate fitness
        let child_performance: Vec<[f32; 5]> = inverted_pendulum::run_physics(&initial_state, &t_end, &dt, &child, &reference_state, &params);
        let child_cost: f32 = inverted_pendulum::cost(&reference_state, &child_performance, &weight_vec);
        cost_vals[i] = child_cost;  // slot in next individual's cost
    }

    // rank order the population in terms of cost
    let mut paired: Vec<_> = prv_generation.iter().cloned().zip(cost_vals.iter().cloned()).collect();
    paired.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
    let (next_generation, cost_vals): (Vec<_>, Vec<_>) = paired.into_iter().unzip();

    // identify the best individual and corresponding cost (?)
    let lowest_cost_this_gen: f32 = cost_vals.iter().cloned().fold(f32::INFINITY, |a, b| a.min(b));
    let best_cost: f32 = prv_best_cost.min(lowest_cost_this_gen);
    let mut best_individual = prv_best_individual;
    if best_cost < prv_best_cost {
        let best_idx: usize = cost_vals.iter().position(|&x| x == best_cost).unwrap();
        best_individual = next_generation[best_idx];
    }

    return (next_generation, cost_vals, best_cost, best_individual);
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

// We derive Deserialize/Serialize so we can persist app state on shutdown.
struct MyApp {
    // Simulation Parameters
    sim_time: f32,
    dt: f32,
    reference_signal: nalgebra::Vector4<f32>,
    initial_condition: nalgebra::Vector4<f32>,

    // GA Parameters
    population_size: usize,
    stochasticity: f32,
    num_elitisim: usize,
    current_generation_num: usize,
    max_generations: usize,
    search_space_lsl: i32,
    search_space_usl: i32,

    // Pendulum Parameters
    params: inverted_pendulum::ModelParameters,

    // Dummy data for plots
    cost_points: Vec<[f64; 2]>,
    lqr_cost: f32,
    angle_points: Vec<[f64; 2]>,
    angle_vel_points: Vec<[f64; 2]>,
    pos_points: Vec<[f64; 2]>,
    pos_vel_points: Vec<[f64; 2]>,

    // application booleans
    lqr_data_available: bool,
    stop_simulation_flag: bool
}

impl Default for MyApp {
    fn default() -> Self {
        // Generate some dummy data for the plots
        let cost_points = (0..=50)
            .map(|i| {
                let x = i as f64;
                // Simulate cost decreasing over generations
                [x, 100.0 * (-x / 20.0).exp() + 10.0 * rand::random::<f64>()]
            })
            .collect();
        
        let angle_points: Vec<[f64; 2]> = (0..=200)
            .map(|i| {
                let t = i as f64 * 0.05;
                // Simulate a damped sine wave for the pendulum angle
                [t, 0.5 * (2.0 * t).cos() * (-t * 0.5).exp()]
            })
            .collect();

        // Dummy data for angular velocity (derivative of angle)
        let angle_vel_points: Vec<[f64; 2]> = angle_points.windows(2).map(|p| {
            let t0 = p[0][0];
            let y0 = p[0][1];
            let t1 = p[1][0];
            let y1 = p[1][1];
            [t0, (y1 - y0) / (t1 - t0)]
        }).collect();

        // Dummy data for position
        let pos_points: Vec<[f64; 2]> = angle_points.iter().map(|p| [p[0], p[1].sin() * 0.2]).collect();

        // Dummy data for position velocity
        let pos_vel_points: Vec<[f64; 2]> = pos_points.windows(2).map(|p| {
            let t0 = p[0][0];
            let y0 = p[0][1];
            let t1 = p[1][0];
            let y1 = p[1][1];
            [t0, (y1 - y0) / (t1 - t0)]
        }).collect();

        Self {
            sim_time: 10.0,
            dt: 0.01,
            reference_signal: nalgebra::Vector4::new(1f32, 0f32, 3.14f32, 0f32),
            initial_condition: nalgebra::Vector4::new(-1f32, 0f32, 3.15f32, 0f32),
            population_size: 25,
            stochasticity: 250f32,
            num_elitisim: 5,
            current_generation_num: 0,
            max_generations: 1000,
            search_space_lsl: -2000i32,
            search_space_usl: 2000i32,
            params: inverted_pendulum::ModelParameters(1f32, 5f32, 2f32, 1f32),
            cost_points,
            lqr_cost: 0f32,
            angle_points,
            angle_vel_points,
            pos_points,
            pos_vel_points,
            lqr_data_available: false,
            stop_simulation_flag: false,
        }
    }
}

impl eframe::App for MyApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        ctx.set_visuals(egui::Visuals::dark());

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
                    if ui.button("Start GA").clicked() { /* Logic to start GA */ }
                    if ui.button("Stop").clicked() { self.stop_simulation_flag = true; }
                    if ui.button("Test Function").clicked() {
                        println!("test function clicked");
                        let mut ind1 = nalgebra::Vector4::new(50f32, 100f32, 150f32, 200f32);
                        inverted_pendulum::mutate(&mut ind1, &self.stochasticity);
                        println!("{:?}", ind1);
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
                ui.add(egui::Slider::new(&mut self.num_elitisim, 0..=self.population_size).text("Elitism"));
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
                    self.pos_points.clear();
                    self.pos_vel_points.clear();
                    self.angle_points.clear();
                    self.angle_vel_points.clear();

                    for data_point in sim_out {
                        let time = data_point[0] as f64;
                        let pos = data_point[1] as f64;
                        let pos_vel = data_point[2] as f64;
                        let angle = data_point[3] as f64;
                        let angle_vel = data_point[4] as f64;

                        self.pos_points.push([time, pos]);
                        self.pos_vel_points.push([time, pos_vel]);
                        self.angle_points.push([time, angle]);
                        self.angle_vel_points.push([time, angle_vel]);
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
                    ui.label("Generation: 42");
                    ui.label("Best Cost: 15.73");
                    ui.label("Elapsed Time: 34s");
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
                            plot_ui.line(Line::new(PlotPoints::from(self.cost_points.clone())));

                            if self.lqr_data_available {
                                let lqr_line = Line::new(vec![[0.0, self.lqr_cost as f64], [self.max_generations as f64, self.lqr_cost as f64]])
                                    .color(Color32::GOLD)
                                    .width(2.0)
                                    .name("LQR Cost");
                                plot_ui.line(lqr_line);
                            }
                        });
                    });

                    // Histograms for controller gains
                    ui.vertical(|ui| {
                        let hist_plot_height = 75.0; // Halved the height to fit four
                        let gain_hist = Plot::new("gain_1_hist")
                            .height(hist_plot_height)
                            .x_axis_label("K_pos")
                            .show_x(true)
                            .show_y(false);
                        gain_hist.show(ui, |plot_ui| {
                            let bars: Vec<Bar> = (-5..=5).map(|i| Bar::new(i as f64, (-(i*i) as f64 / 10.0).exp() * 20.0)).collect();
                            plot_ui.bar_chart(BarChart::new(bars));
                        });

                        let gain_hist_2 = Plot::new("gain_2_hist")
                            .height(hist_plot_height)
                            .x_axis_label("K_vel")
                            .show_x(true)
                            .show_y(false);
                        gain_hist_2.show(ui, |plot_ui| {
                            let bars: Vec<Bar> = (-5..=5).map(|i| Bar::new(i as f64, (-(i*i-2*i) as f64 / 8.0).exp() * 15.0)).collect();
                            plot_ui.bar_chart(BarChart::new(bars));
                        });

                        let gain_hist_3 = Plot::new("gain_3_hist")
                            .height(hist_plot_height)
                            .x_axis_label("K_angle")
                            .show_x(true)
                            .show_y(false);
                        gain_hist_3.show(ui, |plot_ui| {
                            let bars: Vec<Bar> = (-5..=5).map(|i| Bar::new(i as f64, (-(i*i+i) as f64 / 12.0).exp() * 18.0)).collect();
                            plot_ui.bar_chart(BarChart::new(bars));
                        });

                        let gain_hist_4 = Plot::new("gain_4_hist")
                            .height(hist_plot_height)
                            .x_axis_label("K_ang_vel")
                            .show_x(true)
                            .show_y(false);
                        gain_hist_4.show(ui, |plot_ui| {
                            let bars: Vec<Bar> = (-5..=5).map(|i| Bar::new(i as f64, (-(i*i-3*i) as f64 / 9.0).exp() * 22.0)).collect();
                            plot_ui.bar_chart(BarChart::new(bars));
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
                            let line = Line::new(PlotPoints::from(self.angle_points.clone()))
                                .color(Color32::GOLD);
                            if self.lqr_data_available {
                                plot_ui.line(line.name("LQR Solution"));
                            } else {
                                plot_ui.line(line);
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
                            let line = Line::new(PlotPoints::from(self.angle_vel_points.clone()))
                                .color(Color32::GOLD);
                            if self.lqr_data_available {
                                plot_ui.line(line.name("LQR Solution"));
                            } else {
                                plot_ui.line(line);
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
                            let line = Line::new(PlotPoints::from(self.pos_points.clone()))
                                .color(Color32::GOLD);
                            if self.lqr_data_available {
                                plot_ui.line(line.name("LQR Solution"));
                            } else {
                                plot_ui.line(line);
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
                            let line = Line::new(PlotPoints::from(self.pos_vel_points.clone()))
                                .color(Color32::GOLD);
                            if self.lqr_data_available {
                                plot_ui.line(line.name("LQR Solution"));
                            } else {
                                plot_ui.line(line);
                            }
                        });
                    });
                });
            });
        });
    }
}


