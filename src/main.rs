use eframe::egui;
use egui_plot::{Bar, BarChart, Line, Plot, PlotPoints};
mod inverted_pendulum;

fn main() -> Result<(), eframe::Error> {
    let native_options = eframe::NativeOptions::default();
    eframe::run_native(
        "My egui App",
        native_options,
        Box::new(|_cc| Box::new(MyApp::default())),
    )
}

// We derive Deserialize/Serialize so we can persist app state on shutdown.
struct MyApp {
    // GA Parameters
    population_size: usize,
    mutation_rate: f32,

    // Pendulum Parameters
    cart_mass: f64,
    pole_mass: f64,
    pole_length: f64,

    // Dummy data for plots
    cost_points: Vec<[f64; 2]>,
    angle_points: Vec<[f64; 2]>,
    angle_vel_points: Vec<[f64; 2]>,
    pos_points: Vec<[f64; 2]>,
    pos_vel_points: Vec<[f64; 2]>,
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
            population_size: 100,
            mutation_rate: 0.10,
            cart_mass: 1.0,
            pole_mass: 0.5,
            pole_length: 1.0,
            cost_points,
            angle_points,
            angle_vel_points,
            pos_points,
            pos_vel_points,
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
                if ui.button("Start GA Optimization").clicked() { /* Logic to start GA */ }
                if ui.button("Stop").clicked() { /* Logic to stop GA */ }
                if ui.button("Reset").clicked() { /* Logic to reset simulation */ }
            });
            ui.separator();
            ui.collapsing("Genetic Algorithm", |ui| {
                ui.add(egui::DragValue::new(&mut self.population_size).prefix("Population Size: "));
                ui.add(egui::Slider::new(&mut self.mutation_rate, 0.0..=1.0).text("Mutation Rate"));
            });
            ui.separator();
            ui.collapsing("Pendulum Model", |ui| {
                ui.add(egui::DragValue::new(&mut self.cart_mass).speed(0.1).prefix("Cart Mass (kg): "));
                ui.add(egui::DragValue::new(&mut self.pole_mass).speed(0.1).prefix("Pole Mass (kg): "));
                ui.add(egui::DragValue::new(&mut self.pole_length).speed(0.1).prefix("Pole Length (m): "));
            });
            ui.separator();
            ui.collapsing("LQR Baseline", |ui| {
                if ui.button("Calculate LQR Solution").clicked() { /* Logic to calculate LQR */ }
                ui.label("LQR Gains: [k1, k2, k3, k4]");
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
                            .height(300.0);
                        cost_plot.show(ui, |plot_ui| {
                            plot_ui.line(Line::new(PlotPoints::from(self.cost_points.clone())));
                        });
                    });

                    // Histograms for controller gains
                    ui.vertical(|ui| {
                        let hist_plot_height = 75.0; // Halved the height to fit four
                        let gain_hist = Plot::new("gain_1_hist")
                            .height(hist_plot_height)
                            .x_axis_label("K_pos")
                            .show_y(false);
                        gain_hist.show(ui, |plot_ui| {
                            let bars: Vec<Bar> = (-5..=5).map(|i| Bar::new(i as f64, (-(i*i) as f64 / 10.0).exp() * 20.0)).collect();
                            plot_ui.bar_chart(BarChart::new(bars));
                        });

                        let gain_hist_2 = Plot::new("gain_2_hist")
                            .height(hist_plot_height)
                            .x_axis_label("K_vel")
                            .show_y(false);
                        gain_hist_2.show(ui, |plot_ui| {
                            let bars: Vec<Bar> = (-5..=5).map(|i| Bar::new(i as f64, (-(i*i-2*i) as f64 / 8.0).exp() * 15.0)).collect();
                            plot_ui.bar_chart(BarChart::new(bars));
                        });

                        let gain_hist_3 = Plot::new("gain_3_hist")
                            .height(hist_plot_height)
                            .x_axis_label("K_angle")
                            .show_y(false);
                        gain_hist_3.show(ui, |plot_ui| {
                            let bars: Vec<Bar> = (-5..=5).map(|i| Bar::new(i as f64, (-(i*i+i) as f64 / 12.0).exp() * 18.0)).collect();
                            plot_ui.bar_chart(BarChart::new(bars));
                        });

                        let gain_hist_4 = Plot::new("gain_4_hist")
                            .height(hist_plot_height)
                            .x_axis_label("K_ang_vel")
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
                            .height(plot_height);
                        angle_plot.show(ui, |plot_ui| {
                            plot_ui.line(Line::new(PlotPoints::from(self.angle_points.clone())));
                        });
                    });
                    ui.group(|ui| {
                        let angle_vel_plot = Plot::new("angle_vel_plot")
                            .x_axis_label("Time [s]")
                            .y_axis_label("Ang. Vel [rad/s]")
                            .width(ui.available_width())
                            .height(plot_height);
                        angle_vel_plot.show(ui, |plot_ui| {
                            plot_ui.line(Line::new(PlotPoints::from(self.angle_vel_points.clone())));
                        });
                    });
                });
                ui.horizontal(|ui| {
                    ui.group(|ui| {
                        let pos_plot = Plot::new("pos_plot")
                            .x_axis_label("Time [s]")
                            .y_axis_label("Position [m]")
                            .width(ui.available_width() / 2f32)
                            .height(plot_height);
                        pos_plot.show(ui, |plot_ui| {
                            plot_ui.line(Line::new(PlotPoints::from(self.pos_points.clone())));
                        });
                    });
                    ui.group(|ui| {
                        let pos_vel_plot = Plot::new("pos_vel_plot")
                            .x_axis_label("Time [s]")
                            .y_axis_label("Velocity [m/s]")
                            .width(ui.available_width())
                            .height(plot_height);
                        pos_vel_plot.show(ui, |plot_ui| {
                            plot_ui.line(Line::new(PlotPoints::from(self.pos_vel_points.clone())));
                        });
                    });
                });
            });
        });
    }
}


