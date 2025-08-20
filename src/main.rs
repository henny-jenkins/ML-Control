use eframe::egui;
use egui_plot::{Line, Plot, PlotPoints};

fn main() -> Result<(), eframe::Error> {
    let native_options = eframe::NativeOptions::default();
    eframe::run_native(
        "My egui App",
        native_options,
        Box::new(|_cc| Box::new(MyApp::default())),
    )
}

struct MyApp {
    points: Vec<[f64; 2]>,
}

impl Default for MyApp {
    // gives default values for the struct MyApp
    fn default() -> Self {
        // Generate some data for the plot
        let points = (0..=100)
            .map(|i| {
                let x = i as f64 * 0.1;
                [x, x.sin()]
            })
            .collect();

        Self { points }
    }
}

impl eframe::App for MyApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        ctx.set_visuals(egui::Visuals::dark());

        egui::CentralPanel::default().show(ctx, |ui| {
            egui::ScrollArea::vertical().show(ui, |ui| {
                ui.heading("Hello egui!");
                if ui.button("Click me").clicked() {
                    println!("Button clicked!");
                }

                // Add some space before the plot
                ui.add_space(10.0);
                ui.separator();
                ui.heading("My Custom Plot");
                ui.add_space(5.0);
                ui.label(
                    "Plot Controls:\n\
                    - Pan: Drag with primary mouse button\n\
                    - Zoom: Scroll wheel\n\
                    - Zoom to selection: Drag with secondary mouse button\n\
                    - Reset: Double-click"
                );
                ui.add_space(5.0);

                // Define a size for the plots
                let plot_size = egui::Vec2::new(ui.available_width(), 300.0);

                // --- Upper Plot ---
                let upper_plot = Plot::new("upper_plot")
                    .x_axis_label("t [s]")
                    .y_axis_label("State value [dim]");
                
                // Allocate space and draw the plot
                ui.add_sized(plot_size, |ui: &mut egui::Ui| {
                    upper_plot.show(ui, |plot_ui| {
                        let line_points = PlotPoints::from_iter(self.points.iter().map(|p| [p[0], p[1]]));
                        let line = Line::new(line_points);
                        plot_ui.line(line);
                    }).response
                });

                ui.add_space(10.0); // Add some space between plots

                // --- Lower Plot ---
                let lower_plot = Plot::new("lower_plot")
                    .x_axis_label("t [s]")
                    .y_axis_label("State value [dim]");

                // Allocate space and draw the plot
                ui.add_sized(plot_size, |ui: &mut egui::Ui| {
                    lower_plot.show(ui, |plot_ui| {
                        let line_points = PlotPoints::from_iter(self.points.iter().map(|p| [p[0], p[1]]));
                        let line = Line::new(line_points);
                        plot_ui.line(line);
                    }).response
                });
            });
        });
    }
}

