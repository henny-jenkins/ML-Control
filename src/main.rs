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

fn test_func() {
    println!("test func called");
}

struct MyApp {
    points: Vec<[f64; 2]>,
}

impl Default for MyApp {
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
            ui.heading("Hello egui!");
            if ui.button("Click me").clicked() {
                println!("Button clicked!");
            }

            // Add some space before the plot
            ui.add_space(10.0);
            ui.separator();
            ui.heading("My Custom Plot");

            let plot = Plot::new("my_plot")
                .x_axis_label("x")
                .y_axis_label("sin(x)");

            plot.show(ui, |plot_ui| {
                let line_points = PlotPoints::from_iter(self.points.iter().map(|p| [p[0], p[1]]));
                let line = Line::new(line_points);
                plot_ui.line(line);
            });
        });
    }
}

