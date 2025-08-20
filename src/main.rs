use eframe::egui::{self, Rect};

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

        Self {points}
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

            // Create a container for the plot
            egui::Frame::canvas(ui.style()).show(ui, |ui| {
                // Allocate the space for the plot.
                // We use a fixed height and all available width.
                let (response, painter) = ui.allocate_painter(
                    egui::Vec2::new(ui.available_width(), 200.0),
                    egui::Sense::hover(),
                );

                // Get the rectangle of the plot area
                let plot_rect = response.rect;

                // --- Coordinate Transformation ---
                // We need to map our data coordinates to the `plot_rect`.
                // First, let's find the bounding box of our data.
                let data_bounds = self.points.iter().fold(
                    egui::Rect::NOTHING,
                    |acc, &p| {
                        let point = egui::pos2(p[0] as f32, p[1] as f32);
                        acc.union(Rect::from_min_max(point, point))
                    }
                );

                // Then we can create a transform from data space to screen space.
                // The Y-axis is inverted in egui, so we use `from_y_down` for the range.
                let to_screen = egui::emath::RectTransform::from_to(
                    egui::Rect::from_x_y_ranges(data_bounds.x_range(), data_bounds.y_range().max..=data_bounds.y_range().min),
                    plot_rect,
                );

                // --- Draw Grid ---
                let grid_stroke = egui::Stroke::new(1.0, egui::Color32::from_gray(60));
                let num_x_ticks = 5;
                let x_step = data_bounds.width() / (num_x_ticks - 1) as f32;
                for i in 0..num_x_ticks {
                    let x_val = data_bounds.left() + i as f32 * x_step;
                    let screen_x = (to_screen * egui::pos2(x_val, 0.0)).x;
                    painter.line_segment(
                        [egui::pos2(screen_x, plot_rect.top()), egui::pos2(screen_x, plot_rect.bottom())],
                        grid_stroke,
                    );
                }
                let num_y_ticks = 5;
                let y_step = data_bounds.height() / (num_y_ticks - 1) as f32;
                for i in 0..num_y_ticks {
                    let y_val = data_bounds.bottom() + i as f32 * y_step;
                    let screen_y = (to_screen * egui::pos2(0.0, y_val)).y;
                    painter.line_segment(
                        [egui::pos2(plot_rect.left(), screen_y), egui::pos2(plot_rect.right(), screen_y)],
                        grid_stroke,
                    );
                }

                // --- Drawing ---
                // Draw the plot data as a line
                let points_in_screen_space: Vec<egui::Pos2> = self
                    .points
                    .iter()
                    .map(|data_point| {
                        to_screen * egui::pos2(data_point[0] as f32, data_point[1] as f32)
                    })
                    .collect();

                let line = egui::Shape::line(
                    points_in_screen_space,
                    egui::Stroke::new(2.0, egui::Color32::from_rgb(100, 200, 100)),
                );
                painter.add(line);

                // Optional: Draw a line for the x-axis
                painter.line_segment(
                    [
                        to_screen * egui::pos2(data_bounds.left(), 0.0),
                        to_screen * egui::pos2(data_bounds.right(), 0.0),
                    ],
                    egui::Stroke::new(1.0, egui::Color32::GRAY),
                );

                // --- Draw Axis Ticks ---
                let text_color = ui.style().visuals.text_color();
                let font_id = egui::FontId::proportional(12.0);

                // X-axis ticks
                for i in 0..num_x_ticks {
                    let x_val = data_bounds.left() + i as f32 * x_step;
                    let tick_pos_on_axis = to_screen * egui::pos2(x_val, 0.0);
                    painter.line_segment(
                        [tick_pos_on_axis, tick_pos_on_axis + egui::vec2(0.0, 5.0)],
                        egui::Stroke::new(1.0, egui::Color32::GRAY),
                    );
                    painter.text(
                        tick_pos_on_axis + egui::vec2(0.0, 7.0),
                        egui::Align2::CENTER_TOP,
                        format!("{:.1}", x_val),
                        font_id.clone(),
                        text_color,
                    );
                }

                // Y-axis ticks
                for i in 0..num_y_ticks {
                    let y_val = data_bounds.bottom() + i as f32 * y_step;
                    // Find the position on the y-axis (x=0 in data space)
                    let tick_pos_on_axis = to_screen * egui::pos2(0.0, y_val);
                     painter.line_segment(
                        [tick_pos_on_axis, tick_pos_on_axis - egui::vec2(5.0, 0.0)],
                        egui::Stroke::new(1.0, egui::Color32::GRAY),
                    );
                    painter.text(
                        tick_pos_on_axis - egui::vec2(7.0, 0.0),
                        egui::Align2::RIGHT_CENTER,
                        format!("{:.1}", y_val),
                        font_id.clone(),
                        text_color,
                    );
                }

                // --- Draw Axis Labels ---
                let font_id = egui::FontId::proportional(14.0);
                let text_color = ui.style().visuals.text_color();

                // Y-axis label
                painter.text(
                    plot_rect.left_top() + egui::vec2(5.0, 5.0),
                    egui::Align2::LEFT_TOP,
                    "sin(x)",
                    font_id.clone(),
                    text_color,
                );

                // X-axis label
                painter.text(
                    plot_rect.center_bottom() - egui::vec2(0.0, 5.0),
                    egui::Align2::CENTER_BOTTOM,
                    "x",
                    font_id,
                    text_color,
                );
            });
        });
    }
}

