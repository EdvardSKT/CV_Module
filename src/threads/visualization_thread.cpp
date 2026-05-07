#include <atomic>
#include <vector>

#include "raylib.h"

#include "utilities/coordinate_converter.hpp"
#include "threads/queue_thread.hpp"

const double start_x = 0.0;
const double end_x = 3.0;

const double min_y = 0.0;
const double max_y = 1.0;

const int pixels_per_m = 300;

const double conveyor_length_pixels = end_x*pixels_per_m;
const double conveyor_width_pixels = max_y*pixels_per_m;

void visualization_loop(std::atomic<bool>& running, std::vector<BatteryTrack> current_detections)
{

    InitWindow(1000, 400, "Conveyor Tracker View");
    SetTargetFPS(60);

    BeginDrawing();
    ClearBackground(RAYWHITE);

    DrawRectangle(50, 50, conveyor_length_pixels, conveyor_width_pixels, LIGHTGRAY);
    DrawRectangleLines(50, 50, conveyor_length_pixels, conveyor_width_pixels, DARKGRAY);


    while(running && !WindowShouldClose()){
        for(const auto& bat : current_detections){
            double x_percentage = bat.coordinate.x/end_x;
            double y_percentage = bat.coordinate.y/max_y;

            int x_pixel = static_cast<int>(floor(x_percentage*conveyor_length_pixels)) + 50;
            int y_pixels = static_cast<int>(floor(y_percentage*conveyor_width_pixels)) + 50;

            DrawCircle(x_pixel, y_pixels, 10, RED);
        }
    }
}