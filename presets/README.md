# Config presets

Ready-made configs for different arena types. Use with `--config presets/<name>.json`.

- **elevated_maze_arena_detection.json** — Elevated zero / circular mazes. Auto arena crop via edges + contour (Hough disabled by default). Tune `morph_close_ksize`, `canny_low`, `min_area_ratio` as needed.
- **open_field_arena_detection.json** — Open field (white rectangular box). Auto arena crop via brightness threshold and largest white region. Tune `open_field_white_threshold`, `open_field_min_area_ratio`, `open_field_rectangularity_min` as needed.
