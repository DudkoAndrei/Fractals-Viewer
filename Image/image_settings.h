#pragma once

struct ImageSettings {
  unsigned int width{0};
  unsigned int height{0};

  long long offset_x{0};
  long long offset_y{0};

  double scale_x{1};
  double scale_y{1};
};
