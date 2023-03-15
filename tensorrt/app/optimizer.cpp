//
// Created by TYTY on 2023-01-13 013.
//

#include "layers.h"
#include "optimize.h"
#include "logging.h"

static Logger gLogger(Logger::Severity::kINFO);

int main() {
  UDOLayers::registerPlugins();
  //  OptimizationConfig config{256, 256, 1, 1, 64, 2, IOFormat::YUV420, 4, false};
    OptimizationConfig config{1920, 1088, 1, 1, 64, 2, IOFormat::YUV420, 4, true};
//  OptimizationConfig config{768, 544, {1, 3, 3}, {1, 3, 3}, 64, 2, IOFormat::YUV420, 4, true};

  OptimizationContext ctx(config, gLogger, "models");

  ctx.optimize("y4m_any");
}
