#pragma once

#include <array>
#include <vector>
#include <cstdint>
#include <string>
#include <filesystem>

#include <NvInfer.h>

#include "utils.h"
#include "config.h"

template<class T>
struct ModelStuff {
  T feature_extract;
  T feature_fusion;
};

class InferenceSession;

class InferenceContext {
  nvinfer1::ILogger &logger;
  nvinfer1::IRuntime *runtime;
  std::filesystem::path path_prefix;
  ModelStuff<nvinfer1::ICudaEngine *> engine;

  friend class InferenceSession;

 public:
  InferenceConfig config;
  InferenceContext(InferenceConfig config, nvinfer1::ILogger &logger, std::filesystem::path path_prefix);
  bool has_file();
  bool load_engine();

  bool good() {
    return runtime != nullptr && engine.feature_extract != nullptr && engine.feature_fusion != nullptr;
  }
};

class InferenceSession {
  InferenceContext ctx;

  ModelStuff<nvinfer1::IExecutionContext*> context;
  std::vector<void*> cudaBuffers;
  void* executionMemory;
  ModelStuff<int32_t> last_batch, last_offset_in, last_offset_out;
  size_t input_size_;
  size_t output_size_;
  size_t feature_size;
  bool good_;

 public:
  cudaStream_t stream;
  std::array<void*, 2> input;
  std::array<void*, 4> output;

  explicit InferenceSession(InferenceContext& ctx);
  ~InferenceSession();

  bool good() const { return good_; }

  void extractBatch(int32_t offset_in, int32_t offset_out, int32_t batch);
  void fusionBatch(int32_t offset_in, int32_t offset_out, int32_t batch);

  size_t input_size() const { return input_size_; }
  size_t output_size() const { return output_size_; }

  void duplicateExtractOutput(int32_t from, int32_t to);

  bool extract();
  bool fusion();
};
