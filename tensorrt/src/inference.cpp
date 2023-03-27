//
// Created by TYTY on 2021-12-23 023.
//

#include <fstream>
#include <utility>

#include "NvInfer.h"
#include "cuda_runtime_api.h"

#include "inference.h"

//#include "debug/reveal.h"

#define CUDA_CHECK(status)                                                                                             \
  do {                                                                                                                 \
    auto ret = (status);                                                                                               \
    if (ret != 0) {                                                                                                    \
      std::stringstream s;                                                                                             \
      s << "Cuda failure at " __FILE__ ":" << __LINE__ << ": " << cudaGetErrorName(ret);                               \
      logger.log(nvinfer1::ILogger::Severity::kINTERNAL_ERROR, s.str().c_str());                                       \
      return;                                                                                                          \
    }                                                                                                                  \
  } while (0)

#define COND_CHECK(cond, message)                                                                                      \
  do {                                                                                                                 \
    if (!(cond)) {                                                                                                     \
      std::stringstream s;                                                                                             \
      s << "Check failed " __FILE__ ":" << __LINE__ << ": " #cond ", " << message;                                     \
      logger.log(nvinfer1::ILogger::Severity::kINTERNAL_ERROR, s.str().c_str());                                       \
      return;                                                                                                          \
    }                                                                                                                  \
  } while (0)

#define COND_CHECK_EMPTY(cond, message)                                                                                \
  do {                                                                                                                 \
    if (!(cond)) {                                                                                                     \
      std::stringstream s;                                                                                             \
      s << "Check failed " __FILE__ ":" << __LINE__ << ": " #cond ", " << message;                                     \
      logger.log(nvinfer1::ILogger::Severity::kINTERNAL_ERROR, s.str().c_str());                                       \
      return nullptr;                                                                                                  \
    }                                                                                                                  \
  } while (0)

static nvinfer1::ICudaEngine *loadModel(nvinfer1::IRuntime *runtime, nvinfer1::ILogger &logger,
                                        const std::filesystem::path &path) {
  std::ifstream file(path, std::ios::binary);
  COND_CHECK_EMPTY(file.good(), "can't open engine file: " << path);

  file.seekg(0, std::ifstream::end);
  auto size = file.tellg();
  file.seekg(0, std::ifstream::beg);
  auto modelStream = std::make_unique<char[]>(size);
  COND_CHECK_EMPTY(modelStream, "Alloc " << size << " bytes failed.");
  file.read(modelStream.get(), size);
  file.close();

  auto engine = runtime->deserializeCudaEngine(modelStream.get(), size);
  COND_CHECK_EMPTY(runtime, "failed deserializing engine");

  return engine;
}

static std::string fe_engine_name(const InferenceConfig &config) {
  std::stringstream ss;
  ss << "fe_";
  ss << config.input_width << 'x' << config.input_height << '_' << config.scale_factor << "x"
     << "_b" << config.batch_extract << "_l" << config.extraction_layers;
  if (config.format == IOFormat::YUV420) {
    ss << "_yuv1-1";
  }
  if (config.use_fp16) {
    ss << "_fp16";
  }
  if (config.low_mem) {
    ss << "_lm";
  }
  ss << ".engine";
  return ss.str();
}

static std::string ff_engine_name(const InferenceConfig &config) {
  std::stringstream ss;
  ss << "ff_";
  ss << config.input_width << 'x' << config.input_height << '_' << config.scale_factor << "x"
     << "_b" << config.batch_fusion << "_l" << config.extraction_layers;
  if (config.format == IOFormat::YUV420) {
    ss << "_yuv1-1";
  }
  if (config.use_fp16) {
    ss << "_fp16";
  }
  if (config.low_mem) {
    ss << "_lm";
  }
  ss << ".engine";
  return ss.str();
}

InferenceContext::InferenceContext(InferenceConfig config, nvinfer1::ILogger &logger, std::filesystem::path path_prefix)
    : config(config), logger(logger),
      runtime(nvinfer1::createInferRuntime(logger)), path_prefix {std::move(path_prefix)}, engine {} {}

bool InferenceContext::has_file() {
  return exists(path_prefix / fe_engine_name(config)) && exists(path_prefix / ff_engine_name(config));
}

bool InferenceContext::load_engine() {
  engine.feature_extract = loadModel(runtime, logger, path_prefix / fe_engine_name(config));
  engine.feature_fusion = loadModel(runtime, logger, path_prefix / ff_engine_name(config));
  return good();
}

static void *ptr_add(void *b, size_t n) {
  return static_cast<uint8_t *>(b) + n;
}
static size_t alignment(size_t size, size_t alignment) {
  return (size + (alignment - 1)) & (~(alignment - 1));
}

InferenceSession::InferenceSession(InferenceContext &ctx)
    : ctx(ctx), context {ctx.engine.feature_extract->createExecutionContextWithoutDeviceMemory(),
                         ctx.engine.feature_fusion->createExecutionContextWithoutDeviceMemory()},
      last_batch {-1, -1}, last_offset_in {-1, -1}, last_offset_out {-1, -1}, good_ {false}, stream {nullptr},
      executionMemory {nullptr} {
  if (context.feature_extract == nullptr || context.feature_fusion == nullptr) {
    return;
  }

  auto &logger = ctx.logger;
  auto &config = ctx.config;

  CUDA_CHECK(cudaStreamCreate(&stream));

  const size_t eSize = config.use_fp16 ? 2 : 4;
  auto input_height = config.input_height;
  auto input_width = config.input_width;
  auto feature_count = config.feature_count;
  const size_t input_count = input_height * input_width;
  size_t output_count =
      size_t(double(input_width) * config.scale_factor) * size_t(double(input_height) * config.scale_factor);

  auto deviceMemory =
      std::max(ctx.engine.feature_extract->getDeviceMemorySize(), ctx.engine.feature_fusion->getDeviceMemorySize());
  size_t freeMemory {};
  cudaMemGetInfo(&freeMemory, nullptr);
  logger.log(freeMemory > deviceMemory ? nvinfer1::ILogger::Severity::kINFO : nvinfer1::ILogger::Severity::kWARNING,
             ("Device memory: " + std::to_string(freeMemory) + " bytes free, " + std::to_string(deviceMemory) +
              " bytes needed.")
                 .c_str());
  CUDA_CHECK(cudaMallocAsync(&executionMemory, deviceMemory, stream));
  context.feature_extract->setDeviceMemory(executionMemory);
  context.feature_fusion->setDeviceMemory(executionMemory);

  context.feature_extract->setOptimizationProfileAsync(0, stream);
  context.feature_fusion->setOptimizationProfileAsync(0, stream);

  input_size_ = input_count * eSize;
  feature_sizes.resize(config.extraction_layers);
  output_size_ = output_count * eSize;

  if (config.format == IOFormat::RGB) {
    // TODO
  }
  else if (config.format == IOFormat::YUV420) {
    cudaBuffers.resize(2 + config.extraction_layers + 2);

    CUDA_CHECK(cudaMallocAsync(&cudaBuffers[0], config.batch_extract * input_size_, stream));
    CUDA_CHECK(cudaMallocAsync(&cudaBuffers[1], config.batch_extract * input_size_ / 4 * 2, stream));

    auto layer_width = input_width;
    auto layer_height = input_height;
    for (int i = 0; i < config.extraction_layers; ++i) {
      auto layer_size = layer_width * layer_height * feature_count * eSize;
      CUDA_CHECK(cudaMallocAsync(&cudaBuffers[2 + i], (config.batch_extract + 1) * layer_size, stream));
      feature_sizes[i] = layer_size;
      layer_width = (layer_width + 1) / 2;
      layer_height = (layer_width + 1) / 2;
    }

    const auto offset = 2 + config.extraction_layers;
    CUDA_CHECK(cudaMallocAsync(&cudaBuffers[offset], config.batch_fusion * output_size_ * 2, stream));
    CUDA_CHECK(cudaMallocAsync(&cudaBuffers[offset + 1], config.batch_fusion * output_size_ / 4 * 2 * 2, stream));
  }

  good_ = true;
}

InferenceSession::~InferenceSession() {
  auto &logger = ctx.logger;
  if (stream == nullptr) {
    return;
  }

  if (executionMemory != nullptr) {
    CUDA_CHECK(cudaFreeAsync(executionMemory, stream));
  }

  for (auto *p: cudaBuffers) {
    if (p != nullptr) {
      CUDA_CHECK(cudaFreeAsync(p, stream));
    }
  }

  CUDA_CHECK(cudaStreamSynchronize(stream));
  CUDA_CHECK(cudaStreamDestroy(stream));
}

void InferenceSession::extractBatch(int32_t offset_in, int32_t offset_out, int32_t batch) {
  auto &logger = ctx.logger;
  auto &config = ctx.config;
  auto input_height = config.input_height;
  auto input_width = config.input_width;

  COND_CHECK(batch > 0, "invalid extract batch");
  COND_CHECK(offset_in + batch <= config.batch_extract, "invalid extract batch");
  COND_CHECK(offset_out + batch <= config.batch_extract + 1, "invalid extract batch");

  if (batch != last_batch.feature_extract) {
    if (config.format == IOFormat::RGB) {
      // TODO
    }
    else if (config.format == IOFormat::YUV420) {
      context.feature_extract->setInputShape("y", {4, {batch, 1, input_height, input_width}});
      context.feature_extract->setInputShape("uv", {4, {batch, 2, input_height / 2, input_width / 2}});
    }
    last_batch.feature_extract = batch;
  }

  if (offset_in != last_offset_in.feature_extract) {
    if (config.format == IOFormat::RGB) {
      // TODO
    }
    else if (config.format == IOFormat::YUV420) {
      input[0] = ptr_add(cudaBuffers[0], offset_in * input_size_);
      input[1] = ptr_add(cudaBuffers[1], offset_in * input_size_ / 4 * 2);
      context.feature_extract->setTensorAddress("y", input[0]);
      context.feature_extract->setTensorAddress("uv", input[1]);
    }

    last_offset_in.feature_extract = offset_in;
  }

  if (offset_out != last_offset_out.feature_extract) {
    auto baseBuffer = cudaBuffers.data() + (config.format == IOFormat::RGB ? 1 : 2);
    for (int i = 0; i < config.extraction_layers; ++i) {
      context.feature_extract->setTensorAddress(("l" + std::to_string(i)).c_str(),
                                                ptr_add(baseBuffer[i], offset_out * feature_sizes[i]));
    }

    last_offset_out.feature_extract = offset_out;
  }
}

void InferenceSession::fusionBatch(int32_t offset_in, int32_t offset_out, int32_t batch) {
  auto &logger = ctx.logger;
  auto &config = ctx.config;
  auto input_height = config.input_height;
  auto input_width = config.input_width;
  auto feature_count = config.feature_count;

  COND_CHECK(batch > 0, "invalid fusion batch");
  COND_CHECK(offset_in + batch <= config.batch_extract + 1, "invalid fusion batch");
  COND_CHECK(offset_out + batch <= config.batch_fusion, "invalid fusion batch");

  if (batch != last_batch.feature_fusion) {
    auto layer_width = input_width;
    auto layer_height = input_height;
    for (int i = 0; i < config.extraction_layers; ++i) {
      context.feature_fusion->setInputShape(("f0l" + std::to_string(i)).c_str(),
                                            {4, {batch, feature_count, layer_height, layer_width}});
      context.feature_fusion->setInputShape(("f2l" + std::to_string(i)).c_str(),
                                            {4, {batch, feature_count, layer_height, layer_width}});
      layer_width = (layer_width + 1) / 2;
      layer_height = (layer_height + 1) / 2;
    }
    last_batch.feature_fusion = batch;
  }

  auto baseBuffer = cudaBuffers.data() + (config.format == IOFormat::RGB ? 1 : 2);
  if (offset_in != last_offset_in.feature_fusion) {
    for (int i = 0; i < config.extraction_layers; ++i) {
      auto l0 = ptr_add(baseBuffer[i], offset_in * feature_sizes[i]);
      context.feature_fusion->setTensorAddress(("f0l" + std::to_string(i)).c_str(), l0);
      context.feature_fusion->setTensorAddress(("f2l" + std::to_string(i)).c_str(), ptr_add(l0, feature_sizes[i]));
    }

    last_offset_in.feature_fusion = offset_in;
  }

  if (offset_out != last_offset_out.feature_fusion) {
    baseBuffer += config.extraction_layers;
    if (config.format == IOFormat::RGB) {
      // TODO
    }
    else if (config.format == IOFormat::YUV420) {
      output[0] = ptr_add(baseBuffer[0], offset_out * output_size_);
      output[2] = ptr_add(output[0], config.batch_fusion * output_size_);
      output[1] = ptr_add(baseBuffer[1], offset_out * output_size_ / 4 * 2);
      output[3] = ptr_add(output[1], config.batch_fusion * output_size_ / 4 * 2);
      context.feature_fusion->setTensorAddress("h0_y", output[0]);
      context.feature_fusion->setTensorAddress("h1_y", output[2]);
      context.feature_fusion->setTensorAddress("h0_uv", output[1]);
      context.feature_fusion->setTensorAddress("h1_uv", output[3]);
    }

    last_offset_out.feature_fusion = offset_out;
  }
}

void InferenceSession::duplicateExtractOutput(int32_t from, int32_t to) {
  auto &logger = ctx.logger;
  auto &config = ctx.config;

  COND_CHECK(from <= config.batch_extract && to <= config.batch_extract, "invalid index");

  auto baseBuffer = cudaBuffers.data() + (config.format == IOFormat::RGB ? 1 : 2);
  for (int i = 0; i < config.extraction_layers; ++i) {
    CUDA_CHECK(cudaMemcpyAsync(ptr_add(baseBuffer[i], to * feature_sizes[i]),
                               ptr_add(baseBuffer[i], from * feature_sizes[i]), feature_sizes[i],
                               cudaMemcpyDeviceToDevice, stream));
  }
}

bool InferenceSession::extract() {
  return context.feature_extract->enqueueV3(stream);
}

bool InferenceSession::fusion() {
  return context.feature_fusion->enqueueV3(stream);
}
