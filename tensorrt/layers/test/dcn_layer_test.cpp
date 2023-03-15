//
// Created by TYTY on 2022-10-04 004.
//

#include <chrono>
#include <fstream>
#include <memory>
#include <string>

#include "cublas_v2.h"
#include "cuda_fp16.h"
#include "cuda_runtime_api.h"

#include "dcn_layer_impl.h"
#include "md_view.h"

#include "gtest/gtest.h"

#define CUDA_CHECK(status) ASSERT_EQ(status, cudaSuccess)
#define COND_CHECK(cond, message) ASSERT_TRUE(cond)

// small enough to not cause pixel drift
constexpr double Epsilon = 0.5 / 255;

// increase allowed epsilon for fp16.
constexpr double EpsilonHalf = 0.025;

typedef std::chrono::duration<double, std::ratio<1, 1000>> millisecond;

template<class T, size_t N>
void loadFile(const std::string &path, md_view<T, N> &data) {
  std::ifstream file(path, std::ios::binary);
  COND_CHECK(file.good(), "can't open input file.");

  auto size = data.size() * sizeof(T);
  file.read((char *) data.data, size);
  file.close();
}

template<class T, size_t N>
void loadFileNv(const std::string &path, md_view<T, N> &data) {
  std::ifstream file(path, std::ios::binary);
  COND_CHECK(file.good(), "can't open input file.");

  auto size = data.size() * sizeof(T);
  auto tmp = std::make_unique<char[]>(size);
  file.read(tmp.get(), size);
  file.close();

  CUDA_CHECK(cudaMemcpy((void *) data.data, tmp.get(), size, cudaMemcpyHostToDevice));
}

template<size_t N>
void loadFileNvF2H(const std::string &path, md_view<const half, N> &data) {
  std::ifstream file(path, std::ios::binary);
  COND_CHECK(file.good(), "can't open input file.");

  auto size = data.size() * sizeof(float);
  auto tmp = std::make_unique<char[]>(size);
  file.read(tmp.get(), size);
  file.close();

  auto tmpF = (float *)tmp.get();
  auto tmpH = ((half *)tmp.get()) + data.size();
  for (offset_t i = 0; i < data.size(); ++i) {
    offset_t idx = data.size() - i - 1;
    tmpH[idx] = tmpF[idx];
  }

  CUDA_CHECK(cudaMemcpy((void *) data.data, tmpH, size / 2, cudaMemcpyHostToDevice));
}

template<class F>
void ComputeFloat(DCNLayerInput<F> &inputs,
                  DCNLayerOutput<F> &outputs,
                  DCNLayerConfig &config,
                  DCNLayerExtra &extra,
                  cudaStream_t stream,
                  int repeat) {
  for (int i = 0; i < repeat; ++i) {
    auto all_begin = std::chrono::steady_clock::now();
    millisecond elapsed;

    compute(inputs, outputs, config, extra, stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));

    elapsed = std::chrono::steady_clock::now() - all_begin;
    std::cerr << "Inference done after " << elapsed.count() << "ms." << std::endl;
  }
}

void RunTest(const std::string &file_prefix, DCNLayerInput<float> &inputs, DCNLayerOutput<float> &outputs, DCNLayerConfig &config, int repeat = 1) {
  CUDA_CHECK(cudaMalloc((void **) &inputs.input.data, inputs.input.size() * sizeof(float)));
  CUDA_CHECK(cudaMalloc((void **) &inputs.offset.data, inputs.offset.size() * sizeof(float)));
  CUDA_CHECK(cudaMalloc((void **) &inputs.mask.data, inputs.mask.size() * sizeof(float)));
  CUDA_CHECK(cudaMalloc((void **) &inputs.weight.data, inputs.weight.size() * sizeof(float)));
  CUDA_CHECK(cudaMalloc((void **) &inputs.bias.data, inputs.bias.size() * sizeof(float)));
  CUDA_CHECK(cudaMalloc((void **) &inputs.im2col_buffer.data, inputs.im2col_buffer.size() * sizeof(float)));

  CUDA_CHECK(cudaMalloc((void **) &outputs.output.data, outputs.output.size() * sizeof(float)));

  loadFileNv(file_prefix + "input.bin", inputs.input);
  loadFileNv(file_prefix + "offset.bin", inputs.offset);
  loadFileNv(file_prefix + "mask.bin", inputs.mask);
  loadFileNv(file_prefix + "weight.bin", inputs.weight);
  loadFileNv(file_prefix + "bias.bin", inputs.bias);

  cudaStream_t stream;
  CUDA_CHECK(cudaStreamCreate(&stream));

  cublasHandle_t cublas;

  DCNLayerExtra extra{};
  ASSERT_EQ(cublasCreate_v2(&cublas), CUBLAS_STATUS_SUCCESS);
  ASSERT_EQ(cublasSetStream_v2(cublas, stream), CUBLAS_STATUS_SUCCESS);
  extra.cublasHandle = cublas;
  extra.blasMode = 0;

  ComputeFloat(inputs, outputs, config, extra, stream, repeat);

  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaStreamDestroy(stream));

  auto oshape = outputs.output.shape;
  auto output_ref_storage = std::make_unique<float[]>(oshape.count());
  auto output_cpu_storage = std::make_unique<float[]>(oshape.count());
  md_view<float, 4> output_ref{output_ref_storage.get(), oshape};
  md_view<float, 4> output_cpu{output_cpu_storage.get(), oshape};

  loadFile(file_prefix + "output.bin", output_ref);
  CUDA_CHECK(cudaMemcpy((void *) output_cpu.data, outputs.output.data, oshape.count() * sizeof(float), cudaMemcpyDeviceToHost));

  float max = 0;
  double total = 0;

  for (offset_t n = 0; n < oshape[0]; ++n) {
    for (offset_t c = 0; c < oshape[1]; ++c) {
      for (offset_t h = 0; h < oshape[2]; ++h) {
        for (offset_t w = 0; w < oshape[3]; ++w) {
          EXPECT_NEAR(output_ref.at(n, c, h, w), output_cpu.at(n, c, h, w), Epsilon) << "The coordinate is [" << n << "," << c << "," << h << "," << w << "]";
          float diff = std::abs(output_ref.at(n, c, h, w) - output_cpu.at(n, c, h, w));
          total += diff;
          max = diff > max ? diff : max;
        }
      }
    }
  }

  std::cerr << "Diff: max " << max << ", avg " << total / double(oshape.count()) << std::endl;

  ASSERT_EQ(cublasDestroy_v2(cublas), CUBLAS_STATUS_SUCCESS);

  CUDA_CHECK(cudaFree((void *) inputs.input.data));
  CUDA_CHECK(cudaFree((void *) inputs.offset.data));
  CUDA_CHECK(cudaFree((void *) inputs.mask.data));
  CUDA_CHECK(cudaFree((void *) inputs.weight.data));
  CUDA_CHECK(cudaFree((void *) inputs.bias.data));
  CUDA_CHECK(cudaFree((void *) inputs.im2col_buffer.data));

  CUDA_CHECK(cudaFree((void *) outputs.output.data));
}

void RunTest(const std::string &file_prefix, DCNLayerInput<half> &inputs, DCNLayerOutput<half> &outputs, DCNLayerConfig &config, int repeat = 1) {
  CUDA_CHECK(cudaMalloc((void **) &inputs.input.data, inputs.input.size() * sizeof(half)));
  CUDA_CHECK(cudaMalloc((void **) &inputs.offset.data, inputs.offset.size() * sizeof(half)));
  CUDA_CHECK(cudaMalloc((void **) &inputs.mask.data, inputs.mask.size() * sizeof(half)));
  CUDA_CHECK(cudaMalloc((void **) &inputs.weight.data, inputs.weight.size() * sizeof(half)));
  CUDA_CHECK(cudaMalloc((void **) &inputs.bias.data, inputs.bias.size() * sizeof(half)));
  CUDA_CHECK(cudaMalloc((void **) &inputs.im2col_buffer.data, inputs.im2col_buffer.size() * sizeof(half)));

  CUDA_CHECK(cudaMalloc((void **) &outputs.output.data, outputs.output.size() * sizeof(half)));

  loadFileNvF2H(file_prefix + "input.bin", inputs.input);
  loadFileNvF2H(file_prefix + "offset.bin", inputs.offset);
  loadFileNvF2H(file_prefix + "mask.bin", inputs.mask);
  loadFileNvF2H(file_prefix + "weight.bin", inputs.weight);
  loadFileNvF2H(file_prefix + "bias.bin", inputs.bias);

  cudaStream_t stream;
  CUDA_CHECK(cudaStreamCreate(&stream));

  cublasHandle_t cublas;

  DCNLayerExtra extra{};
  ASSERT_EQ(cublasCreate_v2(&cublas), CUBLAS_STATUS_SUCCESS);
  ASSERT_EQ(cublasSetStream_v2(cublas, stream), CUBLAS_STATUS_SUCCESS);
  extra.cublasHandle = cublas;
  extra.blasMode = 0;

  ComputeFloat(inputs, outputs, config, extra, stream, repeat);

  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaStreamDestroy(stream));

  auto oshape = outputs.output.shape;
  auto output_ref_storage = std::make_unique<float[]>(oshape.count());
  auto output_cpu_storage = std::make_unique<half[]>(oshape.count());
  md_view<float, 4> output_ref{output_ref_storage.get(), oshape};
  md_view<half, 4> output_cpu{output_cpu_storage.get(), oshape};

  loadFile(file_prefix + "output.bin", output_ref);
  CUDA_CHECK(cudaMemcpy((void *) output_cpu.data, outputs.output.data, oshape.count() * sizeof(half), cudaMemcpyDeviceToHost));

  float max = 0;
  double total = 0;

  for (offset_t n = 0; n < oshape[0]; ++n) {
    for (offset_t c = 0; c < oshape[1]; ++c) {
      for (offset_t h = 0; h < oshape[2]; ++h) {
        for (offset_t w = 0; w < oshape[3]; ++w) {
          EXPECT_NEAR(output_ref.at(n, c, h, w), output_cpu.at(n, c, h, w), EpsilonHalf) << "The coordinate is [" << n << "," << c << "," << h << "," << w << "]";
          float diff = std::abs(output_ref.at(n, c, h, w) - output_cpu.at(n, c, h, w));
          total += diff;
          max = diff > max ? diff : max;
        }
      }
    }
  }

  std::cerr << "Diff: max " << max << ", avg " << total / double(oshape.count()) << std::endl;

  ASSERT_EQ(cublasDestroy_v2(cublas), CUBLAS_STATUS_SUCCESS);

  CUDA_CHECK(cudaFree((void *) inputs.input.data));
  CUDA_CHECK(cudaFree((void *) inputs.offset.data));
  CUDA_CHECK(cudaFree((void *) inputs.mask.data));
  CUDA_CHECK(cudaFree((void *) inputs.weight.data));
  CUDA_CHECK(cudaFree((void *) inputs.bias.data));
  CUDA_CHECK(cudaFree((void *) inputs.im2col_buffer.data));

  CUDA_CHECK(cudaFree((void *) outputs.output.data));
}

TEST(DCNLayerTest, SmallInput) {
  DCNLayerInput<float> input{
      {nullptr, {1, 1, 5, 5}},
      {nullptr, {1, 1, 3, 3, 2, 5, 5}},
      {nullptr, {1, 1, 3, 3, 5, 5}},
      {nullptr, {1, 1, 3, 3}},
      {nullptr, {1}},

      {nullptr, {1, 1, 3, 3, 5, 5}}};

  DCNLayerOutput<float> output{
      {nullptr, {1, 1, 5, 5}}};

  DCNLayerConfig config{
      {1, 1},
      {1, 1},
      {1, 1},
      1,
      3,
      0.1,
      0};

  RunTest("./small/", input, output, config);
}

TEST(DCNLayerTest, RealCase) {
  DCNLayerInput<float> input{
      {nullptr, {2, 64, 180, 320}},
      {nullptr, {2, 8, 3, 3, 2, 180, 320}},
      {nullptr, {2, 8, 3, 3, 180, 320}},
      {nullptr, {64, 64, 3, 3}},
      {nullptr, {64}},

      {nullptr, {2, 64, 3, 3, 180, 320}}};

  DCNLayerOutput<float> output{
      {nullptr, {2, 64, 180, 320}}};

  DCNLayerConfig config{
      {1, 1},
      {1, 1},
      {1, 1},
      8,
      -1,
      0,
      0};

  for (int i = 0; i < 1000; ++i) {
    RunTest("./real/", input, output, config, 100);
  }

}

TEST(DCNLayerTest, RealCaseHalf) {
  DCNLayerInput<half> input{
      {nullptr, {2, 64, 180, 320}},
      {nullptr, {2, 8, 3, 3, 2, 180, 320}},
      {nullptr, {2, 8, 3, 3, 180, 320}},
      {nullptr, {64, 64, 3, 3}},
      {nullptr, {64}},

      {nullptr, {2, 64, 3, 3, 180, 320}}};

  DCNLayerOutput<half> output{
      {nullptr, {2, 64, 180, 320}}};

  DCNLayerConfig config{
      {1, 1},
      {1, 1},
      {1, 1},
      8,
      -1,
      0,
      0};

  RunTest("./real/", input, output, config, 20);
}
