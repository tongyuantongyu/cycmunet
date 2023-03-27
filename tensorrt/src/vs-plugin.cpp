#include <array>
#include <cmath>
#include <filesystem>
#include <sstream>
#include <string>
#include <string_view>

#include "NvInfer.h"
#include "cuda_runtime_api.h"
#include <cuda_fp16.h>

#include "inference.h"
#include "layers.h"
#include "optimize.h"
#include "reformat.h"

#include "md_view.h"

#include "VSConstants4.h"
#include "VSHelper4.h"
#include "VapourSynth4.h"

// ---------------------------------------------------------------------------------------------------------------------
// Utils

class Logger : public nvinfer1::ILogger {
  typedef void(VS_CC *logMessage_t)(int msgType, const char *msg, VSCore *core) VS_NOEXCEPT;

 public:
  Logger(VSCore *core, logMessage_t logMessage) : core(core), logMessage(logMessage) {}
  void log(Severity severity, const char *message) noexcept override {
    auto severity_int = int32_t(severity);
    if (severity_int < 0 || severity_int > 4) {
      severity_int = 0;
    }
    logMessage(typeMap[severity_int], message, core);
  }

  void log(Severity severity, const std::string_view &message) noexcept {
    auto severity_int = int32_t(severity);
    if (severity_int < 0 || severity_int > 4) {
      severity_int = 0;
    }
    logMessage(typeMap[severity_int], message.data(), core);
  }

 private:
  VSCore *core;
  logMessage_t logMessage;

#if defined(NDEBUG) || defined(_NDEBUG)
  constexpr static VSMessageType trtInfoLevel = VSMessageType::mtDebug;
#else
  constexpr static VSMessageType trtInfoLevel = VSMessageType::mtInformation;
#endif

  constexpr static VSMessageType typeMap[] = {VSMessageType::mtFatal, VSMessageType::mtWarning,
                                              VSMessageType::mtWarning, trtInfoLevel,
                                              VSMessageType::mtDebug};
};

struct scale_ratios_t {
  struct fma {
    float a;
    float b;
  };

  union {
    fma z[3];
    struct {
      fma y, u, v;
    };
    struct {
      fma r, g, b;
    };
  };
};

const std::array<float, 3> default_norm_mean = {0.485, 0.456, 0.406};
const std::array<float, 3> default_norm_std = {0.229, 0.224, 0.225};

struct color_space_t {
  VSColorPrimaries cp;
  VSTransferCharacteristics tc;
  VSMatrixCoefficients mc;
  VSColorRange r;
};

template<size_t N>
static int getFloatArray(const char *name, std::array<float, N> &data, const VSMap *in, const VSAPI *vsapi,
                         const std::array<float, N> &def) {
  int err;

  vsapi->mapGetFloatArray(in, name, &err);
  if (err) {
    data = def;
    return 0;
  }

  for (int i = 0; i < N; ++i) {
    data[i] = vsapi->mapGetFloatSaturated(in, name, i, &err);
    if (err) {
      return -1;
    }
  }

  return 0;
}

static size_t alignment(size_t size, size_t alignment) {
  return (size + (alignment - 1)) & (~(alignment - 1));
}

struct colorPrimariesEntry {
  VSColorPrimaries colorPrimariesEnum;
  float primaries[8];// rX, rY, gX, gY, bX, bY, wX, wY
};

const std::array<colorPrimariesEntry, 11> colorPrimariesTable {
    {{VSC_PRIMARIES_BT709, {0.64f, 0.33f, 0.3f, 0.6f, 0.15f, 0.06f, 0.3127f, 0.329f}},
     {VSC_PRIMARIES_BT470_M, {0.67f, 0.33f, 0.21f, 0.71f, 0.14f, 0.08f, 0.310f, 0.316f}},
     {VSC_PRIMARIES_BT470_BG, {0.64f, 0.33f, 0.29f, 0.60f, 0.15f, 0.06f, 0.3127f, 0.3290f}},
     {VSC_PRIMARIES_ST170_M, {0.630f, 0.340f, 0.310f, 0.595f, 0.155f, 0.070f, 0.3127f, 0.3290f}},
     {VSC_PRIMARIES_ST240_M, {0.630f, 0.340f, 0.310f, 0.595f, 0.155f, 0.070f, 0.3127f, 0.3290f}},
     {VSC_PRIMARIES_FILM, {0.681f, 0.319f, 0.243f, 0.692f, 0.145f, 0.049f, 0.310f, 0.316f}},
     {VSC_PRIMARIES_BT2020, {0.708f, 0.292f, 0.170f, 0.797f, 0.131f, 0.046f, 0.3127f, 0.3290f}},
     {VSC_PRIMARIES_ST428, {1.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.3333f, 0.3333f}},
     {VSC_PRIMARIES_ST431_2, {0.680f, 0.320f, 0.265f, 0.690f, 0.150f, 0.060f, 0.314f, 0.351f}},
     {VSC_PRIMARIES_ST432_1, {0.680f, 0.320f, 0.265f, 0.690f, 0.150f, 0.060f, 0.3127f, 0.3290f}},
     {VSC_PRIMARIES_EBU3213_E, {0.630f, 0.340f, 0.295f, 0.605f, 0.155f, 0.077f, 0.3127f, 0.3290f}}}};

struct matrixCoefficientsEntry {
  VSMatrixCoefficients matrixCoefficientsEnum;
  const float kr;
  const float kb;
};

const std::array<matrixCoefficientsEntry, 6> matrixCoefficientsTable {{{VSC_MATRIX_BT709, 0.2126f, 0.0722f},
                                                                       {VSC_MATRIX_FCC, 0.30f, 0.11f},
                                                                       {VSC_MATRIX_BT470_BG, 0.299f, 0.114f},
                                                                       {VSC_MATRIX_ST170_M, 0.299f, 0.114f},
                                                                       {VSC_MATRIX_ST240_M, 0.212f, 0.087f},
                                                                       {VSC_MATRIX_BT2020_NCL, 0.2627f, 0.0593f}}};

// ---------------------------------------------------------------------------------------------------------------------
// Filter

class CycMuNetFilter {
  int num_frames;
  int loaded_frames;
  int scene_begin_index, scene_begin_index_pending;
  VSColorFamily color_family;
  std::filesystem::path model_path;
  std::filesystem::path model;
  InferenceConfig config;
  InferenceContext *ctx;
  InferenceSession *session;
  const VSFrame *first_frame;
  std::vector<const VSFrame *> requested_frames;
  bool raw_norm;
  scale_ratios_t norm, denorm;
  Logger *logger;
  float y_min, y_max, uv_min, uv_max;
  uint8_t *ioBuffer[2] {};
  uint8_t *ioPointer[6];
  shape_t<2> input_shape_y, input_shape_uv, output_shape_y, output_shape_uv;
  shape_t<2> input_tensor_y, input_tensor_uv, output_tensor_y, output_tensor_uv;

  template<class F, class U>
  std::string readPlane(md_view<F, 2> dst, md_uview<const U, 2> src, md_view<U, 2> cuda_tmp, float a, float b);
  template<class F, class U>
  std::string writePlane(md_uview<U, 2> dst, md_view<const F, 2> src, md_view<U, 2> cuda_tmp, float a, float b,
                         float min, float max);
  template<class U>
  std::string readYUV(offset_t position, const VSFrame *frame, const VSAPI *vsapi);
  template<class U>
  std::string writeYUV(offset_t position, VSFrame *frame, const VSAPI *vsapi);

 public:
  VSNode *node;
  VSVideoInfo vo;
  std::string init1(const VSMap *in, VSCore *core, const VSAPI *vsapi);
  std::string init2(const VSFrame *frame, VSCore *core, const VSAPI *vsapi);
  void requestFrames(int n, VSFrameContext *frameCtx, const VSAPI *vsapi) const;
  std::string prepareFrame(int n, VSFrameContext *frameCtx, const VSAPI *vsapi);
  std::string extractFrame(int n, VSFrame *&frame, VSCore *core, const VSAPI *vsapi);
  ~CycMuNetFilter();

  std::string synchronize() {
    auto err = cudaStreamSynchronize(session->stream);
    if (err != cudaSuccess) {
      return std::string("CycMuNet: failed synchronize CUDA stream: ") + cudaGetErrorName(cudaError_t(err)) + " (" +
             cudaGetErrorString(cudaError_t(err)) + ").";
    }

    return "";
  }
};

std::string CycMuNetFilter::init1(const VSMap *in, VSCore *core, const VSAPI *vsapi) {
  int err;

  logger = new Logger {core, vsapi->logMessage};

  // Get a clip reference from the input arguments. This must be freed later.
  node = vsapi->mapGetNode(in, "clip", 0, nullptr);
  const VSVideoInfo *vi = vsapi->getVideoInfo(node);
  num_frames = vi->numFrames;
  scene_begin_index = 0;
  scene_begin_index_pending = num_frames;

  if (!vsh::isConstantVideoFormat(vi)) {
    return "CycMuNet: only constant format input supported";
  }

  IOFormat format;
  if (vi->format.colorFamily == VSColorFamily::cfRGB) {
    format = IOFormat::RGB;
    return "CycMuNet: RGB input unimplemented";
  }
  else if (vi->format.colorFamily == VSColorFamily::cfYUV) {
    if (vi->format.subSamplingH == 1 && vi->format.subSamplingW == 1) {
      format = IOFormat::YUV420;
    }
    else {
      return "CycMuNet: only support 4:2:0 for YUV IO";
    }
  }
  else {
    return "CycMuNet: only support RGB or YUV format";
  }
  color_family = VSColorFamily(vi->format.colorFamily);

  std::array<float, 3> tmp {};
  err = getFloatArray("norm_mean", tmp, in, vsapi, default_norm_mean);
  if (err) {
    return "CycMuNet: norm_mean should have 3 values";
  }
  denorm.r.b = tmp[0];
  denorm.g.b = tmp[1];
  denorm.b.b = tmp[2];

  err = getFloatArray("norm_std", tmp, in, vsapi, default_norm_std);
  if (err) {
    return "CycMuNet: norm_std should have 3 values";
  }
  denorm.r.a = tmp[0];
  denorm.g.a = tmp[1];
  denorm.b.a = tmp[2];

  auto batch_size = int32_t(vsapi->mapGetInt(in, "batch_size", 0, &err));
  if (err) {
    batch_size = 1;
  }
  else if (batch_size < 1) {
    return "CycMuNet: batch_size should >= 1";
  }

  auto batch_size_fusion = int32_t(vsapi->mapGetInt(in, "batch_size_fusion", 0, &err));
  if (err) {
    batch_size_fusion = 1;
  }
  else if (batch_size_fusion < 1) {
    return "CycMuNet: batch_size_fusion should >= 1";
  }
  else if (batch_size_fusion > batch_size) {
    return "CycMuNet: batch_size_fusion should <= batch_size";
  }
  else if (batch_size % batch_size_fusion != 0) {
    vsapi->logMessage(mtWarning, "CycMuNet: batch_size not multiple of batch_size_fusion, this can be inefficient.",
                      core);
  }

  auto scale_factor = float(vsapi->mapGetFloat(in, "scale_factor", 0, nullptr));
  if (scale_factor <= 1) {
    return "CycMuNet: scale_factor should > 1";
  }

  auto extraction_layers = int32_t(vsapi->mapGetInt(in, "extraction_layers", 0, &err));
  if (err) {
    extraction_layers = 4;
  }
  else if (extraction_layers < 1) {
    return "CycMuNet: extraction_layers should >= 1";
  }

  auto use_fp16 = bool(vsapi->mapGetInt(in, "use_fp16", 0, &err));
  if (err) {
    use_fp16 = false;
  }

  raw_norm = bool(vsapi->mapGetInt(in, "raw_norm", 0, &err));
  if (err) {
    raw_norm = false;
  }

  model_path = vsapi->mapGetData(in, "model_path", 0, &err);
  if (err) {
    model_path = vsapi->getPluginPath(vsapi->getPluginByID("dev.tyty.aim.cycmunet", core));
    model_path = model_path.remove_filename() / "dev.tyty.aim.cycmunet";
  }

  std::string model_name = vsapi->mapGetData(in, "model", 0, &err);
  if (err) {
    model = ".";
  }
  else {
    model = model_name;
  }

  auto low_mem = bool(vsapi->mapGetInt(in, "low_mem", 0, &err));
  if (err) {
    low_mem = false;
  }

  config = {int32_t(vi->width),
            int32_t(vi->height),
            batch_size,
            batch_size_fusion,
            64,
            scale_factor,
            format,
            extraction_layers,
            use_fp16,
            low_mem};

  vo = *vi;
  vo.width = int(double(scale_factor) * vo.width);
  vo.height = int(double(scale_factor) * vo.height);
  vo.numFrames *= 2;
  vo.fpsNum *= 2;
  vsh::reduceRational(&vo.fpsNum, &vo.fpsDen);

  if (color_family == VSColorFamily::cfRGB) {
    size_t input_y_size = alignment((size_t) vi->width * vi->height * vi->format.bytesPerSample, 4096);
    input_shape_y = {vi->height, vi->width};
    input_tensor_y = {config.input_height, config.input_width};

    size_t output_y_size = alignment((size_t) vo.width * vo.height * vo.format.bytesPerSample, 4096);
    output_shape_y = {vo.height, vo.width};
    output_tensor_y = {offset_t(double(config.input_height) * scale_factor),
                       offset_t(double(config.input_width) * scale_factor)};

    err = cudaMalloc((void **) &ioBuffer[0], 3 * input_y_size);
    if (err != cudaSuccess) {
      return "CycMuNet: failed alloc " + std::to_string(3 * input_y_size) +
             " bytes of CUDA memory: " + cudaGetErrorName(cudaError_t(err)) + " (" +
             cudaGetErrorString(cudaError_t(err)) + ").";
    }

    err = cudaMalloc((void **) &ioBuffer[1], 3 * output_y_size);
    if (err != cudaSuccess) {
      return "CycMuNet: failed alloc " + std::to_string(3 * output_y_size) +
             " bytes of CUDA memory: " + cudaGetErrorName(cudaError_t(err)) + " (" +
             cudaGetErrorString(cudaError_t(err)) + ").";
    }

    ioPointer[0] = ioBuffer[0];
    ioPointer[1] = ioBuffer[0] + input_y_size;
    ioPointer[2] = ioBuffer[0] + 2 * input_y_size;
    ioPointer[3] = ioBuffer[1];
    ioPointer[4] = ioBuffer[1] + output_y_size;
    ioPointer[5] = ioBuffer[1] + 2 * output_y_size;
  }
  else {
    size_t input_y_size = alignment((size_t) vi->width * vi->height * vi->format.bytesPerSample, 4096);
    int32_t input_uv_width = (vi->width + (1 << vi->format.subSamplingW) - 1) >> vi->format.subSamplingW;
    int32_t input_uv_height = (vi->height + (1 << vi->format.subSamplingH) - 1) >> vi->format.subSamplingH;
    size_t input_uv_size = alignment((size_t) input_uv_width * input_uv_height * vi->format.bytesPerSample, 4096);
    input_shape_y = {vi->height, vi->width};
    input_shape_uv = {input_uv_height, input_uv_width};
    input_tensor_y = {config.input_height, config.input_width};
    input_tensor_uv = {(input_tensor_y[0] + (1 << vo.format.subSamplingH) - 1) >> vo.format.subSamplingH,
                       (input_tensor_y[1] + (1 << vo.format.subSamplingW) - 1) >> vo.format.subSamplingW};

    size_t output_y_size = alignment((size_t) vo.width * vo.height * vo.format.bytesPerSample, 4096);
    int32_t output_uv_width = (vo.width + (1 << vo.format.subSamplingW) - 1) >> vo.format.subSamplingW;
    int32_t output_uv_height = (vo.height + (1 << vo.format.subSamplingH) - 1) >> vo.format.subSamplingH;
    size_t output_uv_size = alignment((size_t) output_uv_width * output_uv_height * vo.format.bytesPerSample, 4096);
    output_shape_y = {vo.height, vo.width};
    output_shape_uv = {output_uv_height, output_uv_width};
    output_tensor_y = {offset_t(double(config.input_height) * scale_factor),
                       offset_t(double(config.input_width) * scale_factor)};
    output_tensor_uv = {(output_tensor_y[0] + (1 << vo.format.subSamplingH) - 1) >> vo.format.subSamplingH,
                        (output_tensor_y[1] + (1 << vo.format.subSamplingW) - 1) >> vo.format.subSamplingW};

    err = cudaMalloc((void **) &ioBuffer[0], input_y_size + 2 * input_uv_size);
    if (err != cudaSuccess) {
      return "CycMuNet: failed alloc " + std::to_string(input_y_size + 2 * input_uv_size) +
             " bytes of CUDA memory: " + cudaGetErrorName(cudaError_t(err)) + " (" +
             cudaGetErrorString(cudaError_t(err)) + ").";
    }

    err = cudaMalloc((void **) &ioBuffer[1], output_y_size + 2 * output_uv_size);
    if (err != cudaSuccess) {
      return "CycMuNet: failed alloc " + std::to_string(output_y_size + 2 * output_uv_size) +
             " bytes of CUDA memory: " + cudaGetErrorName(cudaError_t(err)) + " (" +
             cudaGetErrorString(cudaError_t(err)) + ").";
    }

    ioPointer[0] = ioBuffer[0];
    ioPointer[1] = ioBuffer[0] + input_y_size;
    ioPointer[2] = ioBuffer[0] + input_y_size + input_uv_size;
    ioPointer[3] = ioBuffer[1];
    ioPointer[4] = ioBuffer[1] + output_y_size;
    ioPointer[5] = ioBuffer[1] + output_y_size + output_uv_size;
  }

  requested_frames = std::vector<const VSFrame *> {size_t(config.batch_extract + 1), nullptr};
  return "";
}

std::string CycMuNetFilter::init2(const VSFrame *frame, VSCore *core, const VSAPI *vsapi) {
  auto frame_prop = vsapi->getFramePropertiesRO(frame);
  first_frame = frame;
  int err;

  color_space_t def, cur;
  if (color_family == VSColorFamily::cfRGB) {
    def = {VSC_PRIMARIES_BT709, VSC_TRANSFER_IEC_61966_2_1, VSC_MATRIX_RGB, VSC_RANGE_FULL};
  }
  else {
    def = {VSC_PRIMARIES_BT709, VSC_TRANSFER_BT709, VSC_MATRIX_BT709, VSC_RANGE_LIMITED};
  }

  cur.r = VSColorRange(vsapi->mapGetInt(frame_prop, "_ColorRange", 0, &err));
  if (err) {
    cur.r = def.r;
  }
  else if (cur.r > VSC_RANGE_LIMITED || cur.r < VSC_RANGE_FULL) {
    vsapi->logMessage(mtWarning, "CycMuNet: input has invalid color range. Assuming default color range.", core);
    cur.r = def.r;
  }

  cur.cp = VSColorPrimaries(vsapi->mapGetInt(frame_prop, "_Primaries", 0, &err));
  if (err) {
    cur.cp = def.cp;
  }
  switch (cur.cp) {
    case VSC_PRIMARIES_UNSPECIFIED: cur.cp = def.cp; break;
    case VSC_PRIMARIES_ST240_M: cur.cp = VSC_PRIMARIES_ST170_M; break;
  }

  cur.tc = VSTransferCharacteristics(vsapi->mapGetInt(frame_prop, "_Transfer", 0, &err));
  if (err) {
    cur.tc = def.tc;
  }
  switch (cur.tc) {
    case VSC_TRANSFER_UNSPECIFIED: cur.tc = def.tc; break;
    case VSC_TRANSFER_BT601:
    case VSC_TRANSFER_BT2020_10:
    case VSC_TRANSFER_BT2020_12: cur.tc = VSC_TRANSFER_BT709; break;
  }

  cur.mc = VSMatrixCoefficients(vsapi->mapGetInt(frame_prop, "_Matrix", 0, &err));
  if (err || cur.mc == VSC_MATRIX_UNSPECIFIED) {
    cur.mc = def.mc;
  }
  if (color_family == VSColorFamily::cfRGB && cur.mc != VSC_MATRIX_RGB) {
    vsapi->logMessage(mtWarning, "CycMuNet: RGB input must uses RGB Matrix.", core);
    cur.mc = VSC_MATRIX_RGB;
  }
  else if (color_family == VSColorFamily::cfYUV) {
    switch (cur.mc) {
      case VSC_MATRIX_RGB:
        vsapi->logMessage(mtWarning, "CycMuNet: YUV input must not use RGB Matrix, reset to default.", core);
        cur.mc = def.mc;
      case VSC_MATRIX_BT470_BG: cur.mc = VSC_MATRIX_ST170_M; break;
      case VSC_MATRIX_CHROMATICITY_DERIVED_NCL:
        switch (cur.cp) {
          case VSC_PRIMARIES_BT709: cur.mc = VSC_MATRIX_BT709; break;
          case VSC_PRIMARIES_BT470_M: cur.mc = VSC_MATRIX_ST170_M; break;
          case VSC_PRIMARIES_BT2020: cur.mc = VSC_MATRIX_BT2020_NCL; break;
        }
    }
  }

  std::string colorspace_folder;
  if (vo.format.colorFamily == VSColorFamily::cfRGB) {
    std::stringstream ss {"rgb_"};
    ss << cur.cp << '_' << cur.tc;
    colorspace_folder = ss.str();
  }
  else {
    std::stringstream ss {"yuv_"};
    ss << cur.cp << '_' << cur.tc << '_' << cur.mc;
    colorspace_folder = ss.str();
  }

  ctx = new InferenceContext {config, *logger, model_path / "engines" / model / colorspace_folder};

  if (!ctx->has_file()) {
    vsapi->logMessage(mtInformation, "CycMuNet: building engine. This will take some time.", core);
    OptimizationContext optimize_ctx {{config.input_width,
                                       config.input_height,
                                       {1, config.batch_extract, config.batch_extract},
                                       {1, config.batch_fusion, config.batch_fusion},
                                       config.feature_count,
                                       config.scale_factor,
                                       config.format,
                                       config.extraction_layers,
                                       config.use_fp16,
                                       config.low_mem},
                                      *logger,
                                      model_path};

    err = optimize_ctx.optimize(model / colorspace_folder);
    if (err) {
      return "CycMuNet: failed building engine for current input dimension";
    }
    vsapi->logMessage(mtInformation, "CycMuNet: done building engine.", core);
  }

  if (!ctx->load_engine()) {
    delete ctx;
    delete logger;
    return "CycMuNet: failed init context";
  }

  session = new InferenceSession {*ctx};
  if (!session->good()) {
    delete session;
    delete ctx;
    delete logger;
    return "CycMuNet: failed init session";
  }

  if (raw_norm) {
    for (int i = 0; i < 3; ++i) {
      norm.z[i].a = 1.0f / denorm.z[i].a;
      norm.z[i].b = -denorm.z[i].b * norm.z[i].a;
    }
  }
  else {
    auto isFloat = vo.format.sampleType == VSSampleType::stFloat;
    auto depth = isFloat ? 8 : vo.format.bitsPerSample;
    if (isFloat && cur.r == VSColorRange::VSC_RANGE_LIMITED) {
      vsapi->logMessage(
          mtWarning, "CycMuNet: Normalization value for limited range floating point input may be inaccurate.", core);
    }

    if (color_family == VSColorFamily::cfYUV) {
      float kr, kg, kb;

      if (cur.mc == VSC_MATRIX_CHROMATICITY_DERIVED_NCL) {
        const float *pPrimaries = nullptr;
        for (const auto &entry: colorPrimariesTable) {
          if (cur.cp == entry.colorPrimariesEnum) {
            pPrimaries = entry.primaries;
            break;
          }
        }

        if (pPrimaries == nullptr) {
          vsapi->logMessage(mtWarning, "CycMuNet: unknown color primary. Assume default (BT.709).", core);
          pPrimaries = colorPrimariesTable[0].primaries;
        }

        const auto [rX, rY, gX, gY, bX, bY, wX, wY] = (const float(&)[8])(*pPrimaries);
        float const rZ = 1.0f - (rX + rY);
        float const gZ = 1.0f - (gX + gY);
        float const bZ = 1.0f - (bX + bY);
        float const wZ = 1.0f - (wX + wY);
        kr = (rY * (wX * (gY * bZ - bY * gZ) + wY * (bX * gZ - gX * bZ) + wZ * (gX * bY - bX * gY))) /
             (wY * (rX * (gY * bZ - bY * gZ) + gX * (bY * rZ - rY * bZ) + bX * (rY * gZ - gY * rZ)));
        kb = (bY * (wX * (rY * gZ - gY * rZ) + wY * (gX * rZ - rX * gZ) + wZ * (rX * gY - gX * rY))) /
             (wY * (rX * (gY * bZ - bY * gZ) + gX * (bY * rZ - rY * bZ) + bX * (rY * gZ - gY * rZ)));
      }
      else {
        bool found = false;
        for (const auto &entry: matrixCoefficientsTable) {
          if (cur.mc == entry.matrixCoefficientsEnum) {
            kr = entry.kr;
            kb = entry.kb;
            found = true;
            break;
          }
        }

        if (!found) {
          return "CycMuNet: unsupported matrix coefficient, can not infer normalization parameter.";
        }
      }

      kg = 1 - kr - kb;

      auto [rs, rm] = denorm.r;
      auto [gs, gm] = denorm.g;
      auto [bs, bm] = denorm.b;
      auto uv_bias = cur.r ? float(1 << (depth - 1)) / float((1 << depth) - 1) : 0.5f;

      denorm.y.b = rm * kr + gm * kg + bm * kb;
      denorm.y.a = std::sqrt(rs * rs * kr * kr + gs * gs * kg * kg + bs * bs * kb * kb);
      denorm.u.b = (bm - denorm.y.b) / (1 - kb) / 2 + uv_bias;
      denorm.u.a = std::sqrt(bs * bs + denorm.y.a * denorm.y.a) / (1 - kb) / 2;
      denorm.v.b = (rm - denorm.y.b) / (1 - kr) / 2 + uv_bias;
      denorm.v.a = std::sqrt(rs * rs + denorm.y.a * denorm.y.a) / (1 - kr) / 2;

      float uv_scale = std::sqrt(float(1 << vo.format.subSamplingW) * float(1 << vo.format.subSamplingH));
      denorm.u.a /= uv_scale;
      denorm.v.a /= uv_scale;
    }

    if (cur.r == VSC_RANGE_FULL) {
      y_max = uv_max = float((1 << depth) - 1);
      y_min = uv_min = 0;
    }
    else {
      y_max = float(235 << (depth - 8));
      uv_max = float(240 << (depth - 8));
      y_min = uv_min = float(16 << (depth - 8));
    }
    if (isFloat) {
      auto unorm = float((1 << depth) - 1);
      y_max /= unorm;
      y_min /= unorm;
      uv_max /= unorm;
      uv_min /= unorm;
    }

    for (int i = 0; i < 3; ++i) {
      float c = (color_family == VSColorFamily::cfRGB) ? (y_max - y_min) : (i == 0 ? y_max - y_min : uv_max - uv_min);
      float d = (color_family == VSColorFamily::cfRGB) ? (y_min) : (i == 0 ? y_min : uv_min);
      norm.z[i].a = 1.0f / denorm.z[i].a;
      norm.z[i].b = -(denorm.z[i].b + d / c) * norm.z[i].a;
      norm.z[i].a /= c;
      denorm.z[i].a *= c;
      denorm.z[i].b = denorm.z[i].b * c + d;
    }
  }

  return "";
}

template<class F, class U>
std::string CycMuNetFilter::readPlane(md_view<F, 2> dst, md_uview<const U, 2> src, md_view<U, 2> cuda_tmp, float a,
                                      float b) {
  int err;
  if (src.is_contiguous()) {
    auto src_c = src.as_view();

    err = cudaMemcpyAsync(cuda_tmp.data, src_c.data, cuda_tmp.size() * sizeof(U), cudaMemcpyHostToDevice,
                          session->stream);
    if (err != cudaSuccess) {
      return "CycMuNet: failed copy " + std::to_string(cuda_tmp.size() * sizeof(U)) +
             " bytes of CUDA memory: " + cudaGetErrorName(cudaError_t(err)) + " (" +
             cudaGetErrorString(cudaError_t(err)) + ").";
    }
  }
  else {
    for (offset_t i = 0; i < src.shape[0]; ++i) {
      auto line = cuda_tmp.at(i);
      err =
          cudaMemcpyAsync(line.data, src.at(i).data, line.size() * sizeof(U), cudaMemcpyHostToDevice, session->stream);
      if (err != cudaSuccess) {
        return "CycMuNet: failed copy " + std::to_string(line.size() * sizeof(U)) +
               " bytes of CUDA memory: " + cudaGetErrorName(cudaError_t(err)) + " (" +
               cudaGetErrorString(cudaError_t(err)) + ").";
      }
    }
  }

  import_pixel<F, U>(dst, cuda_tmp, a, b, session->stream);
  return "";
}

template<class F, class U>
std::string CycMuNetFilter::writePlane(md_uview<U, 2> dst, md_view<const F, 2> src, md_view<U, 2> cuda_tmp, float a,
                                       float b, float min, float max) {
  export_pixel<F, U>(cuda_tmp, src, a, b, min, max, session->stream);

  int err;
  if (dst.is_contiguous()) {
    auto dst_c = dst.as_view();

    err = cudaMemcpyAsync(dst_c.data, cuda_tmp.data, dst_c.size() * sizeof(U), cudaMemcpyDeviceToHost, session->stream);
    if (err != cudaSuccess) {
      return "CycMuNet: failed copy " + std::to_string(dst_c.size() * sizeof(U)) +
             " bytes of CUDA memory: " + cudaGetErrorName(cudaError_t(err)) + " (" +
             cudaGetErrorString(cudaError_t(err)) + ").";
    }
  }
  else {
    for (offset_t i = 0; i < cuda_tmp.shape[0]; ++i) {
      auto line = cuda_tmp.at(i);
      err =
          cudaMemcpyAsync(dst.at(i).data, line.data, line.size() * sizeof(U), cudaMemcpyDeviceToHost, session->stream);
      if (err != cudaSuccess) {
        return "CycMuNet: failed copy " + std::to_string(line.size() * sizeof(U)) +
               " bytes of CUDA memory: " + cudaGetErrorName(cudaError_t(err)) + " (" +
               cudaGetErrorString(cudaError_t(err)) + ").";
      }
    }
  }

  return "";
}

template<class U>
std::string CycMuNetFilter::readYUV(offset_t position, const VSFrame *frame, const VSAPI *vsapi) {
  auto bps = vo.format.bytesPerSample;
  md_uview<const U, 2> input_planes[3] = {
      {reinterpret_cast<const U *>(vsapi->getReadPtr(frame, 0)), input_shape_y, {vsapi->getStride(frame, 0) / bps, 1}},
      {reinterpret_cast<const U *>(vsapi->getReadPtr(frame, 1)), input_shape_uv, {vsapi->getStride(frame, 1) / bps, 1}},
      {reinterpret_cast<const U *>(vsapi->getReadPtr(frame, 2)),
       input_shape_uv,
       {vsapi->getStride(frame, 2) / bps, 1}}};
  md_view<U, 2> input_tmps[3] = {{reinterpret_cast<U *>(ioPointer[0]), input_shape_y},
                                 {reinterpret_cast<U *>(ioPointer[1]), input_shape_uv},
                                 {reinterpret_cast<U *>(ioPointer[2]), input_shape_uv}};
  md_view<uint8_t, 2> y_tensor {static_cast<uint8_t *>(session->input[0]),
                                {config.batch_extract, offset_t(session->input_size())}};
  md_view<uint8_t, 3> uv_tensor {static_cast<uint8_t *>(session->input[1]),
                                 {config.batch_extract, 2, offset_t(session->input_size() / 4)}};

  for (int i = 0; i < 3; ++i) {
    shape_t<2> dim;
    uint8_t *tensor_ptr;
    if (i == 0) {
      dim = input_tensor_y;
      tensor_ptr = y_tensor.at(position).data;
    }
    else {
      dim = input_tensor_uv;
      tensor_ptr = uv_tensor.at(position, i - 1).data;
    }

    std::string result;
    if (config.use_fp16) {
      result = readPlane<half, U>({(half *) tensor_ptr, dim}, input_planes[i], input_tmps[i], norm.z[i].a, norm.z[i].b);
    }
    else {
      result =
          readPlane<float, U>({(float *) tensor_ptr, dim}, input_planes[i], input_tmps[i], norm.z[i].a, norm.z[i].b);
    }
    if (!result.empty()) {
      return result;
    }
  }

  return "";
}

template<class U>
std::string CycMuNetFilter::writeYUV(offset_t position, VSFrame *frame, const VSAPI *vsapi) {
  auto bps = vo.format.bytesPerSample;
  md_uview<U, 2> output_planes[3] = {
      {reinterpret_cast<U *>(vsapi->getWritePtr(frame, 0)), output_shape_y, {vsapi->getStride(frame, 0) / bps, 1}},
      {reinterpret_cast<U *>(vsapi->getWritePtr(frame, 1)), output_shape_uv, {vsapi->getStride(frame, 1) / bps, 1}},
      {reinterpret_cast<U *>(vsapi->getWritePtr(frame, 2)), output_shape_uv, {vsapi->getStride(frame, 2) / bps, 1}}};
  md_view<U, 2> output_tmps[3] = {{reinterpret_cast<U *>(ioPointer[3]), output_shape_y},
                                  {reinterpret_cast<U *>(ioPointer[4]), output_shape_uv},
                                  {reinterpret_cast<U *>(ioPointer[5]), output_shape_uv}};
  int32_t y_idx, uv_idx;
  if (position % 2 == 0) {
    y_idx = 0;
    uv_idx = 1;
  }
  else {
    y_idx = 2;
    uv_idx = 3;
  }

  position /= 2;

  md_view<uint8_t, 2> y_tensor {static_cast<uint8_t *>(session->output[y_idx]),
                                {config.batch_fusion, offset_t(session->output_size())}};
  md_view<uint8_t, 3> uv_tensor {static_cast<uint8_t *>(session->output[uv_idx]),
                                 {config.batch_fusion, 2, offset_t(session->output_size() / 4)}};

  for (int i = 0; i < 3; ++i) {
    shape_t<2> dim;
    uint8_t *tensor_ptr;
    float min, max;
    if (i == 0) {
      dim = output_tensor_y;
      tensor_ptr = y_tensor.at(position).data;
      min = y_min;
      max = y_max;
    }
    else {
      dim = output_tensor_uv;
      tensor_ptr = uv_tensor.at(position, i - 1).data;
      min = uv_min;
      max = uv_max;
    }

    std::string result;
    if (config.use_fp16) {
      result = writePlane<half, U>(output_planes[i], {(half *) tensor_ptr, dim}, output_tmps[i], denorm.z[i].a,
                                   denorm.z[i].b, min, max);
    }
    else {
      result = writePlane<float, U>(output_planes[i], {(float *) tensor_ptr, dim}, output_tmps[i], denorm.z[i].a,
                                    denorm.z[i].b, min, max);
    }
    if (!result.empty()) {
      return result;
    }
  }

  return "";
}

void CycMuNetFilter::requestFrames(int n, VSFrameContext *frameCtx, const VSAPI *vsapi) const {
  auto request = [&](int i) { vsapi->requestFrameFilter(i, node, frameCtx); };
//  auto request = [&](int i) {
//    vsapi->requestFrameFilter(i, node, frameCtx);
//    logger->log(nvinfer1::ILogger::Severity::kWARNING, "Requesting f #" + std::to_string(i));
//  };

  // Get the index of the first frame of current scene
  auto scene_begin_index_current = n / 2 >= scene_begin_index_pending ? scene_begin_index_pending : scene_begin_index;
  auto scene_begin_index_next = n / 2 >= scene_begin_index_pending ? num_frames : scene_begin_index_pending;

  if (n / 2 == scene_begin_index_current && n % 2 == 0) {
    request(scene_begin_index_current);
  }

  int begin = n / 2 + 1;
  auto batch_begin = n - ((n - 2 * scene_begin_index_current) % (2 * config.batch_extract));
  int end = std::min((batch_begin / 2 + 1) + config.batch_extract, scene_begin_index_next);
  if (n == batch_begin && begin != end) {
    for (int i = begin; i < end; ++i) {
      request(i);
    }
  }
  else {
    // Have to at least request one frame
    request(end - 1);
  }
}

std::string CycMuNetFilter::prepareFrame(int n, VSFrameContext *frameCtx, const VSAPI *vsapi) {
  auto loadFrameAt = [&](const VSFrame *frame_in, int32_t offset) -> std::string {
    if (color_family == VSColorFamily::cfRGB) {
      // TODO RGB loader
    }
    else {
      if (vo.format.sampleType == VSSampleType::stFloat) {
        if (vo.format.bytesPerSample == 2) {
          return readYUV<half>(offset, frame_in, vsapi);
        }
        else if (vo.format.bytesPerSample == 4) {
          return readYUV<float>(offset, frame_in, vsapi);
        }
      } else {
        if (vo.format.bytesPerSample == 1) {
          return readYUV<uint8_t>(offset, frame_in, vsapi);
        }
        else if (vo.format.bytesPerSample == 2) {
          return readYUV<uint16_t>(offset, frame_in, vsapi);
        }
      }

      return "CycMuNet: unexpected format";
    }

    return "";
  };

  if (n % 2 == 1) {
    return "";
  }

  int input_index = n / 2;
  if (input_index == scene_begin_index_pending) {
    scene_begin_index = scene_begin_index_pending;
    scene_begin_index_pending = num_frames;
  }
  int extract_index = (input_index - scene_begin_index) % config.batch_extract;
  std::string result;

  // an extract batch starts at this frame
  if (extract_index == 0) {
    // special logic at first frame of scene
    if (input_index == scene_begin_index) {
      session->extractBatch(0, 0, 1);
      requested_frames[0] = first_frame;
      loadFrameAt(requested_frames[0], 0);
      session->extract();
    }
    else {
      session->duplicateExtractOutput(config.batch_extract, 0);
      requested_frames[0] = requested_frames[config.batch_extract];
      requested_frames[config.batch_extract] = nullptr;
    }

    int begin = input_index + 1;
    int end = std::min(begin + config.batch_extract, num_frames);
    session->extractBatch(0, 1, config.batch_extract);
    for (loaded_frames = 0; loaded_frames < end - begin; ++loaded_frames) {
      auto frame_in = vsapi->getFrameFilter(begin + loaded_frames, node, frameCtx);
      assert(frame_in);
      int err;  // ignore key absent error: no scene change info is not critical
      if (vsapi->mapGetInt(vsapi->getFramePropertiesRO(frame_in), "_SceneChangePrev", 0, &err)) {
        first_frame = frame_in;
        // This frame starts next scene. Record its index, and stop at previous frame
        scene_begin_index_pending = begin + loaded_frames;
        break;
      }
      loadFrameAt(frame_in, loaded_frames);
      requested_frames[loaded_frames + 1] = frame_in;
    }

    // This happens if we just reached the end for this scene.
    // We then skip extract, duplicate extract output to match frame count.
    if (loaded_frames == 0) {
      session->duplicateExtractOutput(0, 1);
      loaded_frames = 1;
    }
    else {
      // process all new frames.
      session->extractBatch(0, 1, loaded_frames);
      session->extract();

      if (loaded_frames != config.batch_extract) {
        // We reached end of scene. duplicate extract output of last frame to match frame count.
        session->duplicateExtractOutput(loaded_frames, loaded_frames + 1);
      }
    }
  }

  int fusion_index = extract_index % config.batch_fusion;

  // a fusion batch starts at this frame
  if (fusion_index == 0) {
    int begin = extract_index;
    int end = std::min(begin + config.batch_fusion, loaded_frames + 1);
    session->fusionBatch(begin, 0, end - begin);
    session->fusion();
  }

  return "";
}

std::string CycMuNetFilter::extractFrame(int n, VSFrame *&frame, VSCore *core, const VSAPI *vsapi) {
  int offset = (n - 2 * scene_begin_index) % (2 * config.batch_extract);
  int src_index = offset / 2;
  bool free_src = offset % 2;
  offset %= 2 * config.batch_fusion;

  frame = vsapi->newVideoFrame(&vo.format, vo.width, vo.height, requested_frames[src_index], core);
  if (free_src) {
    vsapi->freeFrame(requested_frames[src_index]);
    requested_frames[src_index] = nullptr;
  }

  if (color_family == VSColorFamily::cfRGB) {
    // TODO RGB extractor
  }
  else {
    if (vo.format.sampleType == VSSampleType::stFloat) {
      if (vo.format.bytesPerSample == 2) {
        return writeYUV<half>(offset, frame, vsapi);
      }
      else if (vo.format.bytesPerSample == 4) {
        return writeYUV<float>(offset, frame, vsapi);
      }
    } else {
      if (vo.format.bytesPerSample == 1) {
        return writeYUV<uint8_t>(offset, frame, vsapi);
      }
      else if (vo.format.bytesPerSample == 2) {
        return writeYUV<uint16_t>(offset, frame, vsapi);
      }
    }

    return "CycMuNet: unexpected format";
  }

  return "";
}

CycMuNetFilter::~CycMuNetFilter() {
  for (auto p: ioBuffer) {
    if (p != nullptr) {
      cudaFree(p);
    }
  }
}

// ---------------------------------------------------------------------------------------------------------------------
// VS API

static const VSFrame *VS_CC cycmunetGetFrame(int n, int activationReason, void *instanceData, void **frameData,
                                             VSFrameContext *frameCtx, VSCore *core, const VSAPI *vsapi) {
  auto *filter = static_cast<CycMuNetFilter *>(instanceData);
  std::string err;

  if (activationReason == arInitial) {
    filter->requestFrames(n, frameCtx, vsapi);
    return nullptr;
  }
  else if (activationReason == arAllFramesReady) {
    if (n == 0) {
      const VSFrame *frame = vsapi->getFrameFilter(0, filter->node, frameCtx);
      filter->init2(frame, core, vsapi);
    }

    err = filter->prepareFrame(n, frameCtx, vsapi);
    if (!err.empty()) {
      vsapi->setFilterError(err.c_str(), frameCtx);
      vsapi->freeNode(filter->node);
      return nullptr;
    }

    VSFrame *out{};
    err = filter->extractFrame(n, out, core, vsapi);
    if (err.empty()) {
      err = filter->synchronize();
    }
    if (!err.empty()) {
      vsapi->setFilterError(err.c_str(), frameCtx);
      vsapi->freeNode(filter->node);
      return nullptr;
    }
    return out;
  }
  return nullptr;
}

static void VS_CC cycmunetFree(void *instanceData, VSCore *, const VSAPI *vsapi) {
  auto *d = static_cast<CycMuNetFilter *>(instanceData);
  vsapi->freeNode(d->node);
  delete d;
}

static void VS_CC cycmunetCreate(const VSMap *in, VSMap *out, void *, VSCore *core, const VSAPI *vsapi) {
  auto filter = new CycMuNetFilter();
  auto err = filter->init1(in, core, vsapi);
  if (!err.empty()) {
    vsapi->mapSetError(out, err.c_str());
    vsapi->freeNode(filter->node);
    return;
  }

  VSFilterDependency deps[] = {{filter->node, rpNoFrameReuse}};
  vsapi->createVideoFilter(out, "CycMuNet", &filter->vo, cycmunetGetFrame, cycmunetFree, fmFrameState, deps, 1, filter,
                           core);
}

static void VS_CC dependencyVersion(const VSMap *, VSMap *out, void *, VSCore *, const VSAPI *vsapi) {
  vsapi->mapSetData(out, "tensorrt_version", std::to_string(getInferLibVersion()).c_str(), -1, ptData, maReplace);

  vsapi->mapSetData(out, "tensorrt_version_build", std::to_string(NV_TENSORRT_VERSION).c_str(), -1, ptData, maReplace);

  int runtime_version;
  cudaRuntimeGetVersion(&runtime_version);
  vsapi->mapSetData(out, "cuda_runtime_version", std::to_string(runtime_version).c_str(), -1, ptData, maReplace);

  vsapi->mapSetData(out, "cuda_runtime_version_build", std::to_string(__CUDART_API_VERSION).c_str(), -1, ptData,
                    maReplace);
}

VS_EXTERNAL_API(void) VapourSynthPluginInit2(VSPlugin *plugin, const VSPLUGINAPI *vspapi) {
  UDOLayers::registerPlugins();
  vspapi->configPlugin("dev.tyty.aim.cycmunet", "cycmunet", "CycMuNet+ Spatial-Temporal Super Resolution Filter",
                       VS_MAKE_VERSION(1, 0), VAPOURSYNTH_API_VERSION, 0, plugin);
  vspapi->registerFunction("CycMuNet",
                           "clip:vnode;"
                           "scale_factor:float;"
                           "batch_size:int:opt;"
                           "batch_size_fusion:int:opt;"
                           "extraction_layers:int:opt;"
                           "use_fp16:int:opt;"
                           "norm_mean:float[]:opt;"
                           "norm_std:float[]:opt;"
                           "raw_norm:int:opt;"
                           "model:data:opt;"
                           "model_path:data:opt;"
                           "low_mem:int:opt;",
                           "clip:vnode;", cycmunetCreate, nullptr, plugin);
  vspapi->registerFunction("CycMuNetVersion", "", "", dependencyVersion, nullptr, plugin);
}
