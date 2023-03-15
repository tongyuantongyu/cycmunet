#include <inttypes.h>
#include <malloc.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <chrono>

#include <cuda_fp16.h>

#include "cuda_runtime_api.h"
#include "inference.h"
#include "layers.h"
#include "logging.h"
#include "reformat.h"

#include "debug/reveal.h"
#include "optimize.h"

#define CUDA_CHECK(status)                                                                                             \
  do {                                                                                                                 \
    auto ret = (status);                                                                                               \
    if (ret != 0) {                                                                                                    \
      std::cerr << "Cuda failure at " __FILE__ ":" << __LINE__ << ": " << ret << std::endl;                            \
      exit(2);                                                                                                         \
    }                                                                                                                  \
  } while (0)

/* Most of this is from x264 */

/* YUV4MPEG2 raw 420 yuv file operation */
typedef struct {
  FILE *fp;
  int width, height;
  int par_width, par_height;
  int next_frame;
  int seq_header_len, frame_header_len;
  int frame_size;
  int csp;
  int fps_num, fps_den;
  int range;
} y4m_input_t;

typedef struct {
  int plane_cnt;
  uint8_t *plane[4];
} image_t;

#define Y4M_MAGIC "YUV4MPEG2"
#define MAX_YUV4_HEADER 160
#define Y4M_FRAME_MAGIC "FRAME"
#define MAX_FRAME_HEADER 80

int open_file_y4m_read(const char *filename, y4m_input_t *y4m) {
  int i, n, d;
  int interlaced;
  char header[MAX_YUV4_HEADER + 10];
  char *tokstart, *tokend, *header_end;
  y4m_input_t h {};

  h.next_frame = 0;

  if (!strcmp(filename, "-"))
    h.fp = stdin;
  else
    h.fp = fopen(filename, "rb");
  if (h.fp == NULL)
    return -1;

  h.frame_header_len = strlen(Y4M_FRAME_MAGIC) + 1;

  /* Read header */
  for (i = 0; i < MAX_YUV4_HEADER; i++) {
    header[i] = fgetc(h.fp);
    if (header[i] == '\n') {
      /* Add a space after last option. Makes parsing "444" vs
               "444alpha" easier. */
      header[i + 1] = 0x20;
      header[i + 2] = 0;
      break;
    }
  }
  if (i == MAX_YUV4_HEADER || strncmp(header, Y4M_MAGIC, strlen(Y4M_MAGIC)))
    return -1;

  /* Scan properties */
  header_end = &header[i + 1]; /* Include space */
  h.seq_header_len = i + 1;
  for (tokstart = &header[strlen(Y4M_MAGIC) + 1]; tokstart < header_end; tokstart++) {
    if (*tokstart == 0x20)
      continue;
    switch (*tokstart++) {
      case 'W': /* Width. Required. */
        h.width = strtol(tokstart, &tokend, 10);
        tokstart = tokend;
        break;
      case 'H': /* Height. Required. */
        h.height = strtol(tokstart, &tokend, 10);
        tokstart = tokend;
        break;
      case 'C': /* Color space */
        if (strncmp("420", tokstart, 3)) {
          fprintf(stderr, "Colorspace unhandled\n");
          return -1;
        }
        tokstart = strchr(tokstart, 0x20);
        break;
      case 'I': /* Interlace type */
        switch (*tokstart++) {
          case 'p': interlaced = 0; break;
          case '?':
          case 't':
          case 'b':
          case 'm':
          default: interlaced = 1; fprintf(stderr, "Warning, this sequence might be interlaced\n");
        }
        break;
      case 'F': /* Frame rate - 0:0 if unknown */
                /* Frame rate in unimportant. */
        if (sscanf(tokstart, "%d:%d", &n, &d) == 2 && n && d) {
          h.fps_num = n;
          h.fps_den = d;
        }
        tokstart = strchr(tokstart, 0x20);
        break;
      case 'A': /* Pixel aspect - 0:0 if unknown */
                /* Don't override the aspect ratio if sar has been explicitly set on the commandline. */
        if (sscanf(tokstart, "%d:%d", &n, &d) == 2 && n && d) {
          h.par_width = n;
          h.par_height = d;
        }
        tokstart = strchr(tokstart, 0x20);
        break;
      case 'X': /* Vendor extensions */
        if (!strncmp("YSCSS=", tokstart, 6)) {
          /* Older nonstandard pixel format representation */
          tokstart += 6;
          if (strncmp("420JPEG", tokstart, 7) && strncmp("420MPEG2", tokstart, 8) && strncmp("420PALDV", tokstart, 8)) {
            fprintf(stderr, "Unsupported extended colorspace\n");
            return -1;
          }
        }
        else if (!strncmp("COLORRANGE=", tokstart, 11)) {
          tokstart += 11;

          if (!strncmp("FULL", tokstart, 4)) {
            h.range = 1;
            tokstart += 4;
          }
          else if (!strncmp("LIMITED", tokstart, 7)) {
            h.range = 0;
            tokstart += 7;
          }
          else {
            fprintf(stderr, "Unsupported extended color range\n");
            return -1;
          }
        }
        tokstart = strchr(tokstart, 0x20);
        break;
    }
  }

  fprintf(stderr, "yuv4mpeg: %ix%i@%i/%ifps, %i:%i\n", h.width, h.height, h.fps_num, h.fps_den, h.par_width,
          h.par_height);

  *y4m = h;
  return 0;
}

int read_frame_y4m(y4m_input_t *handle, image_t *pic, int framenum) {
  int slen = strlen(Y4M_FRAME_MAGIC);
  int i = 0;
  char header[16];
  y4m_input_t *h = handle;

  if (framenum != h->next_frame && framenum != -1) {
    if (fseek(h->fp, (uint64_t) framenum * (3 * (h->width * h->height) / 2 + h->frame_header_len) + h->seq_header_len,
              SEEK_SET))
      return -1;
  }

  /* Read frame header - without terminating '\n' */
  if (fread(header, 1, slen, h->fp) != slen)
    return -1;

  header[slen] = 0;
  if (strncmp(header, Y4M_FRAME_MAGIC, slen)) {
    fprintf(stderr, "Bad header magic (%" PRIx32 " <=> %s)\n", *((uint32_t *) header), header);
    return -1;
  }

  /* Skip most of it */
  while (i < MAX_FRAME_HEADER && fgetc(h->fp) != '\n')
    i++;
  if (i == MAX_FRAME_HEADER) {
    fprintf(stderr, "Bad frame header!\n");
    return -1;
  }
  h->frame_header_len = i + slen + 1;

  if (fread(pic->plane[0], 1, h->width * h->height, h->fp) <= 0 ||
      fread(pic->plane[1], 1, h->width * h->height / 4, h->fp) <= 0 ||
      fread(pic->plane[2], 1, h->width * h->height / 4, h->fp) <= 0)
    return -1;

  h->next_frame = framenum + 1;

  return 0;
}

int open_file_y4m_write(const char *filename, y4m_input_t *y4m) {
  y4m_input_t h {};

  if (!strcmp(filename, "-"))
    h.fp = stdout;
  else
    h.fp = fopen(filename, "wb");
  if (h.fp == NULL)
    return -1;

  fprintf(h.fp, "YUV4MPEG2 ");
  *y4m = h;

  return 0;
}

int write_header_y4m(y4m_input_t *handle) {
  y4m_input_t *h = handle;
  auto f = h->fp;

  fprintf(f, "W%d H%d F%d:%d Ip A%d:%d C420jpeg XYSCSS=420JPEG ", h->width, h->height, h->fps_num, h->fps_den,
          h->par_width, h->par_height);
  if (h->range) {
    fprintf(f, "XCOLORRANGE=FULL");
  }
  else {
    fprintf(f, "XCOLORRANGE=LIMITED");
  }
  fprintf(f, "\n");

  return 0;
}

int write_frame_y4m(y4m_input_t *handle, image_t *pic) {
  y4m_input_t *h = handle;

  fprintf(h->fp, "FRAME\n");
  if (fwrite(pic->plane[0], 1, h->width * h->height, h->fp) <= 0 ||
      fwrite(pic->plane[1], 1, h->width * h->height / 4, h->fp) <= 0 ||
      fwrite(pic->plane[2], 1, h->width * h->height / 4, h->fp) <= 0) {
    return -1;
  }

  return 0;
}

int close_file_y4m(y4m_input_t *handle) {
  y4m_input_t *h = handle;
  if (!h || !h->fp)
    return 0;
  fclose(h->fp);
  return 0;
}

static Logger gLogger(Logger::Severity::kINFO);

static size_t alignment(size_t size, size_t alignment) {
  return (size + (alignment - 1)) & (~(alignment - 1));
}

struct scale_ratios {
  struct fma {
    float a;
    float b;
  };

  union {
    fma r[3];
    struct {
      fma y, u, v;
    };
  };
};

typedef std::chrono::duration<double, std::ratio<1, 1>> second;

int main(int argc, char **argv) {
  UDOLayers::registerPlugins();

  const char *input_name = "-", *output_name = "-";
  if (argc == 2) {
    output_name = argv[1];
  }
  else if (argc == 3) {
    input_name = argv[1];
    output_name = argv[2];
  }

  y4m_input_t input, output;
  if (open_file_y4m_read(input_name, &input) < 0) {
    std::cerr << "Failed open/parse input y4m.\n";
    return 1;
  };

  if (open_file_y4m_write(output_name, &output) < 0) {
    std::cerr << "Failed open output y4m.\n";
    return 1;
  };

  InferenceConfig config {int32_t(alignment(input.width, 32)),
                          int32_t(alignment(input.height, 32)),
                          1,
                          1,
                          64,
                          2.0,
                          IOFormat::YUV420,
                          4,
                          false,
                          false};

  InferenceContext context {config, gLogger, "models/engines"};
  if (!context.has_file()) {
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
                                      gLogger,
                                      "models"};

    if (optimize_ctx.optimize(".")) {
      std::cerr << "Failed create engine.\n";
      return 1;
    }
  }

  if (!context.load_engine()) {
    std::cerr << "Failed load engine.\n";
    return 1;
  }

  InferenceSession session {context};

  output.width = int32_t(double(input.width) * config.scale_factor);
  output.height = int32_t(double(input.height) * config.scale_factor);
  if (input.fps_den % 2 == 0) {
    output.fps_num = input.fps_num;
    output.fps_den = input.fps_den / 2;
  }
  else {
    output.fps_num = input.fps_num * 2;
    output.fps_den = input.fps_den;
  }
  output.par_width = input.par_width;
  output.par_height = input.par_height;
  output.range = input.range;

  write_header_y4m(&output);

  // TODO These are for YUV420
  size_t input_y_size = alignment((size_t) input.width * input.height, 4096);
  int32_t input_uv_width = (input.width + 1) / 2;
  int32_t input_uv_height = (input.height + 1) / 2;
  size_t input_uv_size = alignment((size_t) input_uv_width * input_uv_height, 4096);

  size_t output_y_size = alignment((size_t) output.width * output.height, 4096);
  int32_t output_uv_width = (output.width + 1) / 2;
  int32_t output_uv_height = (output.height + 1) / 2;
  size_t output_uv_size = alignment((size_t) output_uv_width * output_uv_height, 4096);

  int32_t output_tensor_width = int32_t(double(config.input_width) * config.scale_factor);
  int32_t output_tensor_height = int32_t(double(config.input_height) * config.scale_factor);
  int32_t output_tensor_uv_width = (output_tensor_width + 1) / 2;
  int32_t output_tensor_uv_height = (output_tensor_height + 1) / 2;

  uint8_t *input_buffer, *output_buffer;
  CUDA_CHECK(cudaMalloc((void **) &input_buffer, input_y_size + 2 * input_uv_size));
  CUDA_CHECK(cudaMalloc((void **) &output_buffer, output_y_size + 2 * output_uv_size));

  auto input_data = std::make_unique<uint8_t[]>(input_y_size + 2 * input_uv_size);

  image_t frame_in {3,
                    {
                        input_data.get(),
                        input_data.get() + input_y_size,
                        input_data.get() + input_y_size + input_uv_size,
                    }};

  auto output_data = std::make_unique<uint8_t[]>(output_y_size + 2 * output_uv_size);

  image_t frame_out {3,
                     {
                         output_data.get(),
                         output_data.get() + output_y_size,
                         output_data.get() + output_y_size + output_uv_size,
                     }};

  md_view<uint8_t, 2> input_ptrs[3] = {
      {input_buffer, {input.height, input.width}},
      {input_buffer + input_y_size, {input_uv_height, input_uv_width}},
      {input_buffer + input_y_size + input_uv_size, {input_uv_height, input_uv_width}}};

  md_view<uint8_t, 2> output_ptrs[3] = {
      {output_buffer, {output.height, output.width}},
      {output_buffer + output_y_size, {output_uv_height, output_uv_width}},
      {output_buffer + output_y_size + output_uv_size, {output_uv_height, output_uv_width}}};

  scale_ratios norm;
  scale_ratios denorm {{{
      {0.16822528389394978, 0.4585554},
      {0.075699365150944, 0.47363819323806267},
      {0.09021753930074448, 0.5187531388984347},
  }}};

  for (int i = 0; i < 3; ++i) {
    float c = input.range ? 255.0f : (i ? 224.0f : 219.0f);
    float d = input.range ? 0.0f : 16.0f;
    norm.r[i].a = 1.0f / denorm.r[i].a;
    norm.r[i].b = -(denorm.r[i].b + d / c) * norm.r[i].a;
    norm.r[i].a /= c;
    denorm.r[i].a *= c;
    denorm.r[i].b = denorm.r[i].b * c + d;
  }

  std::cerr << "Initialization done: " << input.width << '*' << input.height << '@' << input.fps_num << '/'
            << input.fps_den << "fps -> " << output.width << '*' << output.height << '@' << output.fps_num << '/'
            << output.fps_den << "fps\n";

  auto loadYUVFrame = [&](offset_t position) {
    md_view<uint8_t, 2> y_tensor {static_cast<uint8_t *>(session.input[0]),
                                  {config.batch_extract + 1, offset_t(session.input_size())}};
    md_view<uint8_t, 3> uv_tensor {static_cast<uint8_t *>(session.input[1]),
                                   {config.batch_extract + 1, 2, offset_t(session.input_size() / 4)}};

    for (int i = 0; i < 3; ++i) {
      CUDA_CHECK(cudaMemcpyAsync(input_ptrs[i].data, frame_in.plane[i], input_ptrs[i].size(), cudaMemcpyHostToDevice,
                                 session.stream));
      shape_t<2> dim;
      uint8_t *tensor_ptr;
      if (i == 0) {
        dim = {config.input_height, config.input_width};
        tensor_ptr = y_tensor.at(position).data;
      }
      else {
        dim = {config.input_height / 2, config.input_width / 2};
        tensor_ptr = uv_tensor.at(position, i - 1).data;
      }
      if (config.use_fp16) {
        import_pixel<half, uint8_t>({(half *) tensor_ptr, dim}, input_ptrs[i], norm.r[i].a, norm.r[i].b,
                                    session.stream);
      }
      else {
        import_pixel<float, uint8_t>({(float *) tensor_ptr, dim}, input_ptrs[i], norm.r[i].a, norm.r[i].b,
                                     session.stream);
      }
    }
  };

  auto extractYUVFrame = [&](offset_t position) {
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

    md_view<uint8_t, 2> y_tensor {static_cast<uint8_t *>(session.output[y_idx]),
                                  {config.batch_fusion, offset_t(session.output_size())}};
    md_view<uint8_t, 3> uv_tensor {static_cast<uint8_t *>(session.output[uv_idx]),
                                   {config.batch_fusion, 2, offset_t(session.output_size() / 4)}};

    for (int i = 0; i < 3; ++i) {
      shape_t<2> dim;
      uint8_t *tensor_ptr;
      float min, max;
      if (i == 0) {
        dim = {output_tensor_height, output_tensor_width};
        tensor_ptr = y_tensor.at(position).data;
        min = input.range ? 0 : 16;
        max = input.range ? 255 : 235;
      }
      else {
        dim = {output_tensor_uv_height, output_tensor_uv_width};
        tensor_ptr = uv_tensor.at(position, i - 1).data;
        min = input.range ? 0 : 16;
        max = input.range ? 255 : 240;
      }
      if (config.use_fp16) {
        export_pixel<half, uint8_t>(output_ptrs[i], {(half *) tensor_ptr, dim}, denorm.r[i].a, denorm.r[i].b, min, max,
                                    session.stream);
      }
      else {
        export_pixel<float, uint8_t>(output_ptrs[i], {(float *) tensor_ptr, dim}, denorm.r[i].a, denorm.r[i].b, min,
                                     max, session.stream);
      }

      CUDA_CHECK(cudaMemcpyAsync(frame_out.plane[i], output_ptrs[i].data, output_ptrs[i].size(), cudaMemcpyDeviceToHost,
                                 session.stream));
    }
  };

  if (read_frame_y4m(&input, &frame_in, -1) != 0) {
    std::cerr << "Failed loading first frame.\n";
    return 1;
  }
  session.extractBatch(0, 0, 1);
  loadYUVFrame(0);
  session.extract();

  int32_t loaded_frames = -1;
  bool end = false;
  second this_batch, elapsed;
  uint64_t count = 0;

  auto begin = std::chrono::steady_clock::now();
  while (!end) {
    auto batch_begin = std::chrono::steady_clock::now();

    session.extractBatch(0, 1, config.batch_extract);
    if (loaded_frames != -1) {
      session.duplicateExtractOutput(config.batch_extract, 0);
    }

    for (loaded_frames = 0; loaded_frames < config.batch_extract; ++loaded_frames) {
      if (read_frame_y4m(&input, &frame_in, -1) != 0) {
        break;
      }
      // load frame to correct position.
      loadYUVFrame(loaded_frames);
    }

    // This happens if we just reached the end for this batch.
    // We then skip extract, duplicate extract output to match frame count.
    if (loaded_frames == 0) {
      session.duplicateExtractOutput(0, 1);
      loaded_frames = 1;
      end = true;
    }
    else {
      // process all new frames.
      session.extractBatch(0, 1, loaded_frames);
      session.extract();
    }

    if (loaded_frames != config.batch_extract) {
      // We reached end. duplicate extract output of last frame to match frame count.
      session.duplicateExtractOutput(loaded_frames, loaded_frames + 1);
      end = true;
    }

    for (int fusion_in = 0; fusion_in < loaded_frames; fusion_in += config.batch_fusion) {
      int frames = std::min(config.batch_fusion, loaded_frames - fusion_in);
      session.fusionBatch(fusion_in, 0, frames);
      session.fusion();

      for (int out_frame = 0; out_frame < frames * 2; ++out_frame) {
        extractYUVFrame(out_frame);
        CUDA_CHECK(cudaStreamSynchronize(session.stream));
        write_frame_y4m(&output, &frame_out);
      }
    }

    this_batch = std::chrono::steady_clock::now() - batch_begin;
    double fps = loaded_frames * 2 / this_batch.count();
    double speed = fps / output.fps_num * output.fps_den;
    count += loaded_frames * 2;
    elapsed = std::chrono::steady_clock::now() - begin;
    double average = count / elapsed.count();
    std::cerr << "\r[CycMuNet] " << 2 * loaded_frames << '(' << count << ") frames in " << this_batch.count() * 1000
              << "ms, " << fps << "fps (" << speed << "x), average " << average << "fps.";
    std::cerr.flush();
  }

  std::cerr << std::endl;

  close_file_y4m(&input);
  close_file_y4m(&output);

  CUDA_CHECK(cudaFree(input_buffer));
  CUDA_CHECK(cudaFree(output_buffer));
}