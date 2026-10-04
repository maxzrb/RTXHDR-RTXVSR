#include "video/ffmpeg/ffmpeg_transcode_pipeline.h"
#include "video/ffmpeg/ffmpeg_stream_utils.h"

#include "platform/logging.h"

#include <algorithm>
#include <cerrno>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <iomanip>
#include <limits>
#include <chrono>
#include <sstream>
#include <string_view>
#include <system_error>
#include <thread>
#include <utility>
#include <vector>

#if defined(_WIN32)
#include <d3d11.h>
#include <d3d11_1.h>
#include <dxgi.h>
#include <wrl/client.h>
#endif

extern "C" {
#include <libavcodec/avcodec.h>
#include <libavcodec/packet.h>
#include <libavformat/avio.h>
#include <libavformat/avformat.h>
#include <libavutil/buffer.h>
#include <libavutil/error.h>
#include <libavutil/frame.h>
#include <libavutil/hwcontext.h>
#include <libavutil/mathematics.h>
#include <libavutil/opt.h>
#include <libavutil/pixdesc.h>
#include <libavutil/pixfmt.h>
#include <libavutil/rational.h>
}

#if defined(_WIN32)
#include <libavutil/hwcontext_d3d11va.h>
#endif

namespace vsr {

Result<void> ffmpeg_copy_display_matrix(
    const AVCodecParameters* source,
    AVCodecParameters* destination) {
    if (source == nullptr || destination == nullptr) {
        return Result<void>::Fail({
            "display_matrix_copy_failed",
            "FFmpeg could not copy the input display matrix.",
            "Source or destination codec parameters were null."
        });
    }
    if (source == destination) {
        return Result<void>::Ok();
    }

    const AVPacketSideData* source_matrix = av_packet_side_data_get(
        source->coded_side_data,
        source->nb_coded_side_data,
        AV_PKT_DATA_DISPLAYMATRIX);
    if (source_matrix == nullptr) {
        return Result<void>::Ok();
    }
    constexpr std::size_t display_matrix_size = 9 * sizeof(std::int32_t);
    if (source_matrix->data == nullptr || source_matrix->size != display_matrix_size) {
        return Result<void>::Fail({
            "display_matrix_copy_failed",
            "FFmpeg could not copy the input display matrix.",
            "The input display matrix had an invalid size: " + std::to_string(source_matrix->size) + "."
        });
    }

    av_packet_side_data_remove(
        destination->coded_side_data,
        &destination->nb_coded_side_data,
        AV_PKT_DATA_DISPLAYMATRIX);
    AVPacketSideData* destination_matrix = av_packet_side_data_new(
        &destination->coded_side_data,
        &destination->nb_coded_side_data,
        AV_PKT_DATA_DISPLAYMATRIX,
        source_matrix->size,
        0);
    if (destination_matrix == nullptr) {
        return Result<void>::Fail({
            "display_matrix_copy_failed",
            "FFmpeg could not copy the input display matrix.",
            "Could not allocate " + std::to_string(source_matrix->size) + " bytes of side data."
        });
    }
    std::memcpy(destination_matrix->data, source_matrix->data, source_matrix->size);
    return Result<void>::Ok();
}

namespace {

struct InputFormatContextDeleter {
    void operator()(AVFormatContext* context) const noexcept {
        if (context != nullptr) {
            avformat_close_input(&context);
        }
    }
};

struct OutputFormatContextDeleter {
    void operator()(AVFormatContext* context) const noexcept {
        if (context == nullptr) {
            return;
        }
        if (context->pb != nullptr && context->oformat != nullptr && (context->oformat->flags & AVFMT_NOFILE) == 0) {
            avio_closep(&context->pb);
        }
        avformat_free_context(context);
    }
};

struct CodecContextDeleter {
    void operator()(AVCodecContext* context) const noexcept {
        if (context != nullptr) {
            avcodec_free_context(&context);
        }
    }
};

struct PacketDeleter {
    void operator()(AVPacket* packet) const noexcept {
        if (packet != nullptr) {
            av_packet_free(&packet);
        }
    }
};

struct FrameDeleter {
    void operator()(AVFrame* frame) const noexcept {
        if (frame != nullptr) {
            av_frame_free(&frame);
        }
    }
};

struct BufferRefDeleter {
    void operator()(AVBufferRef* ref) const noexcept {
        if (ref != nullptr) {
            av_buffer_unref(&ref);
        }
    }
};

using InputFormatContextPtr = std::unique_ptr<AVFormatContext, InputFormatContextDeleter>;
using OutputFormatContextPtr = std::unique_ptr<AVFormatContext, OutputFormatContextDeleter>;
using CodecContextPtr = std::unique_ptr<AVCodecContext, CodecContextDeleter>;
using PacketPtr = std::unique_ptr<AVPacket, PacketDeleter>;
using FramePtr = std::unique_ptr<AVFrame, FrameDeleter>;
using BufferRefPtr = std::unique_ptr<AVBufferRef, BufferRefDeleter>;

class TemporaryOutputCleanupGuard {
public:
    explicit TemporaryOutputCleanupGuard(std::filesystem::path path) : path_(std::move(path)) {}
    ~TemporaryOutputCleanupGuard() {
        if (active_) {
            std::error_code ignored;
            std::filesystem::remove(path_, ignored);
        }
    }

    void release() { active_ = false; }

private:
    std::filesystem::path path_;
    bool active_ = true;
};

static Error ffmpeg_error(const char* code, const char* message, int ffmpeg_code) {
    char buffer[AV_ERROR_MAX_STRING_SIZE] = {};
    av_strerror(ffmpeg_code, buffer, sizeof(buffer));
    return {code, message, buffer};
}

Result<void> convert_software_frame_for_d3d11_upload(
    const AVFrame* source,
    AVFrame* destination,
    AVPixelFormat destination_format) {
    if (source == nullptr || destination == nullptr) {
        return Result<void>::Fail({
            "software_decode_conversion_failed",
            "FFmpeg could not convert the software-decoded frame for D3D11 upload.",
            "Source or destination frame was null."
        });
    }

    const auto source_format = static_cast<AVPixelFormat>(source->format);
    const int width = source->width;
    const int height = source->height;
    const int chroma_width = (width + 1) / 2;
    const int chroma_height = (height + 1) / 2;

    if (source_format == AV_PIX_FMT_YUV420P && destination_format == AV_PIX_FMT_NV12) {
        for (int y = 0; y < height; ++y) {
            std::memcpy(
                destination->data[0] + y * destination->linesize[0],
                source->data[0] + y * source->linesize[0],
                static_cast<std::size_t>(width));
        }
        for (int y = 0; y < chroma_height; ++y) {
            const auto* source_u = source->data[1] + y * source->linesize[1];
            const auto* source_v = source->data[2] + y * source->linesize[2];
            auto* destination_uv = destination->data[1] + y * destination->linesize[1];
            for (int x = 0; x < chroma_width; ++x) {
                destination_uv[x * 2] = source_u[x];
                destination_uv[x * 2 + 1] = source_v[x];
            }
        }
        return Result<void>::Ok();
    }

    if (source_format == AV_PIX_FMT_YUV420P10LE && destination_format == AV_PIX_FMT_P010LE) {
        for (int y = 0; y < height; ++y) {
            const auto* source_y = reinterpret_cast<const std::uint16_t*>(
                source->data[0] + y * source->linesize[0]);
            auto* destination_y = reinterpret_cast<std::uint16_t*>(
                destination->data[0] + y * destination->linesize[0]);
            for (int x = 0; x < width; ++x) {
                destination_y[x] = static_cast<std::uint16_t>((source_y[x] & 0x03ffU) << 6U);
            }
        }
        for (int y = 0; y < chroma_height; ++y) {
            const auto* source_u = reinterpret_cast<const std::uint16_t*>(
                source->data[1] + y * source->linesize[1]);
            const auto* source_v = reinterpret_cast<const std::uint16_t*>(
                source->data[2] + y * source->linesize[2]);
            auto* destination_uv = reinterpret_cast<std::uint16_t*>(
                destination->data[1] + y * destination->linesize[1]);
            for (int x = 0; x < chroma_width; ++x) {
                destination_uv[x * 2] = static_cast<std::uint16_t>((source_u[x] & 0x03ffU) << 6U);
                destination_uv[x * 2 + 1] = static_cast<std::uint16_t>((source_v[x] & 0x03ffU) << 6U);
            }
        }
        return Result<void>::Ok();
    }

    // 兼容常见 planar YUV 的 4:2:2/4:4:4 及 8/10/12/16 位输入，避免增加 DLL 依赖。
    const AVPixFmtDescriptor* descriptor = av_pix_fmt_desc_get(source_format);
    if (descriptor == nullptr || !(descriptor->flags & AV_PIX_FMT_FLAG_PLANAR) ||
        (descriptor->flags & (AV_PIX_FMT_FLAG_RGB | AV_PIX_FMT_FLAG_BE)) ||
        descriptor->nb_components < 3 || descriptor->comp[0].depth > 16) {
        return Result<void>::Fail({"software_decode_format_unsupported",
            "Software decoder produced an unsupported pixel format.",
            av_get_pix_fmt_name(source_format) ? av_get_pix_fmt_name(source_format) : "unknown"});
    }
    const int depth = descriptor->comp[0].depth;
    const bool ten_bit = destination_format == AV_PIX_FMT_P010LE;
    auto read_sample = [&](int plane, int x, int y) -> unsigned int {
        const std::uint8_t* row = source->data[plane] + y * source->linesize[plane];
        return depth > 8 ? reinterpret_cast<const std::uint16_t*>(row)[x] : row[x];
    };
    auto convert_depth = [&](unsigned int value) -> unsigned int {
        const int target_depth = ten_bit ? 10 : 8;
        value = depth > target_depth ? value >> (depth - target_depth) : value << (target_depth - depth);
        return ten_bit ? value << 6 : value;
    };
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            const unsigned int value = convert_depth(read_sample(0, x, y));
            std::uint8_t* row = destination->data[0] + y * destination->linesize[0];
            if (ten_bit) reinterpret_cast<std::uint16_t*>(row)[x] = static_cast<std::uint16_t>(value);
            else row[x] = static_cast<std::uint8_t>(value);
        }
    }
    for (int y = 0; y < chroma_height; ++y) {
        for (int x = 0; x < chroma_width; ++x) {
            for (int plane = 1; plane <= 2; ++plane) {
                unsigned int sum = 0, count = 0;
                const int x_first = (2 * x) >> descriptor->log2_chroma_w;
                const int y_first = (2 * y) >> descriptor->log2_chroma_h;
                const int x_end = (std::min(2 * x + 2, width) + (1 << descriptor->log2_chroma_w) - 1) >> descriptor->log2_chroma_w;
                const int y_end = (std::min(2 * y + 2, height) + (1 << descriptor->log2_chroma_h) - 1) >> descriptor->log2_chroma_h;
                for (int sy = y_first; sy < y_end; ++sy) {
                    for (int sx = x_first; sx < x_end; ++sx) { sum += read_sample(plane, sx, sy); ++count; }
                }
                const unsigned int value = convert_depth((sum + count / 2) / count);
                std::uint8_t* row = destination->data[1] + y * destination->linesize[1];
                const int offset = 2 * x + plane - 1;
                if (ten_bit) reinterpret_cast<std::uint16_t*>(row)[offset] = static_cast<std::uint16_t>(value);
                else row[offset] = static_cast<std::uint8_t>(value);
            }
        }
    }
    return Result<void>::Ok();
}

Result<void> set_required_encoder_option(AVCodecContext* context, const char* name, const char* value) {
    const int result = av_opt_set(context->priv_data, name, value, 0);
    if (result < 0) {
        return Result<void>::Fail(ffmpeg_error("encoder_option_failed", "FFmpeg could not set a required NVENC option.", result));
    }
    return Result<void>::Ok();
}

Result<void> set_required_encoder_option_double(AVCodecContext* context, const char* name, double value) {
    const int result = av_opt_set_double(context->priv_data, name, value, 0);
    if (result < 0) {
        return Result<void>::Fail(ffmpeg_error("encoder_option_failed", "FFmpeg could not set a required NVENC option.", result));
    }
    return Result<void>::Ok();
}

Result<void> set_required_encoder_option_int(AVCodecContext* context, const char* name, std::int64_t value) {
    const int result = av_opt_set_int(context->priv_data, name, value, 0);
    if (result < 0) {
        return Result<void>::Fail(ffmpeg_error("encoder_option_failed", "FFmpeg could not set a required NVENC option.", result));
    }
    return Result<void>::Ok();
}

// Blocks the calling pipeline thread while the job is paused. Cancel always
// wins over pause so a stopped job never waits on the pause flag.
void wait_while_job_paused(CancellationToken& cancellation) {
    bool logged = false;
    while (cancellation.paused.load() && !cancellation.requested.load()) {
        if (!logged) {
            log_info("Pipeline paused by request.");
            logged = true;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
    }
    if (logged && !cancellation.requested.load()) {
        log_info("Pipeline resumed.");
    }
}

void set_optional_encoder_option_int(AVCodecContext* context, const char* name, std::int64_t value) {
    const int result = av_opt_set_int(context->priv_data, name, value, 0);
    if (result < 0) {
        log_info(std::string("Skipping unsupported NVENC option: ") + name);
    }
}

Result<void> configure_nvenc_quality(AVCodecContext* context, const OutputSettings& output, bool enable_split_encode) {
    for (const auto& option : {
             std::pair<const char*, const char*>{"preset", "p7"},
             std::pair<const char*, const char*>{"tune", "hq"},
             std::pair<const char*, const char*>{"rc", "vbr"},
         }) {
        const auto configured = set_required_encoder_option(context, option.first, option.second);
        if (!configured.ok()) {
            return configured;
        }
    }

    const auto cq = set_required_encoder_option_double(context, "cq", nvenc_target_quality(output.video_codec));
    if (!cq.ok()) {
        return cq;
    }

    for (const auto& option : {
             std::pair<const char*, std::int64_t>{"spatial-aq", 1},
             std::pair<const char*, std::int64_t>{"temporal-aq", 1},
             std::pair<const char*, std::int64_t>{"aq-strength", 8},
             std::pair<const char*, std::int64_t>{"rc-lookahead", 32},
             std::pair<const char*, std::int64_t>{"multipass", 2},
         }) {
        const auto configured = set_required_encoder_option_int(context, option.first, option.second);
        if (!configured.ok()) {
            return configured;
        }
    }
    // Caller-provided overrides win over the built-in defaults above. Option
    // names were already validated against the allowlist in validate_request.
    // Invalid values (e.g. unsupported tune presets) fall back to the default
    // with a warning instead of failing the whole job.
    for (const auto& [name, value] : output.encoder_options) {
        const int result = av_opt_set(context->priv_data, name.c_str(), value.c_str(), 0);
        if (result < 0) {
            log_info(std::string("Skipping unsupported NVENC option: ") + name);
        }
    }
    if (enable_split_encode) {
        // On GPUs with multiple NVENC engines, encode horizontal strips concurrently.
        // This keeps the existing P7/HQ, AQ, lookahead and full-resolution multipass
        // quality configuration; single-engine GPUs are handled by the driver.
        set_optional_encoder_option_int(context, "split_encode_mode", 1);
    }
    return Result<void>::Ok();
}

std::int64_t source_video_bit_rate(const AVFormatContext* format, const AVStream* stream, const AVCodecContext* decoder) {
    if (stream != nullptr && stream->codecpar != nullptr && stream->codecpar->bit_rate > 0) {
        return stream->codecpar->bit_rate;
    }
    if (decoder != nullptr && decoder->bit_rate > 0) {
        return decoder->bit_rate;
    }
    if (format != nullptr && format->bit_rate > 0) {
        return format->bit_rate;
    }
    return 0;
}

static double progress_from_frames(std::int64_t frames_done, std::int64_t frames_total) {
    return ffmpeg_progress_from_frames(frames_done, frames_total);
}

int find_primary_video_stream(const AVFormatContext* format) {
    for (unsigned int index = 0; index < format->nb_streams; ++index) {
        const AVStream* stream = format->streams[index];
        if (stream != nullptr && stream->codecpar != nullptr && stream->codecpar->codec_type == AVMEDIA_TYPE_VIDEO) {
            return static_cast<int>(index);
        }
    }
    return -1;
}

bool valid_rational(AVRational value) {
    return value.num > 0 && value.den > 0;
}

std::string codec_details(const AVStream* stream) {
    if (stream == nullptr || stream->codecpar == nullptr) {
        return "unknown stream";
    }
    const char* codec_name = avcodec_get_name(stream->codecpar->codec_id);
    std::ostringstream details;
    details << "stream=" << stream->index << ", codec=" << (codec_name != nullptr ? codec_name : "unknown");
    return details.str();
}

std::int64_t estimate_total_frames(const AVFormatContext* format, const AVStream* stream, AVRational frame_rate) {
    if (stream->nb_frames > 0) {
        return stream->nb_frames;
    }
    if (stream->duration != AV_NOPTS_VALUE && stream->duration > 0 && valid_rational(stream->time_base) &&
        valid_rational(frame_rate)) {
        const double seconds = static_cast<double>(stream->duration) * av_q2d(stream->time_base);
        return static_cast<std::int64_t>(std::llround(seconds * av_q2d(frame_rate)));
    }
    if (format->duration != AV_NOPTS_VALUE && format->duration > 0 && valid_rational(frame_rate)) {
        const double seconds = static_cast<double>(format->duration) / static_cast<double>(AV_TIME_BASE);
        return static_cast<std::int64_t>(std::llround(seconds * av_q2d(frame_rate)));
    }
    return 0;
}

int even_scaled_dimension(int value, double scale) {
    const double scaled = static_cast<double>(value) * scale;
    auto rounded = static_cast<long long>(std::llround(scaled));
    rounded = std::max<long long>(2, rounded);
    if ((rounded % 2) != 0) {
        ++rounded;
    }
    return static_cast<int>(std::min<long long>(rounded, std::numeric_limits<int>::max()));
}

AVPixelFormat choose_d3d11_format(AVCodecContext*, const AVPixelFormat* formats) {
    for (const AVPixelFormat* format = formats; *format != AV_PIX_FMT_NONE; ++format) {
        if (*format == AV_PIX_FMT_D3D11) {
            return *format;
        }
    }
    return AV_PIX_FMT_NONE;
}

#if defined(_WIN32)

Error hresult_error(const char* code, const char* message, HRESULT result) {
    std::ostringstream details;
    details << "HRESULT 0x" << std::uppercase << std::hex << static_cast<unsigned long>(result);
    return {code, message, details.str()};
}

struct D3d11DevicePair {
    Microsoft::WRL::ComPtr<ID3D11Device> device;
    Microsoft::WRL::ComPtr<ID3D11DeviceContext> context;
};

Result<D3d11DevicePair> create_d3d11_device() {
    D3d11DevicePair pair;
    D3D_FEATURE_LEVEL created_feature_level = {};
    const D3D_FEATURE_LEVEL feature_levels_with_11_1[] = {
        D3D_FEATURE_LEVEL_11_1,
        D3D_FEATURE_LEVEL_11_0,
        D3D_FEATURE_LEVEL_10_1,
        D3D_FEATURE_LEVEL_10_0,
    };
    const D3D_FEATURE_LEVEL feature_levels[] = {
        D3D_FEATURE_LEVEL_11_0,
        D3D_FEATURE_LEVEL_10_1,
        D3D_FEATURE_LEVEL_10_0,
    };

    const UINT flags = D3D11_CREATE_DEVICE_VIDEO_SUPPORT | D3D11_CREATE_DEVICE_BGRA_SUPPORT;

    // Optimus laptops and virtual display adapters can shift the default DXGI
    // adapter (an iGPU or a virtual adapter may be picked as the default).
    // NVENC and NGX both require an NVIDIA device, so enumerate adapters and
    // prefer the NVIDIA one explicitly; fall back to the system default only
    // when no NVIDIA adapter is present.
    Microsoft::WRL::ComPtr<IDXGIAdapter1> nvidia_adapter;
    {
        Microsoft::WRL::ComPtr<IDXGIFactory1> factory;
        if (SUCCEEDED(CreateDXGIFactory1(__uuidof(IDXGIFactory1), &factory))) {
            for (UINT index = 0;; ++index) {
                Microsoft::WRL::ComPtr<IDXGIAdapter1> candidate;
                if (factory->EnumAdapters1(index, &candidate) == DXGI_ERROR_NOT_FOUND) break;
                DXGI_ADAPTER_DESC1 desc{};
                if (FAILED(candidate->GetDesc1(&desc))) continue;
                if (desc.VendorId == 0x10DE) {
                    nvidia_adapter = candidate;
                    break;
                }
            }
        }
    }

    auto open_device = [&](const D3D_FEATURE_LEVEL* levels, UINT count) {
        return D3D11CreateDevice(
            nvidia_adapter ? nvidia_adapter.Get() : nullptr,
            nvidia_adapter ? D3D_DRIVER_TYPE_UNKNOWN : D3D_DRIVER_TYPE_HARDWARE,
            nullptr,
            flags,
            levels,
            count,
            D3D11_SDK_VERSION,
            pair.device.ReleaseAndGetAddressOf(),
            &created_feature_level,
            pair.context.ReleaseAndGetAddressOf());
    };

    HRESULT result = open_device(feature_levels_with_11_1, static_cast<UINT>(std::size(feature_levels_with_11_1)));
    if (result == E_INVALIDARG) {
        result = open_device(feature_levels, static_cast<UINT>(std::size(feature_levels)));
    }
    if (FAILED(result)) {
        return Result<D3d11DevicePair>::Fail(hresult_error(
            "d3d11_device_create_failed",
            "A D3D11 hardware video device could not be created.",
            result));
    }
    if (nvidia_adapter) {
        DXGI_ADAPTER_DESC1 desc{};
        if (SUCCEEDED(nvidia_adapter->GetDesc1(&desc))) {
            std::wstring adapter_name(desc.Description);
            log_info("D3D11 device created on NVIDIA adapter: " +
                std::string(adapter_name.begin(), adapter_name.end()));
        }
    }

    return Result<D3d11DevicePair>::Ok(std::move(pair));
}

Result<BufferRefPtr> create_ffmpeg_d3d11_device(ID3D11Device* device, ID3D11DeviceContext* context) {
    AVBufferRef* raw_device_ref = av_hwdevice_ctx_alloc(AV_HWDEVICE_TYPE_D3D11VA);
    if (raw_device_ref == nullptr) {
        return Result<BufferRefPtr>::Fail({"d3d11va_device_alloc_failed", "FFmpeg could not allocate a D3D11VA device context.", ""});
    }
    BufferRefPtr device_ref(raw_device_ref);

    auto* hw_device = reinterpret_cast<AVHWDeviceContext*>(device_ref->data);
    auto* d3d11 = reinterpret_cast<AVD3D11VADeviceContext*>(hw_device->hwctx);
    d3d11->device = device;
    d3d11->device->AddRef();
    d3d11->device_context = context;
    d3d11->device_context->AddRef();

    const int result = av_hwdevice_ctx_init(device_ref.get());
    if (result < 0) {
        return Result<BufferRefPtr>::Fail(ffmpeg_error(
            "d3d11va_device_init_failed",
            "FFmpeg could not initialize the D3D11VA device context.",
            result));
    }

    return Result<BufferRefPtr>::Ok(std::move(device_ref));
}

int decoder_pixel_depth(const AVCodecContext* decoder) {
    if (decoder == nullptr) {
        return 8;
    }

    const AVPixelFormat formats[] = {
        decoder->sw_pix_fmt,
        decoder->pix_fmt
    };
    for (const AVPixelFormat format : formats) {
        const AVPixFmtDescriptor* descriptor = av_pix_fmt_desc_get(format);
        if (descriptor != nullptr && descriptor->nb_components > 0) {
            return descriptor->comp[0].depth;
        }
    }
    return 8;
}

DXGI_FORMAT dxgi_format_for_rtx_input_surface(const AVCodecContext* decoder) {
    // Match NVIDIA's TrueHDR sample: the NGX input surface follows the source
    // bit depth, independently of whether TrueHDR is enabled. Promoting an
    // 8-bit SDR source to packed RGB10 before NGX corrupts its channel layout
    // on the affected driver path.
    return decoder_pixel_depth(decoder) > 8
        ? DXGI_FORMAT_R10G10B10A2_UNORM
        : DXGI_FORMAT_R8G8B8A8_UNORM;
}

DXGI_FORMAT dxgi_format_for_rtx_output_surface(const AVCodecContext* decoder, bool hdr_enabled) {
    return hdr_enabled || decoder_pixel_depth(decoder) > 8
        ? DXGI_FORMAT_R10G10B10A2_UNORM
        : DXGI_FORMAT_R8G8B8A8_UNORM;
}

DXGI_FORMAT dxgi_format_for_encoder_surface(AVPixelFormat format) {
    switch (format) {
    case AV_PIX_FMT_X2BGR10:
        return DXGI_FORMAT_R10G10B10A2_UNORM;
    case AV_PIX_FMT_NV12:
        return DXGI_FORMAT_NV12;
    case AV_PIX_FMT_P010LE:
        return DXGI_FORMAT_P010;
    default:
        return DXGI_FORMAT_UNKNOWN;
    }
}

DXGI_COLOR_SPACE_TYPE d3d11_input_color_space(const AVCodecContext* decoder) {
    const bool full_range = decoder != nullptr && decoder->color_range == AVCOL_RANGE_JPEG;
    const bool bt2020 = decoder != nullptr &&
        (decoder->colorspace == AVCOL_SPC_BT2020_NCL || decoder->color_primaries == AVCOL_PRI_BT2020);
    const bool pq = decoder != nullptr && decoder->color_trc == AVCOL_TRC_SMPTE2084;

    if (bt2020 && pq) {
        return DXGI_COLOR_SPACE_YCBCR_STUDIO_G2084_LEFT_P2020;
    }
    if (bt2020) {
        return full_range ? DXGI_COLOR_SPACE_YCBCR_FULL_G22_LEFT_P2020 : DXGI_COLOR_SPACE_YCBCR_STUDIO_G22_LEFT_P2020;
    }
    return full_range ? DXGI_COLOR_SPACE_YCBCR_FULL_G22_LEFT_P709 : DXGI_COLOR_SPACE_YCBCR_STUDIO_G22_LEFT_P709;
}

DXGI_COLOR_SPACE_TYPE d3d11_rtx_input_color_space(const AVCodecContext* decoder) {
    const bool bt2020 = decoder != nullptr &&
        (decoder->colorspace == AVCOL_SPC_BT2020_NCL || decoder->color_primaries == AVCOL_PRI_BT2020);
    const bool pq = decoder != nullptr && decoder->color_trc == AVCOL_TRC_SMPTE2084;
    if (bt2020 && pq) {
        return DXGI_COLOR_SPACE_RGB_FULL_G2084_NONE_P2020;
    }
    return bt2020 ? DXGI_COLOR_SPACE_RGB_FULL_G22_NONE_P2020 : DXGI_COLOR_SPACE_RGB_FULL_G22_NONE_P709;
}

DXGI_COLOR_SPACE_TYPE d3d11_encoder_output_color_space(const AVCodecContext* encoder) {
    const bool full_range = encoder != nullptr && encoder->color_range == AVCOL_RANGE_JPEG;
    const bool bt2020 = encoder != nullptr &&
        (encoder->colorspace == AVCOL_SPC_BT2020_NCL || encoder->color_primaries == AVCOL_PRI_BT2020);
    const bool pq = encoder != nullptr && encoder->color_trc == AVCOL_TRC_SMPTE2084;
    if (bt2020 && pq) {
        return DXGI_COLOR_SPACE_YCBCR_STUDIO_G2084_LEFT_P2020;
    }
    if (bt2020) {
        return full_range ? DXGI_COLOR_SPACE_YCBCR_FULL_G22_LEFT_P2020 : DXGI_COLOR_SPACE_YCBCR_STUDIO_G22_LEFT_P2020;
    }
    return full_range ? DXGI_COLOR_SPACE_YCBCR_FULL_G22_LEFT_P709 : DXGI_COLOR_SPACE_YCBCR_STUDIO_G22_LEFT_P709;
}

Result<void> create_texture(
    ID3D11Device* device,
    int width,
    int height,
    DXGI_FORMAT format,
    UINT bind_flags,
    Microsoft::WRL::ComPtr<ID3D11Texture2D>& texture) {
    D3D11_TEXTURE2D_DESC desc = {};
    desc.Width = static_cast<UINT>(width);
    desc.Height = static_cast<UINT>(height);
    desc.MipLevels = 1;
    desc.ArraySize = 1;
    desc.Format = format;
    desc.SampleDesc.Count = 1;
    desc.Usage = D3D11_USAGE_DEFAULT;
    desc.BindFlags = bind_flags;

    const HRESULT result = device->CreateTexture2D(&desc, nullptr, texture.ReleaseAndGetAddressOf());
    if (FAILED(result)) {
        return Result<void>::Fail(hresult_error(
            "d3d11_texture_create_failed",
            "A D3D11 texture required by the hardware pipeline could not be created.",
            result));
    }
    return Result<void>::Ok();
}

class D3d11VideoConverter {
public:
    Result<void> initialize(
        ID3D11Device* device,
        ID3D11DeviceContext* context,
        ID3D11Texture2D* input,
        ID3D11Texture2D* output,
        int input_width,
        int input_height,
        int output_width,
        int output_height,
        DXGI_COLOR_SPACE_TYPE input_color_space,
        DXGI_COLOR_SPACE_TYPE output_color_space) {
        HRESULT result = device->QueryInterface(IID_PPV_ARGS(video_device_.GetAddressOf()));
        if (FAILED(result)) {
            return Result<void>::Fail(hresult_error("d3d11_video_device_required", "The D3D11 device does not expose video processing support.", result));
        }
        result = context->QueryInterface(IID_PPV_ARGS(video_context_.GetAddressOf()));
        if (FAILED(result)) {
            return Result<void>::Fail(hresult_error("d3d11_video_context_required", "The D3D11 immediate context does not expose video processing support.", result));
        }

        D3D11_TEXTURE2D_DESC input_desc = {};
        D3D11_TEXTURE2D_DESC output_desc = {};
        input->GetDesc(&input_desc);
        output->GetDesc(&output_desc);
        D3D11_VIDEO_PROCESSOR_CONTENT_DESC content_desc = {};
        content_desc.InputFrameFormat = D3D11_VIDEO_FRAME_FORMAT_PROGRESSIVE;
        content_desc.InputWidth = static_cast<UINT>(input_width);
        content_desc.InputHeight = static_cast<UINT>(input_height);
        content_desc.OutputWidth = static_cast<UINT>(output_width);
        content_desc.OutputHeight = static_cast<UINT>(output_height);
        content_desc.Usage = D3D11_VIDEO_USAGE_PLAYBACK_NORMAL;
        result = video_device_->CreateVideoProcessorEnumerator(&content_desc, enumerator_.GetAddressOf());
        if (FAILED(result)) {
            return Result<void>::Fail(hresult_error("d3d11_video_processor_enumerator_failed", "D3D11 video processor enumeration failed.", result));
        }

        UINT format_flags = 0;
        result = enumerator_->CheckVideoProcessorFormat(input_desc.Format, &format_flags);
        if (FAILED(result) || (format_flags & D3D11_VIDEO_PROCESSOR_FORMAT_SUPPORT_INPUT) == 0) {
            return Result<void>::Fail(hresult_error("d3d11_video_processor_input_format_unsupported", "The D3D11 video processor cannot use the decoded frame format as input.", FAILED(result) ? result : E_INVALIDARG));
        }
        format_flags = 0;
        result = enumerator_->CheckVideoProcessorFormat(output_desc.Format, &format_flags);
        if (FAILED(result) || (format_flags & D3D11_VIDEO_PROCESSOR_FORMAT_SUPPORT_OUTPUT) == 0) {
            return Result<void>::Fail(hresult_error("d3d11_video_processor_output_format_unsupported", "The D3D11 video processor cannot write the RTX texture format.", FAILED(result) ? result : E_INVALIDARG));
        }
        result = video_device_->CreateVideoProcessor(enumerator_.Get(), 0, processor_.GetAddressOf());
        if (FAILED(result)) {
            return Result<void>::Fail(hresult_error("d3d11_video_processor_create_failed", "D3D11 video processor creation failed.", result));
        }

        RECT output_rect = {0, 0, static_cast<LONG>(output_width), static_cast<LONG>(output_height)};
        video_context_->VideoProcessorSetStreamFrameFormat(processor_.Get(), 0, D3D11_VIDEO_FRAME_FORMAT_PROGRESSIVE);
        video_context_->VideoProcessorSetStreamOutputRate(processor_.Get(), 0, D3D11_VIDEO_PROCESSOR_OUTPUT_RATE_NORMAL, TRUE, nullptr);
        video_context_->VideoProcessorSetStreamSourceRect(processor_.Get(), 0, FALSE, nullptr);
        video_context_->VideoProcessorSetStreamDestRect(processor_.Get(), 0, TRUE, &output_rect);
        video_context_->VideoProcessorSetOutputTargetRect(processor_.Get(), TRUE, &output_rect);
        Microsoft::WRL::ComPtr<ID3D11VideoContext1> video_context1;
        if (SUCCEEDED(context->QueryInterface(IID_PPV_ARGS(video_context1.GetAddressOf())))) {
            video_context1->VideoProcessorSetStreamColorSpace1(processor_.Get(), 0, input_color_space);
            video_context1->VideoProcessorSetOutputColorSpace1(processor_.Get(), output_color_space);
        }

        return Result<void>::Ok();
    }

    Result<void> convert(ID3D11Texture2D* input, UINT input_slice, ID3D11Texture2D* output, UINT output_slice) {
        ID3D11VideoProcessorInputView* input_view = nullptr;
        for (auto& cached : input_views_) {
            if (cached.texture.Get() == input && cached.slice == input_slice) {
                input_view = cached.view.Get();
                break;
            }
        }
        if (input_view == nullptr) {
            D3D11_TEXTURE2D_DESC input_desc = {};
            input->GetDesc(&input_desc);
            if (input_slice >= input_desc.ArraySize) {
                return Result<void>::Fail({"d3d11_input_slice_invalid", "The D3D11 texture slice is out of range.", std::to_string(input_slice)});
            }
            D3D11_VIDEO_PROCESSOR_INPUT_VIEW_DESC input_view_desc = {};
            input_view_desc.ViewDimension = D3D11_VPIV_DIMENSION_TEXTURE2D;
            input_view_desc.Texture2D.MipSlice = 0;
            input_view_desc.Texture2D.ArraySlice = input_slice;
            CachedInputView cached;
            cached.texture = input;
            cached.slice = input_slice;
            const HRESULT result = video_device_->CreateVideoProcessorInputView(
                input,
                enumerator_.Get(),
                &input_view_desc,
                cached.view.GetAddressOf());
            if (FAILED(result)) {
                return Result<void>::Fail(hresult_error("d3d11_video_processor_input_view_failed", "D3D11 video processor input view creation failed.", result));
            }
            input_views_.push_back(std::move(cached));
            input_view = input_views_.back().view.Get();
        }
        ID3D11VideoProcessorOutputView* output_view = nullptr;
        for (auto& cached : output_views_) {
            if (cached.texture.Get() == output && cached.slice == output_slice) {
                output_view = cached.view.Get();
                break;
            }
        }
        if (output_view == nullptr) {
            D3D11_TEXTURE2D_DESC output_desc = {};
            output->GetDesc(&output_desc);
            D3D11_VIDEO_PROCESSOR_OUTPUT_VIEW_DESC output_view_desc = {};
            if (output_desc.ArraySize > 1) {
                output_view_desc.ViewDimension = D3D11_VPOV_DIMENSION_TEXTURE2DARRAY;
                output_view_desc.Texture2DArray.MipSlice = 0;
                output_view_desc.Texture2DArray.FirstArraySlice = output_slice;
                output_view_desc.Texture2DArray.ArraySize = 1;
            } else {
                output_view_desc.ViewDimension = D3D11_VPOV_DIMENSION_TEXTURE2D;
                output_view_desc.Texture2D.MipSlice = 0;
            }
            CachedOutputView cached;
            cached.texture = output;
            cached.slice = output_slice;
            const HRESULT result = video_device_->CreateVideoProcessorOutputView(
                output,
                enumerator_.Get(),
                &output_view_desc,
                cached.view.GetAddressOf());
            if (FAILED(result)) {
                return Result<void>::Fail(hresult_error("d3d11_video_processor_output_view_failed", "D3D11 video processor output view creation failed.", result));
            }
            output_views_.push_back(std::move(cached));
            output_view = output_views_.back().view.Get();
        }

        D3D11_VIDEO_PROCESSOR_STREAM stream = {};
        stream.Enable = TRUE;
        stream.pInputSurface = input_view;
        const HRESULT result = video_context_->VideoProcessorBlt(processor_.Get(), output_view, 0, 1, &stream);
        if (FAILED(result)) {
            return Result<void>::Fail(hresult_error("d3d11_video_processor_blt_failed", "D3D11 video processor conversion failed.", result));
        }
        // Do not Flush here. The conversion and the following NGX evaluation use
        // the same immediate context, so D3D11 command ordering already preserves
        // the dependency while allowing the driver to batch submissions.
        return Result<void>::Ok();
    }

private:
    struct CachedInputView {
        Microsoft::WRL::ComPtr<ID3D11Texture2D> texture;
        UINT slice = 0;
        Microsoft::WRL::ComPtr<ID3D11VideoProcessorInputView> view;
    };

    struct CachedOutputView {
        Microsoft::WRL::ComPtr<ID3D11Texture2D> texture;
        UINT slice = 0;
        Microsoft::WRL::ComPtr<ID3D11VideoProcessorOutputView> view;
    };

    Microsoft::WRL::ComPtr<ID3D11VideoDevice> video_device_;
    Microsoft::WRL::ComPtr<ID3D11VideoContext> video_context_;
    Microsoft::WRL::ComPtr<ID3D11VideoProcessorEnumerator> enumerator_;
    Microsoft::WRL::ComPtr<ID3D11VideoProcessor> processor_;
    std::vector<CachedInputView> input_views_;
    std::vector<CachedOutputView> output_views_;
};

Result<BufferRefPtr> create_encoder_frames_context(
    AVBufferRef* hw_device_ref,
    AVPixelFormat sw_format,
    int width,
    int height) {
    AVBufferRef* raw_frames_ref = av_hwframe_ctx_alloc(hw_device_ref);
    if (raw_frames_ref == nullptr) {
        return Result<BufferRefPtr>::Fail({"encoder_frames_alloc_failed", "FFmpeg could not allocate encoder D3D11 frames.", ""});
    }
    BufferRefPtr frames_ref(raw_frames_ref);

    auto* frames = reinterpret_cast<AVHWFramesContext*>(frames_ref->data);
    frames->format = AV_PIX_FMT_D3D11;
    frames->sw_format = sw_format;
    frames->width = width;
    frames->height = height;
    frames->initial_pool_size = 0;

    auto* d3d11_frames = reinterpret_cast<AVD3D11VAFramesContext*>(frames->hwctx);
    d3d11_frames->BindFlags = D3D11_BIND_RENDER_TARGET | D3D11_BIND_SHADER_RESOURCE;

    const int result = av_hwframe_ctx_init(frames_ref.get());
    if (result < 0) {
        return Result<BufferRefPtr>::Fail(ffmpeg_error(
            "encoder_frames_init_failed",
            "FFmpeg could not initialize encoder D3D11 frames.",
            result));
    }

    return Result<BufferRefPtr>::Ok(std::move(frames_ref));
}

Result<void> drain_encoder(AVCodecContext* encoder, AVFormatContext* output, AVStream* stream, AVPacket* packet) {
    for (;;) {
        av_packet_unref(packet);
        const int result = avcodec_receive_packet(encoder, packet);
        if (result == AVERROR(EAGAIN) || result == AVERROR_EOF) {
            return Result<void>::Ok();
        }
        if (result < 0) {
            return Result<void>::Fail(ffmpeg_error(
                "encoder_receive_packet_failed",
                "NVENC did not return an encoded packet.",
                result));
        }

        packet->stream_index = stream->index;
        if (packet->duration == 0) {
            // D3D11VA-decoded frames carry no duration, so NVENC packets lose
            // it too. movenc then computes a track duration one frame short
            // and the demuxer's edit list marks the final frame as DISCARD.
            // time_base is 1/fps, so exactly one unit is one frame.
            packet->duration = 1;
        }
        av_packet_rescale_ts(packet, encoder->time_base, stream->time_base);
        // 输入 PTS 已在编码前归零，保留编码器重排产生的负 DTS。

        const int write_result = av_interleaved_write_frame(output, packet);
        if (write_result < 0) {
            return Result<void>::Fail(ffmpeg_error(
                "output_packet_write_failed",
                "FFmpeg could not mux an encoded packet.",
                write_result));
        }
    }
}

class RtxShutdownGuard {
public:
    explicit RtxShutdownGuard(RtxProcessor* rtx) : rtx_(rtx) {}
    ~RtxShutdownGuard() {
        if (active_ && rtx_ != nullptr) {
            rtx_->shutdown();
        }
    }

    void activate() { active_ = true; }
    void release() {
        if (active_ && rtx_ != nullptr) {
            rtx_->shutdown();
        }
        active_ = false;
    }

private:
    RtxProcessor* rtx_ = nullptr;
    bool active_ = false;
};

#endif

} // namespace

FfmpegTranscodePipeline::FfmpegTranscodePipeline(std::unique_ptr<RtxProcessor> rtx) : rtx_(std::move(rtx)) {}

Result<void> FfmpegTranscodePipeline::run(
    const TranscodeRequest& request,
    CancellationToken& cancellation,
    ProgressCallback progress,
    WarningCallback warning) {
    log_info("FFmpeg pipeline starting: " + request.input_path + " -> " + request.output_path);
    const auto report_warning = [&warning](const std::string& message) {
        log_info("FFmpeg pipeline warning: " + message);
        if (warning) {
            warning(message);
        }
    };
    const std::filesystem::path input_path = path_from_utf8(request.input_path);
    std::error_code input_status_error;
    const bool input_exists = std::filesystem::exists(input_path, input_status_error);
    if (input_status_error) {
        return Result<void>::Fail({
            "input_access_failed",
            "Input file could not be checked.",
            request.input_path + ": " + input_status_error.message()
        });
    }
    if (!input_exists) {
        return Result<void>::Fail({"input_not_found", "Input file does not exist.", request.input_path});
    }
    const bool frame_pipe_output = !request.output.frame_pipe_path.empty();
    const std::filesystem::path final_output_path = path_from_utf8(request.output_path);
    if (!frame_pipe_output) {
        const auto output_target = ffmpeg_validate_output_target(final_output_path);
        if (!output_target.ok()) {
            return Result<void>::Fail(output_target.error());
        }
    }
    const std::filesystem::path temporary_output_path = frame_pipe_output
        ? path_from_utf8(request.output.frame_pipe_path)
        : ffmpeg_temporary_output_path(final_output_path);
    TemporaryOutputCleanupGuard temporary_output_cleanup(temporary_output_path);

    if (rtx_ == nullptr) {
        return Result<void>::Fail({
            "rtx_processor_required",
            "The FFmpeg hardware pipeline requires an RTX processor.",
            "Build with VSR_ENABLE_RTX_SDK=ON to enable RTX processing."
        });
    }
    auto canceled_result = [](const std::string& detail) -> Result<void> {
        log_info("FFmpeg pipeline canceled: " + detail + ".");
        return Result<void>::Fail({"job_canceled", "Job was canceled.", ""});
    };
    if (cancellation.requested.load()) {
        return canceled_result("before initialization");
    }

    if (progress) {
        JobProgress validating;
        validating.stage = JobStage::validating;
        validating.progress = 0.01;
        progress(validating);
    }

#if !defined(_WIN32)
    (void)progress_from_frames;
    return Result<void>::Fail({
        "d3d11_required",
        "The FFmpeg hardware pipeline requires Windows D3D11.",
        "Run the hardware backend on Windows with D3D11VA and NVENC support."
    });
#else
    AVFormatContext* raw_input = nullptr;
    int result = avformat_open_input(&raw_input, request.input_path.c_str(), nullptr, nullptr);
    if (result < 0) {
        return Result<void>::Fail(ffmpeg_error(
            "input_open_failed",
            "FFmpeg could not open the input file.",
            result));
    }
    InputFormatContextPtr input(raw_input);

    result = avformat_find_stream_info(input.get(), nullptr);
    if (result < 0) {
        return Result<void>::Fail(ffmpeg_error(
            "stream_info_failed",
            "FFmpeg could not read stream information.",
            result));
    }

    // Blu-ray m2ts sources commonly carry a large start timestamp baseline
    // (e.g. 600s). Strip it from every muxed packet so the output timeline
    // starts at zero, mirroring the ffmpeg CLI's default behavior for direct
    // libavformat muxing.
    const std::int64_t input_start_time_us =
        (input->start_time != AV_NOPTS_VALUE && input->start_time > 0) ? input->start_time : 0;

    const int video_stream_index = find_primary_video_stream(input.get());
    if (video_stream_index < 0) {
        return Result<void>::Fail({"video_stream_missing", "Input file does not contain a video stream.", request.input_path});
    }
    AVStream* input_stream = input->streams[video_stream_index];
    const AVCodec* decoder = avcodec_find_decoder(input_stream->codecpar->codec_id);
    if (decoder == nullptr) {
        return Result<void>::Fail({"decoder_missing", "FFmpeg does not have a decoder for the primary video stream.", ""});
    }

    AVRational frame_rate = av_guess_frame_rate(input.get(), input_stream, nullptr);
    if (!valid_rational(frame_rate)) {
        frame_rate = input_stream->avg_frame_rate;
    }
    if (!valid_rational(frame_rate)) {
        frame_rate = {30, 1};
    }
    const std::int64_t frames_total = estimate_total_frames(input.get(), input_stream, frame_rate);
    if (progress) {
        JobProgress probing;
        probing.stage = JobStage::probing;
        probing.progress = 0.05;
        probing.frames_total = frames_total;
        progress(probing);
    }

    const auto d3d11 = create_d3d11_device();
    if (!d3d11.ok()) {
        return Result<void>::Fail(d3d11.error());
    }
    if (progress) {
        JobProgress initializing;
        initializing.stage = JobStage::initializing_gpu;
        initializing.progress = 0.08;
        initializing.frames_total = frames_total;
        progress(initializing);
    }

    auto hw_device = create_ffmpeg_d3d11_device(d3d11.value().device.Get(), d3d11.value().context.Get());
    if (!hw_device.ok()) {
        return Result<void>::Fail(hw_device.error());
    }

    // 先查解码器硬解配置，再验证当前显卡能否解出首帧。
    bool has_d3d11_decoder = false;
    for (int index = 0; const AVCodecHWConfig* config = avcodec_get_hw_config(decoder, index); ++index) {
        if (config->device_type == AV_HWDEVICE_TYPE_D3D11VA &&
            config->pix_fmt == AV_PIX_FMT_D3D11 &&
            (config->methods & AV_CODEC_HW_CONFIG_METHOD_HW_DEVICE_CTX)) {
            has_d3d11_decoder = true;
            break;
        }
    }
    bool software_decode_fallback = !has_d3d11_decoder ||
        input_stream->codecpar->width < 320 || input_stream->codecpar->height < 240;
    CodecContextPtr decoder_context;
    auto open_decoder = [&]() -> int {
        decoder_context.reset(avcodec_alloc_context3(decoder));
        if (!decoder_context) return AVERROR(ENOMEM);
        int opened = avcodec_parameters_to_context(decoder_context.get(), input_stream->codecpar);
        if (opened < 0) return opened;
        decoder_context->pkt_timebase = input_stream->time_base;
        if (!software_decode_fallback) {
            decoder_context->get_format = choose_d3d11_format;
            decoder_context->hw_device_ctx = av_buffer_ref(hw_device.value().get());
            if (decoder_context->hw_device_ctx == nullptr) return AVERROR(ENOMEM);
        }
        return avcodec_open2(decoder_context.get(), decoder, nullptr);
    };
    result = open_decoder();
    if (result < 0 && !software_decode_fallback) {
        log_info("Hardware decoder initialization failed; retrying software decode.");
        software_decode_fallback = true;
        result = open_decoder();
    }
    if (result < 0) {
        return Result<void>::Fail(ffmpeg_error("decoder_open_failed",
            "FFmpeg could not open the video decoder.", result));
    }

    // 保留首帧探测时的所有流数据包，稍后顺序重放，不丢起始音视频或字幕。
    std::vector<PacketPtr> decoder_probe_packets;
    if (!software_decode_fallback) {
        FramePtr first_frame(av_frame_alloc());
        if (!first_frame) return Result<void>::Fail({"decoder_probe_alloc_failed", "Could not allocate decoder probe frame.", ""});
        int decoded = AVERROR(EAGAIN);
        while (decoded == AVERROR(EAGAIN)) {
            if (cancellation.requested.load()) return canceled_result("during decoder probe");
            wait_while_job_paused(cancellation);
            PacketPtr probe_packet(av_packet_alloc());
            if (!probe_packet) return Result<void>::Fail({"decoder_probe_alloc_failed", "Could not allocate decoder probe packet.", ""});
            int read = av_read_frame(input.get(), probe_packet.get());
            if (read == AVERROR_EOF) {
                decoded = avcodec_send_packet(decoder_context.get(), nullptr);
                if (decoded >= 0) decoded = avcodec_receive_frame(decoder_context.get(), first_frame.get());
                break;
            }
            if (read < 0) return Result<void>::Fail(ffmpeg_error("input_read_failed", "Could not read decoder probe packet.", read));
            const bool video_packet = probe_packet->stream_index == video_stream_index;
            decoder_probe_packets.push_back(std::move(probe_packet));
            if (!video_packet) continue;
            decoded = avcodec_send_packet(decoder_context.get(), decoder_probe_packets.back().get());
            if (decoded >= 0) decoded = avcodec_receive_frame(decoder_context.get(), first_frame.get());
        }
        if (decoded < 0 || first_frame->format != AV_PIX_FMT_D3D11) {
            log_info("Hardware first-frame decode failed; retrying software decode and D3D11 upload.");
            software_decode_fallback = true;
        }
        result = open_decoder();
        if (result < 0) return Result<void>::Fail(ffmpeg_error("decoder_open_failed", "Could not reopen video decoder.", result));
    }
    log_info(std::string("Video decode route: ") + (software_decode_fallback ? "software + D3D11 upload" : "D3D11VA") +
        "; codec=" + decoder->name + "; size=" + std::to_string(decoder_context->width) + "x" + std::to_string(decoder_context->height));

    BufferRefPtr decoder_upload_frames;
    const AVPixelFormat decoder_upload_format = decoder_pixel_depth(decoder_context.get()) > 8
        ? AV_PIX_FMT_P010LE
        : AV_PIX_FMT_NV12;
    if (software_decode_fallback) {
        auto upload_frames = create_encoder_frames_context(
            hw_device.value().get(),
            decoder_upload_format,
            decoder_context->width,
            decoder_context->height);
        if (!upload_frames.ok()) {
            return Result<void>::Fail(upload_frames.error());
        }
        decoder_upload_frames = std::move(upload_frames.value());
    }

    AVFormatContext* raw_output = nullptr;
    const std::string final_output_string = path_to_utf8(final_output_path);
    const std::string temporary_output_string = path_to_utf8(temporary_output_path);
    // "auto" guesses the muxer from the final output file extension, so every
    // container FFmpeg supports can be targeted directly; feature combinations
    // the target container cannot hold surface as the muxer's own errors when
    // the header is written.
    const char* output_format_name = nullptr;
    if (frame_pipe_output) {
        output_format_name = "rawvideo";
    } else if (request.output.container != "auto" && !request.output.container.empty()) {
        output_format_name = request.output.container.c_str();
    } else {
        const AVOutputFormat* guessed = av_guess_format(nullptr, final_output_string.c_str(), nullptr);
        if (guessed == nullptr) {
            return Result<void>::Fail({
                "unsupported_container",
                "FFmpeg has no muxer for this output file extension.",
                final_output_string});
        }
        output_format_name = guessed->name;
    }
    result = avformat_alloc_output_context2(&raw_output, nullptr, output_format_name, temporary_output_string.c_str());
    if (result < 0 || raw_output == nullptr) {
        return Result<void>::Fail(result < 0
            ? ffmpeg_error("output_context_alloc_failed", "FFmpeg could not create an output context.", result)
            : Error{"output_context_alloc_failed", "FFmpeg could not create an output context.", temporary_output_string});
    }
    OutputFormatContextPtr output(raw_output);

    const bool hdr_enabled = request.processing.hdr.enabled;
    const char* encoder_name = frame_pipe_output ? "rawvideo" : ffmpeg_nvenc_encoder_name(request.output);
    const bool av1_encoder = std::string_view(encoder_name) == "av1_nvenc";
    const bool hevc_encoder = std::string_view(encoder_name) == "hevc_nvenc";
    const AVCodec* encoder = avcodec_find_encoder_by_name(encoder_name);
    if (encoder == nullptr) {
        return Result<void>::Fail({
            "video_encoder_missing",
            "FFmpeg does not expose the required video encoder.",
            encoder_name
        });
    }

    const double scale = request.processing.vsr.enabled ? request.processing.vsr.scale : 1.0;
    const int output_width = even_scaled_dimension(decoder_context->width, scale);
    const int output_height = even_scaled_dimension(decoder_context->height, scale);
    // Keep TrueHDR's packed RGB10 output intact. NVENC accepts X2BGR10
    // directly; an extra D3D11 VideoProcessor RGB10 -> P010 pass corrupts
    // chroma on the affected driver path.
    const AVPixelFormat encoder_sw_format = hdr_enabled
        ? AV_PIX_FMT_X2BGR10
        : (request.output.pixel_format == "p010le" ? AV_PIX_FMT_P010LE : AV_PIX_FMT_NV12);
    const bool ten_bit_output = hdr_enabled || encoder_sw_format == AV_PIX_FMT_P010LE;
    const DXGI_FORMAT encoder_dxgi_format = dxgi_format_for_encoder_surface(encoder_sw_format);
    const DXGI_FORMAT rtx_input_dxgi_format = dxgi_format_for_rtx_input_surface(decoder_context.get());
    const DXGI_FORMAT rtx_output_dxgi_format = dxgi_format_for_rtx_output_surface(decoder_context.get(), hdr_enabled);
    if (encoder_dxgi_format == DXGI_FORMAT_UNKNOWN) {
        return Result<void>::Fail({"encoder_format_unsupported", "The requested encoder texture format is not supported.", ""});
    }

    auto encoder_frames = create_encoder_frames_context(
        hw_device.value().get(),
        encoder_sw_format,
        output_width,
        output_height);
    if (!encoder_frames.ok()) {
        return Result<void>::Fail(encoder_frames.error());
    }

    CodecContextPtr encoder_context(avcodec_alloc_context3(encoder));
    if (!encoder_context) {
        return Result<void>::Fail({"encoder_context_alloc_failed", "FFmpeg could not allocate the NVENC context.", ""});
    }
    encoder_context->width = output_width;
    encoder_context->height = output_height;
    encoder_context->time_base = av_inv_q(frame_rate);
    encoder_context->framerate = frame_rate;
    encoder_context->pix_fmt = frame_pipe_output ? encoder_sw_format : AV_PIX_FMT_D3D11;
    encoder_context->sw_pix_fmt = encoder_sw_format;
    // 保留 NVENC 预设的 B 帧设置：UHQ 与强制零 B 帧组合会初始化失败。
    // 尾帧时长由 drain_encoder 补齐，并保留重排产生的负 DTS。
    encoder_context->sample_aspect_ratio = decoder_context->sample_aspect_ratio;
    const std::int64_t source_bit_rate = source_video_bit_rate(input.get(), input_stream, decoder_context.get());
    const std::int64_t target_bit_rate = ffmpeg_recommended_nvenc_bitrate(
        source_bit_rate,
        decoder_context->width,
        decoder_context->height,
        output_width,
        output_height,
        av_q2d(frame_rate),
        hdr_enabled);
    encoder_context->bit_rate = target_bit_rate;
    encoder_context->rc_max_rate = target_bit_rate + (target_bit_rate / 2);
    encoder_context->rc_buffer_size = target_bit_rate * 2;
    if (!frame_pipe_output) {
        encoder_context->hw_frames_ctx = av_buffer_ref(encoder_frames.value().get());
        if (encoder_context->hw_frames_ctx == nullptr) {
            return Result<void>::Fail({"encoder_frames_ref_failed", "FFmpeg could not reference the encoder D3D11 frames.", ""});
        }
    }
    if (output->oformat != nullptr && (output->oformat->flags & AVFMT_GLOBALHEADER) != 0) {
        encoder_context->flags |= AV_CODEC_FLAG_GLOBAL_HEADER;
    }
    if (hdr_enabled) {
        encoder_context->color_primaries = AVCOL_PRI_BT2020;
        encoder_context->color_trc = AVCOL_TRC_SMPTE2084;
        encoder_context->colorspace = AVCOL_SPC_BT2020_NCL;
        encoder_context->color_range = AVCOL_RANGE_MPEG;
    }
    if (ten_bit_output && !frame_pipe_output) {
        if (hevc_encoder) {
            encoder_context->profile = 2;
            const auto profile = set_required_encoder_option(encoder_context.get(), "profile", "main10");
            if (!profile.ok()) {
                return Result<void>::Fail(profile.error());
            }
            const auto tier = set_required_encoder_option(encoder_context.get(), "tier", "main");
            if (!tier.ok()) {
                return Result<void>::Fail(tier.error());
            }
        } else if (av1_encoder) {
            encoder_context->profile = AV_PROFILE_AV1_MAIN;
            const auto high_bit_depth = set_required_encoder_option(encoder_context.get(), "highbitdepth", "1");
            if (!high_bit_depth.ok()) {
                return Result<void>::Fail(high_bit_depth.error());
            }
        }
    }
    if (!hdr_enabled) {
        encoder_context->color_primaries = decoder_context->color_primaries;
        encoder_context->color_trc = decoder_context->color_trc;
        encoder_context->colorspace = decoder_context->colorspace;
        encoder_context->color_range = decoder_context->color_range;
    }
    if (!frame_pipe_output) {
        const auto quality_configured = configure_nvenc_quality(
            encoder_context.get(),
            request.output,
            hevc_encoder || av1_encoder);
        if (!quality_configured.ok()) {
            return Result<void>::Fail(quality_configured.error());
        }
    }
    if (av1_encoder && !frame_pipe_output) {
        const auto level = select_av1_level(
            output_width, output_height, av_q2d(frame_rate),
            encoder_context->rc_max_rate, encoder_context->rc_buffer_size);
        if (!level.ok()) {
            return Result<void>::Fail(level.error());
        }
        const auto set_level = set_required_encoder_option_int(encoder_context.get(), "level", level.value().index);
        if (!set_level.ok()) {
            return set_level;
        }
        const auto set_tier = set_required_encoder_option_int(encoder_context.get(), "tier", level.value().tier);
        if (!set_tier.ok()) {
            return set_tier;
        }
        log_info("AV1 level configured: index=" + std::to_string(level.value().index) +
            ", tier=" + std::to_string(level.value().tier));
    }
    if (!frame_pipe_output) {
        log_info(
            "NVENC quality configured: encoder=" + std::string(encoder_name) +
            ", cq=" + std::to_string(nvenc_target_quality(request.output.video_codec)) +
            ", target_bitrate=" + std::to_string(target_bit_rate) +
            ", maxrate=" + std::to_string(encoder_context->rc_max_rate) +
            ", buffer=" + std::to_string(encoder_context->rc_buffer_size));
    } else {
        log_info("RTX frame pipe configured: format=" + std::string(av_get_pix_fmt_name(encoder_sw_format)));
    }

    result = avcodec_open2(encoder_context.get(), encoder, nullptr);
    if (result < 0) {
        return Result<void>::Fail(ffmpeg_error(
            "encoder_open_failed",
            "FFmpeg could not open the video encoder.",
            result));
    }

    AVStream* output_stream = avformat_new_stream(output.get(), nullptr);
    if (output_stream == nullptr) {
        return Result<void>::Fail({"output_stream_create_failed", "FFmpeg could not create the output video stream.", ""});
    }
    output_stream->time_base = encoder_context->time_base;
    result = avcodec_parameters_from_context(output_stream->codecpar, encoder_context.get());
    if (result < 0) {
        return Result<void>::Fail(ffmpeg_error(
            "encoder_parameters_failed",
            "FFmpeg could not copy encoder parameters to the output stream.",
            result));
    }
    const auto display_matrix_copied = ffmpeg_copy_display_matrix(
        input_stream->codecpar,
        output_stream->codecpar);
    if (!display_matrix_copied.ok()) {
        return Result<void>::Fail(display_matrix_copied.error());
    }

    struct CopiedStream {
        AVStream* output_stream = nullptr;
    };
    std::vector<CopiedStream> copied_streams(input->nb_streams);
    int audio_ordinal = 0;
    int subtitle_ordinal = 0;
    if (ffmpeg_requests_stream_copy(request.output)) {
        for (unsigned int index = 0; index < input->nb_streams; ++index) {
            if (static_cast<int>(index) == video_stream_index) {
                continue;
            }

            AVStream* source_stream = input->streams[index];
            if (source_stream == nullptr || source_stream->codecpar == nullptr) {
                continue;
            }

            const AVMediaType media_type = source_stream->codecpar->codec_type;
            if (media_type == AVMEDIA_TYPE_AUDIO) {
                const int current_ordinal = audio_ordinal++;
                if (request.output.audio_mode != "copy") {
                    continue;
                }
                if (request.output.audio_stream_indices.has_value() &&
                    !std::binary_search(request.output.audio_stream_indices->begin(),
                        request.output.audio_stream_indices->end(), current_ordinal)) {
                    continue;
                }
                // MPEG-TS/Blu-ray demuxing often leaves audio parameters unset.
                // TrueHD is a fixed-rate codec and Matroska refuses to write an
                // audio track without a sample rate, so fill in the known value.
                if (source_stream->codecpar->codec_id == AV_CODEC_ID_TRUEHD &&
                    source_stream->codecpar->sample_rate == 0) {
                    source_stream->codecpar->sample_rate = 48000;
                }
                if (source_stream->codecpar->sample_rate <= 0 ||
                    source_stream->codecpar->ch_layout.nb_channels <= 0) {
                    report_warning(
                        "Skipped audio stream with incomplete stream parameters (" + codec_details(source_stream) + ").");
                    continue;
                }
            } else if (media_type == AVMEDIA_TYPE_SUBTITLE) {
                const int current_ordinal = subtitle_ordinal++;
                if (request.output.subtitle_mode != "copy-compatible") {
                    continue;
                }
                if (request.output.subtitle_stream_indices.has_value() &&
                    !std::binary_search(request.output.subtitle_stream_indices->begin(),
                        request.output.subtitle_stream_indices->end(), current_ordinal)) {
                    continue;
                }
            } else {
                continue;
            }

            AVStream* copied_stream = avformat_new_stream(output.get(), nullptr);
            if (copied_stream == nullptr) {
                return Result<void>::Fail({"output_stream_create_failed", "FFmpeg could not create a copied output stream.", codec_details(source_stream)});
            }
            result = avcodec_parameters_copy(copied_stream->codecpar, source_stream->codecpar);
            if (result < 0) {
                return Result<void>::Fail(ffmpeg_error(
                    "copy_stream_parameters_failed",
                    "FFmpeg could not copy input stream parameters to the output.",
                    result));
            }
            copied_stream->codecpar->codec_tag = 0;
            copied_stream->time_base = source_stream->time_base;
            copied_streams[index].output_stream = copied_stream;
        }
    }

    const auto initialized = rtx_->initialize(d3d11.value().device.Get(), request.processing);
    if (!initialized.ok()) {
        return Result<void>::Fail(initialized.error());
    }
    RtxShutdownGuard rtx_shutdown(rtx_.get());
    rtx_shutdown.activate();

    if (output->oformat != nullptr && (output->oformat->flags & AVFMT_NOFILE) == 0) {
        result = avio_open(&output->pb, temporary_output_string.c_str(), AVIO_FLAG_WRITE);
        if (result < 0) {
            return Result<void>::Fail(ffmpeg_error(
                "output_open_failed",
                "FFmpeg could not open the output file for writing.",
                result));
        }
    }

    result = avformat_write_header(output.get(), nullptr);
    if (result < 0) {
        return Result<void>::Fail(ffmpeg_error(
            "output_header_failed",
            (std::string("FFmpeg could not write the ") +
             (output->oformat != nullptr && output->oformat->name != nullptr ? output->oformat->name : "output") +
             " header. The target container likely cannot hold one of the streams.").c_str(),
            result));
    }

    PacketPtr packet(av_packet_alloc());
    PacketPtr encoder_packet(av_packet_alloc());
    FramePtr decoded_frame(av_frame_alloc());
    FramePtr encoder_frame(av_frame_alloc());
    FramePtr downloaded_frame(av_frame_alloc());
    FramePtr converted_decoder_frame(av_frame_alloc());
    FramePtr uploaded_decoder_frame(av_frame_alloc());
    if (!packet || !encoder_packet || !decoded_frame || !encoder_frame || !downloaded_frame
        || !converted_decoder_frame || !uploaded_decoder_frame) {
        return Result<void>::Fail({"ffmpeg_frame_alloc_failed", "FFmpeg could not allocate decode buffers.", ""});
    }

    constexpr std::size_t rtx_input_ring_size = 4;
    std::vector<Microsoft::WRL::ComPtr<ID3D11Texture2D>> rtx_input_textures;
    std::vector<Microsoft::WRL::ComPtr<ID3D11Texture2D>> rtx_output_textures;
    D3d11VideoConverter decoder_to_rtx_converter;
    D3d11VideoConverter rtx_to_encoder_converter;
    bool decoder_to_rtx_converter_initialized = false;
    bool rtx_to_encoder_converter_initialized = false;
    std::int64_t frames_done = 0;
    auto progress_sample_time = std::chrono::steady_clock::now();
    std::int64_t progress_sample_frames = 0;
    double smoothed_fps = 0.0;
    std::int64_t eta_seconds = 0;
    const DXGI_COLOR_SPACE_TYPE video_processor_input_color = d3d11_input_color_space(decoder_context.get());
    const DXGI_COLOR_SPACE_TYPE rtx_input_color = d3d11_rtx_input_color_space(decoder_context.get());
    const DXGI_COLOR_SPACE_TYPE rtx_output_color = hdr_enabled
        ? DXGI_COLOR_SPACE_RGB_FULL_G2084_NONE_P2020
        : d3d11_rtx_input_color_space(decoder_context.get());
    const DXGI_COLOR_SPACE_TYPE encoder_output_color = d3d11_encoder_output_color_space(encoder_context.get());

    auto emit_progress = [&](JobStage stage, std::int64_t done) {
        if (!progress) {
            return;
        }
        const auto now = std::chrono::steady_clock::now();
        const double sample_seconds = std::chrono::duration<double>(now - progress_sample_time).count();
        const std::int64_t sample_frames = done - progress_sample_frames;
        if (sample_frames > 0 && (smoothed_fps <= 0.0 || sample_seconds >= 0.25)) {
            const auto metrics = ffmpeg_progress_metrics(
                done,
                frames_total,
                sample_frames,
                sample_seconds,
                smoothed_fps);
            smoothed_fps = metrics.fps;
            eta_seconds = metrics.eta_seconds;
            progress_sample_time = now;
            progress_sample_frames = done;
        } else if (smoothed_fps > 0.0 && frames_total > done) {
            eta_seconds = static_cast<std::int64_t>(std::ceil(
                static_cast<double>(frames_total - done) / smoothed_fps));
        }
        JobProgress job_progress;
        job_progress.stage = stage;
        job_progress.progress = progress_from_frames(done, frames_total);
        job_progress.frames_done = done;
        job_progress.frames_total = frames_total;
        job_progress.fps = smoothed_fps;
        job_progress.eta_seconds = eta_seconds;
        progress(job_progress);
    };

    auto process_decoded_frame = [&](AVFrame* frame) -> Result<void> {
        wait_while_job_paused(cancellation);
        if (cancellation.requested.load()) {
            return canceled_result("before processing decoded frame");
        }
        AVFrame* d3d11_frame = frame;
        if (software_decode_fallback) {
            AVFrame* upload_source = frame;
            const auto source_format = static_cast<AVPixelFormat>(frame->format);
            if (source_format != decoder_upload_format) {
                av_frame_unref(converted_decoder_frame.get());
                converted_decoder_frame->format = decoder_upload_format;
                converted_decoder_frame->width = frame->width;
                converted_decoder_frame->height = frame->height;
                result = av_frame_get_buffer(converted_decoder_frame.get(), 32);
                if (result < 0) {
                    return Result<void>::Fail(ffmpeg_error(
                        "software_decode_buffer_failed",
                        "FFmpeg could not allocate the low-resolution upload frame.",
                        result));
                }
                const auto converted = convert_software_frame_for_d3d11_upload(
                    frame,
                    converted_decoder_frame.get(),
                    decoder_upload_format);
                if (!converted.ok()) {
                    return converted;
                }
                upload_source = converted_decoder_frame.get();
            }

            av_frame_unref(uploaded_decoder_frame.get());
            result = av_hwframe_get_buffer(decoder_upload_frames.get(), uploaded_decoder_frame.get(), 0);
            if (result < 0) {
                return Result<void>::Fail(ffmpeg_error(
                    "decoder_upload_buffer_failed",
                    "FFmpeg could not allocate a D3D11 frame for software-decoded input.",
                    result));
            }
            result = av_hwframe_transfer_data(uploaded_decoder_frame.get(), upload_source, 0);
            if (result < 0) {
                return Result<void>::Fail(ffmpeg_error(
                    "decoder_upload_failed",
                    "FFmpeg could not upload the software-decoded frame to D3D11.",
                    result));
            }
            d3d11_frame = uploaded_decoder_frame.get();
        }

        if (d3d11_frame->format != AV_PIX_FMT_D3D11 || d3d11_frame->data[0] == nullptr) {
            return Result<void>::Fail({
                "d3d11_frame_required",
                "The hardware pipeline requires FFmpeg D3D11VA frames.",
                "Use a Windows FFmpeg build with D3D11VA and NVENC support."
            });
        }

        auto* decoded_texture = reinterpret_cast<ID3D11Texture2D*>(d3d11_frame->data[0]);
        const UINT decoded_slice = static_cast<UINT>(reinterpret_cast<intptr_t>(d3d11_frame->data[1]));

        if (rtx_input_textures.empty()) {
            rtx_input_textures.resize(rtx_input_ring_size);
            rtx_output_textures.resize(rtx_input_ring_size);
            for (std::size_t index = 0; index < rtx_input_ring_size; ++index) {
                const auto created = create_texture(
                    d3d11.value().device.Get(),
                    decoder_context->width,
                    decoder_context->height,
                    rtx_input_dxgi_format,
                    D3D11_BIND_RENDER_TARGET | D3D11_BIND_SHADER_RESOURCE | D3D11_BIND_UNORDERED_ACCESS,
                    rtx_input_textures[index]);
                if (!created.ok()) {
                    return Result<void>::Fail(created.error());
                }
                const auto output_created = create_texture(
                    d3d11.value().device.Get(),
                    output_width,
                    output_height,
                    rtx_output_dxgi_format,
                    D3D11_BIND_RENDER_TARGET | D3D11_BIND_SHADER_RESOURCE | D3D11_BIND_UNORDERED_ACCESS,
                    rtx_output_textures[index]);
                if (!output_created.ok()) {
                    return Result<void>::Fail(output_created.error());
                }
            }
            const auto converter_initialized = decoder_to_rtx_converter.initialize(
                d3d11.value().device.Get(),
                d3d11.value().context.Get(),
                decoded_texture,
                rtx_input_textures.front().Get(),
                decoder_context->width,
                decoder_context->height,
                decoder_context->width,
                decoder_context->height,
                video_processor_input_color,
                rtx_input_color);
            if (!converter_initialized.ok()) {
                return Result<void>::Fail(converter_initialized.error());
            }
            decoder_to_rtx_converter_initialized = true;
        }

        if (!decoder_to_rtx_converter_initialized) {
            return Result<void>::Fail({"d3d11_video_converter_uninitialized", "The D3D11 video converter was not initialized.", ""});
        }
        const std::size_t rtx_ring_index = static_cast<std::size_t>(frames_done) % rtx_input_textures.size();
        auto& rtx_input_texture = rtx_input_textures[rtx_ring_index];
        auto& rtx_output_texture = rtx_output_textures[rtx_ring_index];
        const auto converted = decoder_to_rtx_converter.convert(decoded_texture, decoded_slice, rtx_input_texture.Get(), 0);
        if (!converted.ok()) {
            return Result<void>::Fail(converted.error());
        }

        av_frame_unref(encoder_frame.get());
        result = av_hwframe_get_buffer(encoder_frames.value().get(), encoder_frame.get(), 0);
        if (result < 0) {
            return Result<void>::Fail(ffmpeg_error(
                "encoder_frame_get_buffer_failed",
                "FFmpeg could not allocate a D3D11 encoder frame.",
                result));
        }

        auto* output_texture = reinterpret_cast<ID3D11Texture2D*>(encoder_frame->data[0]);
        const UINT output_slice = static_cast<UINT>(reinterpret_cast<intptr_t>(encoder_frame->data[1]));
        if (output_texture == nullptr) {
            return Result<void>::Fail({"encoder_frame_texture_missing", "The D3D11 encoder frame did not expose a texture.", ""});
        }
        D3D11_TEXTURE2D_DESC output_desc = {};
        output_texture->GetDesc(&output_desc);
        if (output_desc.Format != encoder_dxgi_format) {
            return Result<void>::Fail({
                "encoder_texture_format_mismatch",
                "The NVENC D3D11 frame format does not match the requested YUV format.",
                ""
            });
        }

        if (!hdr_enabled && !rtx_to_encoder_converter_initialized) {
            const auto converter_initialized = rtx_to_encoder_converter.initialize(
                d3d11.value().device.Get(),
                d3d11.value().context.Get(),
                rtx_output_texture.Get(),
                output_texture,
                output_width,
                output_height,
                output_width,
                output_height,
                rtx_output_color,
                encoder_output_color);
            if (!converter_initialized.ok()) {
                return Result<void>::Fail(converter_initialized.error());
            }
            rtx_to_encoder_converter_initialized = true;
        }

        emit_progress(JobStage::processing_rtx, frames_done);
        RtxDx11Frame rtx_frame;
        rtx_frame.input = rtx_input_texture.Get();
        rtx_frame.output = rtx_output_texture.Get();
        rtx_frame.input_width = static_cast<std::uint32_t>(decoder_context->width);
        rtx_frame.input_height = static_cast<std::uint32_t>(decoder_context->height);
        rtx_frame.output_width = static_cast<std::uint32_t>(output_width);
        rtx_frame.output_height = static_cast<std::uint32_t>(output_height);

        const auto processed = rtx_->process(rtx_frame, request.processing);
        if (!processed.ok()) {
            return Result<void>::Fail(processed.error());
        }

        if (hdr_enabled) {
            d3d11.value().context->CopySubresourceRegion(
                output_texture,
                D3D11CalcSubresource(0, output_slice, 1),
                0,
                0,
                0,
                rtx_output_texture.Get(),
                0,
                nullptr);
        } else {
            const auto encoder_converted = rtx_to_encoder_converter.convert(
                rtx_output_texture.Get(),
                0,
                output_texture,
                output_slice);
            if (!encoder_converted.ok()) {
                return Result<void>::Fail(encoder_converted.error());
            }
        }

        encoder_frame->pts = frame->pts == AV_NOPTS_VALUE
            ? frames_done
            : av_rescale_q(frame->pts, input_stream->time_base, encoder_context->time_base);
        // 在送入编码器前归零；NVENC 的重排 DTS 依赖输入 PTS，封装前再减偏移过晚。
        if (frame->pts != AV_NOPTS_VALUE && input_start_time_us > 0) {
            encoder_frame->pts -= av_rescale_q(input_start_time_us, AV_TIME_BASE_Q, encoder_context->time_base);
        }
        encoder_frame->duration = frame->duration > 0
            ? av_rescale_q(frame->duration, input_stream->time_base, encoder_context->time_base)
            : 0;
        encoder_frame->width = output_width;
        encoder_frame->height = output_height;
        encoder_frame->format = AV_PIX_FMT_D3D11;

        AVFrame* frame_to_encode = encoder_frame.get();
        if (frame_pipe_output) {
            av_frame_unref(downloaded_frame.get());
            result = av_hwframe_transfer_data(downloaded_frame.get(), encoder_frame.get(), 0);
            if (result < 0) {
                return Result<void>::Fail(ffmpeg_error(
                    "frame_download_failed",
                    "FFmpeg could not download the RTX frame for the 3FUI encoder.",
                    result));
            }
            downloaded_frame->pts = encoder_frame->pts;
            downloaded_frame->duration = encoder_frame->duration;
            downloaded_frame->color_primaries = encoder_context->color_primaries;
            downloaded_frame->color_trc = encoder_context->color_trc;
            downloaded_frame->colorspace = encoder_context->colorspace;
            downloaded_frame->color_range = encoder_context->color_range;
            frame_to_encode = downloaded_frame.get();
        }

        result = avcodec_send_frame(encoder_context.get(), frame_to_encode);
        if (result < 0) {
            return Result<void>::Fail(ffmpeg_error(
                "encoder_send_frame_failed",
                "The video encoder could not accept a processed frame.",
                result));
        }
        ++frames_done;
        emit_progress(JobStage::encoding, frames_done);

        return drain_encoder(encoder_context.get(), output.get(), output_stream, encoder_packet.get());
    };

    auto receive_decoder_frames = [&]() -> Result<void> {
        for (;;) {
            av_frame_unref(decoded_frame.get());
            result = avcodec_receive_frame(decoder_context.get(), decoded_frame.get());
            if (result == AVERROR(EAGAIN) || result == AVERROR_EOF) {
                return Result<void>::Ok();
            }
            if (result < 0) {
                return Result<void>::Fail(ffmpeg_error(
                    "decoder_receive_frame_failed",
                    "FFmpeg could not receive a decoded frame.",
                    result));
            }

            const auto processed = process_decoded_frame(decoded_frame.get());
            if (!processed.ok()) {
                return Result<void>::Fail(processed.error());
            }
        }
    };

    std::size_t decoder_probe_index = 0;
    for (;;) {
        if (cancellation.requested.load()) {
            return canceled_result("while reading input");
        }
        wait_while_job_paused(cancellation);
        av_packet_unref(packet.get());
        result = decoder_probe_index < decoder_probe_packets.size()
            ? av_packet_ref(packet.get(), decoder_probe_packets[decoder_probe_index++].get())
            : av_read_frame(input.get(), packet.get());
        if (result == AVERROR_EOF) {
            break;
        }
        if (result < 0) {
            return Result<void>::Fail(ffmpeg_error(
                "input_read_failed",
                "FFmpeg could not read the next input packet.",
                result));
        }
        if (packet->stream_index != video_stream_index) {
            if (packet->stream_index >= 0 && static_cast<std::size_t>(packet->stream_index) < copied_streams.size()) {
                AVStream* copied_stream = copied_streams[packet->stream_index].output_stream;
                if (copied_stream != nullptr) {
                    AVStream* source_stream = input->streams[packet->stream_index];
                    packet->stream_index = copied_stream->index;
                    av_packet_rescale_ts(packet.get(), source_stream->time_base, copied_stream->time_base);
                    if (input_start_time_us > 0) {
                        const auto shift = av_rescale_q(input_start_time_us, AV_TIME_BASE_Q, copied_stream->time_base);
                        if (packet->pts != AV_NOPTS_VALUE) {
                            packet->pts = std::max<std::int64_t>(0, packet->pts - shift);
                        }
                        if (packet->dts != AV_NOPTS_VALUE) {
                            packet->dts = std::max<std::int64_t>(0, packet->dts - shift);
                        }
                    }
                    const int write_result = av_interleaved_write_frame(output.get(), packet.get());
                    if (write_result < 0) {
                        return Result<void>::Fail(ffmpeg_error(
                            "output_copy_packet_write_failed",
                            "FFmpeg could not mux a copied stream packet.",
                            write_result));
                    }
                }
            }
            continue;
        }

        emit_progress(JobStage::decoding, frames_done);
        result = avcodec_send_packet(decoder_context.get(), packet.get());
        if (result < 0) {
            return Result<void>::Fail(ffmpeg_error(
                "decoder_send_packet_failed",
                "FFmpeg could not send a packet to the decoder.",
                result));
        }

        const auto received = receive_decoder_frames();
        if (!received.ok()) {
            return Result<void>::Fail(received.error());
        }
    }

    if (cancellation.requested.load()) {
        return canceled_result("before decoder flush");
    }
    result = avcodec_send_packet(decoder_context.get(), nullptr);
    if (result < 0) {
        return Result<void>::Fail(ffmpeg_error(
            "decoder_flush_failed",
            "FFmpeg could not flush the decoder.",
            result));
    }
    const auto decoded_flush = receive_decoder_frames();
    if (!decoded_flush.ok()) {
        return Result<void>::Fail(decoded_flush.error());
    }

    if (cancellation.requested.load()) {
        return canceled_result("before encoder flush");
    }
    result = avcodec_send_frame(encoder_context.get(), nullptr);
    if (result < 0) {
        return Result<void>::Fail(ffmpeg_error(
            "encoder_flush_failed",
            "The video encoder could not flush.",
            result));
    }
    const auto encoded_flush = drain_encoder(encoder_context.get(), output.get(), output_stream, encoder_packet.get());
    if (!encoded_flush.ok()) {
        return Result<void>::Fail(encoded_flush.error());
    }

    if (progress) {
        JobProgress muxing;
        muxing.stage = JobStage::muxing;
        muxing.progress = 0.99;
        muxing.frames_done = frames_done;
        // 容器时长只能给出估算值；解码完成后用实际处理帧数纠正总数。
        muxing.frames_total = frames_done;
        muxing.fps = smoothed_fps;
        muxing.eta_seconds = 0;
        progress(muxing);
    }

    if (cancellation.requested.load()) {
        return canceled_result("before final muxing");
    }
    result = av_write_trailer(output.get());
    if (result < 0) {
        return Result<void>::Fail(ffmpeg_error(
            "output_trailer_failed",
            "FFmpeg could not write the MP4 trailer.",
            result));
    }

    if (output->pb != nullptr && output->oformat != nullptr && (output->oformat->flags & AVFMT_NOFILE) == 0) {
        result = avio_closep(&output->pb);
        if (result < 0) {
            return Result<void>::Fail(ffmpeg_error(
                "output_close_failed",
                "FFmpeg could not close the temporary output file.",
                result));
        }
    }

    if (!frame_pipe_output) {
        const auto replaced = ffmpeg_replace_output_file(temporary_output_path, final_output_path);
        if (!replaced.ok()) {
            return Result<void>::Fail(replaced.error());
        }
    }
    temporary_output_cleanup.release();

    rtx_shutdown.release();
    log_info("FFmpeg pipeline completed: " + request.output_path);
    if (progress) {
        JobProgress finalizing;
        finalizing.stage = JobStage::finalizing;
        finalizing.progress = 1.0;
        finalizing.frames_done = frames_done;
        finalizing.frames_total = frames_done;
        finalizing.fps = smoothed_fps;
        finalizing.eta_seconds = 0;
        progress(finalizing);
    }
    return Result<void>::Ok();
#endif
}

} // namespace vsr
