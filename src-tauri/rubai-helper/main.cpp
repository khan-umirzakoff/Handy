#include "whisper.h"

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <thread>
#include <vector>

namespace {

bool read_exact(void *data, size_t size) {
    auto *cursor = static_cast<unsigned char *>(data);
    while (size > 0) {
        const size_t read = std::fread(cursor, 1, size, stdin);
        if (read == 0) {
            return false;
        }
        cursor += read;
        size -= read;
    }
    return true;
}

bool write_exact(const void *data, size_t size) {
    const auto *cursor = static_cast<const unsigned char *>(data);
    while (size > 0) {
        const size_t written = std::fwrite(cursor, 1, size, stdout);
        if (written == 0) {
            return false;
        }
        cursor += written;
        size -= written;
    }
    return true;
}

bool write_status(uint32_t status, const std::string &message) {
    const uint32_t length = static_cast<uint32_t>(message.size());
    return write_exact(&status, sizeof(status)) &&
           write_exact(&length, sizeof(length)) &&
           write_exact(message.data(), message.size()) &&
           std::fflush(stdout) == 0;
}

void quiet_whisper_log(enum ggml_log_level, const char *, void *) {}

} // namespace

int main(int argc, char **argv) {
    if (argc != 2) {
        return 2;
    }

    whisper_log_set(quiet_whisper_log, nullptr);

    whisper_context_params context_params = whisper_context_default_params();
    context_params.use_gpu = true;
    context_params.flash_attn = true;

    whisper_context *context = whisper_init_from_file_with_params(argv[1], context_params);
    if (context == nullptr) {
        write_status(1, "Rubai modelini yuklab bo'lmadi");
        return 3;
    }

    if (!write_status(0, "ready")) {
        whisper_free(context);
        return 4;
    }

    for (;;) {
        uint32_t sample_count = 0;
        if (!read_exact(&sample_count, sizeof(sample_count)) || sample_count == 0) {
            break;
        }

        std::vector<float> samples(sample_count);
        if (!read_exact(samples.data(), samples.size() * sizeof(float))) {
            break;
        }

        whisper_full_params params =
            whisper_full_default_params(WHISPER_SAMPLING_BEAM_SEARCH);
        params.language = "uz";
        params.translate = false;
        params.beam_search.beam_size = 5;
        const unsigned int hardware_threads = std::thread::hardware_concurrency();
        params.n_threads = static_cast<int>(std::max(1u, std::min(4u, hardware_threads)));
        params.no_timestamps = true;
        params.print_progress = false;
        params.print_realtime = false;
        params.print_special = false;
        params.print_timestamps = false;
        params.suppress_blank = true;
        params.no_speech_thold = 0.25f;
        params.logprob_thold = -1.0f;

        if (whisper_full(context, params, samples.data(), static_cast<int>(samples.size())) != 0) {
            if (!write_status(1, "Rubai transkripsiyasi bajarilmadi")) {
                break;
            }
            continue;
        }

        std::string text;
        const int segment_count = whisper_full_n_segments(context);
        for (int i = 0; i < segment_count; ++i) {
            const char *segment = whisper_full_get_segment_text(context, i);
            if (segment != nullptr) {
                text += segment;
            }
        }

        if (!write_status(0, text)) {
            break;
        }
    }

    whisper_free(context);
    return 0;
}
