// Experiment adapter for the pinned Mooncake CPU wheel. Keep the upstream
// admission, deferral, size limits, file writer and completion handler.
#include <atomic>
#include <cstdint>
#include <cstdlib>
#include <string>
#include <unordered_map>
#include <vector>
#include <tl/expected.hpp>

namespace {
std::atomic<std::uint64_t> calls{0}, mixed_plans{0}, emitted{0};
bool split = false;
}

extern "C" void configure_bucket_layout(int separate) { split = separate != 0; }
extern "C" std::uint64_t layout_calls() { return calls.load(); }
extern "C" std::uint64_t layout_mixed_plans() { return mixed_plans.load(); }
extern "C" std::uint64_t layout_emitted() { return emitted.load(); }

namespace mooncake {
enum class ErrorCode : std::int32_t;
using Keys = std::vector<std::string>;
using Plans = std::vector<Keys>;
using Sizes = std::unordered_map<std::string, std::int64_t>;
using Status = tl::expected<void, ErrorCode>;
class BucketStorageBackend {
public:
    Status AllocateOffloadingBuckets(const Sizes&, Plans&, Keys*);
    Status GroupOffloadingKeysByBucket(const Sizes&, Plans&, Keys*);
};

Status BucketStorageBackend::AllocateOffloadingBuckets(
        const Sizes& sizes, Plans& out, Keys* deferred) {
    Plans plans;
    auto result = GroupOffloadingKeysByBucket(sizes, plans, deferred);
    if (!result) return result;
    ++calls;
    for (auto& plan : plans) {
        Keys mla, kda;
        for (const auto& key : plan) {
            if (key.find("@store_format:attention_head_major@") != std::string::npos)
                mla.push_back(key);
            else if (key.find("@store_format:mamba_state@") != std::string::npos)
                kda.push_back(key);
            else
                std::abort(); // This experiment only accepts the two pinned formats.
        }
        if (!mla.empty() && !kda.empty()) ++mixed_plans;
        if (split && !mla.empty() && !kda.empty()) {
            out.push_back(std::move(mla));
            out.push_back(std::move(kda));
            emitted += 2;
        } else {
            out.push_back(std::move(plan));
            ++emitted;
        }
    }
    return result;
}
} // namespace mooncake
