// Passive client-side boundaries of the pinned n15 wheel. All calls use the
// real service.
#include "observer_symbols.inc"
#include "replica.h"
#include "rpc_types.h"
#include "storage_backend.h"

#include <cstdlib>
#include <dlfcn.h>
#include <fstream>
#include <mutex>
#include <numeric>
#include <time.h>

namespace {
using namespace mooncake;
using Strings = std::vector<std::string>;
using Status = tl::expected<void, ErrorCode>;
using Allocation =
    std::vector<tl::expected<std::vector<Replica::Descriptor>, ErrorCode>>;
using Slices = std::unordered_map<std::string, std::vector<Slice>>;
using Completion = std::function<ErrorCode(
    const Strings &, std::vector<StorageObjectMetadata> &)>;

template <typename F> F real(const char *name) {
  static void *handle =
      dlopen(std::getenv("KV_SSD_OBSERVED_DSO"), RTLD_LAZY | RTLD_NOLOAD);
  void *symbol = handle ? dlsym(handle, name) : nullptr;
  if (!symbol)
    std::abort(); // An incompatible measurement must fail visibly.
  return reinterpret_cast<F>(symbol);
}

long long now() {
  timespec t;
  clock_gettime(CLOCK_MONOTONIC, &t);
  return static_cast<long long>(t.tv_sec) * 1000000000LL + t.tv_nsec;
}

void record(long long stamp, const char *stage, const std::string &key,
            long long bytes, long long bucket, int code = 0) {
  static std::mutex lock;
  std::lock_guard<std::mutex> guard(lock);
  static std::ofstream out(std::getenv("KV_SSD_OBSERVER_LOG"), std::ios::app);
  if (key.find_first_of("\t\r\n") != key.npos)
    std::abort();
  out << stamp << '\t' << stage << '\t' << key << '\t' << bytes << '\t'
      << bucket << '\t' << code << std::endl;
  if (!out)
    std::abort();
}
} // namespace

// On the pinned x86-64 C++ ABI a member call passes this as its first explicit
// argument. Aliases are resolved at build time from the verified wheel,
// including all template arguments.
extern "C" Allocation allocate(void *, const Strings &,
                               const std::vector<std::vector<uint64_t>> &,
                               const ReplicateConfig &) asm(OBS_ALLOCATE);
Allocation allocate(void *self, const Strings &keys,
                    const std::vector<std::vector<uint64_t>> &sizes,
                    const ReplicateConfig &config) {
  static auto call = real<decltype(&allocate)>(OBS_ALLOCATE);
  auto result = call(self, keys, sizes, config);
  auto t = now();
  for (size_t i = 0; i < result.size(); ++i)
    record(t, "allocated", keys.at(i),
           std::accumulate(sizes.at(i).begin(), sizes.at(i).end(), uint64_t{0}),
           -1, result[i] ? 0 : static_cast<int>(result[i].error()));
  return result;
}

extern "C" std::vector<Status> ready(void *, const std::vector<ObjectMeta> &,
                                     ReplicaType) asm(OBS_READY);
std::vector<Status> ready(void *self, const std::vector<ObjectMeta> &keys,
                          ReplicaType type) {
  static auto call = real<decltype(&ready)>(OBS_READY);
  auto result = call(self, keys, type);
  auto t = now();
  for (size_t i = 0; i < result.size(); ++i)
    record(t, "memory_ready", keys.at(i).key, -1, -1,
           result[i] ? 0 : static_cast<int>(result[i].error()));
  return result;
}

extern "C" tl::expected<int64_t, ErrorCode>
submit(void *, const Slices &, Completion,
       StorageBackendInterface::EvictionHandler) asm(OBS_SUBMIT);
tl::expected<int64_t, ErrorCode>
submit(void *self, const Slices &batch, Completion done,
       StorageBackendInterface::EvictionHandler evict) {
  static auto call = real<decltype(&submit)>(OBS_SUBMIT);
  auto t = now();
  for (const auto &[key, parts] : batch) {
    size_t bytes = 0;
    for (const auto &part : parts)
      bytes += part.size;
    record(t, "write_submit", key, bytes, -1);
  }
  return call(self, batch, std::move(done), std::move(evict));
}

extern "C" Status write_bucket(void *, int64_t, std::shared_ptr<BucketMetadata>,
                               std::vector<iovec> &) asm(OBS_WRITE);
Status write_bucket(void *self, int64_t bucket,
                    std::shared_ptr<BucketMetadata> metadata,
                    std::vector<iovec> &iov) {
  static auto call = real<decltype(&write_bucket)>(OBS_WRITE);
  auto start = now();
  auto result = call(self, bucket, metadata, iov);
  auto end = now();
  for (size_t i = 0; i < metadata->keys.size(); ++i) {
    auto size = metadata->metadatas.at(i).data_size;
    record(start, "write_start", metadata->keys[i], size, bucket);
    record(end, "file_write_end", metadata->keys[i], size, bucket,
           result ? 0 : static_cast<int>(result.error()));
  }
  return result;
}

extern "C" Status
publish(void *, const UUID &, const std::vector<OffloadTaskItem> &,
        const std::vector<StorageObjectMetadata> &) asm(OBS_PUBLISH);
Status publish(void *self, const UUID &client,
               const std::vector<OffloadTaskItem> &tasks,
               const std::vector<StorageObjectMetadata> &metas) {
  static auto call = real<decltype(&publish)>(OBS_PUBLISH);
  auto result = call(self, client, tasks, metas);
  auto t = now();
  for (size_t i = 0; i < tasks.size(); ++i)
    record(t, "disk_ack", tasks[i].key, metas.at(i).data_size,
           metas[i].bucket_id, result ? 0 : static_cast<int>(result.error()));
  return result;
}

using ReadSlices = std::unordered_map<std::string, Slice>;
using ReadSpans = std::unordered_map<std::string, StorageValueSpan>;
size_t length(const Slice &part) { return part.size; }
size_t length(const StorageValueSpan &part) { return part.slice.size; }

template <typename Map, typename F>
Status read_batch(void *self, Map &batch, F call) {
  auto start = now();
  auto result = call(self, batch);
  auto end = now();
  for (const auto &[key, part] : batch) {
    record(start, "read_submit", key, length(part), -1);
    record(end, "read_complete", key, length(part), -1,
           result ? 0 : static_cast<int>(result.error()));
  }
  return result;
}

extern "C" Status read_slices(void *, ReadSlices &) asm(OBS_READ);
Status read_slices(void *self, ReadSlices &batch) {
  static auto call = real<decltype(&read_slices)>(OBS_READ);
  return read_batch(self, batch, call);
}
extern "C" Status read_spans(void *, ReadSpans &) asm(OBS_READ_SPANS);
Status read_spans(void *self, ReadSpans &batch) {
  static auto call = real<decltype(&read_spans)>(OBS_READ_SPANS);
  return read_batch(self, batch, call);
}
