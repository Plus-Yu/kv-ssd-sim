// A metadata-only driver for the pinned wheel's MasterService. No transfer
// engine is started and the addresses mounted below are never dereferenced.
#include "master_service.h"
#include "storage_backend.h"

#include <atomic>
#include <chrono>
#include <iostream>
#include <set>
#include <sstream>

namespace {
std::atomic<int64_t> elapsed_ns{0};
constexpr int64_t epoch_ns = 1700000000000000000LL;
}  // namespace

// The child executable exports this symbol to the Mooncake DSO. The parent
// Python process, its scheduler, and the host keep their own clocks.
std::chrono::system_clock::time_point std::chrono::system_clock::now() noexcept {
    return time_point(std::chrono::nanoseconds(epoch_ns + elapsed_ns.load()));
}

namespace mooncake::test {
// The pinned upstream header explicitly friends this test/driver interface.
class MasterServiceTestPeer {
   public:
    static void TakeClockOwnership(MasterService& m) {
        m.eviction_running_ = false;
        m.client_monitor_running_ = false;
        m.eviction_thread_.join();
        m.client_monitor_thread_.join();
    }
    static StorageUsage Usage(const MasterService& m) {
        return m.segment_manager_.GetMemoryUsage();
    }
    static constexpr int64_t period_ns = MasterService::kEvictionThreadSleepMs * 1000000;
    static bool UnderPressure(const MasterService& m) {
        return Usage(m).used_ratio() > m.eviction_high_watermark_ratio_ ||
               (m.need_mem_eviction_ && m.eviction_ratio_ > 0);
    }
    static void Round(MasterService& m) {
        const double used_ratio = Usage(m).used_ratio();
        const double eviction_ratio_ = m.eviction_ratio_;
        const double eviction_high_watermark_ratio_ = m.eviction_high_watermark_ratio_;
        auto BatchEvict = [&](double target, double lower) { m.BatchEvict(target, lower); };
        // Exact pressure-to-round calculation from the pinned EvictionThreadFunc.
#include "native_round.inc"
    }
};
}  // namespace mooncake::test

namespace mooncake {
// File writes are virtual. Grouping and selection execute in the wheel.
class ReplayBucketIndex {
    BucketStorageBackend backend_;
    int64_t generation_ = 0;
    static BucketBackendConfig config() {
        BucketBackendConfig cfg;
        cfg.eviction_policy = BucketEvictionPolicy::FIFO;
        cfg.flush_partial_buckets = true;
        return cfg;
    }

   public:
    ReplayBucketIndex() : backend_(FileStorageConfig{}, config()) {
        backend_.aligned_io_buffer_.reset();  // No direct I/O in the fixture.
    }
    auto group(const std::unordered_map<std::string, int64_t>& sizes) {
        std::vector<std::vector<std::string>> groups;
        auto status = backend_.AllocateOffloadingBuckets(sizes, groups);
        if (!status) throw std::runtime_error("native bucket grouping failed");
        return groups;
    }
    int64_t publish(const std::vector<std::string>& keys) {
        auto bucket = std::make_shared<BucketMetadata>();
        bucket->keys = keys;
        backend_.buckets_.emplace(++generation_, std::move(bucket));
        return generation_;
    }
    std::vector<std::string> oldest() {
        const auto it = backend_.SelectEvictionCandidate();
        if (it == backend_.buckets_.end()) return {};
        return it->second->keys;
    }
    void erase() {
        auto it = backend_.SelectEvictionCandidate();
        if (it == backend_.buckets_.end()) throw std::logic_error("missing disk candidate");
        backend_.buckets_.erase(it);
    }
};
}  // namespace mooncake

namespace {
using namespace mooncake;
using Peer = test::MasterServiceTestPeer;

template <class T>
T take(tl::expected<T, ErrorCode> value) {
    if (!value) throw std::runtime_error("Mooncake error " + std::to_string(int(value.error())));
    return std::move(*value);
}
void check(tl::expected<void, ErrorCode> value) {
    if (!value) throw std::runtime_error("Mooncake error " + std::to_string(int(value.error())));
}

// Preserve the original bytes for the master's hashing and tie breaking.
std::string decode(const std::string& hex) {
    if (hex.empty() || hex.size() % 2) throw std::invalid_argument("invalid key");
    std::string out;
    for (size_t i = 0; i < hex.size(); i += 2)
        out.push_back(static_cast<char>(std::stoul(hex.substr(i, 2), nullptr, 16)));
    return out;
}
std::string encode(const std::string& text) {
    constexpr char hex[] = "0123456789abcdef";
    std::string out;
    for (unsigned char c : text) {
        out += hex[c >> 4];
        out += hex[c & 15];
    }
    return out;
}

class Replay {
    MasterService master_;
    const TenantId tenant_ = TenantId::Default();
    std::vector<UUID> clients_;
    std::vector<std::string> segments_;
    std::vector<std::unique_ptr<ReplayBucketIndex>> disks_;
    std::set<std::string> resident_;  // Observation only, never passed to BatchEvict.
    int64_t next_tick_ = Peer::period_ns;
    static int mask(const GetReplicaListResponse& response) {
        int state = 0;
        for (const auto& r : response.replicas) {
            if (r.is_memory_replica()) state |= 1;
            if (r.is_local_disk_replica()) state |= 2;
        }
        return state;
    }
    int inspect(const std::string& key) {
        auto r = master_.GetReplicaListForAdmin(key, tenant_);
        if (!r && r.error() == ErrorCode::OBJECT_NOT_FOUND) return 0;
        return mask(take(std::move(r)));
    }
    void changes(int64_t tick) {
        for (auto it = resident_.begin(); it != resident_.end();) {
            const int state = inspect(*it);
            if (state & 1) {
                ++it;
                continue;
            }
            std::cout << "evicted " << tick << ' ' << encode(*it) << ' ' << state << '\n';
            it = resident_.erase(it);
        }
    }

   public:
    Replay(int clients, uint64_t capacity, int64_t quota, const MasterServiceConfig& cfg)
        : master_(cfg) {
        Peer::TakeClockOwnership(master_);
        // Each segment has its own address space and offset allocator. Segment
        // endpoints distinguish identical addresses on different virtual clients.
        for (int i = 0; i < clients; ++i) {
            UUID id{1, uint64_t(i + 1)};
            clients_.push_back(id);
            disks_.push_back(std::make_unique<ReplayBucketIndex>());
            segments_.push_back("replay-" + std::to_string(i));
            Segment segment{};
            segment.id = {2, uint64_t(i + 1)};
            segment.name = segment.te_endpoint = segments_.back();
            segment.base = 1ULL << 32;
            segment.size = capacity;
            check(master_.MountSegment(segment, id));
            check(master_.MountLocalDiskSegment(id, true));
            check(master_.ReportSsdCapacity(id, quota));
        }
        if (Peer::Usage(master_).capacity_bytes != clients * capacity)
            throw std::runtime_error("mounted capacity mismatch");
    }
    void advance(int64_t until) {
        if (until < elapsed_ns || until > INT64_MAX - epoch_ns)
            throw std::invalid_argument("virtual time out of range");
        while (next_tick_ <= until && Peer::UnderPressure(master_)) {
            elapsed_ns = next_tick_;
            const auto before = Peer::Usage(master_).used_bytes;
            Peer::Round(master_);
            const auto after = Peer::Usage(master_).used_bytes;
            std::cout << "round " << next_tick_ << ' ' << before << ' ' << after << '\n';
            if (before != after) changes(next_tick_);
            next_tick_ += Peer::period_ns;
        }
        // With no pressure, an arbitrary number of empty ticks has no effect.
        next_tick_ = (until / Peer::period_ns + 1) * Peer::period_ns;
        elapsed_ns = until;
    }
    std::string dispatch(const std::string& op, std::istringstream& in) {
        std::ostringstream result;
        auto key = [&]() {
            std::string h;
            in >> h;
            return decode(h);
        };
        auto integer = [&]() {
            int64_t n;
            if (!(in >> n)) throw std::invalid_argument("missing integer");
            return n;
        };
        if (op == "put_start") {
            const auto writer = integer();
            const auto size = integer();
            const auto k = key();
            ReplicateConfig cfg;
            cfg.preferred_segments = {segments_.at(writer)};
            auto allocated = master_.PutStart(clients_.at(writer), k, tenant_, size, cfg);
            if (!allocated) return std::to_string(int(allocated.error()));
            if (allocated->size() != 1 || !allocated->front().is_memory_replica())
                throw std::runtime_error("unexpected allocation");
            const auto& endpoint =
                allocated->front().get_memory_descriptor().buffer_descriptor.transport_endpoint_;
            auto found = std::find(segments_.begin(), segments_.end(), endpoint);
            if (found == segments_.end()) throw std::runtime_error("unknown allocation endpoint");
            const auto owner = found - segments_.begin();
            result << 0 << ' ' << owner;
        } else if (op == "put_end") {
            const auto writer = integer();
            const auto k = key();
            check(master_.PutEnd(clients_.at(writer), k, tenant_, ReplicaType::MEMORY));
            resident_.insert(k);
        } else if (op == "poll") {
            const auto owner = integer();
            auto tasks = take(master_.OffloadObjectHeartbeat(clients_.at(owner), true));
            std::unordered_map<std::string, int64_t> sizes;
            std::unordered_map<std::string, std::string> users;
            sizes.reserve(tasks.size());
            for (const auto& task : tasks) {
                auto scoped = TenantId(task.tenant_id).MakeScopedKey(task.key);
                sizes.emplace(scoped, task.size);
                users.emplace(scoped, task.key);
            }
            for (const auto& bucket : disks_[owner]->group(sizes)) {
                if (bucket.empty()) continue;
                result << ' ';
                bool first = true;
                for (const auto& scoped : bucket) {
                    if (!first) result << ',';
                    result << encode(users.at(scoped));
                    first = false;
                    users.erase(scoped);
                }
            }
            // Native grouping skips some tasks, for example oversize values.
            // Explicitly NACK them just as FileStorage::OffloadObjects does.
            for (const auto& entry : users) result << " -" << encode(entry.second);
        } else if (op == "source") {
            std::vector<std::string> keys;
            std::string h;
            while (in >> h) keys.push_back(decode(h));
            for (auto& r : master_.BatchGetReplicaList(keys, tenant_))
                result << mask(take(std::move(r))) << ' ';
        } else if (op == "nack") {
            const auto owner = integer(), size = integer();
            const auto k = key();
            OffloadTaskItem task{.tenant_id = tenant_.value(), .key = k, .size = size};
            StorageObjectMetadata metadata{};
            metadata.data_size = -1;
            metadata.transport_endpoint = segments_.at(owner);
            check(master_.NotifyOffloadSuccess(clients_[owner], {task}, {metadata}));
            result << inspect(k);
        } else if (op == "publish") {
            const auto owner = integer();
            std::vector<OffloadTaskItem> tasks;
            std::vector<StorageObjectMetadata> metadata;
            std::vector<std::string> keys;
            int64_t size, offset = 0;
            while (in >> size) {
                auto k = key();
                keys.push_back(k);
                tasks.push_back({.tenant_id = tenant_.value(), .key = k, .size = size});
                metadata.push_back({0, offset, int64_t(k.size()), size, segments_.at(owner)});
                offset += size;
            }
            auto bucket_id = disks_[owner]->publish(keys);
            for (auto& m : metadata) m.bucket_id = bucket_id;
            check(master_.NotifyOffloadSuccess(clients_.at(owner), tasks, metadata));
            result << bucket_id;
        } else if (op == "exists") {
            std::vector<std::string> keys;
            std::string h;
            while (in >> h) keys.push_back(decode(h));
            for (auto value : master_.BatchExistKey(keys, tenant_)) result << int(take(value));
        } else if (op == "get") {
            auto r = take(master_.GetReplicaList(key(), tenant_));
            result << mask(r) << ' ' << r.lease_ttl_ms;
        } else if (op == "inspect") {
            result << inspect(key());
        } else if (op == "disk_candidate") {
            for (const auto& k : disks_.at(integer())->oldest()) result << encode(k) << ' ';
        } else if (op == "remove_disk") {
            const auto owner = integer();
            auto keys = disks_.at(owner)->oldest();
            for (const auto& k : keys) {
                check(
                    master_.EvictDiskReplica(clients_[owner], k, tenant_, ReplicaType::LOCAL_DISK));
                result << encode(k) << ':' << inspect(k) << ' ';
            }
            disks_[owner]->erase();
        } else if (op == "clear_memory") {
            const auto writer = integer(), owner = integer();
            const auto k = key();
            if (take(master_.BatchReplicaClear({k}, clients_.at(writer), segments_.at(owner)))
                    .size() != 1)
                throw std::runtime_error("controlled memory clear refused");
            resident_.erase(k);
            result << inspect(k);
        } else if (op == "usage") {
            auto usage = Peer::Usage(master_);
            result << usage.used_bytes << ' ' << usage.capacity_bytes << ' ' << Peer::period_ns;
        } else if (op != "advance") {
            throw std::invalid_argument("unknown replay operation");
        }
        return result.str();
    }
};
}  // namespace

int main(int argc, char** argv) try {
    if (argc != 7)
        throw std::invalid_argument("expected clients segment quota lease_ms high_watermark ratio");
    mooncake::MasterServiceConfig cfg;
    cfg.default_kv_lease_ttl = std::stoull(argv[4]);
    cfg.eviction_high_watermark_ratio = std::stod(argv[5]);
    cfg.eviction_ratio = std::stod(argv[6]);
    cfg.enable_offload = true;
    cfg.memory_allocator = mooncake::BufferAllocatorType::OFFSET;
    Replay replay(std::stoi(argv[1]), std::stoull(argv[2]), std::stoll(argv[3]), cfg);
    std::cout << "ready 1\n" << std::flush;
    std::string line;
    while (std::getline(std::cin, line)) {
        std::istringstream in(line);
        std::string op;
        int64_t at;
        if (!(in >> op >> at)) throw std::invalid_argument("missing operation or time");
        if (op == "close") break;
        replay.advance(at);
        auto result = replay.dispatch(op, in);
        std::cout << "ok " << result << '\n' << std::flush;
    }
    return 0;
} catch (const std::exception& e) {
    std::cerr << "object replay master: " << e.what() << '\n';
    return 1;
}
