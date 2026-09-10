#pragma once

// Compact host identity for the model inputs licensed by the resident KV/GDN state.

#include <ninfer/targets/qwen3_6/prepared_prompt.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <span>
#include <vector>

namespace ninfer::targets::qwen3_6::detail {

struct PrefixHash128 {
    std::uint64_t lo = 0;
    std::uint64_t hi = 0;

    [[nodiscard]] friend bool operator==(const PrefixHash128& a, const PrefixHash128& b) noexcept {
        return a.lo == b.lo && a.hi == b.hi;
    }
    [[nodiscard]] friend bool operator!=(const PrefixHash128& a, const PrefixHash128& b) noexcept {
        return !(a == b);
    }
};

class ResidentPrefixIdentity {
public:
    void reserve(std::size_t tokens);
    void clear() noexcept;
    void assign(const PreparedPromptData& prompt);
    void swap(ResidentPrefixIdentity& other) noexcept;
    void append_generated(std::size_t count, std::int32_t rope_delta,
                          std::optional<std::uint32_t> execution_split_after = std::nullopt);
    void truncate(std::size_t tokens);

    [[nodiscard]] std::size_t size() const noexcept { return token_types_.size(); }

    [[nodiscard]] bool matches(const PreparedPromptData& prompt, std::size_t count) const;
    // V2-T8 r9: frontier-reuse variant of matches() that does not compare the
    // rewrite-execution-frontier list. The state restored at a record's execution frontier is
    // the complete state of that record's execution (every internal split already baked in),
    // so the candidate's execution structure over the matched prefix is irrelevant to
    // AppendAtFrontier reuse. Token types, all three position axes, and vision items are still
    // compared. Only plan_match's frontier path uses this; the checkpoint path stays strict
    // (matches()).
    [[nodiscard]] bool frontier_matches(const PreparedPromptData& prompt, std::size_t count) const;
    [[nodiscard]] bool equals(const ResidentPrefixIdentity& other) const;
    [[nodiscard]] bool prefix_equals(const ResidentPrefixIdentity& other, std::size_t count) const;
    // Packed serialization + hash-chain accessors (V2-T8 RAM-KV cache); added to the
    // execution-split design without disturbing the decoder's methods (shared members only).
    [[nodiscard]] std::span<const std::uint8_t> token_types() const noexcept {
        return std::span<const std::uint8_t>(token_types_);
    }
    [[nodiscard]] std::span<const std::int32_t> positions(std::size_t axis) const {
        return positions_.at(axis);
    }
    [[nodiscard]] std::span<const std::uint32_t> rewrite_execution_frontiers() const noexcept {
        return std::span<const std::uint32_t>(rewrite_execution_frontiers_);
    }
    [[nodiscard]] std::span<const VisionItem> vision_items() const noexcept {
        return std::span<const VisionItem>(vision_items_);
    }
    [[nodiscard]] std::size_t packed_bytes() const;
    void pack(void* dst) const;
    void unpack(const void* src, std::size_t bytes);
    void test_tamper_content_digest(std::size_t item, std::uint8_t byte);


private:
    std::vector<std::uint8_t> token_types_;
    std::array<std::vector<std::int32_t>, 3> positions_;
    std::vector<VisionItem> vision_items_;
    std::vector<std::uint32_t> rewrite_execution_frontiers_;
    bool identity_matches_fields(const PreparedPromptData& prompt, std::size_t count,
                                 bool check_execution_splits) const;
};

// One rolling digest per token frontier. This is only a content shortlist: exact token and
// ResidentPrefixIdentity comparison remains authoritative for reuse. Keeping it separate from the
// exact identity avoids retaining hash-only state in immutable capture backings.
class PrefixShortlistDigests {
public:
    void reserve(std::size_t tokens);
    void clear() noexcept;
    void assign(const PreparedPromptData& prompt);
    void swap(PrefixShortlistDigests& other) noexcept;
    void append_generated(std::span<const TokenId> tokens, std::int32_t rope_delta,
                          std::optional<std::uint32_t> execution_split_after = std::nullopt);
    void truncate(std::size_t tokens);

    [[nodiscard]] std::size_t size() const noexcept {
        return digests_.empty() ? 0 : digests_.size() - 1U;
    }

    [[nodiscard]] std::array<std::uint64_t, 2> at(std::size_t frontier) const;

private:
    std::vector<std::array<std::uint64_t, 2>> digests_;
};

[[nodiscard]] bool prefix_matches(const PreparedPromptData& prompt,
                                  std::span<const TokenId> resident_tokens,
                                  const ResidentPrefixIdentity& resident_identity,
                                  std::size_t count);
// V2-T8 r9: prefix_matches with the frontier-tolerant identity check (see
// ResidentPrefixIdentity::frontier_matches).
[[nodiscard]] bool frontier_prefix_matches(const PreparedPromptData& prompt,
                                           std::span<const TokenId> resident_tokens,
                                           const ResidentPrefixIdentity& resident_identity,
                                           std::size_t count);
[[nodiscard]] std::vector<PrefixHash128> prefix_hash_chain(const PreparedPromptData& prompt);

[[nodiscard]] PrefixHash128 prefix_hash_at(std::span<const TokenId> tokens,
                                           const ResidentPrefixIdentity& identity,
                                           std::size_t count);

// Longest-common-prefix ladder (V2-T8): the frontier points {F} ∪ {2^j ≤ F} ∪
// {F - 2^j ≥ 1}, sorted unique. The rolling prefix hash is sampled at exactly these
// points on one pass (prefix_hash_ladder), so a plan_match candidate can locate the
// deepest ladder point its chain agrees with a captured record -- O(ladder), not
// O(tokens) -- without the record holding a per-frontier hash vector. For this
// hybrid state model an in-flight ladder point is diagnostic only: a reusable point
// must additionally carry a state snapshot (the frontier or the checkpoint
// frontier), which the matcher verifies separately.
[[nodiscard]] std::vector<std::uint32_t> hash_ladder_points(std::uint32_t frontier);
[[nodiscard]] std::vector<PrefixHash128>
prefix_hash_ladder(std::span<const TokenId> tokens, const ResidentPrefixIdentity& identity,
                   std::span<const std::uint32_t> points);
// First index in [begin, end) where the candidate prompt stops matching the resident
// ledger (token, type, or any RoPE position); nullopt = identical throughout the
// window. The window is bounded by the caller.
[[nodiscard]] std::optional<std::uint32_t>
first_divergence(const PreparedPromptData& candidate, std::span<const TokenId> resident_tokens,
                 const ResidentPrefixIdentity& resident, std::size_t begin, std::size_t end);

} // namespace ninfer::targets::qwen3_6::detail
