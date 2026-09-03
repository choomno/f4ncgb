#pragma once

#include <bit>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace f4ncgb {
namespace internal {

// An open-addressed, linearly probed map from a 32 bit key to a 32 bit value.
struct probe_map {
  struct entry {
    uint32_t key = 0;
    uint32_t val = 0;
  };

  // Keys are stored biased by one so that a zeroed entry reads as empty.
  // Key 0 is the id of the empty monomial
  static constexpr uint64_t mixer = 0x9e3779b97f4a7c15ull;

  inline size_t index(uint32_t stored) const {
    return static_cast<size_t>((static_cast<uint64_t>(stored) * mixer)
                               >> shift_);
  }

  // Sizes for `expected` entries and empties the table.
  void reset(size_t expected) {
    size_t want = 1024;
    while(want * max_load_num < expected * max_load_den)
      want <<= 1;

    if(want != entries_.size())
      entries_.assign(want, entry{});
    else
      std::fill(entries_.begin(), entries_.end(), entry{});

    mask_ = want - 1;
    shift_ = 64 - static_cast<size_t>(std::countr_zero(want));
    limit_ = want * max_load_num / max_load_den;
    size_ = 0;
  }

  inline void prefetch(uint32_t key) const {
    __builtin_prefetch(&entries_[index(key + 1)]);
  }

  // Returns the slot holding the value, and whether this call created it. The
  // pointer is valid until the next insertion that grows the table.
  inline std::pair<uint32_t*, bool> find_or_insert(uint32_t key, uint32_t val) {
    const uint32_t stored = key + 1;
    size_t i = index(stored);
    for(;;) {
      entry& e = entries_[i];
      if(e.key == stored)
        return { &e.val, false };
      if(e.key == 0) {
        if(size_ >= limit_) {
          grow();
          return find_or_insert(key, val);
        }
        e.key = stored;
        e.val = val;
        size_++;
        return { &e.val, true };
      }
      i = (i + 1) & mask_;
    }
  }

  size_t size() const { return size_; }
  size_t slots() const { return entries_.size(); }
  size_t bytes() const { return entries_.capacity() * sizeof(entry); }

  private:
  // Linear probing degrades sharply near a full table, and eight entries share
  // a cache line, so the chains this leaves are mostly free to walk.
  static constexpr size_t max_load_num = 7;
  static constexpr size_t max_load_den = 10;

  inline void insert_fresh(uint32_t stored, uint32_t val) {
    size_t i = index(stored);
    while(entries_[i].key != 0)
      i = (i + 1) & mask_;
    entries_[i].key = stored;
    entries_[i].val = val;
    size_++;
  }

  void grow() {
    std::vector<entry> old;
    old.swap(entries_);

    const size_t want = old.size() * 2;
    entries_.assign(want, entry{});
    mask_ = want - 1;
    shift_ = 64 - static_cast<size_t>(std::countr_zero(want));
    limit_ = want * max_load_num / max_load_den;
    size_ = 0;

    for(const entry& e : old)
      if(e.key != 0)
        insert_fresh(e.key, e.val);
  }

  std::vector<entry> entries_;
  size_t mask_ = 0;
  size_t shift_ = 64;
  size_t limit_ = 0;
  size_t size_ = 0;
};

}
}
