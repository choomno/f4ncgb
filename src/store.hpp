#pragma once

#include <algorithm>
#include <concepts>
#include <cstdint>
#include <cstdlib>
#include <format>
#include <fstream>
#include <iterator>
#include <limits>
#include <memory>
#include <new>
#include <span>
#include <type_traits>
#include <utility>
#include <vector>

// The profiler shows boost::alignment to take a while, but this seems
// to be an artifact from waiting for memory to appear. It makes no
// difference if this is changed, so it will remain at this approch.
#include <boost/align/align_down.hpp>
#include <boost/align/align_up.hpp>

#include <boost/align/aligned_alloc.hpp>

#include <boost/container/small_vector.hpp>

#include <boost/container_hash/hash.hpp>
#include <boost/unordered/unordered_flat_map.hpp>

#include "coeff.hpp"
#include "diagnostics.hpp"
#include "f4ncgb.hpp"

#include "ambiguity.hpp"

#ifdef F4NCGB_ENABLE_STORE_TRACE
#include "store_tracer.hpp"
#endif

#ifdef __linux__
#include <sys/mman.h>
#endif

#if defined(__GNUC__) && !defined(__clang__)
#pragma GCC diagnostic push
// Older GCC compilers warn about unknown warnings, which is really annoying.
#pragma GCC diagnostic ignored "-Wpragmas"
#pragma GCC diagnostic ignored "-Walloc-size"
#endif

namespace f4ncgb {

namespace internal {

/// Pools are allocated in multiples of a huge page, so that the kernel can back
/// them with one where that is supported.
constexpr size_t huge_page_size = 2097152;// 2^21, 2MB

/// Largest pool the index type can address. Capped under AddressSanitizer,
/// which cannot shadow an allocation of that size.
template<typename I, size_t element_size>
consteval size_t
addressable_capacity() {
#if defined(__has_feature)
#if __has_feature(address_sanitizer)
  size_t absolute_max = 0x10000000000;
#else
  size_t absolute_max = std::numeric_limits<size_t>::max();
#endif
#else
  size_t absolute_max = std::numeric_limits<size_t>::max();
#endif
  return std::min(static_cast<size_t>(std::numeric_limits<I>::max())
                    * element_size,
                  absolute_max);
}

/// Reserves at least `capacity` bytes, aligned to a huge page. Falls back to
/// ever smaller alignments when that fails, and reports what was actually
/// taken -- which may be less than asked for, so callers have to bound
/// themselves by `allocated` and not by what they requested.
inline std::byte*
allocate_pool(size_t capacity, size_t& allocated) {
  // Rounded up. Rounding down handed back a zero-byte pool for any store whose
  // index type cannot address a whole huge page, which the small-index
  // instantiations in the tests then wrote straight past.
  const size_t pages = (capacity + huge_page_size - 1) / huge_page_size;

  allocated = huge_page_size * pages;
  auto* pool = reinterpret_cast<std::byte*>(
    boost::alignment::aligned_alloc(huge_page_size, allocated));

  for(size_t divisor = 1; !pool; divisor *= 2) {
    if(divisor > 32)
      throw std::bad_alloc();
    allocated = (huge_page_size / divisor) * pages;
    pool = reinterpret_cast<std::byte*>(std::malloc(allocated));
  }

#ifdef __linux__
  madvise(pool, allocated, MADV_HUGEPAGE);
#endif
  return pool;
}
struct monomial_store_overrun_exception : public std::exception {
  virtual const char* what() const throw() {
    return "monomial_store is overful";
  }
};
struct length_overrun_exception : public std::exception {
  size_t size;
  size_t capacity;
  std::string msg;
  length_overrun_exception(size_t size, size_t capacity)
    : size(size)
    , capacity(capacity)
    , msg(std::format("tried to create a length of {} but the highest possible "
                      "fitting length is {}",
                      size,
                      capacity)) {}
  virtual const char* what() const throw() { return msg.c_str(); }
};
struct scratch_insertion_with_zero_length_exception : public std::exception {
  size_t size;
  std::string msg;
  scratch_insertion_with_zero_length_exception(size_t size)
    : size(size)
    , msg(std::format(
        "Tried to call insert_scratch() on a store where the current metadata "
        "wasn't assigned a length, size of the store was {}",
        size)) {}
  virtual const char* what() const throw() { return msg.c_str(); }
};
struct coefficient_overrun_exception : public std::exception {
  size_t size;
  size_t capacity;
  std::string msg;
  coefficient_overrun_exception(size_t size, size_t capacity)
    : size(size)
    , capacity(capacity)
    , msg(std::format("tried to get a coefficient {} the capacity is only {}",
                      size,
                      capacity)) {}
  virtual const char* what() const throw() { return msg.c_str(); }
};

template<typename length_type = uint8_t>
struct metadata {
  length_type length = 0;
};

static_assert(sizeof(metadata<uint8_t>) == sizeof(uint8_t));
static_assert(alignof(metadata<uint8_t>) == alignof(uint8_t));

template<typename T>
concept metadata_concept = requires(T t) {
  { t.length } -> std::convertible_to<size_t>;
};

consteval bool
is_power_of_two(size_t n) {
  return (n & (n - 1)) == 0;
}

template<typename T>
concept value_concept = requires { is_power_of_two(sizeof(T)); };

/// A store holds some instances of structures like [metadata, V_1, ...
/// V_metadata.length].
template<class B,
         metadata_concept M = metadata<>,
         value_concept V = uint8_t,
         typename I = uint32_t>
class store {
  protected:
  using self = store<B, M, V, I>;
  using V_span = std::span<const V>;
  using V_span_it = V_span::iterator;

  struct pool_deleter {
    void operator()(std::byte pool[]) { free(pool); }
  };

  public:
  __attribute__((always_inline)) inline static bool V_equal(
    const std::span<const V>& a,
    const std::span<const V>& b) noexcept {
    return std::equal(a.begin(), a.end(), b.begin(), b.end());
  }

  struct V_span_equality_struct {
    __attribute__((always_inline)) inline bool operator()(
      const std::span<const V>& a,
      const std::span<const V>& b) const noexcept {
      return V_equal(a, b);
    }
  };

#ifdef F4NCGB_ENABLE_STORE_DUMP
  std::string dump_output_path_;
  void dump_to_binary() {
    const size_t bytes = static_cast<size_t>(size_) * unit;
    msg("Dumping %zu bytes to %s", bytes, dump_output_path_.c_str());
    std::ofstream of(dump_output_path_, std::ios::binary | std::ios::out);
    for(size_t i = 0; i < bytes; ++i) {
      of << static_cast<uint8_t>(pool_[i]);
    }
  }
#endif

  friend B;

  static_assert(
    alignof(M) >= alignof(V),
    "Alignment of M needs to be stricter, as elements are aligned this way.");

  /// Metadata is never destroyed individually -- entries are variable-sized, so
  /// there is no way to walk to the i-th one without walking all of them, and
  /// the pool is freed in one piece. Nothing here has ever needed a metadata
  /// type that owns anything, so this is asserted rather than worked around.
  static_assert(std::is_trivially_destructible_v<M>,
                "Metadata must be trivially destructible: the pool is released "
                "as a whole and entries are never destructed individually.");

  protected:
  /// Ids and size_ count `unit`-sized slots, not bytes -- an entry always
  /// starts at a multiple of alignof(M), so the low bits carry nothing. That is
  /// what makes the whole pool reachable: at one id per byte an `I` of 32 bits
  /// would stop at 4 GB, a quarter of what capacity() reserves.
  static constexpr size_t unit = alignof(M);

  consteval static size_t capacity() {
    return addressable_capacity<I, unit>();
  }

  constexpr inline std::byte* slot(I id) const {
    return pool_.get() + static_cast<size_t>(id) * unit;
  }

  constexpr inline M& get_metadata_from_id(I id) {
    return *reinterpret_cast<M*>(std::assume_aligned<alignof(M)>(slot(id)));
  }
  constexpr inline M& get_metadata_from_id(I id) const {
    return const_cast<self*>(this)->get_metadata_from_id(id);
  }
  constexpr inline V* get_value_from_id(I id) {
    return reinterpret_cast<V*>(
      std::assume_aligned<alignof(V)>(boost::alignment::align_up(
        slot(id) + sizeof(M), alignof(V))));
  }
  constexpr inline const V* get_value_from_id(I id) const {
    return const_cast<self*>(this)->get_value_from_id(id);
  }

  I size_ = 0;
  I real_size_ = 0;
  I inserted_count_ = 0;
  M zero_metadata_;

  /// Bytes an entry may actually occupy: what the allocator handed out, but
  /// never more than an id of type `I` can reach. allocate_pool() rounds up to
  /// whole huge pages and may also fall back to far less than requested, so
  /// neither bound implies the other.
  size_t usable_ = 0;

  std::unique_ptr<std::byte[], pool_deleter> pool_;

  /// Bytes an entry holding `space` values can take at worst: the metadata, the
  /// padding in front of the values, the values, and the padding that aligns
  /// the next entry.
  static constexpr size_t entry_bound(size_t space) {
    return sizeof(M) + alignof(V) + space * sizeof(V) + unit;
  }

  /// Work on the current tip but do not commit anything. The tip can later be
  /// committed using insert_scratch.
  inline std::pair<M&, V*> new_scratch(I space = 32) {
    if(static_cast<size_t>(size_) * unit + entry_bound(space) > usable_)
      throw monomial_store_overrun_exception();

    std::byte* ptr = slot(size_);
    M* metadata = new(std::assume_aligned<alignof(M)>(ptr)) M;

    // Advance the pointer to the first possible location of V.
    V* vv = reinterpret_cast<V*>(std::assume_aligned<alignof(V)>(
      boost::alignment::align_up(ptr + sizeof(M), alignof(V))));

    return std::pair<M&, V*>(*metadata, vv);
  }

  inline I scratch_id() const { return size_ + 1; }

  inline I insert_scratch() {
    I id = size_;

    std::byte* ptr_start = slot(size_);

    M* metadata
      = reinterpret_cast<M*>(std::assume_aligned<alignof(M)>(ptr_start));
    if(metadata->length == 0) {
      throw scratch_insertion_with_zero_length_exception(size_);
    }

    std::byte* ptr = static_cast<std::byte*>(
      boost::alignment::align_up(ptr_start + sizeof(M), alignof(V)));

    std::byte* ptr_end = ptr + metadata->length * sizeof(V);
    std::byte* ptr_next = static_cast<std::byte*>(
      boost::alignment::align_up(ptr_end, alignof(M)));

    reinterpret_cast<B*>(this)->new_entry(id);

    size_ += static_cast<I>(static_cast<size_t>(ptr_next - ptr_start) / unit);
    ++inserted_count_;
    real_size_ = size_;

    return id + 1;
  }

  /// Called by insert_scratch() once an entry is committed, for derived stores
  /// that keep something alongside the pool. Deliberately not the place to put
  /// an index: monomial_store needs to decide insert-or-found before the entry
  /// exists, which a post-commit hook cannot express.
  inline void new_entry(I id) { (void)id; }

  public:
  using length_type = decltype(M::length);
  using index_type = I;
  using value_type = V;

  size_t pool_capacity_ = 0;

  store() {
    pool_.reset(allocate_pool(capacity(), pool_capacity_));
    usable_ = std::min(pool_capacity_, capacity());
  }
  ~store() {
#ifdef F4NCGB_ENABLE_STORE_DUMP
    if(dump_output_path_ != "") {
      dump_to_binary();
    }
#endif
    // M is trivially destructible (asserted above), so freeing the pool is the
    // whole teardown. Derived stores that own something per entry -- as
    // polynomial_store does for its coefficients -- destroy it themselves.
  }

  inline M& get_metadata(I id) noexcept {
    if(id == 0) {
      return zero_metadata_;
    }
    return get_metadata_from_id(id - 1);
  }
  inline const M& get_metadata(I id) const noexcept {
    return const_cast<self*>(this)->get_metadata(id);
  }

  inline length_type get_length(I id) const noexcept {
    return get_metadata(id).length;
  }

  inline void reset() { size_ = real_size_; }

  inline const std::span<const V> operator[](I id) const noexcept {
    if(id == 0)
      return std::span<V>();

    size_t len = get_length(id);

    const V* start = get_value_from_id(id - 1);

#ifdef F4NCGB_ENABLE_STORE_TRACE
    reinterpret_cast<B*>(const_cast<self*>(this))->tracer_.access(id);
#endif

    return std::span<const V>(start, len);
  }

  I next_pos(I pos) const {
    if(pos == 0)
      return 1;
    else
      --pos;
    // Increment the position to the next valid index.
    std::byte* ptr_start = slot(pos);

    std::byte* ptr = static_cast<std::byte*>(
      boost::alignment::align_up(ptr_start + sizeof(M), alignof(V)));

    std::byte* ptr_end = ptr + get_length(pos + 1) * sizeof(V);

    std::byte* ptr_next = static_cast<std::byte*>(
      boost::alignment::align_up(ptr_end, alignof(M)));

    return 1 + pos
           + static_cast<I>(static_cast<size_t>(ptr_next - ptr_start) / unit);
  }

  std::ostream& print(std::ostream& o) {
    o << "size: " << size_ << std::endl;
    for(auto pos : (*this)) {
      o << pos << ": =";
      for(auto v : (*this)[pos]) {
        o << " " << static_cast<int>(v);
      }
      o << ";" << std::endl;
    }
    return o;
  }

  struct pos_iterator {
    using iterator_category = std::forward_iterator_tag;
    using difference_type = std::ptrdiff_t;
    using value_type = I;
    using pointer = value_type*;
    using reference = const value_type&;

    pos_iterator(const self& s, I pos = 0)
      : s(s)
      , pos(pos) {}

    reference operator*() const { return pos; }
    pointer operator->() { return &pos; }
    pos_iterator& operator++() {
      pos = s.next_pos(pos);
      return *this;
    }
    pos_iterator operator++(int) {
      pos_iterator tmp = *this;
      ++(*this);
      return tmp;
    }
    friend bool operator==(const pos_iterator& a, const pos_iterator& b) {
      return &a.s == &b.s && a.pos == b.pos;
    };
    friend bool operator!=(const pos_iterator& a, const pos_iterator& b) {
      return &a.s != &b.s || a.pos != b.pos;
    };

    pos_iterator operator+(I count) const {
      pos_iterator it(*this);
      for(I i = 0; i < count; ++i) {
        ++it;
      }
      return it;
    }

    private:
    const self& s;
    I pos;
  };

  pos_iterator begin() const { return pos_iterator(*this, 0); }
  pos_iterator end() const { return pos_iterator(*this, size_ + 1); }
  /// Number of entries plus the 0 element. Walks the store, because entries are
  /// variable-sized and inserted_count_ also counts the scratch polynomials
  /// that polynomial_store::multiply() rolls back.
  size_t size() const { return (size_t)std::distance(begin(), end()); }

#ifdef F4NCGB_ENABLE_STORE_DUMP
  void set_binary_dump_path(const std::string& p) { dump_output_path_ = p; }
#endif
};
}
}

#if defined(__GNUC__) && !defined(__clang__)
#pragma GCC diagnostic pop
#endif
