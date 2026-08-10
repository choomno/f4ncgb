#ifndef MONOMIAL_TRIE_H
#define MONOMIAL_TRIE_H

#include <cstdint>
#include <span>
#include <vector>

#include "store.hpp"

namespace f4ncgb {

template<internal::value_concept V, typename I>
struct monomial_trie {
  std::vector<uint64_t> nodes;
  /// For each variable, the nodes reached by an edge labelled with it. Lets
  /// inclusions() look at the handful of nodes a word could possibly start at
  /// instead of walking the whole trie.
  std::vector<std::vector<uint64_t>> entered_by;
  size_t child_size;

  using match = std::pair<I, size_t>;

  monomial_trie(size_t n)
    : entered_by(n + 1)
    , child_size(n) {
    nodes.reserve((child_size + 1) * 1024);
    // insert root
    nodes.insert(nodes.end(), child_size + 1, 0);
  }
  // -----------------------------------------------------------------

  void print() {
    size_t i = 0;
    while(i < nodes.size()) {
      std::cout << "nodes[" << i << "] = " << nodes[i];
      i++;
      std::cout << " : [";
      for(size_t j = 1; j < child_size + 1; j++)
        std::cout << nodes[i++] << ", ";
      std::cout << "]\n";
    }
  }
  // -----------------------------------------------------------------

  size_t size() { return nodes.size(); }
  // -----------------------------------------------------------------

  /// Descends along [first, last), creating the nodes that are missing, and
  /// returns the node the path ends at. The three inserts below differ only in
  /// the direction they read the entry and in what they do with that node.
  template<typename It>
  size_t inline walk_creating(It first, It last) {
    size_t cur_idx = 0;

    for(It it = first; it != last; ++it) {
      const auto c = *it;
      if(nodes[cur_idx + c] == 0) {
        nodes[cur_idx + c] = nodes.size();
        nodes.insert(nodes.end(), child_size + 1, 0);
        entered_by[c].push_back(nodes[cur_idx + c]);
      }
      cur_idx = nodes[cur_idx + c];
    }

    links_valid = false;
    return cur_idx;
  }
  // -----------------------------------------------------------------

  void inline insert(const std::span<const V>& entry, I id) {
    const size_t cur_idx = walk_creating(entry.begin(), entry.end());
    assert(nodes[cur_idx] == 0);
    nodes[cur_idx] = id;
  }
  // -----------------------------------------------------------------

  void inline insert_rev(const std::span<const V>& entry, I id) {
    nodes[walk_creating(entry.rbegin(), entry.rend())] = id;
  }
  // -----------------------------------------------------------------

  void inline clear() {
    nodes.resize(child_size + 1);
    std::fill(nodes.begin(), nodes.end(), 0);
    for(auto& nodes_of_var : entered_by)
      nodes_of_var.clear();
    links_valid = false;
  }
  // -----------------------------------------------------------------

  /// Like insert(), but returns the id that was stored at that node before.
  I inline insert_chain(const std::span<const V>& entry, I id) {
    const size_t cur_idx = walk_creating(entry.begin(), entry.end());
    I old_id = static_cast<I>(nodes[cur_idx]);
    nodes[cur_idx] = id;
    return old_id;
  }
  // -----------------------------------------------------------------
  size_t inline starts_sequence(size_t idx, const std::span<const V>& word) {
    uint64_t cur_idx = idx;
    for(auto c : word) {
      cur_idx = nodes[cur_idx + c];
      if(cur_idx == 0)
        break;
    }
    return cur_idx;
  }
  // -----------------------------------------------------------------

  /// Every word stored in the subtree at `idx`, reported with `depth`.
  ///
  /// Overlaps want one fixed depth for the whole subtree -- how far the two
  /// monomials reach into each other is decided by where the subtree starts.
  /// Inclusions want it to grow with the descent, because there the depth is
  /// the length of the part trailing the occurrence.
  template<bool depth_grows>
  void inline collect_words(size_t idx,
                            std::vector<match>& res,
                            size_t depth) const {
    uint64_t word = nodes[idx];
    if(word != 0)
      res.emplace_back(word, depth);

    // Recursively visit all children
    for(size_t i = idx + 1; i < idx + 1 + child_size; i++) {
      uint64_t child_idx = nodes[i];
      if(child_idx != 0)
        collect_words<depth_grows>(
          child_idx, res, depth_grows ? depth + 1 : depth);
    }
  }
  // -----------------------------------------------------------------
  /// All stored words that have `word` as a factor, each with the length of the
  /// part that follows the occurrence.
  ///
  /// An occurrence can only begin at a node entered by the word's first
  /// variable, which is a small fraction of the trie. Walking every node
  /// instead is quadratic in the size of the basis, and dominates examples that
  /// build a large one.
  ///
  /// Matches come out in a different order than a walk over all nodes would
  /// produce. That is fine: gebauer_moeller() sorts the ambiguities it collects
  /// into a canonical order before acting on them.
  void inline inclusions(const std::span<const V>& word,
                         std::vector<match>& inclusions) {
    if(word.empty())
      return;

    const std::span<const V> rest = word.subspan(1);
    for(const uint64_t start : entered_by[word[0]]) {
      size_t end_idx = starts_sequence(start, rest);
      if(end_idx == 0)
        continue;
      collect_words<true>(end_idx, inclusions, 0);
    }
  }

  // -----------------------------------------------------------------

  void inline overlaps_rev(const std::span<const V>& word,
                           std::vector<match>& overlaps) {

    long D = static_cast<long>(word.size());
    for(long d = 1; d < D; d++) {
      uint64_t cur_idx = 0;
      // iterate over suffix of length d
      // but in reversed order
      for(auto it = word.rbegin() + static_cast<long>(d); it != word.rend();
          it++) {
        cur_idx = nodes[cur_idx + *it];
        if(cur_idx == 0)
          break;
      }
      if(cur_idx) {
        // compute overlaps, but skip current node
        // as this is an inclusion
        for(size_t i = 1; i < child_size + 1; i++) {
          uint64_t child_idx = nodes[cur_idx + i];
          if(child_idx != 0) {
            assert(D >= d);
            collect_words<false>(child_idx,
                                 overlaps,
                                 static_cast<size_t>(D)
                                   - static_cast<size_t>(d));
          }
        }
      }
    }
  }
  // -----------------------------------------------------------------

  size_t inline prefixes(const std::span<const V>& prefix,
                         std::vector<match>& res,
                         size_t depth) {
    size_t cur_idx = 0;

    // check if prefix actually appears
    for(auto c : prefix) {
      uint64_t word = nodes[cur_idx];
      if(word)
        res.emplace_back(word, depth);
      assert(cur_idx + c < nodes.size());
      cur_idx = nodes[cur_idx + c];
      // no more matching prefixes
      if(cur_idx == 0)
        return cur_idx;
    }
    if(nodes[cur_idx])
      res.emplace_back(nodes[cur_idx], depth);
    return cur_idx;
  }
  // -----------------------------------------------------------------

  inline const std::vector<match>& get_prefixes(
    const std::span<const V>& word) {
    match_buffer.clear();
    prefixes(word, match_buffer, 0);
    return match_buffer;
  }
  // -----------------------------------------------------------------

  /// The occurrence a reducer should be built from: among the stored words that
  /// divide `dividend`, the one `prefers` ranks first, and among its
  /// occurrences the last one. Returns word 0 when nothing divides.
  ///
  /// This is the question a pattern matcher answers, so it is answered the way
  /// one does: read `dividend` once, following a failure link whenever the next
  /// variable cannot be read. Restarting at the root for each of the suffixes
  /// instead costs a walk per suffix, and those walks are the bulk of symbolic
  /// preprocessing on examples with long monomials.
  ///
  /// The winner is picked as occurrences turn up, so the full list -- which the
  /// one caller would only scan once and throw away -- never gets built.
  /// `prefers(a, b)` is true when word a ranks before word b.
  /// Every occurrence of a stored word in `dividend`, as (word, position).
  ///
  /// This is the question a pattern matcher answers, so it is answered the way
  /// one does: read `dividend` once, following a failure link whenever the next
  /// variable cannot be read. Restarting at the root for each suffix instead
  /// costs a walk per suffix.
  ///
  /// The caller wants one occurrence out of this and picks it afterwards.
  /// Choosing during the walk instead, so the vector is never built, measured
  /// exactly the same on 4nilp5s-10 -- the intermediate vector is not what this
  /// costs. Collecting keeps the trie free of the caller's ordering rule, so it
  /// is the shape that stayed.
  ///
  /// Occurrences come out ordered by where they end rather than where they
  /// start; find_reducer(), the only caller, states which one it wants outright
  /// rather than relying on the order.
  const std::vector<match>& divisors(const std::span<const V>& dividend) {
    if(!links_valid)
      build_links();
    match_buffer.clear();
    uint64_t cur = 0;
    for(size_t pos = 0; pos < dividend.size(); pos++) {
      const auto c = dividend[pos];

      // fall back to shorter suffixes until this variable can be read
      while(!nodes[cur + c] and cur != 0)
        cur = fail_link[node_number(cur)];
      cur = nodes[cur + c];
      const size_t cur_no = node_number(cur);

      // the word ending here, then the shorter ones ending here
      if(nodes[cur])
        match_buffer.emplace_back(static_cast<I>(nodes[cur]),
                                  pos + 1 - node_depth[cur_no]);
      for(uint64_t d = dict_link[cur_no]; d;) {
        const size_t d_no = node_number(d);
        match_buffer.emplace_back(static_cast<I>(nodes[d]),
                                  pos + 1 - node_depth[d_no]);
        d = dict_link[d_no];
      }
    }
    return match_buffer;
  }

  // -----------------------------------------------------------------

  /// Rebuilt on demand after an insertion, which costs one breadth-first pass
  /// over the trie. The trie only grows between F4 iterations, while divisors()
  /// runs once per column of the next matrix, so this is paid once per
  /// iteration.
  void build_links() {
    const size_t node_size = child_size + 1;
    const size_t count = nodes.size() / node_size;

    fail_link.assign(count, 0);
    dict_link.assign(count, 0);
    node_depth.assign(count, 0);

    // breadth first, so that a node's failure link is known before its children
    // need it. Nodes of depth 1 keep the defaults: they fall back to the root.
    std::vector<uint64_t> queue;
    queue.reserve(count);
    for(size_t c = 1; c <= child_size; c++) {
      const uint64_t child = nodes[c];
      if(child) {
        node_depth[node_number(child)] = 1;
        queue.push_back(child);
      }
    }

    for(size_t q = 0; q < queue.size(); q++) {
      const uint64_t cur = queue[q];
      const size_t cur_no = node_number(cur);

      for(size_t c = 1; c <= child_size; c++) {
        const uint64_t child = nodes[cur + c];
        if(!child)
          continue;
        const size_t child_no = node_number(child);
        node_depth[child_no] = node_depth[cur_no] + 1;

        // longest proper suffix of this node that is itself a node
        uint64_t f = fail_link[cur_no];
        while(!nodes[f + c] and f != 0)
          f = fail_link[node_number(f)];
        const uint64_t fl = nodes[f + c];
        fail_link[child_no] = fl;

        // nearest shorter suffix that is a stored word
        dict_link[child_no] = nodes[fl] ? fl : dict_link[node_number(fl)];

        queue.push_back(child);
      }
    }

    links_valid = true;
  }
  // -----------------------------------------------------------------
  void inline overlaps_and_inclusions(const std::span<const V>& word,
                                      std::vector<match>& overlaps,
                                      std::vector<match>& inclusions) {

    size_t D = word.size();
    for(size_t d = 1; d <= D; d++) {

      // compute inclusions
      uint64_t cur_idx = prefixes(word.last(d), inclusions, D - d);
      if(cur_idx == 0)
        continue;

      // compute overlaps, but skip current node
      // as this is an inclusion
      for(size_t i = cur_idx + 1; i < cur_idx + 1 + child_size; i++) {
        uint64_t child_idx = nodes[i];
        if(child_idx)
          collect_words<false>(child_idx, overlaps, d);
      }
    }
  }

  private:
  /// Returned by divisors() and get_prefixes(). They share it because neither
  /// result outlives the other's next call, and reusing it keeps both from
  /// reallocating on every query.
  std::vector<match> match_buffer;

  size_t inline node_number(uint64_t idx) const {
    return idx / (child_size + 1);
  }

  /// Aho-Corasick links, by node number. `fail_link` is the longest proper
  /// suffix of a node that is a node itself, `dict_link` the nearest one that
  /// is a stored word, `node_depth` the length of the word ending at the node.
  std::vector<uint64_t> fail_link;
  std::vector<uint64_t> dict_link;
  std::vector<uint32_t> node_depth;
  bool links_valid = false;
};
}

#endif// MONOMIAL_TRIE_H
