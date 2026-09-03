#ifndef MONOMIAL_TRIE_H
#define MONOMIAL_TRIE_H

#include <bit>
#include <cassert>
#include <cstdint>
#include <iostream>
#include <span>
#include <vector>

#include "store.hpp"

namespace f4ncgb {

template<internal::value_concept V, typename I>
struct monomial_trie {
  std::vector<uint64_t> nodes;
  std::vector<std::vector<uint64_t>> entered_by;
  size_t child_size;

  using match = std::pair<I, size_t>;

  static constexpr uint64_t inverse_mod_2_64(uint64_t d) {
    uint64_t x = 1;
    for(int i = 0; i < 6; i++)
      x *= 2 - d * x;
    return x;
  }

  monomial_trie(size_t n)
    : entered_by(n + 1)
    , child_size(n) {
    const size_t node_size = child_size + 1;
    node_shift_ = static_cast<size_t>(std::countr_zero(node_size));
    node_inv_ = inverse_mod_2_64(node_size >> node_shift_);
    nodes.reserve(node_size * 1024);
    // insert root
    nodes.insert(nodes.end(), node_size, 0);
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

  // Bytes held by the trie
  size_t bytes() const {
    size_t n = nodes.capacity() * sizeof(uint64_t)
               + fail_link.capacity() * sizeof(uint64_t)
               + min_word.capacity() * sizeof(word_ref)
               + node_weight.capacity() * sizeof(uint32_t)
               + match_buffer.capacity() * sizeof(match)
               + walk_stack.capacity() * sizeof(std::pair<uint64_t, size_t>);
    for(const auto& e : entered_by)
      n += e.capacity() * sizeof(uint64_t);
    return n;
  }
  // -----------------------------------------------------------------

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

  /// `weight` is carried through to for_each_shortest_divisor() untouched. It
  /// lets find_reducer() rank a candidate without resolving it back to a
  /// polynomial
  void inline insert(const std::span<const V>& entry,
                     I id,
                     uint32_t weight = 0) {
    const size_t cur_idx = walk_creating(entry.begin(), entry.end());
    assert(nodes[cur_idx] == 0);
    nodes[cur_idx] = id;
    if(node_weight.size() < nodes.size())
      node_weight.resize(nodes.size(), 0);
    node_weight[cur_idx] = weight;
  }
  // -----------------------------------------------------------------

  void inline insert_rev(const std::span<const V>& entry, I id) {
    nodes[walk_creating(entry.rbegin(), entry.rend())] = id;
  }
  // -----------------------------------------------------------------

  void inline clear() {
    nodes.resize(child_size + 1);
    std::fill(nodes.begin(), nodes.end(), 0);
    node_weight.clear();
    for(auto& nodes_of_var : entered_by)
      nodes_of_var.clear();
    links_valid = false;
  }
  // -----------------------------------------------------------------

  // like insert(), but returns the id that was stored at that node before.
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

  template<bool depth_grows>
  void inline collect_words(size_t idx, std::vector<match>& res, size_t depth) {
    walk_stack.clear();
    walk_stack.emplace_back(idx, depth);

    while(!walk_stack.empty()) {
      const auto [cur, d] = walk_stack.back();
      walk_stack.pop_back();

      uint64_t word = nodes[cur];
      if(word != 0)
        res.emplace_back(word, d);

      for(size_t i = cur + child_size; i > cur; i--) {
        uint64_t child_idx = nodes[i];
        if(child_idx != 0)
          walk_stack.emplace_back(child_idx, depth_grows ? d + 1 : d);
      }
    }
  }
  // -----------------------------------------------------------------
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

  template<typename Visit>
  void for_each_shortest_divisor(const std::span<const V>& dividend,
                                 Visit visit) {
    if(!links_valid)
      build_links();
    uint64_t cur = 0;
    for(size_t pos = 0; pos < dividend.size(); pos++) {
      const auto c = dividend[pos];

      while(!nodes[cur + c] and cur != 0)
        cur = fail_link[node_number(cur)];
      cur = nodes[cur + c];

      const word_ref w = min_word[node_number(cur)];
      if(w.id)
        visit(static_cast<I>(w.id), pos + 1 - w.len, w.weight);
    }
  }

  // -----------------------------------------------------------------

  void build_links() {
    const size_t node_size = child_size + 1;
    const size_t count = nodes.size() / node_size;

    fail_link.assign(count, 0);
    min_word.assign(count, word_ref{});
    std::vector<uint64_t> dict_link(count, 0);
    std::vector<uint32_t> node_depth(count, 0);

    // breadth first, so that a node's failure link is known before its children
    // need it. Nodes of depth 1 keep the defaults: they fall back to the root.
    std::vector<uint64_t> queue;
    queue.reserve(count);
    for(size_t c = 1; c <= child_size; c++) {
      const uint64_t child = nodes[c];
      if(child) {
        const size_t child_no = node_number(child);
        node_depth[child_no] = 1;
        // no proper suffix to fall back on, so the node's own word is all there
        // can be
        if(nodes[child])
          min_word[child_no]
            = { static_cast<I>(nodes[child]), 1, weight_of(child) };
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
        const uint32_t depth = node_depth[cur_no] + 1;
        node_depth[child_no] = depth;

        // longest proper suffix of this node that is itself a node
        uint64_t f = fail_link[cur_no];
        while(!nodes[f + c] and f != 0)
          f = fail_link[node_number(f)];
        const uint64_t fl = nodes[f + c];
        fail_link[child_no] = fl;

        // nearest shorter suffix that is a stored word
        const uint64_t dl = nodes[fl] ? fl : dict_link[node_number(fl)];
        dict_link[child_no] = dl;

        if(dl)
          min_word[child_no] = min_word[node_number(dl)];
        else if(nodes[child])
          min_word[child_no]
            = { static_cast<I>(nodes[child]), depth, weight_of(child) };

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
  std::vector<match> match_buffer;

  std::vector<std::pair<uint64_t, size_t>> walk_stack;

  size_t node_shift_ = 0;
  uint64_t node_inv_ = 1;
  size_t inline node_number(uint64_t idx) const {
    return (idx >> node_shift_) * node_inv_;
  }

  // a stored word, its length, and whatever insert() attached to it
  struct word_ref {
    I id = 0;
    uint32_t len = 0;
    uint32_t weight = 0;
  };

  // Aho-Corasick state, by node number. `fail_link` is the longest proper
  // suffix of a node that is a node itself; `min_word` is the shortest stored
  // word that is a suffix of the word ending at the node
  std::vector<uint64_t> fail_link;
  std::vector<word_ref> min_word;
  std::vector<uint32_t> node_weight;
  uint32_t weight_of(uint64_t idx) const {
    return idx < node_weight.size() ? node_weight[idx] : 0;
  }
  bool links_valid = false;
};
}

#endif// MONOMIAL_TRIE_H
