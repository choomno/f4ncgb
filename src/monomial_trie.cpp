#include "monomial_trie.hpp"

namespace f4ncgb {
// We instantiate this variant directly, so that the other compilation unit does
// not have to reproduce the same class several times.
template struct f4ncgb::monomial_trie<uint8_t, uint32_t>;
}
