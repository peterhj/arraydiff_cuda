#ifndef ARRAYDIFF_STR_UTIL_HH
#define ARRAYDIFF_STR_UTIL_HH

#include <sstream>
#include <string>
#include <vector>

namespace arraydiff {

using std::string;
using std::vector;

vector<string> split(const string& s, char delim) {
  vector<string> elems;
  std::stringstream stream;
  stream.str(s);
  string elem_buf;
  while (std::getline(stream, elem_buf, delim)) {
    elems.push_back(elem_buf);
  }
  return elems;
}

} // namespace arraydiff

#endif
