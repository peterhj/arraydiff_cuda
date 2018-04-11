#include "nvarraydiff/op.hh"

#include <iostream>
#include <memory>

using namespace nvarraydiff;
using std::function;
using std::shared_ptr;

int main(int argc, const char** argv) {
  shared_ptr<SrcOp<float> > scalar_src(new SrcOp<float>([](auto txn, auto node_id) { return 3.14f; }));
  std::clog << "DEBUG: op label: " << scalar_src->_debug_op_label() << std::endl;
  return 0;
}
