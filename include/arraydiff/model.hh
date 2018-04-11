#ifndef ARRAYDIFF_MODEL_HH
#define ARRAYDIFF_MODEL_HH

#include "arraydiff/op.hh"

#include <memory>

namespace arraydiff {

using std::shared_ptr;

class CategoricalLoss {
public:
  shared_ptr<Op> batch_logit;
  shared_ptr<Op> batch_loss;
  shared_ptr<Op> sum_loss;
  VarIdSet batch_input_vars;
  VarIdSet batch_label_vars;
  VarIdSet batch_logit_vars;
  VarIdSet batch_loss_vars;
  VarIdSet params;
  VarIdSet grads;
};

} // namespace arraydiff

#endif
