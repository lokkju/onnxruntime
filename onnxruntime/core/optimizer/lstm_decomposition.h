// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/optimizer/graph_transformer.h"

namespace onnxruntime {

/**
@class LSTMDecomposition

Transformer that decomposes LSTM nodes into primitive ops
(MatMul, Split, Sigmoid, Tanh, Mul, Add) that are already
supported by execution providers like WebGPU.

Currently handles forward-only LSTM with seq_len=1 (streaming mode).
Nodes with seq_len>1 or bidirectional are left unchanged.
*/
class LSTMDecomposition : public GraphTransformer {
 public:
  LSTMDecomposition(const InlinedHashSet<std::string_view>& compatible_execution_providers = {}) noexcept;

 private:
  Status ApplyImpl(Graph& graph, bool& modified, int graph_level, const logging::Logger& logger) const override;
};

}  // namespace onnxruntime
