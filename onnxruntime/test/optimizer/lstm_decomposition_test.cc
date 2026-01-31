// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/graph/graph.h"
#include "core/optimizer/lstm_decomposition.h"
#include "test/optimizer/graph_transform_test_fixture.h"
#include "test/unittest_util/graph_transform_test_builder.h"
#include "gtest/gtest.h"

namespace onnxruntime {
namespace test {

namespace {

// Count nodes of a given op type in a graph.
int CountOpsInGraph(const Graph& graph, const std::string& op_type) {
  int count = 0;
  for (const auto& node : graph.Nodes()) {
    if (node.OpType() == op_type) ++count;
  }
  return count;
}

// Build a minimal LSTM graph.
//   x_shape: shape for X input. std::nullopt => no shape info at all.
//   If x_shape has values, -1 dims become symbolic.
void BuildLSTMGraph(ModelTestBuilder& builder,
                    const std::optional<std::vector<int64_t>>& x_shape,
                    int64_t hidden_size,
                    int64_t input_size,
                    const std::string& direction,
                    bool add_bias,
                    bool add_initial_states) {
  int64_t num_directions = (direction == "bidirectional") ? 2 : 1;
  int64_t batch_size = 1;

  // X input — use the optional-shape overload so we can pass nullopt for no shape.
  NodeArg* x = builder.MakeInput<float>(x_shape);

  // W: [num_directions, 4*hidden_size, input_size]
  auto* W = builder.MakeInitializer<float>(
      {num_directions, 4 * hidden_size, input_size}, 0.0f, 0.1f);

  // R: [num_directions, 4*hidden_size, hidden_size]
  auto* R = builder.MakeInitializer<float>(
      {num_directions, 4 * hidden_size, hidden_size}, 0.0f, 0.1f);

  std::vector<NodeArg*> lstm_inputs = {x, W, R};

  // B (optional): [num_directions, 8*hidden_size]
  if (add_bias) {
    auto* B = builder.MakeInitializer<float>(
        {num_directions, 8 * hidden_size}, 0.0f, 0.01f);
    lstm_inputs.push_back(B);
  } else {
    lstm_inputs.push_back(builder.MakeEmptyInput());
  }

  // sequence_lens — empty
  lstm_inputs.push_back(builder.MakeEmptyInput());

  // initial_h, initial_c
  if (add_initial_states) {
    auto* h0 = builder.MakeInitializer<float>(
        {num_directions, batch_size, hidden_size}, 0.0f, 0.1f);
    auto* c0 = builder.MakeInitializer<float>(
        {num_directions, batch_size, hidden_size}, 0.0f, 0.1f);
    lstm_inputs.push_back(h0);
    lstm_inputs.push_back(c0);
  }

  // Outputs: Y (unused), Y_h, Y_c (unused)
  auto* Y = builder.MakeIntermediate();
  auto* Y_h = builder.MakeOutput();
  auto* Y_c = builder.MakeIntermediate();

  auto& lstm_node = builder.AddNode("LSTM", lstm_inputs, {Y, Y_h, Y_c});
  lstm_node.AddAttribute("hidden_size", hidden_size);
  if (direction != "forward") {
    lstm_node.AddAttribute("direction", direction);
  }
}

}  // namespace

// Test 1: LSTM with static seq_len=1, forward direction.
// Should be decomposed into MatMul/Split/Sigmoid/Tanh/Mul/Add.
TEST_F(GraphTransformationTests, LSTMDecomposition_Forward_SeqLen1) {
  constexpr int64_t seq_len = 1, batch_size = 1, input_size = 8, hidden_size = 4;

  auto pre_graph_checker = [](Graph& graph) -> Status {
    TEST_RETURN_IF_NOT(CountOpsInGraph(graph, "LSTM") == 1);
    return Status::OK();
  };

  auto post_graph_checker = [](Graph& graph) -> Status {
    TEST_RETURN_IF_NOT(CountOpsInGraph(graph, "LSTM") == 0);
    TEST_RETURN_IF_NOT(CountOpsInGraph(graph, "MatMul") > 0);
    TEST_RETURN_IF_NOT(CountOpsInGraph(graph, "Sigmoid") > 0);
    TEST_RETURN_IF_NOT(CountOpsInGraph(graph, "Tanh") > 0);
    return Status::OK();
  };

  auto build_test_case = [&](ModelTestBuilder& builder) {
    BuildLSTMGraph(builder,
                   std::vector<int64_t>{seq_len, batch_size, input_size},
                   hidden_size, input_size,
                   "forward", /*add_bias=*/true, /*add_initial_states=*/true);
  };

  auto transformer = std::make_unique<LSTMDecomposition>();
  ASSERT_STATUS_OK(TestGraphTransformer(build_test_case, 14, *logger_,
                                        std::move(transformer),
                                        TransformerLevel::Level2, 1,
                                        pre_graph_checker, post_graph_checker));
}

// Test 2: LSTM with NO shape info on X input (simulates Nemotron case).
// Should still be decomposed after the fix.
TEST_F(GraphTransformationTests, LSTMDecomposition_Forward_DynamicSeqLen) {
  constexpr int64_t input_size = 8, hidden_size = 4;

  auto pre_graph_checker = [](Graph& graph) -> Status {
    TEST_RETURN_IF_NOT(CountOpsInGraph(graph, "LSTM") == 1);
    return Status::OK();
  };

  auto post_graph_checker = [](Graph& graph) -> Status {
    TEST_RETURN_IF_NOT(CountOpsInGraph(graph, "LSTM") == 0);
    TEST_RETURN_IF_NOT(CountOpsInGraph(graph, "MatMul") > 0);
    return Status::OK();
  };

  auto build_test_case = [&](ModelTestBuilder& builder) {
    BuildLSTMGraph(builder,
                   std::nullopt,  // no shape info
                   hidden_size, input_size,
                   "forward", /*add_bias=*/true, /*add_initial_states=*/true);
  };

  auto transformer = std::make_unique<LSTMDecomposition>();
  ASSERT_STATUS_OK(TestGraphTransformer(build_test_case, 14, *logger_,
                                        std::move(transformer),
                                        TransformerLevel::Level2, 1,
                                        pre_graph_checker, post_graph_checker));
}

// Test 3: Bidirectional LSTM should NOT be decomposed.
TEST_F(GraphTransformationTests, LSTMDecomposition_Bidirectional_Skipped) {
  constexpr int64_t seq_len = 1, batch_size = 1, input_size = 8, hidden_size = 4;

  auto pre_graph_checker = [](Graph& graph) -> Status {
    TEST_RETURN_IF_NOT(CountOpsInGraph(graph, "LSTM") == 1);
    return Status::OK();
  };

  auto post_graph_checker = [](Graph& graph) -> Status {
    TEST_RETURN_IF_NOT(CountOpsInGraph(graph, "LSTM") == 1);
    return Status::OK();
  };

  auto build_test_case = [&](ModelTestBuilder& builder) {
    BuildLSTMGraph(builder,
                   std::vector<int64_t>{seq_len, batch_size, input_size},
                   hidden_size, input_size,
                   "bidirectional", /*add_bias=*/true, /*add_initial_states=*/true);
  };

  auto transformer = std::make_unique<LSTMDecomposition>();
  ASSERT_STATUS_OK(TestGraphTransformer(build_test_case, 14, *logger_,
                                        std::move(transformer),
                                        TransformerLevel::Level2, 1,
                                        pre_graph_checker, post_graph_checker));
}

// Test 4: LSTM with static seq_len > 1 should NOT be decomposed.
TEST_F(GraphTransformationTests, LSTMDecomposition_SeqLenGt1_Skipped) {
  constexpr int64_t seq_len = 4, batch_size = 1, input_size = 8, hidden_size = 4;

  auto pre_graph_checker = [](Graph& graph) -> Status {
    TEST_RETURN_IF_NOT(CountOpsInGraph(graph, "LSTM") == 1);
    return Status::OK();
  };

  auto post_graph_checker = [](Graph& graph) -> Status {
    TEST_RETURN_IF_NOT(CountOpsInGraph(graph, "LSTM") == 1);
    return Status::OK();
  };

  auto build_test_case = [&](ModelTestBuilder& builder) {
    BuildLSTMGraph(builder,
                   std::vector<int64_t>{seq_len, batch_size, input_size},
                   hidden_size, input_size,
                   "forward", /*add_bias=*/true, /*add_initial_states=*/true);
  };

  auto transformer = std::make_unique<LSTMDecomposition>();
  ASSERT_STATUS_OK(TestGraphTransformer(build_test_case, 14, *logger_,
                                        std::move(transformer),
                                        TransformerLevel::Level2, 1,
                                        pre_graph_checker, post_graph_checker));
}

}  // namespace test
}  // namespace onnxruntime
