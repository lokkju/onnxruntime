// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <functional>

#include "core/optimizer/lstm_decomposition.h"
#include "core/optimizer/initializer.h"
#include "core/optimizer/utils.h"
#include "core/graph/graph_utils.h"
#include "core/framework/tensorprotoutils.h"

using namespace onnxruntime::common;

namespace onnxruntime {

LSTMDecomposition::LSTMDecomposition(
    const InlinedHashSet<std::string_view>& compatible_execution_providers) noexcept
    : GraphTransformer("LSTMDecomposition", compatible_execution_providers) {
}

namespace {

NodeArg* AddInt64Initializer(Graph& graph, const char* name,
                             gsl::span<const int64_t> shape,
                             gsl::span<const int64_t> data) {
  ONNX_NAMESPACE::TensorProto proto;
  proto.set_name(graph.GenerateNodeArgName(name));
  proto.set_data_type(ONNX_NAMESPACE::TensorProto_DataType_INT64);
  for (auto d : shape) {
    proto.add_dims(d);
  }
  utils::SetRawDataInTensorProto(proto, data.data(), data.size() * sizeof(int64_t));
  return &graph_utils::AddInitializerWithOrtValue(graph, proto);
}

// Add a node with one output, return the Node pointer and output NodeArg.
std::pair<Node*, NodeArg*> AddNode1(Graph& graph, const char* op_type,
                                    gsl::span<NodeArg*> inputs,
                                    const std::string& ep_type,
                                    std::function<void(Node&)> configure = nullptr) {
  auto out_name = graph.GenerateNodeArgName(op_type);
  auto* out_arg = &graph.GetOrCreateNodeArg(out_name, nullptr);
  Node& n = graph.AddNode(graph.GenerateNodeName(op_type), op_type, "", inputs, {out_arg});
  n.SetExecutionProviderType(ep_type);
  if (configure) configure(n);
  return {&n, out_arg};
}

// Add a node with N outputs, return the Node pointer and output NodeArgs.
std::pair<Node*, std::vector<NodeArg*>> AddNodeN(Graph& graph, const char* op_type,
                                                  gsl::span<NodeArg*> inputs,
                                                  int num_outputs,
                                                  const std::string& ep_type) {
  std::vector<NodeArg*> outs;
  outs.reserve(num_outputs);
  for (int j = 0; j < num_outputs; ++j) {
    auto nm = graph.GenerateNodeArgName(std::string(op_type) + "_o" + std::to_string(j));
    outs.push_back(&graph.GetOrCreateNodeArg(nm, nullptr));
  }
  Node& n = graph.AddNode(graph.GenerateNodeName(op_type), op_type, "", inputs, outs);
  n.SetExecutionProviderType(ep_type);
  return {&n, std::move(outs)};
}

}  // namespace

/*
  Decompose LSTM (seq_len=1, forward-only, no peephole) into primitive ops.

  ONNX LSTM gate order: i (input), o (output), f (forget), c (cell) = IOFC.

  Equations (default activations: Sigmoid, Tanh, Tanh):
    gates = X @ W^T + H_prev @ R^T + Wb + Rb
    it = Sigmoid(gate_i), ot = Sigmoid(gate_o), ft = Sigmoid(gate_f), ct = Tanh(gate_c)
    Ct = ft * C_prev + it * ct
    Ht = ot * Tanh(Ct)
*/
Status LSTMDecomposition::ApplyImpl(Graph& graph, bool& modified, int graph_level,
                                    const logging::Logger& logger) const {
  GraphViewer graph_viewer(graph);
  auto& order = graph_viewer.GetNodesInTopologicalOrder();

  for (NodeIndex idx : order) {
    auto* node = graph.GetNode(idx);
    if (node == nullptr || node->OpType() != "LSTM") continue;
    ORT_RETURN_IF_ERROR(Recurse(*node, modified, graph_level, logger));

    Node& lstm = *node;
    auto& attrs = lstm.GetAttributes();

    // Only forward direction.
    if (attrs.count("direction")) {
      const auto& dir = attrs.at("direction").s();
      if (dir != "forward") continue;
    }

    // Only layout=0.
    if (attrs.count("layout") && attrs.at("layout").i() != 0) continue;

    // No peephole.
    if (lstm.InputDefs().size() > 7 && lstm.InputDefs()[7]->Exists()) continue;

    // Only seq_len=1 (skip if statically known to be >1; allow if unknown).
    auto* X_def = lstm.MutableInputDefs()[0];
    if (!X_def) continue;
    if (X_def->Shape() && X_def->Shape()->dim_size() >= 1) {
      auto& seq_dim = X_def->Shape()->dim(0);
      if (seq_dim.has_dim_value() && seq_dim.dim_value() != 1) continue;
    }

    int64_t hidden_size = 0;
    if (attrs.count("hidden_size")) hidden_size = attrs.at("hidden_size").i();
    if (hidden_size <= 0) continue;

    // Need initial states.
    if (lstm.InputDefs().size() <= 6) continue;
    auto* init_h_def = lstm.MutableInputDefs()[5];
    auto* init_c_def = lstm.MutableInputDefs()[6];
    if (!init_h_def || !init_h_def->Exists() || !init_c_def || !init_c_def->Exists()) continue;

    auto ep = lstm.GetExecutionProviderType();
    auto* X = lstm.MutableInputDefs()[0];
    auto* W = lstm.MutableInputDefs()[1];
    auto* R = lstm.MutableInputDefs()[2];
    NodeArg* B = (lstm.InputDefs().size() > 3 && lstm.InputDefs()[3]->Exists())
                     ? lstm.MutableInputDefs()[3]
                     : nullptr;

    // Shared axis-0 initializer for Squeeze/Unsqueeze.
    const int64_t ax0_val[] = {0};
    const int64_t ax0_shape[] = {1};
    auto* axes_0 = AddInt64Initializer(graph, "lstm_ax0", ax0_shape, ax0_val);

    // Squeeze away num_directions dim (axis 0).
    NodeArg* sq_x_in[] = {X, axes_0};
    auto* Xt = AddNode1(graph, "Squeeze", sq_x_in, ep).second;          // [batch, input_size]

    NodeArg* sq_w_in[] = {W, axes_0};
    auto* Ws = AddNode1(graph, "Squeeze", sq_w_in, ep).second;          // [4H, input_size]

    NodeArg* sq_r_in[] = {R, axes_0};
    auto* Rs = AddNode1(graph, "Squeeze", sq_r_in, ep).second;          // [4H, H]

    NodeArg* sq_h_in[] = {init_h_def, axes_0};
    auto* Ht_prev = AddNode1(graph, "Squeeze", sq_h_in, ep).second;     // [batch, H]

    NodeArg* sq_c_in[] = {init_c_def, axes_0};
    auto* Ct_prev = AddNode1(graph, "Squeeze", sq_c_in, ep).second;     // [batch, H]

    // Transpose W and R for MatMul: [4H, N] -> [N, 4H]
    NodeArg* tw_in[] = {Ws};
    auto* WT = AddNode1(graph, "Transpose", tw_in, ep, [](Node& n) {
      n.AddAttribute("perm", std::vector<int64_t>{1, 0});
    }).second;

    NodeArg* tr_in[] = {Rs};
    auto* RT = AddNode1(graph, "Transpose", tr_in, ep, [](Node& n) {
      n.AddAttribute("perm", std::vector<int64_t>{1, 0});
    }).second;

    // gates = Xt @ W^T + Ht_prev @ R^T
    NodeArg* mm1_in[] = {Xt, WT};
    auto* xW = AddNode1(graph, "MatMul", mm1_in, ep).second;            // [batch, 4H]

    NodeArg* mm2_in[] = {Ht_prev, RT};
    auto* hR = AddNode1(graph, "MatMul", mm2_in, ep).second;            // [batch, 4H]

    NodeArg* add_g_in[] = {xW, hR};
    auto* gates = AddNode1(graph, "Add", add_g_in, ep).second;          // [batch, 4H]

    // Add bias: B = [1, 8H] -> squeeze -> [8H] -> split Wb[4H]+Rb[4H] -> add -> [4H]
    if (B) {
      NodeArg* sq_b_in[] = {B, axes_0};
      auto* Bs = AddNode1(graph, "Squeeze", sq_b_in, ep).second;

      int64_t four_h = 4 * hidden_size;
      const int64_t sp_sizes[] = {four_h, four_h};
      const int64_t sp_shape[] = {2};
      auto* sp_init = AddInt64Initializer(graph, "lstm_bsplit", sp_shape, sp_sizes);

      NodeArg* spb_in[] = {Bs, sp_init};
      auto [sp_node, sp_outs] = AddNodeN(graph, "Split", spb_in, 2, ep);
      sp_node->AddAttribute("axis", static_cast<int64_t>(0));

      NodeArg* ab_in[] = {sp_outs[0], sp_outs[1]};
      auto* bias = AddNode1(graph, "Add", ab_in, ep).second;            // [4H]

      NodeArg* abg_in[] = {gates, bias};
      gates = AddNode1(graph, "Add", abg_in, ep).second;                // [batch, 4H]
    }

    // Split gates into 4 x [batch, H]: i, o, f, c (IOFC)
    int64_t h = hidden_size;
    const int64_t gs_sizes[] = {h, h, h, h};
    const int64_t gs_shape[] = {4};
    auto* gs_init = AddInt64Initializer(graph, "lstm_gsplit", gs_shape, gs_sizes);

    NodeArg* sg_in[] = {gates, gs_init};
    auto [sg_node, gparts] = AddNodeN(graph, "Split", sg_in, 4, ep);
    sg_node->AddAttribute("axis", static_cast<int64_t>(1));

    auto* i_gate = gparts[0];
    auto* o_gate = gparts[1];
    auto* f_gate = gparts[2];
    auto* c_gate = gparts[3];

    // Gate activations
    NodeArg* si_in[] = {i_gate};
    auto* it = AddNode1(graph, "Sigmoid", si_in, ep).second;
    NodeArg* so_in[] = {o_gate};
    auto* ot = AddNode1(graph, "Sigmoid", so_in, ep).second;
    NodeArg* sf_in[] = {f_gate};
    auto* ft = AddNode1(graph, "Sigmoid", sf_in, ep).second;
    NodeArg* tc_in[] = {c_gate};
    auto* ct_act = AddNode1(graph, "Tanh", tc_in, ep).second;

    // Ct = ft * Ct_prev + it * ct
    NodeArg* mf_in[] = {ft, Ct_prev};
    auto* ft_Cp = AddNode1(graph, "Mul", mf_in, ep).second;
    NodeArg* mi_in[] = {it, ct_act};
    auto* it_ct = AddNode1(graph, "Mul", mi_in, ep).second;
    NodeArg* ac_in[] = {ft_Cp, it_ct};
    auto* Ct_new = AddNode1(graph, "Add", ac_in, ep).second;

    // Ht = ot * Tanh(Ct)
    NodeArg* tC_in[] = {Ct_new};
    auto* tanh_Ct = AddNode1(graph, "Tanh", tC_in, ep).second;
    NodeArg* mh_in[] = {ot, tanh_Ct};
    auto* Ht_new = AddNode1(graph, "Mul", mh_in, ep).second;

    // --- Rewire outputs ---
    // Collect edges and LSTM output defs before modifying the graph.
    auto input_edges = graph_utils::GraphEdge::GetNodeInputEdges(lstm);
    auto output_edges = graph_utils::GraphEdge::GetNodeOutputEdges(lstm);
    auto& lstm_outs = lstm.MutableOutputDefs();

    // Get the original LSTM output NodeArgs (Y, Y_h, Y_c).
    NodeArg* orig_Y = (lstm_outs.size() > 0 && lstm_outs[0]->Exists()) ? lstm_outs[0] : nullptr;
    NodeArg* orig_Y_h = (lstm_outs.size() > 1 && lstm_outs[1]->Exists()) ? lstm_outs[1] : nullptr;
    NodeArg* orig_Y_c = (lstm_outs.size() > 2 && lstm_outs[2]->Exists()) ? lstm_outs[2] : nullptr;

    // Remove the original LSTM node first.
    graph_utils::GraphEdge::RemoveGraphEdges(graph, input_edges);
    graph_utils::GraphEdge::RemoveGraphEdges(graph, output_edges);
    graph.RemoveNode(lstm.Index());

    // Now create Unsqueeze nodes that output directly to the original LSTM output NodeArgs.
    // This avoids creating intermediate NodeArgs that become orphaned.

    // Helper: add a node whose output is an existing NodeArg.
    auto AddNodeWithOutput = [&](const char* op_type, gsl::span<NodeArg*> inputs,
                                 NodeArg* output_arg) -> Node& {
      Node& n = graph.AddNode(graph.GenerateNodeName(op_type), op_type, "",
                               inputs, {output_arg});
      n.SetExecutionProviderType(ep);
      return n;
    };

    // Y_h: [batch, H] -> [1, batch, H]
    Node* y_h_producer = nullptr;
    if (orig_Y_h) {
      NodeArg* uh_in[] = {Ht_new, axes_0};
      y_h_producer = &AddNodeWithOutput("Unsqueeze", uh_in, orig_Y_h);
    }

    // Y_c: [batch, H] -> [1, batch, H]
    Node* y_c_producer = nullptr;
    if (orig_Y_c) {
      NodeArg* uc_in[] = {Ct_new, axes_0};
      y_c_producer = &AddNodeWithOutput("Unsqueeze", uc_in, orig_Y_c);
    }

    // Y: [batch, H] -> [1, batch, H] -> [1, 1, batch, H]
    Node* y_producer = nullptr;
    if (orig_Y) {
      // First unsqueeze: [batch, H] -> [1, batch, H] (intermediate)
      NodeArg* uy1_in[] = {Ht_new, axes_0};
      auto* y_intermediate = AddNode1(graph, "Unsqueeze", uy1_in, ep).second;
      // Second unsqueeze: [1, batch, H] -> [1, 1, batch, H] -> orig_Y
      NodeArg* uy2_in[] = {y_intermediate, axes_0};
      y_producer = &AddNodeWithOutput("Unsqueeze", uy2_in, orig_Y);
    }

    // Re-add output edges from new producer nodes to downstream consumers.
    Node* producers[] = {y_producer, y_h_producer, y_c_producer};
    for (const auto& edge : output_edges) {
      auto src_idx = static_cast<size_t>(edge.src_arg_index);
      if (src_idx < 3 && producers[src_idx]) {
        graph.AddEdge(producers[src_idx]->Index(), edge.dst_node, 0, edge.dst_arg_index);
      }
    }

    modified = true;
  }

  return Status::OK();
}

}  // namespace onnxruntime
