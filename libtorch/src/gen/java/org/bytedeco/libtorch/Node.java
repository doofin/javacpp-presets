// Targeted by JavaCPP version 1.5.4-SNAPSHOT: DO NOT EDIT THIS FILE

package org.bytedeco.libtorch;

import org.bytedeco.javacpp.BytePointer;
import org.bytedeco.javacpp.Loader;
import org.bytedeco.javacpp.Pointer;
import org.bytedeco.javacpp.annotation.*;


//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//                               Node
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// A `Node` is an abstract class that represents an operation taking zero
// or more input `Variable`s and producing zero or more output `Variable`s. All
// functions in PyTorch's autograd machinery derive from this class and
// override its `apply` method. Instances of such subclasses will then be
// invokeable via the call operator.
//
//                    Nodes in the Autograd Graph
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// When viewing the autograd system as a graph, `Node`s are the vertices or
// nodes, connected to each other via (directed) `Edge`s, which themselves are
// represented via (`Node`, input_nr) pairs. `Variable`s are the outputs to
// and inputs of `Node`s, and travel between these edges during execution
// of the graph. When two or more `Edge`s (from different sources) point at the
// same input to a `Node`, the values produced along all of these edges are
// implicitly summed prior to being forwarded to the target `Node`.
//
//                              Hierarchy
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// Subclasses usually represent differentiable functions as well as their
// gradient operators. Note, however, that due to the very general definition
// of a `Node` taking *zero* or more inputs and producing *zero* or more
// outputs, uses of `Node`s are flexible and extend beyond purely
// mathematical operations. For example, the `AccumulateGrad` function is a
// *sink*: it takes one input, but produces no outputs, instead accumulating
// the input as a side effect. At the other extreme, the `GraphRoot` function
// receives no inputs from other functions, but produces multiple outputs.
//
//                              Interface
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// The most important method on `Node` is the call operator, which takes in
// a list of variables and produces a list of variables. The precise size of
// these lists can be determined with `num_inputs()` and `num_outputs()`.
// `Node`s are stitched together via their `next_edge` interface, which let
// you manipulate the set of outgoing edges of a `Node`. You can add an
// edge with `add_next_edge()`, retrieve an edge with `next_edge(index)` and
// iterate over them via the `next_edges()` method. Other methods exist for
// integration with the JIT and other parts of PyTorch. Every `Node` has a
// *sequence number* that increases monotonically in the order of `Node`
// construction. It can be retrieved via the `sequence_nr()` method. Note that
// this sequence number is *thread local*. This means that when `Node`s
// `A`, `B` and `C` are created consecutively in the same thread, their
// sequence numbers will be ordered `A` < `B` < `C`. If, however, `A` and `B`
// are created in one thread and `C` is created in a new thread, there are *no
// guarantees* w.r.t. the ordering of `C` relative to `A` or `B`.
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
@Namespace("torch::autograd") @NoOffset @Properties(inherit = org.bytedeco.libtorch.presets.libtorch.class)
public class Node extends Pointer {
    static { Loader.load(); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public Node(Pointer p) { super(p); }

  /** Construct a new {@code Node} with the given {@code next_edges}. {@code sequence_nr} is
   *  a (currently THE) hint to prioritization in the backward() pass, with
   *  higher sequence numbers prioritized before lower sequence numbers. */

  /** Nodes are neither copyable nor moveable. */
  
  
  
  

  /** Evaluates the function on the given inputs and returns the result of the
   *  function call. */
  

  // Graph Connectivity API
  //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

  // Inputs. NOTE: inputs of the grad_fn correspond to Tensor outputs of the
  // forward function.

  // Marker for expected undefined input
  @Opaque public static class undefined_input extends Pointer {
      /** Empty constructor. Calls {@code super((Pointer)null)}. */
      public undefined_input() { super((Pointer)null); }
      /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
      public undefined_input(Pointer p) { super(p); }
  }

  /** Adds the type and shape metadata for a new input. Returns the index of
   *  of the new input. */

  public native @Cast("uint32_t") @NoException int add_input_metadata(@Const @ByRef Tensor t);

  /** Adds a placeholder for an input that will not be used. */
  public native @Cast("uint32_t") @NoException int add_input_metadata(@ByVal undefined_input u);

  public native @Cast("uint32_t") @NoException int num_inputs();

  public native @Const @ByRef InputMetadata input_metadata(@Cast("size_t") long index);

  /**
   * Note: Function Streams
   * A function's stream (for a given device type) is the stream of the first
   * element of its input buffer on a device of that type.
   *
   * If all elements are on the same device they MUST share a stream. If
   * elements are on different devices (across multiple GPUs, for example)
   * they may have different streams.
   */
  public native @ByVal @Cast("c10::optional<c10::Stream>*") Pointer stream(@Cast("const c10::DeviceType") short device_type);

  public native void clear_input_metadata();

  // Outputs ("Next Edges")

  public native @Const @ByRef @NoException Edge next_edge(@Cast("size_t") long index);

  public native void set_next_edge(@Cast("size_t") long index, @ByVal Edge edge);

  public native void add_next_edge(@ByVal Edge edge);

  

  public native @Cast("torch::autograd::edge_list*") @ByRef @NoException EdgeVector next_edges();

  public native @Cast("uint32_t") @NoException int num_outputs();

  // Miscellaneous Methods
  //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

  /** The sequence number of this {@code Node}. */
  public native @Cast("uint64_t") @NoException long sequence_nr();

  /** Returns the name of the dynamic type of the function, for debugging. */
  public native @StdString BytePointer name();

  /** Returns true if the particular output edge is active, and that particular
   *  output of this function should be computed. */
  public native @Cast("bool") boolean should_compute_output(@Cast("size_t") long output_edge_index);

  /** Returns true if any of the output edges in any of the ranges are active. */

  /** Returns the {@code PyObject} stored for this {@code Node} (for Python
   *  interaction). */
  public native @Cast("PyObject*") @NoException Pointer pyobj();

  /** Sets the {@code PyObject} stored for this {@code Node} (for Python interaction). */
  public native @NoException void set_pyobj(@Cast("PyObject*") Pointer pyobj);

  /** Returns the anomaly metadata stored for this {@code Node}.
   *  If none exist, creates a new empty one. */
  public native @NoException AnomalyMetadata metadata();

  // Hook API
  //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

  

  

  // delete a post hook matching the key
  

  

  

  

  

  // Customization Points for Subclasses
  //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

  /** Releases saved variables if the operation won't be reused. */
  public native void release_variables();

  /** Called before an apply if {@code release_variables()} is going to be called.
   *  Allows larger ops like {@code InterpreterAutogradFunction} to incrementally
   *  release variables as they run. */
  public native void will_release_variables();

  /** Returns true if this function is traceable. An op is traceable if all
   *  operations happening within {@code apply()} are performed on autograd
   *  {@code Variables} (i.e. apply mostly instantiates and applies other functions). */
  public native @Cast("bool") boolean is_traceable();

  /** A {@code Node} is said to pass state transparently to backward, if the
   *  state consists only of (Saved)Variables and only non-variable objects
   *  that parameterize the operation in some way that defines the graph
   *  structure AND the backward function is traceable. In particular,
   *  parametrization MUST NOT depend on the data of any {@code Variable}.
   *  TODO: it might be possible to handle cases where backward is
   *  non-traceable but state passing could be considered transparent. This
   *  will probably depend on saved_variable_list being mutable.
   *  NOTE: this value matters only if is_traceable() returns false. */
  public native @Cast("bool") boolean passes_state_transparently();

  public static native @Cast("uint64_t") long peek_at_next_sequence_nr();
}
