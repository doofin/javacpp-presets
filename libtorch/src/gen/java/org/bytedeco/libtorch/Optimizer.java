// Targeted by JavaCPP version 1.5.4-SNAPSHOT: DO NOT EDIT THIS FILE

package org.bytedeco.libtorch;

import org.bytedeco.javacpp.Loader;
import org.bytedeco.javacpp.Pointer;
import org.bytedeco.javacpp.annotation.*;


@Namespace("torch::optim") @NoOffset @Properties(inherit = org.bytedeco.libtorch.presets.libtorch.class)
public class Optimizer extends Pointer {
    static { Loader.load(); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public Optimizer(Pointer p) { super(p); }

  // The copy constructor is deleted, because the user should use the
  // `state_dict` / `load_state_dict` API to copy an optimizer instead.
  

  /** Constructs the {@code Optimizer} from a vector of parameters. */

  /** Adds the given param_group to the optimizer's param_group list. */
  /** A loss function closure, which is expected to return the loss value. */
  public native @ByVal @Cast("torch::Tensor*") Pointer step(@ByVal(nullValue = "torch::optim::Optimizer::LossClosure(nullptr)") @Cast("torch::optim::Optimizer::LossClosure*") Pointer closure);
  public native @ByVal @Cast("torch::Tensor*") Pointer step();

  /** Adds the given vector of parameters to the optimizer's parameter list. */
  public native void add_parameters(@Cast("torch::Tensor*") @StdVector Pointer parameters);

  /** Zeros out the gradients of all parameters. */
  public native void zero_grad();

  /** Provides a const reference to the parameters in the first param_group this optimizer holds. */

  /** Provides a reference to the parameters in the first param_group this optimizer holds. */
  public native @Cast("torch::Tensor*") @StdVector @NoException Pointer parameters();

  /** Returns the number of parameters referenced by the optimizer. */
  public native @Cast("size_t") @NoException long size();

  public native @ByRef @NoException OptimizerOptions defaults();

  /** Provides a reference to the param_groups this optimizer holds. */

  /** Provides a const reference to the param_groups this optimizer holds. */

  /** Provides a reference to the state this optimizer holds */
  public native @Cast("ska::flat_hash_map<std::string,std::unique_ptr<torch::optim::OptimizerParamState> >*") @ByRef @NoException Pointer state();

  /** Provides a const reference to the state this optimizer holds */

  /** Serializes the optimizer state into the given {@code archive}. */
  public native void save(@ByRef OutputArchive archive);

  /** Deserializes the optimizer state from the given {@code archive}. */
  public native void load(@ByRef InputArchive archive);
}
