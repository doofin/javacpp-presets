// Targeted by JavaCPP version 1.5.4-SNAPSHOT: DO NOT EDIT THIS FILE

package org.bytedeco.libtorch;

import org.bytedeco.javacpp.Loader;
import org.bytedeco.javacpp.Pointer;
import org.bytedeco.javacpp.annotation.Namespace;
import org.bytedeco.javacpp.annotation.Properties;
 // namespace functional

// ============================================================================

/** Options for the {@code BCEWithLogitsLoss} module.
 * 
 *  Example:
 *  <pre>{@code
 *  BCEWithLogitsLoss model(BCEWithLogitsLossOptions().reduction(torch::kNone).weight(weight));
 *  }</pre> */
@Namespace("torch::nn") @Properties(inherit = org.bytedeco.libtorch.presets.libtorch.class)
public class BCEWithLogitsLossOptions extends Pointer {
    static { Loader.load(); }
    /** Default native constructor. */
    public BCEWithLogitsLossOptions() { super((Pointer)null); allocate(); }
    /** Native array allocator. Access with {@link Pointer#position(long)}. */
    public BCEWithLogitsLossOptions(long size) { super((Pointer)null); allocateArray(size); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public BCEWithLogitsLossOptions(Pointer p) { super(p); }
    private native void allocate();
    private native void allocateArray(long size);
    @Override public BCEWithLogitsLossOptions position(long position) {
        return (BCEWithLogitsLossOptions)super.position(position);
    }
    @Override public BCEWithLogitsLossOptions getPointer(long i) {
        return new BCEWithLogitsLossOptions(this).position(position + i);
    }

  /** A manual rescaling weight given to the loss of each batch element.
   *  If given, has to be a Tensor of size {@code nbatch}. */
  /** Specifies the reduction to apply to the output. Default: Mean */
  /** A weight of positive examples.
   *  Must be a vector with length equal to the number of classes. */
}
