// Targeted by JavaCPP version 1.5.4-SNAPSHOT: DO NOT EDIT THIS FILE

package org.bytedeco.libtorch;

import org.bytedeco.javacpp.Loader;
import org.bytedeco.javacpp.Pointer;
import org.bytedeco.javacpp.annotation.Namespace;
import org.bytedeco.javacpp.annotation.Properties;
 // namespace functional

// ============================================================================

/** Options for the {@code MultiLabelSoftMarginLoss} module.
 * 
 *  Example:
 *  <pre>{@code
 *  MultiLabelSoftMarginLoss model(MultiLabelSoftMarginLossOptions().reduction(torch::kNone).weight(weight));
 *  }</pre> */
@Namespace("torch::nn") @Properties(inherit = org.bytedeco.libtorch.presets.libtorch.class)
public class MultiLabelSoftMarginLossOptions extends Pointer {
    static { Loader.load(); }
    /** Default native constructor. */
    public MultiLabelSoftMarginLossOptions() { super((Pointer)null); allocate(); }
    /** Native array allocator. Access with {@link Pointer#position(long)}. */
    public MultiLabelSoftMarginLossOptions(long size) { super((Pointer)null); allocateArray(size); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public MultiLabelSoftMarginLossOptions(Pointer p) { super(p); }
    private native void allocate();
    private native void allocateArray(long size);
    @Override public MultiLabelSoftMarginLossOptions position(long position) {
        return (MultiLabelSoftMarginLossOptions)super.position(position);
    }
    @Override public MultiLabelSoftMarginLossOptions getPointer(long i) {
        return new MultiLabelSoftMarginLossOptions(this).position(position + i);
    }


  /** A manual rescaling weight given to each
   *  class. If given, it has to be a Tensor of size {@code C}. Otherwise, it is
   *  treated as if having all ones. */

  /** Specifies the reduction to apply to the output: 'none' | 'mean' | 'sum'.
   *  'none': no reduction will be applied, 'mean': the sum of the output will
   *  be divided by the number of elements in the output, 'sum': the output will
   *  be summed. Default: 'mean' */
}
