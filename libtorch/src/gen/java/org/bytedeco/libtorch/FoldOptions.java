// Targeted by JavaCPP version 1.5.4-SNAPSHOT: DO NOT EDIT THIS FILE

package org.bytedeco.libtorch;

import org.bytedeco.javacpp.Loader;
import org.bytedeco.javacpp.Pointer;
import org.bytedeco.javacpp.annotation.ByVal;
import org.bytedeco.javacpp.annotation.Cast;
import org.bytedeco.javacpp.annotation.Namespace;
import org.bytedeco.javacpp.annotation.Properties;


/** Options for the {@code Fold} module.
 * 
 *  Example:
 *  <pre>{@code
 *  Fold model(FoldOptions({8, 8}, {3, 3}).dilation(2).padding({2, 1}).stride(2));
 *  }</pre> */
@Namespace("torch::nn") @Properties(inherit = org.bytedeco.libtorch.presets.libtorch.class)
public class FoldOptions extends Pointer {
    static { Loader.load(); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public FoldOptions(Pointer p) { super(p); }

  public FoldOptions(@ByVal @Cast("torch::ExpandingArray<2>*") Pointer output_size, @ByVal @Cast("torch::ExpandingArray<2>*") Pointer kernel_size) { super((Pointer)null); allocate(output_size, kernel_size); }
  private native void allocate(@ByVal @Cast("torch::ExpandingArray<2>*") Pointer output_size, @ByVal @Cast("torch::ExpandingArray<2>*") Pointer kernel_size);

  /** describes the spatial shape of the large containing tensor of the sliding
   *  local blocks. It is useful to resolve the ambiguity when multiple input
   *  shapes map to same number of sliding blocks, e.g., with stride > 0. */

  /** the size of the sliding blocks */

  /** controls the spacing between the kernel points; also known as the à trous
   *  algorithm. */

  /** controls the amount of implicit zero-paddings on both sides for padding
   *  number of points for each dimension before reshaping. */

  /** controls the stride for the sliding blocks. */
}
