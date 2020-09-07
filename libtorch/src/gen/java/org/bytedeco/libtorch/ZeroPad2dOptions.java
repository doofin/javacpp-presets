// Targeted by JavaCPP version 1.5.4-SNAPSHOT: DO NOT EDIT THIS FILE

package org.bytedeco.libtorch;

import org.bytedeco.javacpp.Loader;
import org.bytedeco.javacpp.Pointer;
import org.bytedeco.javacpp.annotation.ByVal;
import org.bytedeco.javacpp.annotation.Namespace;
import org.bytedeco.javacpp.annotation.Properties;


// ============================================================================

/** Options for the {@code ZeroPad2d} module.
 * 
 *  Example:
 *  <pre>{@code
 *  ZeroPad2d model(ZeroPad2dOptions({1, 1, 2, 0}));
 *  }</pre> */
@Namespace("torch::nn") @Properties(inherit = org.bytedeco.libtorch.presets.libtorch.class)
public class ZeroPad2dOptions extends Pointer {
    static { Loader.load(); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public ZeroPad2dOptions(Pointer p) { super(p); }

  public ZeroPad2dOptions(@ByVal ExpandingArray4 padding) { super((Pointer)null); allocate(padding); }
  private native void allocate(@ByVal ExpandingArray4 padding);

  /** The size of the padding.
   *  - If it is {@code int}, uses the same padding in all boundaries.
   *  - If it is a 4-{@code tuple} (for ZeroPad2d), uses (padding_left, padding_right, padding_top, padding_bottom). */
}
