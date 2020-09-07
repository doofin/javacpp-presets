// Targeted by JavaCPP version 1.5.4-SNAPSHOT: DO NOT EDIT THIS FILE

package org.bytedeco.libtorch;

import org.bytedeco.javacpp.Loader;
import org.bytedeco.javacpp.Pointer;
import org.bytedeco.javacpp.annotation.Cast;
import org.bytedeco.javacpp.annotation.Namespace;
import org.bytedeco.javacpp.annotation.Properties;


/** Options for {@code torch::nn::functional::group_norm}.
 * 
 *  Example:
 *  <pre>{@code
 *  namespace F = torch::nn::functional;
 *  F::group_norm(input, F::GroupNormFuncOptions(2).eps(2e-5));
 *  }</pre> */
@Namespace("torch::nn::functional") @Properties(inherit = org.bytedeco.libtorch.presets.libtorch.class)
public class GroupNormFuncOptions extends Pointer {
    static { Loader.load(); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public GroupNormFuncOptions(Pointer p) { super(p); }

  /* implicit */ public GroupNormFuncOptions(@Cast("int64_t") long num_groups) { super((Pointer)null); allocate(num_groups); }
private native void allocate(@Cast("int64_t") long num_groups);

  /** number of groups to separate the channels into */

  /** a value added to the denominator for numerical stability. Default: 1e-5 */
}
