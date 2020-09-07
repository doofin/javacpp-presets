// Targeted by JavaCPP version 1.5.4-SNAPSHOT: DO NOT EDIT THIS FILE

package org.bytedeco.libtorch;

import org.bytedeco.javacpp.Loader;
import org.bytedeco.javacpp.Pointer;
import org.bytedeco.javacpp.annotation.Namespace;
import org.bytedeco.javacpp.annotation.Properties;


/** Options for the {@code Dropout} module.
 * 
 *  Example:
 *  <pre>{@code
 *  Dropout model(DropoutOptions().p(0.42).inplace(true));
 *  }</pre> */
@Namespace("torch::nn") @Properties(inherit = org.bytedeco.libtorch.presets.libtorch.class)
public class DropoutOptions extends Pointer {
    static { Loader.load(); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public DropoutOptions(Pointer p) { super(p); }

  /* implicit */ public DropoutOptions(double p/*=0.5*/) { super((Pointer)null); allocate(p); }
private native void allocate(double p/*=0.5*/);
public DropoutOptions() { super((Pointer)null); allocate(); }
private native void allocate();

  /** The probability of an element to be zeroed. Default: 0.5 */

  /** can optionally do the operation in-place. Default: False */
}
