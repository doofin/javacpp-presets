// Targeted by JavaCPP version 1.5.4-SNAPSHOT: DO NOT EDIT THIS FILE

package org.bytedeco.libtorch;

import org.bytedeco.javacpp.Loader;
import org.bytedeco.javacpp.Pointer;
import org.bytedeco.javacpp.annotation.Cast;
import org.bytedeco.javacpp.annotation.Namespace;
import org.bytedeco.javacpp.annotation.Properties;


/** Options for the {@code PixelShuffle} module.
 * 
 *  Example:
 *  <pre>{@code
 *  PixelShuffle model(PixelShuffleOptions(5));
 *  }</pre> */
@Namespace("torch::nn") @Properties(inherit = org.bytedeco.libtorch.presets.libtorch.class)
public class PixelShuffleOptions extends Pointer {
    static { Loader.load(); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public PixelShuffleOptions(Pointer p) { super(p); }

  public PixelShuffleOptions(@Cast("int64_t") long upscale_factor) { super((Pointer)null); allocate(upscale_factor); }
  private native void allocate(@Cast("int64_t") long upscale_factor);

  /** Factor to increase spatial resolution by */
}
