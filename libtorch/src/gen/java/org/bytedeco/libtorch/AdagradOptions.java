// Targeted by JavaCPP version 1.5.4-SNAPSHOT: DO NOT EDIT THIS FILE

package org.bytedeco.libtorch;

import org.bytedeco.javacpp.Loader;
import org.bytedeco.javacpp.Pointer;
import org.bytedeco.javacpp.annotation.ByRef;
import org.bytedeco.javacpp.annotation.Namespace;
import org.bytedeco.javacpp.annotation.Properties;
 // namespace torch

@Namespace("torch::optim") @Properties(inherit = org.bytedeco.libtorch.presets.libtorch.class)
public class AdagradOptions extends Pointer {
    static { Loader.load(); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public AdagradOptions(Pointer p) { super(p); }

  public AdagradOptions(double lr/*=1e-2*/) { super((Pointer)null); allocate(lr); }
  private native void allocate(double lr/*=1e-2*/);
  public AdagradOptions() { super((Pointer)null); allocate(); }
  private native void allocate();
  public native void serialize(@ByRef InputArchive archive);
  public native void serialize(@ByRef OutputArchive archive);
  
}
