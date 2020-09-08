// Targeted by JavaCPP version 1.5.4-SNAPSHOT: DO NOT EDIT THIS FILE

package org.bytedeco.libtorch;

import org.bytedeco.javacpp.Loader;
import org.bytedeco.javacpp.Pointer;
import org.bytedeco.javacpp.annotation.*;


@Namespace("torch::optim") @NoOffset @Properties(inherit = org.bytedeco.libtorch.presets.libtorch.class)
public class LBFGS extends Optimizer {
    static { Loader.load(); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public LBFGS(Pointer p) { super(p); }

   public LBFGS(
          @StdVector Tensor params,
          @ByVal(nullValue = "torch::optim::LBFGSOptions({})") LBFGSOptions defaults) { super((Pointer)null); allocate(params, defaults); }
   private native void allocate(
          @StdVector Tensor params,
          @ByVal(nullValue = "torch::optim::LBFGSOptions({})") LBFGSOptions defaults);
  public native void save(@ByRef OutputArchive archive);
  public native void load(@ByRef InputArchive archive);
}
