// Targeted by JavaCPP version 1.5.4-SNAPSHOT: DO NOT EDIT THIS FILE

package org.bytedeco.libtorch;

import org.bytedeco.javacpp.Loader;
import org.bytedeco.javacpp.Pointer;
import org.bytedeco.javacpp.annotation.Name;
import org.bytedeco.javacpp.annotation.Properties;
@Name("torch::nn::Cloneable<torch::nn::IdentityImpl>") @Properties(inherit = org.bytedeco.libtorch.presets.libtorch.class)
public class CloneableIdentityImpl extends Module {
    static { Loader.load(); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public CloneableIdentityImpl(Pointer p) { super(p); }


  /** {@code reset()} must perform initialization of all members with reference
   *  semantics, most importantly parameters, buffers and submodules. */
  public native void reset();

  /** Performs a recursive "deep copy" of the {@code Module}, such that all parameters
   *  and submodules in the cloned module are different from those in the
   *  original module. */
  
}
