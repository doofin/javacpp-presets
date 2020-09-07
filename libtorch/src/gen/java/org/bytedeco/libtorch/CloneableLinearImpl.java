// Targeted by JavaCPP version 1.5.4-SNAPSHOT: DO NOT EDIT THIS FILE

package org.bytedeco.libtorch;

import org.bytedeco.javacpp.Loader;
import org.bytedeco.javacpp.Pointer;
import org.bytedeco.javacpp.annotation.Name;
import org.bytedeco.javacpp.annotation.Properties;

/** The {@code clone()} method in the base {@code Module} class does not have knowledge of
 *  the concrete runtime type of its subclasses. Therefore, {@code clone()} must
 *  either be called from within the subclass, or from a base class that has
 *  knowledge of the concrete type. {@code Cloneable} uses the CRTP to gain
 *  knowledge of the subclass' static type and provide an implementation of the
 *  {@code clone()} method. We do not want to use this pattern in the base class,
 *  because then storing a module would always require templatizing it. */@Name("torch::nn::Cloneable<torch::nn::LinearImpl>") @Properties(inherit = org.bytedeco.libtorch.presets.libtorch.class)
public class CloneableLinearImpl extends Module {
    static { Loader.load(); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public CloneableLinearImpl(Pointer p) { super(p); }


  /** {@code reset()} must perform initialization of all members with reference
   *  semantics, most importantly parameters, buffers and submodules. */
  public native void reset();

  /** Performs a recursive "deep copy" of the {@code Module}, such that all parameters
   *  and submodules in the cloned module are different from those in the
   *  original module. */
  
}

