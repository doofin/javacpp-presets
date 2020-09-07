// Targeted by JavaCPP version 1.5.4-SNAPSHOT: DO NOT EDIT THIS FILE

package org.bytedeco.libtorch;

import org.bytedeco.javacpp.Loader;
import org.bytedeco.javacpp.Pointer;
import org.bytedeco.javacpp.annotation.*;


// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Identity ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

/** A placeholder identity operator that is argument-insensitive.
 *  See https://pytorch.org/docs/master/nn.html#torch.nn.Identity to learn
 *  about the exact behavior of this module. */
@Namespace("torch::nn") @Properties(inherit = org.bytedeco.libtorch.presets.libtorch.class)
public class IdentityImpl extends CloneableIdentityImpl {
    static { Loader.load(); }
    /** Default native constructor. */
    public IdentityImpl() { super((Pointer)null); allocate(); }
    /** Native array allocator. Access with {@link Pointer#position(long)}. */
    public IdentityImpl(long size) { super((Pointer)null); allocateArray(size); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public IdentityImpl(Pointer p) { super(p); }
    private native void allocate();
    private native void allocateArray(long size);
    @Override public IdentityImpl position(long position) {
        return (IdentityImpl)super.position(position);
    }
    @Override public IdentityImpl getPointer(long i) {
        return new IdentityImpl(this).position(position + i);
    }

  public native void reset();

  /** Pretty prints the {@code Identity} module into the given {@code stream}. */
  public native void pretty_print(@Cast("std::ostream*") @ByRef Pointer stream);

  public native @ByVal @Cast("torch::Tensor*") Pointer forward(@Cast("const torch::Tensor*") @ByRef Pointer input);
}
