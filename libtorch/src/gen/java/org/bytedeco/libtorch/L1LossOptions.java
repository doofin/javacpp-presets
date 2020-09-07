// Targeted by JavaCPP version 1.5.4-SNAPSHOT: DO NOT EDIT THIS FILE

package org.bytedeco.libtorch;

import org.bytedeco.javacpp.Loader;
import org.bytedeco.javacpp.Pointer;
import org.bytedeco.javacpp.annotation.ByVal;
import org.bytedeco.javacpp.annotation.Namespace;
import org.bytedeco.javacpp.annotation.Properties;


/** Options for the {@code L1Loss} module.
 * 
 *  Example:
 *  <pre>{@code
 *  L1Loss model(L1LossOptions(torch::kNone));
 *  }</pre> */
@Namespace("torch::nn") @Properties(inherit = org.bytedeco.libtorch.presets.libtorch.class)
public class L1LossOptions extends Pointer {
    static { Loader.load(); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public L1LossOptions(Pointer p) { super(p); }
    /** Native array allocator. Access with {@link Pointer#position(long)}. */
    public L1LossOptions(long size) { super((Pointer)null); allocateArray(size); }
    private native void allocateArray(long size);
    @Override public L1LossOptions position(long position) {
        return (L1LossOptions)super.position(position);
    }
    @Override public L1LossOptions getPointer(long i) {
        return new L1LossOptions(this).position(position + i);
    }


  public L1LossOptions() { super((Pointer)null); allocate(); }
  private native void allocate();
public L1LossOptions(@ByVal kNone reduction) { super((Pointer)null); allocate(reduction); }
private native void allocate(@ByVal kNone reduction);
public L1LossOptions(@ByVal kMean reduction) { super((Pointer)null); allocate(reduction); }
private native void allocate(@ByVal kMean reduction);
public L1LossOptions(@ByVal kSum reduction) { super((Pointer)null); allocate(reduction); }
private native void allocate(@ByVal kSum reduction);

  /** Specifies the reduction to apply to the output. */
}
