// Targeted by JavaCPP version 1.5.4-SNAPSHOT: DO NOT EDIT THIS FILE

package org.bytedeco.libtorch;

import org.bytedeco.javacpp.Loader;
import org.bytedeco.javacpp.Pointer;
import org.bytedeco.javacpp.annotation.ByRef;
import org.bytedeco.javacpp.annotation.Namespace;
import org.bytedeco.javacpp.annotation.Properties;
import org.bytedeco.javacpp.annotation.UniquePtr;


@Namespace("torch::optim") @Properties(inherit = org.bytedeco.libtorch.presets.libtorch.class)
public class OptimizerOptions extends Pointer {
    static { Loader.load(); }
    /** Default native constructor. */
    public OptimizerOptions() { super((Pointer)null); allocate(); }
    /** Native array allocator. Access with {@link Pointer#position(long)}. */
    public OptimizerOptions(long size) { super((Pointer)null); allocateArray(size); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public OptimizerOptions(Pointer p) { super(p); }
    private native void allocate();
    private native void allocateArray(long size);
    @Override public OptimizerOptions position(long position) {
        return (OptimizerOptions)super.position(position);
    }
    @Override public OptimizerOptions getPointer(long i) {
        return new OptimizerOptions(this).position(position + i);
    }

  public native @UniquePtr OptimizerOptions clone();
  public native void serialize(@ByRef InputArchive archive);
  public native void serialize(@ByRef OutputArchive archive);
}