// Targeted by JavaCPP version 1.5.1-SNAPSHOT: DO NOT EDIT THIS FILE

package org.bytedeco.cpython;

import java.nio.*;
import org.bytedeco.javacpp.*;
import org.bytedeco.javacpp.annotation.*;

import static org.bytedeco.cpython.global.python.*;


@Properties(inherit = org.bytedeco.cpython.presets.python.class)
public class wrapperbase extends Pointer {
    static { Loader.load(); }
    /** Default native constructor. */
    public wrapperbase() { super((Pointer)null); allocate(); }
    /** Native array allocator. Access with {@link Pointer#position(long)}. */
    public wrapperbase(long size) { super((Pointer)null); allocateArray(size); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public wrapperbase(Pointer p) { super(p); }
    private native void allocate();
    private native void allocateArray(long size);
    @Override public wrapperbase position(long position) {
        return (wrapperbase)super.position(position);
    }

    public native @Cast("const char*") BytePointer name(); public native wrapperbase name(BytePointer setter);
    public native int offset(); public native wrapperbase offset(int setter);
    public native Pointer function(); public native wrapperbase function(Pointer setter);
    public native wrapperfunc wrapper(); public native wrapperbase wrapper(wrapperfunc setter);
    public native @Cast("const char*") BytePointer doc(); public native wrapperbase doc(BytePointer setter);
    public native int flags(); public native wrapperbase flags(int setter);
    public native PyObject name_strobj(); public native wrapperbase name_strobj(PyObject setter);
}
