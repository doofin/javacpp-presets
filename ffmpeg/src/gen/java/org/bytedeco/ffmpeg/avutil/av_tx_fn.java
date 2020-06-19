// Targeted by JavaCPP version 1.5.4-SNAPSHOT: DO NOT EDIT THIS FILE

package org.bytedeco.ffmpeg.avutil;

import java.nio.*;
import org.bytedeco.javacpp.*;
import org.bytedeco.javacpp.annotation.*;

import static org.bytedeco.javacpp.presets.javacpp.*;

import static org.bytedeco.ffmpeg.global.avutil.*;


/**
 * Function pointer to a function to perform the transform.
 *
 * \note Using a different context than the one allocated during av_tx_init()
 * is not allowed.
 *
 * @param s the transform context
 * @param out the output array
 * @param in the input array
 * @param stride the input or output stride in bytes
 *
 * The out and in arrays must be aligned to the maximum required by the CPU
 * architecture.
 * The stride must follow the constraints the transform type has specified.
 */
@Properties(inherit = org.bytedeco.ffmpeg.presets.avutil.class)
public class av_tx_fn extends FunctionPointer {
    static { Loader.load(); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public    av_tx_fn(Pointer p) { super(p); }
    protected av_tx_fn() { allocate(); }
    private native void allocate();
    public native void call(AVTXContext s, Pointer out, Pointer in, @Cast("ptrdiff_t") long stride);
}
