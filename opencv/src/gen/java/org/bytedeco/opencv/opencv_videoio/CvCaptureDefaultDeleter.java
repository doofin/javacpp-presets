// Targeted by JavaCPP version 1.5-SNAPSHOT: DO NOT EDIT THIS FILE

package org.bytedeco.opencv.opencv_videoio;

import java.nio.*;
import org.bytedeco.javacpp.*;
import org.bytedeco.javacpp.annotation.*;

import org.bytedeco.opencv.opencv_core.*;
import static org.bytedeco.opencv.global.opencv_core.*;
import org.bytedeco.opencv.opencv_imgproc.*;
import static org.bytedeco.opencv.global.opencv_imgproc.*;
import static org.bytedeco.opencv.global.opencv_imgcodecs.*;

import static org.bytedeco.opencv.global.opencv_videoio.*;


@Name("cv::DefaultDeleter<CvCapture>") @Properties(inherit = org.bytedeco.opencv.presets.opencv_videoio.class)
public class CvCaptureDefaultDeleter extends Pointer {
    static { Loader.load(); }
    /** Default native constructor. */
    public CvCaptureDefaultDeleter() { super((Pointer)null); allocate(); }
    /** Native array allocator. Access with {@link Pointer#position(long)}. */
    public CvCaptureDefaultDeleter(long size) { super((Pointer)null); allocateArray(size); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public CvCaptureDefaultDeleter(Pointer p) { super(p); }
    private native void allocate();
    private native void allocateArray(long size);
    @Override public CvCaptureDefaultDeleter position(long position) {
        return (CvCaptureDefaultDeleter)super.position(position);
    }
 public native @Name("operator ()") void apply(CvCapture obj); }