// Targeted by JavaCPP version 1.5-SNAPSHOT: DO NOT EDIT THIS FILE

package org.bytedeco.javacpp.opencv_cudafilters;

import java.nio.*;
import org.bytedeco.javacpp.*;
import org.bytedeco.javacpp.annotation.*;

import org.bytedeco.javacpp.opencv_core.*;
import static org.bytedeco.javacpp.opencv_core.opencv_core.*;
import org.bytedeco.javacpp.opencv_imgproc.*;
import static org.bytedeco.javacpp.opencv_imgproc.opencv_imgproc.*;
import org.bytedeco.javacpp.opencv_cudaarithm.*;
import static org.bytedeco.javacpp.opencv_cudaarithm.opencv_cudaarithm.*;

import static org.bytedeco.javacpp.opencv_cudafilters.opencv_cudafilters.*;


/** \addtogroup cudafilters
 *  \{
<p>
/** \brief Common interface for all CUDA filters :
 */
@Namespace("cv::cuda") @Properties(inherit = org.bytedeco.javacpp.opencv_cudafilters.opencv_cudafilters_presets.class)
public class Filter extends Algorithm {
    static { Loader.load(); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public Filter(Pointer p) { super(p); }

    /** \brief Applies the specified filter to the image.
    <p>
    @param src Input image.
    @param dst Output image.
    @param stream Stream for the asynchronous version.
     */
    public native void apply(@ByVal Mat src, @ByVal Mat dst, @ByRef(nullValue = "cv::cuda::Stream::Null()") Stream stream);
    public native void apply(@ByVal Mat src, @ByVal Mat dst);
    public native void apply(@ByVal UMat src, @ByVal UMat dst, @ByRef(nullValue = "cv::cuda::Stream::Null()") Stream stream);
    public native void apply(@ByVal UMat src, @ByVal UMat dst);
    public native void apply(@ByVal GpuMat src, @ByVal GpuMat dst, @ByRef(nullValue = "cv::cuda::Stream::Null()") Stream stream);
    public native void apply(@ByVal GpuMat src, @ByVal GpuMat dst);
}