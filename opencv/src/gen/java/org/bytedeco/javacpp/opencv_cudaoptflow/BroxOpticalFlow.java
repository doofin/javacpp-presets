// Targeted by JavaCPP version 1.5-SNAPSHOT: DO NOT EDIT THIS FILE

package org.bytedeco.javacpp.opencv_cudaoptflow;

import java.nio.*;
import org.bytedeco.javacpp.*;
import org.bytedeco.javacpp.annotation.*;

import org.bytedeco.javacpp.opencv_core.*;
import static org.bytedeco.javacpp.opencv_core.opencv_core.*;
import org.bytedeco.javacpp.opencv_imgproc.*;
import static org.bytedeco.javacpp.opencv_imgproc.opencv_imgproc.*;
import org.bytedeco.javacpp.opencv_imgcodecs.*;
import static org.bytedeco.javacpp.opencv_imgcodecs.opencv_imgcodecs.*;
import org.bytedeco.javacpp.opencv_videoio.*;
import static org.bytedeco.javacpp.opencv_videoio.opencv_videoio.*;
import org.bytedeco.javacpp.opencv_highgui.*;
import static org.bytedeco.javacpp.opencv_highgui.opencv_highgui.*;
import org.bytedeco.javacpp.opencv_flann.*;
import static org.bytedeco.javacpp.opencv_flann.opencv_flann.*;
import org.bytedeco.javacpp.opencv_features2d.*;
import static org.bytedeco.javacpp.opencv_features2d.opencv_features2d.*;
import org.bytedeco.javacpp.opencv_calib3d.*;
import static org.bytedeco.javacpp.opencv_calib3d.opencv_calib3d.*;
import org.bytedeco.javacpp.opencv_objdetect.*;
import static org.bytedeco.javacpp.opencv_objdetect.opencv_objdetect.*;
import org.bytedeco.javacpp.opencv_ximgproc.*;
import static org.bytedeco.javacpp.opencv_ximgproc.opencv_ximgproc.*;
import org.bytedeco.javacpp.opencv_video.*;
import static org.bytedeco.javacpp.opencv_video.opencv_video.*;
import org.bytedeco.javacpp.opencv_optflow.*;
import static org.bytedeco.javacpp.opencv_optflow.opencv_optflow.*;
import org.bytedeco.javacpp.opencv_cudaarithm.*;
import static org.bytedeco.javacpp.opencv_cudaarithm.opencv_cudaarithm.*;
import org.bytedeco.javacpp.opencv_cudafilters.*;
import static org.bytedeco.javacpp.opencv_cudafilters.opencv_cudafilters.*;
import org.bytedeco.javacpp.opencv_cudaimgproc.*;
import static org.bytedeco.javacpp.opencv_cudaimgproc.opencv_cudaimgproc.*;
import org.bytedeco.javacpp.opencv_cudawarping.*;
import static org.bytedeco.javacpp.opencv_cudawarping.opencv_cudawarping.*;

import static org.bytedeco.javacpp.opencv_cudaoptflow.opencv_cudaoptflow.*;


//
// BroxOpticalFlow
//

/** \brief Class computing the optical flow for two images using Brox et al Optical Flow algorithm (\cite Brox2004).
 */
@Namespace("cv::cuda") @Properties(inherit = org.bytedeco.javacpp.opencv_cudaoptflow.opencv_cudaoptflow_presets.class)
public class BroxOpticalFlow extends DenseOpticalFlow {
    static { Loader.load(); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public BroxOpticalFlow(Pointer p) { super(p); }

    public native double getFlowSmoothness();
    public native void setFlowSmoothness(double alpha);

    public native double getGradientConstancyImportance();
    public native void setGradientConstancyImportance(double gamma);

    public native double getPyramidScaleFactor();
    public native void setPyramidScaleFactor(double scale_factor);

    /** number of lagged non-linearity iterations (inner loop) */
    public native int getInnerIterations();
    public native void setInnerIterations(int inner_iterations);

    /** number of warping iterations (number of pyramid levels) */
    public native int getOuterIterations();
    public native void setOuterIterations(int outer_iterations);

    /** number of linear system solver iterations */
    public native int getSolverIterations();
    public native void setSolverIterations(int solver_iterations);

    public static native @Ptr BroxOpticalFlow create(
                double alpha/*=0.197*/,
                double gamma/*=50.0*/,
                double scale_factor/*=0.8*/,
                int inner_iterations/*=5*/,
                int outer_iterations/*=150*/,
                int solver_iterations/*=10*/);
    public static native @Ptr BroxOpticalFlow create();
}