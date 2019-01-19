/*
 * Copyright (C) 2014-2018 Samuel Audet
 *
 * Licensed either under the Apache License, Version 2.0, or (at your option)
 * under the terms of the GNU General Public License as published by
 * the Free Software Foundation (subject to the "Classpath" exception),
 * either version 2, or any later version (collectively, the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *     http://www.gnu.org/licenses/
 *     http://www.gnu.org/software/classpath/license.html
 *
 * or as provided in the LICENSE.txt file that accompanied this code.
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package org.bytedeco.javacpp.opencv_features2d;

import org.bytedeco.javacpp.annotation.Platform;
import org.bytedeco.javacpp.annotation.Properties;
import org.bytedeco.javacpp.opencv_flann.opencv_flann_presets;
import org.bytedeco.javacpp.opencv_highgui.opencv_highgui_presets;
import org.bytedeco.javacpp.tools.InfoMap;
import org.bytedeco.javacpp.tools.InfoMapper;

/**
 *
 * @author Samuel Audet
 */
@Properties(inherit = {opencv_highgui_presets.class, opencv_flann_presets.class}, value = {
    @Platform(include = "<opencv2/features2d.hpp>", link = "opencv_features2d@.4.0"),
    @Platform(value = "ios", preload = "libopencv_features2d"),
    @Platform(value = "windows", link = "opencv_features2d401")},
        target = "org.bytedeco.javacpp.opencv_features2d",
        global = "opencv_features2d"
)
public class opencv_features2d_presets implements InfoMapper {
    public void map(InfoMap infoMap) {
//        infoMap.put(new Info("cv::Feature2D").pointerTypes("Feature2D").virtualize());
    }
}