// Targeted by JavaCPP version 1.5.4-SNAPSHOT: DO NOT EDIT THIS FILE

package org.bytedeco.libtorch;

import org.bytedeco.javacpp.Pointer;
import org.bytedeco.javacpp.annotation.Namespace;
import org.bytedeco.javacpp.annotation.Opaque;
import org.bytedeco.javacpp.annotation.Properties;


// Forward declarations confuse Doxygen
// #ifndef DOXYGEN_SHOULD_SKIP_THIS
@Namespace("at") @Opaque @Properties(inherit = org.bytedeco.libtorch.presets.libtorch.class)
public class Tensor extends Pointer {
    /** Empty constructor. Calls {@code super((Pointer)null)}. */
    public Tensor() { super((Pointer)null); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public Tensor(Pointer p) { super(p); }
}
