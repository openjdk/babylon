/*
 * Copyright (c) 2024, Oracle and/or its affiliates. All rights reserved.
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * This code is free software; you can redistribute it and/or modify it
 * under the terms of the GNU General Public License version 2 only, as
 * published by the Free Software Foundation.  Oracle designates this
 * particular file as subject to the "Classpath" exception as provided
 * by Oracle in the LICENSE file that accompanied this code.
 *
 * This code is distributed in the hope that it will be useful, but WITHOUT
 * ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
 * FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License
 * version 2 for more details (a copy is included in the LICENSE file that
 * accompanied this code).
 *
 * You should have received a copy of the GNU General Public License version
 * 2 along with this work; if not, write to the Free Software Foundation,
 * Inc., 51 Franklin St, Fifth Floor, Boston, MA 02110-1301 USA.
 *
 * Please contact Oracle, 500 Oracle Parkway, Redwood Shores, CA 94065 USA
 * or visit www.oracle.com if you need additional information or have any
 * questions.
 */
package hat;

import optkl.util.Regex;

/**
 * Used by kernels to extract thread information at runtime.
 * <p>
 * In reality these calls will be redirected to backend runtime equiv during code gen.
 * @author Gary Frost
 */
public interface KernelContext {
     Regex threadAccessRegex = Regex.of("(([GLB][SI][XYZ])|WRS|barrier)");

    /**
     * Marker called by kernel code which is mapped to a barrier implementation in the target language.
     */
    static void barrier() {
        // empty method - this is just a marker for the HAT Kernels
    }
    static int GIX(){return 0;};
    static int GIY(){return 0;};
    static int GIZ(){return 0;};
    static int GSX(){return 0;};
    static int GSY(){return 0;};
    static int GSZ(){return 0;};
    static int BIX(){return 0;};
    static int BIY(){return 0;};
    static int BIZ(){return 0;};
    static int BSX(){return 0;};
    static int BSY(){return 0;};
    static int BSZ(){return 0;};
    static int LIX(){return 0;};
    static int LIY(){return 0;};
    static int LIZ(){return 0;};
    static int LSX(){return 0;};
    static int LSY(){return 0;};
    static int LSZ(){return 0;};
    static int WRS(){return 0;};
}
