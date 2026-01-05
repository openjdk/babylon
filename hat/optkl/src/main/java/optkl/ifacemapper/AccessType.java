/*
 * Copyright (c) 2025, Oracle and/or its affiliates. All rights reserved.
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
package optkl.ifacemapper;

import java.lang.annotation.Annotation;

public enum AccessType {
    NOT_BUFFER((byte) 0),
    NA((byte) 1),
    RO((byte) 2),
    WO((byte) 4),
    RW((byte) 6);

    public final byte value;

    AccessType(byte i) {
        value = i;
    }

    public static AccessType of(byte i) {
        return switch (i) {
            case (byte)0 -> NOT_BUFFER;
            case (byte)1 -> NA;
            case (byte)2 -> RO;
            case (byte)4 -> WO;
            case (byte)6 -> RO;
            default -> throw new IllegalStateException("No access type for " + i);
        };
    }

    public static AccessType of(Annotation annotation) {
        return switch (annotation) {
            case MappableIface.RO ro -> RO;
            case MappableIface.RW rw -> RW;
            case MappableIface.WO wo -> WO;
            default -> NA;
        };
    }
}
