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

package jdk.incubator.code.tools.renderer;

import jdk.incubator.code.Block;
import jdk.incubator.code.Value;
import java.util.HashMap;
import java.util.Map;

public final class GlobalValueBlockNaming {
    final Map<Value, String> gvn;
    final Map<Block, String> gbn;
    int valueOrdinal = 0;
    int blockOrdinal = 0;

    GlobalValueBlockNaming() {
        this.gvn = new HashMap<>();
        this.gbn = new HashMap<>();
    }

    String getValueName(Value _v) {
        return gvn.computeIfAbsent(_v, v -> String.valueOf(valueOrdinal++));
    }

    String getBlockName(Block _b) {
        return gbn.computeIfAbsent(_b, b -> "block_" + blockOrdinal++);
    }
}
