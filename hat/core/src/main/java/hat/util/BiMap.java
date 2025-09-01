
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
package hat.util;

import jdk.incubator.code.Block;
import jdk.incubator.code.Op;

import java.util.LinkedHashMap;
import java.util.Map;

public class BiMap<T1 extends Block.Parameter, T2 extends Op> {
    public Map<T1, T2> t1ToT2 = new LinkedHashMap<>();
    public Map<T2, T1> t2ToT1 = new LinkedHashMap<>();

    public void add(T1 t1, T2 t2) {
        t1ToT2.put(t1, t2);
        t2ToT1.put(t2, t1);
    }

    public T1 get(T2 t2) {
        return t2ToT1.get(t2);
    }

    public T2 get(T1 t1) {
        return t1ToT2.get(t1);
    }

    public boolean containsKey(T1 t1) {
        return t1ToT2.containsKey(t1);
    }

    public boolean containsKey(T2 t2) {
        return t2ToT1.containsKey(t2);
    }
}
