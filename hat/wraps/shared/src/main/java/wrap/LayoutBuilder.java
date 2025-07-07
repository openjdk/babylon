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
package wrap;

import java.lang.foreign.GroupLayout;
import java.lang.foreign.MemoryLayout;
import java.lang.foreign.ValueLayout;
import java.util.ArrayList;
import java.util.List;
import java.util.function.Consumer;

import static java.lang.foreign.MemoryLayout.structLayout;
import static java.lang.foreign.MemoryLayout.unionLayout;
import static java.lang.foreign.ValueLayout.JAVA_INT;
import static java.lang.foreign.ValueLayout.JAVA_LONG;

public class LayoutBuilder {
    //String name;
    List<MemoryLayout> layouts = new ArrayList<>();

    // final public MemoryLayout memoryLayout;
    LayoutBuilder() {
        //this.memoryLayout = memoryLayout;
    }

    LayoutBuilder struct(String name, Consumer<LayoutBuilder> consumer) {
        LayoutBuilder lb = new LayoutBuilder();
        consumer.accept(lb);
        MemoryLayout layout = structLayout(lb.layouts.toArray(new MemoryLayout[0]));
        if (name != null) {
            layout.withName(name);
        }
        layouts.add(layout);
        return this;
    }

    LayoutBuilder union(String name, Consumer<LayoutBuilder> consumer) {
        LayoutBuilder lb = new LayoutBuilder();
        consumer.accept(lb);
        MemoryLayout layout = unionLayout(lb.layouts.toArray(new MemoryLayout[0]));
        if (name != null) {
            layout.withName(name);
        }
        layouts.add(layout);
        return this;
    }

    public LayoutBuilder i32(String name) {
        layouts.add(JAVA_INT.withName(name));
        return this;
    }
    public LayoutBuilder i64(String name) {
        layouts.add(JAVA_LONG.withName(name));
        return this;
    }

    public MemoryLayout memoryLayout(){
        return layouts.getFirst();
    }

    public LayoutBuilder i8Seq(String name, long elementCount) {
        layouts.add(MemoryLayout.sequenceLayout(elementCount, ValueLayout.JAVA_BYTE).withName(name));
        return this;
    }
    public static LayoutBuilder structBuilder(String name, Consumer<LayoutBuilder> consumer) {
        return new LayoutBuilder().struct(name, consumer);
    }
    public static GroupLayout structOf(String name, Consumer<LayoutBuilder> consumer) {
        return (GroupLayout) structBuilder(name, consumer).memoryLayout();
    }
}
