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
package hat.opcodebuilders;


import java.lang.reflect.code.TypeElement;
import java.lang.reflect.code.Value;

public abstract class StyledOpCodeBuilder<T extends StyledOpCodeBuilder<T>> extends OpCodeBuilder<T> {


    public T defaultStyle(Runnable r) {
        r.run();
        return self();
    }

    public T valueStyle(Runnable r) {
        return defaultStyle(r);
    }

    public T opNameStyle(Runnable r) {
        return defaultStyle(r);
    }

    public T dquoteStyle(Runnable r) {
        return defaultStyle(r);
    }

    public T typeNameStyle(Runnable r) {
        return defaultStyle(r);
    }

    public T atStyle(Runnable r) {
        return defaultStyle(r);
    }

    @Override
    public T value(Value v) {
        return valueStyle(() -> super.value(v));
    }

    @Override
    public T opName(String name) {
        return opNameStyle(() -> super.opName(name));
    }

    @Override
    public T dquote(String name) {
        return dquoteStyle(() -> super.dquote(name));
    }

    @Override
    public T typeName(TypeElement typeDesc) {
        return typeNameStyle(() -> super.typeName(typeDesc));
    }

    @Override
    public T at() {
        return atStyle(() -> super.at());
    }
}


