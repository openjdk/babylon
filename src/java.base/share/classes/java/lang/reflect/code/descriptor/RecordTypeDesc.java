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

package java.lang.reflect.code.descriptor;

import java.lang.reflect.code.descriptor.impl.RecordTypeDescImpl;
import java.lang.reflect.code.type.JavaType;
import java.lang.reflect.code.TypeElement;
import java.util.List;
import java.util.stream.Stream;

/**
 * The symbolic description of a Java record type.
 */
public sealed interface RecordTypeDesc permits RecordTypeDescImpl {
    TypeElement recordType();

    /**
     * The symbolic description a Java record component.
     * @param type the type of the component
     * @param name the name of the component
     */
    record ComponentDesc(TypeElement type, String name) {}

    List<ComponentDesc> components();

    MethodDesc methodForComponent(int i);

    // Factories

    static RecordTypeDesc recordType(Class<? extends Record> c) {
        List<ComponentDesc> components = Stream.of(c.getRecordComponents())
                .map(rc -> new ComponentDesc(JavaType.type(rc.getType()), rc.getName()))
                .toList();
        return recordType(JavaType.type(c), components);
    }

    static RecordTypeDesc recordType(TypeElement recordType, ComponentDesc... components) {
        return recordType(recordType, List.of(components));
    }

    static RecordTypeDesc recordType(TypeElement recordType, List<ComponentDesc> components) {
        return new RecordTypeDescImpl(recordType, components);
    }

    // Copied code in jdk.compiler module throws UOE
    static RecordTypeDesc ofString(String s) {
/*__throw new UnsupportedOperationException();__*/        return java.lang.reflect.code.parser.impl.DescParser.parseRecordTypeDesc(s);
    }
}
