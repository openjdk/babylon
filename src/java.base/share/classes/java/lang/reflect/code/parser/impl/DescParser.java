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

package java.lang.reflect.code.parser.impl;

import java.lang.reflect.code.descriptor.*;
import java.lang.reflect.code.descriptor.impl.*;
import java.lang.reflect.code.type.CoreTypeFactory;
import java.lang.reflect.code.TypeElement;
import java.lang.reflect.code.type.TypeDefinition;
import java.lang.reflect.code.type.impl.TypeDefinitionImpl;
import java.util.ArrayList;
import java.util.List;

public final class DescParser {
    private DescParser() {}

    /**
     * Parse a type descriptor from its serialized textual form.
     * @param desc the serialized type descriptor
     * @return the type descriptor
     */
    public static TypeDefinition parseTypeDefinition(String desc) {
        Scanner s = Scanner.factory().newScanner(desc);
        s.nextToken();
        return parseTypeDefinition(s);
    }

    /**
     * Parse a method type descriptor from its serialized textual form.
     * @param desc the serialized method type descriptor
     * @return the method type descriptor
     */
    public static MethodTypeDesc parseMethodTypeDesc(String desc) {
        Scanner s = Scanner.factory().newScanner(desc);
        s.nextToken();
        return parseMethodTypeDesc(s);
    }

    /**
     * Parse a method descriptor from its serialized textual form.
     *
     * @param desc the serialized method descriptor
     * @return the method descriptor
     */
    public static MethodDesc parseMethodDesc(String desc) {
        Scanner s = Scanner.factory().newScanner(desc);
        s.nextToken();
        return parseMethodDesc(s);
    }

    /**
     * Parse a field descriptor from its serialized textual form.
     *
     * @param desc the serialized field descriptor
     * @return the field descriptor
     */
    public static FieldDesc parseFieldDesc(String desc) {
        Scanner s = Scanner.factory().newScanner(desc);
        s.nextToken();
        return parseFieldDesc(s);
    }

    /**
     * Parse a record type descriptor from its serialized textual form.
     *
     * @param desc the serialized record type descriptor
     * @return the record type descriptor
     */
    public static RecordTypeDesc parseRecordTypeDesc(String desc) {
        Scanner s = Scanner.factory().newScanner(desc);
        s.nextToken();
        return parseRecordTypeDesc(s);
    }

    public static TypeDefinition parseTypeDefinition(Lexer l) {
        // Type
        Tokens.Token t = l.accept(Tokens.TokenKind.IDENTIFIER);
        StringBuilder identifier = new StringBuilder();
        identifier.append(t.name());
        while (l.acceptIf(Tokens.TokenKind.DOT)) {
            identifier.append(Tokens.TokenKind.DOT.name);
            t = l.accept(Tokens.TokenKind.IDENTIFIER);
            identifier.append(t.name());
        }

        // Type parameters
        List<TypeDefinition> args;
        if (l.token().kind == Tokens.TokenKind.LT) {
            args = new ArrayList<>();
            do {
                l.nextToken();
                TypeDefinition arg = parseTypeDefinition(l);
                args.add(arg);
            } while (l.token().kind == Tokens.TokenKind.COMMA);
            l.accept(Tokens.TokenKind.GT);
        } else {
            args = List.of();
        }

        // @@@ Enclosed/inner classes, separated by $ which may also be parameterized

        // Parse array-like syntax []+
        int dims = 0;
        while (l.acceptIf(Tokens.TokenKind.LBRACKET)) {
            l.accept(Tokens.TokenKind.RBRACKET);
            dims++;
        }

        TypeDefinition td = new TypeDefinition(identifier.toString(), args);
        if (dims > 0) {
            // If array-like then type definition becomes a child with identifier [+
            return new TypeDefinition("[".repeat(dims), List.of(td));
        } else {
            return td;
        }
    }

    static TypeElement parseTypeElement(Lexer l) {
        TypeDefinition typeDesc = parseTypeDefinition(l);
        return CoreTypeFactory.CORE_TYPE_FACTORY.constructType(typeDesc);
    }

    static MethodTypeDesc parseMethodTypeDesc(Lexer l) {
        List<TypeElement> ptypes = new ArrayList<>();
        l.accept(Tokens.TokenKind.LPAREN);
        if (l.token().kind != Tokens.TokenKind.RPAREN) {
            ptypes.add(parseTypeElement(l));
            while (l.acceptIf(Tokens.TokenKind.COMMA)) {
                ptypes.add(parseTypeElement(l));
            }
        }
        l.accept(Tokens.TokenKind.RPAREN);
        TypeElement rtype = parseTypeElement(l);
        return new MethodTypeDescImpl(rtype, ptypes);
    }

    static MethodDescImpl parseMethodDesc(Lexer l) {
        TypeElement refType = parseTypeElement(l);

        l.accept(Tokens.TokenKind.COLCOL);

        String methodName;
        if (l.acceptIf(Tokens.TokenKind.LT)) {
            // Special name such as "<new>"
            Tokens.Token t = l.accept(Tokens.TokenKind.IDENTIFIER);
            l.accept(Tokens.TokenKind.GT);
            methodName = "<" + t.name() + ">";
        } else {
            methodName = l.accept(Tokens.TokenKind.IDENTIFIER).name();
        }

        MethodTypeDesc mtype = parseMethodTypeDesc(l);

        return new MethodDescImpl(refType, methodName, mtype);
    }

    static FieldDescImpl parseFieldDesc(Lexer l) {
        TypeElement refType = parseTypeElement(l);

        l.accept(Tokens.TokenKind.COLCOL);

        String fieldName = l.accept(Tokens.TokenKind.IDENTIFIER).name();

        MethodTypeDesc mtype = parseMethodTypeDesc(l);
        if (!mtype.parameters().isEmpty()) {
            throw new IllegalArgumentException();
        }
        return new FieldDescImpl(refType, fieldName, mtype.returnType());
    }

    static RecordTypeDesc parseRecordTypeDesc(Lexer l) {
        List<RecordTypeDesc.ComponentDesc> components = new ArrayList<>();
        l.accept(Tokens.TokenKind.LPAREN);
        if (l.token().kind != Tokens.TokenKind.RPAREN) {
            do {
                TypeElement componentType = parseTypeElement(l);
                String componentName = l.accept(Tokens.TokenKind.IDENTIFIER).name();

                components.add(new RecordTypeDesc.ComponentDesc(componentType, componentName));
            } while(l.acceptIf(Tokens.TokenKind.COMMA));
        }
        l.accept(Tokens.TokenKind.RPAREN);
        TypeElement recordType = parseTypeElement(l);
        return new RecordTypeDescImpl(recordType, components);
    }
}
