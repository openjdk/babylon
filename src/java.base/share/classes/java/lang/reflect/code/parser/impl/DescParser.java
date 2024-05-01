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

import java.lang.reflect.code.parser.impl.Tokens.Token;
import java.lang.reflect.code.parser.impl.Tokens.TokenKind;
import java.lang.reflect.code.type.*;
import java.lang.reflect.code.CodeType;
import java.lang.reflect.code.type.RecordTypeRef;
import java.lang.reflect.code.type.impl.FieldRefImpl;
import java.lang.reflect.code.type.impl.MethodRefImpl;
import java.lang.reflect.code.type.impl.RecordTypeRefImpl;
import java.util.ArrayList;
import java.util.List;

public final class DescParser {
    private DescParser() {}

    /**
     * Parse a type definition from its serialized textual form.
     * @param desc the serialized type definition
     * @return the type definition
     */
    public static CodeType.ExternalizedCodeType parseExternalizedCodeType(String desc) {
        Scanner s = Scanner.factory().newScanner(desc);
        s.nextToken();
        return parseExternalizedCodeType(s);
    }

    /**
     * Parse a method reference from its serialized textual form.
     *
     * @param desc the serialized method reference
     * @return the method reference
     */
    public static MethodRef parseMethodRef(String desc) {
        Scanner s = Scanner.factory().newScanner(desc);
        s.nextToken();
        return parseMethodRef(s);
    }

    /**
     * Parse a field reference from its serialized textual form.
     *
     * @param desc the serialized field reference
     * @return the field reference
     */
    public static FieldRef parseFieldRef(String desc) {
        Scanner s = Scanner.factory().newScanner(desc);
        s.nextToken();
        return parseFieldRef(s);
    }

    /**
     * Parse a record type reference from its serialized textual form.
     *
     * @param desc the serialized record type reference
     * @return the record type reference
     */
    public static RecordTypeRef parseRecordTypeRef(String desc) {
        Scanner s = Scanner.factory().newScanner(desc);
        s.nextToken();
        return parseRecordTypeRef(s);
    }

    public static CodeType.ExternalizedCodeType parseExternalizedCodeType(Lexer l) {
        StringBuilder identifier = new StringBuilder();
        if (l.token().kind == TokenKind.HASH) {
            // Quoted identifier
            Token t = l.token();
            while (t.kind != TokenKind.LT) {
                identifier.append(t.kind == TokenKind.IDENTIFIER ? t.name() : t.kind.name);
                l.nextToken();
                t = l.token();
            }
        } else {
            // Qualified identifier
            Tokens.Token t = l.accept(TokenKind.IDENTIFIER,
                    TokenKind.PLUS, TokenKind.SUB);
            identifier.append(t.kind == TokenKind.IDENTIFIER ? t.name() : t.kind.name);
            while (l.acceptIf(Tokens.TokenKind.DOT)) {
                identifier.append(Tokens.TokenKind.DOT.name);
                t = l.accept(Tokens.TokenKind.IDENTIFIER);
                identifier.append(t.name());
            }
        }

        // Type parameters
        List<CodeType.ExternalizedCodeType> args;
        if (l.token().kind == Tokens.TokenKind.LT) {
            args = new ArrayList<>();
            do {
                l.nextToken();
                CodeType.ExternalizedCodeType arg = parseExternalizedCodeType(l);
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

        CodeType.ExternalizedCodeType td = new CodeType.ExternalizedCodeType(identifier.toString(), args);
        if (dims > 0) {
            // If array-like then type definition becomes a child with identifier [+
            return new CodeType.ExternalizedCodeType("[".repeat(dims), List.of(td));
        } else {
            return td;
        }
    }

    static CodeType parseCodeType(Lexer l) {
        CodeType.ExternalizedCodeType typeDesc = parseExternalizedCodeType(l);
        return CoreTypeFactory.CORE_TYPE_FACTORY.constructType(typeDesc);
    }

    // (T, T, T, T)R
    static FunctionType parseMethodType(Lexer l) {
        List<CodeType> ptypes = new ArrayList<>();
        l.accept(Tokens.TokenKind.LPAREN);
        if (l.token().kind != Tokens.TokenKind.RPAREN) {
            ptypes.add(parseCodeType(l));
            while (l.acceptIf(Tokens.TokenKind.COMMA)) {
                ptypes.add(parseCodeType(l));
            }
        }
        l.accept(Tokens.TokenKind.RPAREN);
        CodeType rtype = parseCodeType(l);
        return FunctionType.functionType(rtype, ptypes);
    }

    static MethodRef parseMethodRef(Lexer l) {
        CodeType refType = parseCodeType(l);

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

        FunctionType mtype = parseMethodType(l);

        return new MethodRefImpl(refType, methodName, mtype);
    }

    static FieldRef parseFieldRef(Lexer l) {
        CodeType refType = parseCodeType(l);

        l.accept(Tokens.TokenKind.COLCOL);

        String fieldName = l.accept(Tokens.TokenKind.IDENTIFIER).name();

        FunctionType mtype = parseMethodType(l);
        if (!mtype.parameterTypes().isEmpty()) {
            throw new IllegalArgumentException();
        }
        return new FieldRefImpl(refType, fieldName, mtype.returnType());
    }

    static RecordTypeRef parseRecordTypeRef(Lexer l) {
        List<RecordTypeRef.ComponentRef> components = new ArrayList<>();
        l.accept(Tokens.TokenKind.LPAREN);
        if (l.token().kind != Tokens.TokenKind.RPAREN) {
            do {
                CodeType componentType = parseCodeType(l);
                String componentName = l.accept(Tokens.TokenKind.IDENTIFIER).name();

                components.add(new RecordTypeRef.ComponentRef(componentType, componentName));
            } while(l.acceptIf(Tokens.TokenKind.COMMA));
        }
        l.accept(Tokens.TokenKind.RPAREN);
        CodeType recordType = parseCodeType(l);
        return new RecordTypeRefImpl(recordType, components);
    }
}
