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

package jdk.incubator.code.extern.impl;

import jdk.incubator.code.extern.ExternalizedCodeType;
import jdk.incubator.code.extern.impl.Tokens.TokenKind;

import java.util.ArrayList;
import java.util.List;

public final class DescParser {
    private DescParser() {}

    /**
     * Parse an externalized code type from its serialized textual form.
     * @param desc the serialized externalized code type
     * @return the externalized code type
     */
    public static ExternalizedCodeType parseExCodeType(String desc) {
        Scanner s = Scanner.factory().newScanner(desc);
        s.nextToken();
        return parseExCodeType(s);
    }

    //    ExType:
    //        ExTypeName
    //        ExTypeName '<' ExType* '>'
    //
    //    ExTypeName:
    //        ExIdentPart
    //        ExIdentPart ExIdentSep ExIdentPart
    //
    //    ExIdentPart:
    //        ident
    //        string
    //
    //    ExIdentSep:
    //        '.'
    //        ':'
    public static ExternalizedCodeType parseExCodeType(Lexer l) {
        StringBuilder identifier = new StringBuilder();
        identifier.append(parseExCodeTypeNamePart(l));
        while (l.is(TokenKind.DOT) || l.is(TokenKind.COLON)) {
            identifier.append(l.token().kind.name);
            l.nextToken();
            identifier.append(parseExCodeTypeNamePart(l));
        }
        List<ExternalizedCodeType> args = new ArrayList<>();
        if (l.is(TokenKind.LT)) {
            l.accept(TokenKind.LT);
            args.add(parseExCodeType(l));
            while (l.is(TokenKind.COMMA)) {
                l.accept(TokenKind.COMMA);
                args.add(parseExCodeType(l));
            }
            l.accept(TokenKind.GT);
        }
        return new ExternalizedCodeType(identifier.toString(), args);
    }

    private static String parseExCodeTypeNamePart(Lexer l) {
        String namePart = switch (l.token().kind) {
            case IDENTIFIER -> l.token().name();
            case STRINGLITERAL -> "\"" + l.token().stringVal() + "\"";
            default -> throw l.unexpected();
        };
        l.nextToken();
        return namePart;
    }
}
