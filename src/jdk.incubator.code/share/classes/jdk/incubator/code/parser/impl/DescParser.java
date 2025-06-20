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

package jdk.incubator.code.parser.impl;

import jdk.incubator.code.ExternalizableTypeElement;
import jdk.incubator.code.parser.impl.Tokens.TokenKind;

import java.util.ArrayList;
import java.util.List;

public final class DescParser {
    private DescParser() {}

    /**
     * Parse an externalized type element from its serialized textual form.
     * @param desc the serialized externalized type element
     * @return the externalized type element
     */
    public static ExternalizableTypeElement.ExternalizedTypeElement parseExTypeElem(String desc) {
        Scanner s = Scanner.factory().newScanner(desc);
        s.nextToken();
        return parseExTypeElem(s);
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
    public static ExternalizableTypeElement.ExternalizedTypeElement parseExTypeElem(Lexer l) {
        StringBuilder identifier = new StringBuilder();
        identifier.append(parseExTypeNamePart(l));
        while (l.is(TokenKind.DOT) || l.is(TokenKind.COLON)) {
            identifier.append(l.token().kind.name);
            l.nextToken();
            identifier.append(parseExTypeNamePart(l));
        }
        List<ExternalizableTypeElement.ExternalizedTypeElement> args = new ArrayList<>();
        if (l.is(TokenKind.LT)) {
            l.accept(TokenKind.LT);
            args.add(parseExTypeElem(l));
            while (l.is(TokenKind.COMMA)) {
                l.accept(TokenKind.COMMA);
                args.add(parseExTypeElem(l));
            }
            l.accept(TokenKind.GT);
        }
        return new ExternalizableTypeElement.ExternalizedTypeElement(identifier.toString(), args);
    }

    private static String parseExTypeNamePart(Lexer l) {
        String namePart = switch (l.token().kind) {
            case IDENTIFIER -> l.token().name();
            case STRINGLITERAL -> "\"" + l.token().stringVal() + "\"";
            default -> throw l.unexpected();
        };
        l.nextToken();
        return namePart;
    }
}
