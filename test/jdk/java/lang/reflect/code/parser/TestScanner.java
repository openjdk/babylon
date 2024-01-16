/*
 * Copyright (c) 2024, Oracle and/or its affiliates. All rights reserved.
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * This code is free software; you can redistribute it and/or modify it
 * under the terms of the GNU General Public License version 2 only, as
 * published by the Free Software Foundation.
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

import org.testng.Assert;
import org.testng.annotations.DataProvider;
import org.testng.annotations.Test;

import java.lang.reflect.code.parser.impl.Scanner;
import java.lang.reflect.code.parser.impl.Tokens;
import java.util.ArrayList;
import java.util.List;

import static java.lang.reflect.code.parser.impl.Tokens.TokenKind.*;

/*
 * @test
 * @modules java.base/java.lang.reflect.code.parser.impl
 * @run testng TestScanner
 */

public class TestScanner {

    @DataProvider
    Object[][] data() {
        return new Object[][] {
                {"java.lang.Integer", List.of(IDENTIFIER, DOT, IDENTIFIER, DOT, IDENTIFIER)},
                {"java.lang.Integer", List.of("java", DOT, "lang", DOT, "Integer")},
                {"a<a<a, a>,a<a, a>>", List.of("a", LT,
                        "a", LT, "a", COMMA, "a", GT,
                        COMMA,
                        "a", LT, "a", COMMA, "a", GT,
                        GT)},
                {"_->(){}[],.=><?:;+-&^@", List.of(
                        UNDERSCORE,
                        ARROW,
                        LPAREN,
                        RPAREN,
                        LBRACE,
                        RBRACE,
                        LBRACKET,
                        RBRACKET,
                        COMMA,
                        DOT,
                        EQ,
                        GT,
                        LT,
                        QUES,
                        COLON,
                        SEMI,
                        PLUS,
                        SUB,
                        AMP,
                        CARET,
                        MONKEYS_AT
                        )},
                {"%1 %a %_1", List.of(VALUE_IDENTIFIER, VALUE_IDENTIFIER, VALUE_IDENTIFIER)},
                {"%1 %a %_1", List.of("%1", "%a", "%_1")},
                {"\"abc\\n\"", List.of(STRINGLITERAL)},
                {"\"abc \\b \\f \\n \\r \\t \\' \\\" \\\\\"", List.of("abc \b \f \n \r \t \' \" \\")},
        };
    }

    @Test(dataProvider = "data")
    public void test(String content, List<Object> expectedTokens) {
        Scanner.Factory factory = Scanner.factory();

        Scanner s = factory.newScanner(content);
        s.nextToken();
        List<Tokens.Token> actualTokens = new ArrayList<>();
        while (s.token().kind != EOF) {
            actualTokens.add(s.token());
            s.nextToken();
        }

        Assert.assertEquals(actualTokens.size(), expectedTokens.size());
        for (int i = 0; i < expectedTokens.size(); i++) {
            Object e = expectedTokens.get(i);
            Tokens.Token a = actualTokens.get(i);
            if (e instanceof Tokens.TokenKind t) {
                Assert.assertEquals(a.kind, t);
            } else if (e instanceof String v) {
                String as = switch (a.kind.tag) {
                    case NAMED -> a.name();
                    case STRING, NUMERIC -> a.stringVal();
                    case DEFAULT -> a.kind.name;
                };
                Assert.assertEquals(as, v);
            } else {
                assert false;
            }
        }
    }
}
