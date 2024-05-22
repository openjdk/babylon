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


import hat.text.CodeBuilder;

import java.lang.reflect.code.Block;
import java.lang.reflect.code.Body;
import java.lang.reflect.code.CodeItem;
import java.lang.reflect.code.Op;
import java.lang.reflect.code.TypeElement;
import java.lang.reflect.code.Value;
//import java.lang.reflect.code.writer.OpWriter;
import java.lang.reflect.code.type.JavaType;
import java.util.HashMap;
import java.util.Map;
import java.util.function.Function;

public abstract class OpCodeBuilder<T extends OpCodeBuilder<T>> extends CodeBuilder<T> {

    static final class GlobalValueBlockNaming implements Function<CodeItem, String> {
        final Map<CodeItem, String> gn;
        int valueOrdinal = 0;
        int blockOrdinal = 0;

        GlobalValueBlockNaming() {
            this.gn = new HashMap<>();
        }

        @Override
        public String apply(CodeItem codeItem) {
            return switch (codeItem) {
                case Block block -> gn.computeIfAbsent(block, _b -> "block_" + blockOrdinal++);
                case Value value -> gn.computeIfAbsent(value, _v -> String.valueOf(valueOrdinal++));
                default -> throw new IllegalStateException("Unexpected code item: " + codeItem);
            };
        }
    }

    static final class AttributeMapper {
        static String toString(Object value) {
          throw new IllegalStateException();
        }
    }

    // Copied from com.sun.tools.javac.util.Convert
    static String quote(String s) {
        StringBuilder buf = new StringBuilder();
        for (int i = 0; i < s.length(); i++) {
            buf.append(quote(s.charAt(i)));
        }
        return buf.toString();
    }

    /**
     * Escapes a character if it has an escape sequence or is
     * non-printable ASCII.  Leaves non-ASCII characters alone.
     */
    static String quote(char ch) {
        return switch (ch) {
            case '\b' -> "\\b";
            case '\f' -> "\\f";
            case '\n' -> "\\n";
            case '\r' -> "\\r";
            case '\t' -> "\\t";
            case '\'' -> "\\'";
            case '\"' -> "\\\"";
            case '\\' -> "\\\\";
            default -> (isPrintableAscii(ch))
                    ? String.valueOf(ch)
                    : String.format("\\u%04x", (int) ch);
        };
    }
    static boolean isPrintableAscii(char ch) {
        return ch >= ' ' && ch <= '~';
    }
    final GlobalValueBlockNaming gn;

    public OpCodeBuilder() {
        this.gn = new GlobalValueBlockNaming();
    }

    public T opName(String name) {
        return append(name);
    }

    T blockName(Block b) {
        return blockName(gn.apply(b));
    }

    T blockName(String s) {
        return hat().append(s);
    }

    T valueName(Value v) {
        return append(gn.apply(v));
    }

    T value(Value v) {
        return percent().valueName(v);
    }

    T typeName(TypeElement typeElement) {
        return append(typeElement.toString());
    }

    T valueDeclaration(Value v) {
        return value(v).space().colon().space().typeName(v.type());
    }

    T attributeName(String name) {
        return append(name);
    }



    T attributeValue(Object value) {
        //what is a null?
      //  if (value == Op.NULL_ATTRIBUTE_VALUE) {
        //    return nullKeyword();
       // } else {
            return dquote(value.toString());
       // }
    }

    T attribute(String name, Object value) {
        return at().when(!name.isEmpty(), _ -> attributeName(name).equals()).attributeValue(value);
    }

    T returnType(Body body) {
        return typeName(body.bodyType().returnType());
    }


    T successor(Block.Reference successor) {
        blockName(successor.targetBlock());
        if (!successor.arguments().isEmpty()) {
            paren(_ -> commaSeparated(successor.arguments(), this::value));
        }
        return self();
    }


    T block(Block block) {
        if (!block.isEntryBlock()) {
            blockName(block);
            zeroOrOneOrMore(block.parameters(),
                    _->{},
                    (one)->{},
                    (params)-> paren(_ -> commaSeparated(params, this::valueDeclaration))
            );
            colon().nl();
        }
        indent(_ ->
                block.ops().forEach(op -> {
                    Op.Result opr = op.result();
                    if (!opr.type().equals(JavaType.VOID)) {
                        valueDeclaration(opr).space().equals().space();
                    }
                    op(op).nl();
                })
        );
        return self();
    }

    T body(Body body) {
        paren(_ -> commaSeparated(body.entryBlock().parameters(), this::valueDeclaration));
        returnType(body).space().rarrow().space();
        return brace(_ -> nl().indent(_ -> nlSeparated(body.blocks(), this::block)));
    }

    public T op(Op op) {
        opName(op.opName());

        op.operands().forEach((operand) -> space().value(operand));

        op.successors().forEach((successor -> space().successor(successor)));


        //op.attributes().forEach((key, value) -> space().attribute(key, value));

        zeroOrOneOrMore(op.bodies(),
                _ -> {/*zero*/},
                this::body,
                (more) -> {
                    nl().indent(_ -> {
                        indent(_ -> {
                            nlSeparated(more, this::body);
                        });
                    });
                });

        return semicolon();
    }

}


