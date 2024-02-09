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

package jdk.code.tools.renderer;

import java.lang.reflect.code.Block;
import java.lang.reflect.code.Body;
import java.lang.reflect.code.op.CoreOps;
import java.lang.reflect.code.Op;
import java.lang.reflect.code.Value;

import java.io.*;
import java.lang.reflect.code.type.JavaType;
import java.nio.charset.StandardCharsets;

/**
 * Created by gfrost
 */
public final class SRRenderer extends CommonRenderer<SRRenderer> {

    static class AttributeMapper {
        static String toString(Object value) {
            if (value instanceof Integer i && i >= 0) {
                return Integer.toString(i);
            } else if (value == null) {
                return "null";
            } else {
                return "\"" + quote(value.toString()) + "\"";
            }
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
        switch (ch) {
            case '\b':  return "\\b";
            case '\f':  return "\\f";
            case '\n':  return "\\n";
            case '\r':  return "\\r";
            case '\t':  return "\\t";
            case '\'':  return "\\'";
            case '\"':  return "\\\"";
            case '\\':  return "\\\\";
            default:
                return (isPrintableAscii(ch))
                        ? String.valueOf(ch)
                        : String.format("\\u%04x", (int) ch);
        }
    }

    /**
     * Is a character printable ASCII?
     */
    static boolean isPrintableAscii(char ch) {
        return ch >= ' ' && ch <= '~';
    }

    SRRenderer() {
        super();
    }

    SRRenderer caretLabelTarget(String name) {
        return caret().labelTarget(name).self();
    }

    SRRenderer atIdentifier(String name) {
        return at().identifier(name).self();
    }

    SRRenderer percentLiteral(String name) {
        return percent().literal(name).self();
    }

    SRRenderer spaceColonSpace() {
        return space().colon().space();
    }

    SRRenderer spaceEqualSpace() {
        return space().equal().space();
    }

    public void write(Op op) {
        GlobalValueBlockNaming gn = new GlobalValueBlockNaming();
        write(gn, op);
        nl();
    }

    public void write(GlobalValueBlockNaming gn, Block.Reference successor) {
        caretLabelTarget(gn.getBlockName(successor.targetBlock()));
        if (!successor.arguments().isEmpty()) {
            oparen().commaSeparatedList();
            for (var a : successor.arguments()) {
                commaSeparator();
                percentLiteral(gn.getValueName(a));
            }
            cparen();
        }

    }

    public void write(GlobalValueBlockNaming gn, Op op) {
        keyword(op.opName());
        if (!op.operands().isEmpty()) {
            space().spaceSeparatedList();
            for (var v : op.operands()) {
                spaceSeparator();
                percentLiteral(gn.getValueName(v));
            }
        }
        if (!op.successors().isEmpty()) {
            space().spaceSeparatedList();
            for (Block.Reference sb : op.successors()) {
                spaceSeparator();
                write(gn, sb);
            }
        }

        if (!op.attributes().isEmpty()) {
            space().spaceSeparatedList();
            for (var e : op.attributes().entrySet()) {
                spaceSeparator();
                String name = e.getKey();
                if (!name.isEmpty()) {
                    atIdentifier(name).equal().identifier(AttributeMapper.toString(e.getValue()));
                } else {
                    atIdentifier(AttributeMapper.toString(e.getValue()));
                }
            }
        }

        if (!op.bodies().isEmpty()) {
            int nBodies = op.bodies().size();
            if (nBodies == 1) {
                space();
            } else {
                nl().in().in();
            }
            // @@@ separated list state does not nest as state.first gets overwritten
            boolean first = true;
            for (Body body : op.bodies()) {
                if (!first) {
                    nl();
                }
                write(gn, body);
                first = false;
            }
            if (nBodies > 1) {
                out().out();
            }
        }

        semicolon();
    }

    public void write(GlobalValueBlockNaming gn, Block block, boolean isEntryBlock) {
        if (!isEntryBlock) {
            caretLabelTarget(gn.getBlockName(block));
            if (!block.parameters().isEmpty()) {
                oparen().commaSpaceSeparatedList();
                for (var v : block.parameters()) {
                    commaSpaceSeparator();
                    writeValueDecl(gn, v);
                }
                cparen();
            }
            colon().nl();
        }
        in();
        for (Op op : block.ops()) {
            Op.Result or = op.result();
            if (!or.type().equals(JavaType.VOID)) {
                writeValueDecl(gn, or);
                spaceEqualSpace();
            }
            write(gn, op);
            nl();
        }
        out();
    }

    public void write(GlobalValueBlockNaming gn, Body body) {
        Block eb = body.entryBlock();
        oparen().commaSpaceSeparatedList();
        for (var v : eb.parameters()) {
            commaSpaceSeparator();
            writeValueDecl(gn, v);
        }
        cparen().type(body.descriptor().returnType().toString()).space().rarrow().space().obrace().nl();
        in();
        boolean isEntryBlock = true;
        for (Block b : body.blocks()) {
            if (!isEntryBlock) {
                nl();
            }
            write(gn, b, isEntryBlock);
            isEntryBlock = false;
        }
        out();
        cbrace();
    }

    public void writeValueDecl(GlobalValueBlockNaming gn, Value v) {
        percentLiteral(gn.getValueName(v)).spaceColonSpace().type(v.type().toString());
    }

    // @@@ Not used
    public void write(GlobalValueBlockNaming gn, CoreOps.FuncOp fRep) {
        this.append(fRep.opName());// w.write(name);
        if (!fRep.operands().isEmpty()) {
            space().spaceSeparatedList();
            for (var v : fRep.operands()) {
                spaceSeparator();
                percentLiteral(gn.getValueName(v));
            }
        }
        if (!fRep.successors().isEmpty()) {
            space().spaceSeparatedList();
            for (Block.Reference sb : fRep.successors()) {
                spaceSeparator();
                write(gn, sb);
            }
        }
        if (!fRep.attributes().isEmpty()) {
            space();
            for (var e : fRep.attributes().entrySet()) {
                String name = e.getKey();
                String value = AttributeMapper.toString(e.getValue());
                op("@");
                if (name.isEmpty()) {
                    literal(value);
                } else {
                    identifier(name).equal().literal(value);
                }
            }
        }
        if (!fRep.bodies().isEmpty()) {
            space().newlineSeparatedList();
            for (Body body : fRep.bodies()) {
                newlineSeparator();
                write(gn, body);
            }
        }

    }

    public static void write(Writer writer, Op op) {
        new SRRenderer().writer(writer).write(op);
    }

    public static void write(OutputStream out, Op op) {
        write(new OutputStreamWriter(out, StandardCharsets.UTF_8), op);
    }

    public static String stringify(Op op) {
        StringWriter sw = new StringWriter();
        write(sw, op);
        return sw.toString();
    }

    public static String colorize(TextRenderer.TokenColorMap tokenColorMap, Op op) {
        StringWriter sw = new StringWriter();
        new SRRenderer().writer(sw).colorize(tokenColorMap).write(op);
        return sw.toString();
    }

    public static String colorize(Op op) {
        return colorize(new TextRenderer.TokenColorMap(), op);
    }
}
