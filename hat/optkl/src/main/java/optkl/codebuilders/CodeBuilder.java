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
package optkl.codebuilders;

import jdk.incubator.code.Op;
import jdk.incubator.code.dialect.core.CoreOp;
import jdk.incubator.code.dialect.java.JavaOp;
import optkl.util.StreamMutable;

import java.util.function.Consumer;
import java.util.stream.Stream;

/**
 * Extends the base TextBuilder to add common constructs/keywords for generating C99/Java style code.
 *
 * @author Gary Frost
 */
public abstract class CodeBuilder<T extends CodeBuilder<T>> extends TextBuilder<T> implements CodeRenderer<T> {

    public T semicolon() {
        return symbol(";");
    }

    public T semicolonNl() {
        return semicolon().nl();
    }

    public T comma() {
        return symbol(",");
    }

    final public T commaSpace() {
        return comma().space();
    }

    public T tilde() {
        return symbol("~");
    }

    public T dot() {
        return symbol(".");
    }

    public T leftShift() {
        return symbol("<<");
    }

    public T rightShift() {
        return symbol(">>");
    }

    public T rightShift(int v) {
        return rightShift().intValue(v);
    }

    public T leftShift(int v) {
        return leftShift().intValue(v);
    }

    public T equals() {
        return symbol("=");
    }

    public T assign() {
        return space().equals().space();
    }

    public T dollar() {
        return symbol("$");
    }

    public T plusplus() {
        return symbol("++");
    }


    public T minusminus() {
        return symbol("--");
    }

    public T lineComment(String line) {
        return comment("//").space().comment(line).nl();
    }

    @Override
    public T constant(String text) {
        return emitText(text);
    }


    public T blockComment(String block) {
        return comment("/*").nl().comment(block).nl().symbol("*/").nl();
    }

    public T blockInlineComment(String block) {
        return comment("/*").space().comment(block).space().comment("*/");
    }

    public T newKeyword() {
        return keyword("new");
    }


    public T staticKeyword() {
        return keyword("static");
    }

    public T constexprKeyword() {
        return keyword("constexpr");
    }

    public T constKeyword() {
        return keyword("const");
    }

    public T explicitKeyword() {
        return keyword("explicit");
    }

    public T virtualKeyword() {
        return keyword("virtual");
    }

    public T ifKeyword() {
        return keyword("if");
    }

    public T whileKeyword() {
        return keyword("while");
    }


    public T breakKeyword() {
        return keyword("break");
    }

    public T gotoKeyword() {
        return keyword("goto");
    }

    public T continueKeyword() {
        return keyword("continue");
    }


    public T colon() {
        return symbol(":");
    }


    public T nullConst() {
        return symbol("NULL");
    }


    public T elseKeyword() {
        return keyword("else");
    }


    public T returnKeyword() {
        return keyword("return");
    }

    public T returnKeyword(String identifier) {
        return returnKeyword().space().identifier(identifier);
    }

    public T switchKeyword() {
        return keyword("switch");
    }


    public T caseKeyword() {
        return keyword("case");
    }


    public T defaultKeyword() {
        return keyword("default");
    }

    public T doKeyword() {
        return keyword("do");
    }

    public T forKeyword() {
        return keyword("for");
    }

    public T ampersand() {
        return symbol("&");
    }

    public T addressOf(String identifier) {
        return ampersand().identifier(identifier);
    }

    public T asterisk() {
        return symbol("*");
    }

    public T dereference(String identifier) {
        return asterisk().identifier(identifier);
    }

    public T mul() {
        return asterisk();
    }

    public T percent() {
        return symbol("%");
    }

    public T mod() {
        return percent();
    }

    public T slash() {
        return symbol("/");
    }

    public T div() {
        return slash();
    }

    public T plus() {
        return symbol("+");
    }

    public T add() {
        return plus();
    }

    public T minus() {
        return symbol("-");
    }

    public T sub() {
        return minus();
    }

    public T lt() {
        return symbol("<");
    }

    public T eq() {
        return equals().equals();
    }

    public T lte() {
        return lt().equals();
    }

    public T gte() {
        return gt().equals();
    }

    public T pling() {
        return symbol("!");
    }

    public T gt() {
        return symbol(">");
    }

    public T condAnd() {
        return symbol("&&");
    }

    public T condOr() {
        return symbol("||");
    }

    public T oparen() {
        return symbol("(");
    }

    public final T paren(Consumer<T> consumer) {
        return oparen().accept(consumer).cparen();
    }

    public T ocparen() {
        return oparen().cparen();
    }

    public T parenWhen(boolean value, Consumer<T> consumer) {
        if (value) {
            oparen().accept(consumer).cparen();
        } else {
            accept(consumer);
        }
        return self();
    }

    public T semicolonTerminated(Consumer<T> consumer) {
        return accept(consumer).semicolon();
    }

    public T semicolonNlTerminated(Consumer<T> consumer) {
        return semicolonTerminated(consumer).nl();
    }

    public T obrace() {
        return symbol("{");
    }

    public T indent(Consumer<T> ct) {
        return in().accept(ct).out();
    }

    public T nlIndentNl(Consumer<T> ct) {
        return nl().indent(ct).nl();
    }

    public T braceNlIndented(Consumer<T> ct) {
        return obrace().nlIndentNl(ct).cbrace();
    }

    public T parenNlIndented(Consumer<T> ct) {
        return oparen().nlIndentNl(ct).cparen();
    }

    public T brace(Consumer<T> ct) {
        return obrace().indent(ct).cbrace();
    }

    public T ocsbrace() {
        return osbrace().csbrace();
    }

    public T ocbrace() {
        return obrace().cbrace();
    }

    public T sbrace(Consumer<T> ct) {
        return osbrace().accept(ct).csbrace();
    }

    public T accept(Consumer<T> ct) {
        ct.accept(self());
        return self();
    }


    public T ochevron() {
        return rawochevron();
    }

    final public T rawochevron() {
        return emitText("<");
    }

    public T bar() {
        return symbol("|");
    }

    public T cchevron() {
        return rawcchevron();
    }

    public T chevron(Consumer<T> ct) {
        return rawochevron().indent(ct).rawcchevron();
    }

    final public T rawcchevron() {
        return emitText(">");
    }

    public T osbrace() {
        return symbol("[");
    }

    public T cparen() {
        return symbol(")");
    }

    public T cbrace() {
        return symbol("}");
    }


    public T csbrace() {
        return symbol("]");
    }

    public T underscore() {
        return symbol("_");
    }

    public T dquote() {
        return symbol("\"");
    }

    public T odquote() {
        return dquote();
    }

    public T cdquote() {
        return dquote();
    }

    public T squote() {
        return symbol("'");
    }

    public T osquote() {
        return squote();
    }

    public T csquote() {
        return squote();
    }

    public T dquote(String string) {
        return odquote().escaped(string).cdquote();
    }

    public T at() {
        return symbol("@");
    }

    public T hat() {
        return symbol("^");
    }

    public T squote(String txt) {
        return osquote().escaped(txt).csquote();
    }

    public T rarrow() {
        return symbol("->");
    }

    public T larrow() {
        return symbol("<-");
    }


    public T questionMark() {
        return symbol("?");
    }

    public T hash() {
        return symbol("#");
    }

    public T when(boolean c, Consumer<T> consumer) {
        if (c) {
            accept(consumer);
        }
        return self();
    }

    public T either(boolean c, Consumer<T> lhs, Consumer<T> rhs) {
        if (c) {
            accept(lhs);
        } else {
            accept(rhs);
        }
        return self();
    }

    public <I> T separated(Iterable<I> iterable, Consumer<T> separator, Consumer<I> consumer) {
        var first = StreamMutable.of(true);
        iterable.forEach(t -> {
            if (first.get()) {
                first.set(false);
            } else {
                separator.accept(self());
            }
            consumer.accept(t);
        });
        return self();
    }

    public <I> T commaSpaceSeparated(Iterable<I> iterable, Consumer<I> consumer) {
        return separated(iterable, _ -> commaSpace(), consumer);
    }

    public T commaSpaceSeparated(Consumer<T>... consumers) {
        for (int i = 0; i < consumers.length; i++) {
            if (i > 0) {
                commaSpace();
            }
            consumers[i].accept(self());
        }
        return self();
    }

    public T args(Consumer<T>... consumers) {
        return commaSpaceSeparated(consumers);
    }

    public <I> T commaSeparated(Iterable<I> iterable, Consumer<I> consumer) {
        return separated(iterable, _ -> comma(), consumer);
    }

    public <I> T commaNlSeparated(Iterable<I> iterable, Consumer<I> consumer) {
        return separated(iterable, _ -> comma().nl(), consumer);
    }

    public <I> T barSeparated(Iterable<I> iterable, Consumer<I> consumer) {
        return separated(iterable, _ -> bar(), consumer);
    }

    public <I> T semicolonNlSeparated(Iterable<I> iterable, Consumer<I> consumer) {
        return separated(iterable, _ -> semicolonNl(), consumer);
    }

    public <I> T nlSeparated(Iterable<I> iterable, Consumer<I> consumer) {
        return separated(iterable, _ -> nl(), consumer);
    }

    public <I> T separated(Stream<I> stream, Consumer<T> separator, Consumer<I> consumer) {
        var first = StreamMutable.of(true);
        stream.forEach(t -> {
            if (first.get()) {
                first.set(false);
            } else {
                separator.accept(self());
            }
            consumer.accept(t);
        });
        return self();
    }

    public <I> T commaSpaceSeparated(Stream<I> stream, Consumer<I> consumer) {
        return separated(stream, _ -> commaSpace(), consumer);
    }

    public <I> T commaSeparated(Stream<I> stream, Consumer<I> consumer) {
        return separated(stream, _ -> comma(), consumer);
    }

    public <I> T nlSeparated(Stream<I> stream, Consumer<I> consumer) {
        return separated(stream, _ -> nl(), consumer);
    }

    public final T s32Type() {
        return typeName("int");
    }

    public final T s32Type(String identifier) {
        return s32Type().space().identifier(identifier);
    }

    public final T intConstZero() {
        return constant("0");
    }

    public final T intConstOne() {
        return constant("1");
    }

    public final T intConstTwo() {
        return constant("2");
    }

    public final T voidType() {
        return typeName("void");
    }

    public final T s08Type() {
        return typeName("char");
    }

    public final T s08Type(String name) {
        return s08Type().space().identifier(name);
    }

    public final T f32Type() {
        return typeName("float");
    }

    public final T f32Type(String identifier) {
        return f32Type().space().identifier(identifier);
    }

    public final T s64Type() {
        return typeName("long");
    }

    public final T f64Type() {
        return typeName("double");
    }

    public final T boolType() {
        return typeName("char");
    }

    public final T s16Type() {
        return typeName("short");
    }

    public final T s16Type(String identifier) {
        return s16Type().space().identifier(identifier);
    }


    @Override
    public final T comment(String text) {
        return emitText(text);
    }

    @Override
    public T identifier(String text) {
        return emitText(text);
    }

    @Override
    public T reserved(String text) {
        return emitText(text);
    }

    @Override
    public T label(String text) {
        return emitText(text);
    }

    @Override
    public final T symbol(String text) {
        return emitText(text);
    }

    @Override
    public final T typeName(String text) {
        return emitText(text);
    }

    @Override
    public final T keyword(String text) {
        return emitText(text);
    }

    @Override
    public final T literal(String text) {
        return emitText(text);
    }

    @Override
    public T nl() {
        return super.nl();
    }

    @Override
    public T space() {
        return emitText(" ");
    }

    public T builtin(String text) {
        return emitText(text);
    }

    public T composeIdentifier(String preffix, String postfix) {
        return identifier(preffix + postfix);
    }

    public String toCamelExceptFirst(String s) {
        String[] parts = s.split("_");
        StringBuilder camelCaseString = new StringBuilder();
        for (String part : parts) {
            camelCaseString.append(camelCaseString.isEmpty()
                    ? part.toLowerCase()
                    : part.substring(0, 1).toUpperCase() + part.substring(1).toLowerCase());
        }
        return camelCaseString.toString();
    }

    public final T sizeArray(int size) {
        return sbrace( _ -> constant(Integer.toString(size)));
    }

    public final T oracleCopyright(){
        return blockComment("""
                * Copyright (c) 2025, Oracle and/or its affiliates. All rights reserved.
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
                * questions."""
        );
    }


    public final T varName(CoreOp.VarOp varOp) {
        return identifier(varOp.varName());
    }
    public final  T funcName(CoreOp.FuncCallOp funcCallOp){
        return identifier(funcCallOp.funcName());
    }
    public final T funcName(CoreOp.FuncOp funcOp) {
        return identifier(funcOp.funcName());
    }
    public final T fieldName(JavaOp.FieldAccessOp fieldAccessOp) {
        return identifier(fieldAccessOp.fieldDescriptor().name());
    }
    public final T funcName(JavaOp.InvokeOp invokeOp){
        return identifier(invokeOp.invokeDescriptor().name());
    }


    protected final T camel(String value) {
        return identifier(Character.toString(Character.toLowerCase(value.charAt(0)))).identifier(value.substring(1));
    }

    public final T camelJoin(String prefix, String suffix) {
        return camel(prefix).identifier(Character.toString(Character.toUpperCase(suffix.charAt(0)))).identifier(suffix.substring(1));
    }

    public T symbol(Op op) {
        return switch (op) {
            case JavaOp.ModOp o -> percent();
            case JavaOp.MulOp o -> mul();
            case JavaOp.DivOp o -> div();
            case JavaOp.AddOp o -> add();
            case JavaOp.SubOp o -> sub();
            case JavaOp.LtOp o -> lt();
            case JavaOp.GtOp o -> gt();
            case JavaOp.LeOp o -> lte();
            case JavaOp.GeOp o -> gte();
            case JavaOp.AshrOp o -> cchevron().cchevron();
            case JavaOp.LshlOp o -> ochevron().ochevron();
            case JavaOp.LshrOp o -> cchevron().cchevron();
            case JavaOp.NeqOp o -> pling().equals();
            case JavaOp.NegOp o -> minus();
            case JavaOp.EqOp o -> equals().equals();
            case JavaOp.NotOp o -> pling();
            case JavaOp.AndOp o -> ampersand();
            case JavaOp.OrOp o -> bar();
            case JavaOp.XorOp o -> hat();
            case JavaOp.ConditionalAndOp o -> condAnd();
            case JavaOp.ConditionalOrOp o -> condOr();
            default -> throw new IllegalStateException("Unexpected value: " + op);
        };
    }
}
