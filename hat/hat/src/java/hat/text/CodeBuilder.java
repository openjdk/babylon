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
package hat.text;


import hat.util.StreamCounter;

import java.util.Collection;
import java.util.function.Consumer;

/**
 * Extends the base TextBuilder to add common constructs/keywords for generating code.
 *
 * @author Gary Frost
 */
public abstract class CodeBuilder<T extends CodeBuilder<T>> extends TextBuilder<T> {

    public T semicolon() {
        return symbol(";");
    }

    public T semicolonIf(boolean c) {
        if (c) {
            return semicolon();
        } else {
            return self();
        }
    }

    public T semicolonNl() {
        return semicolon().nl();
    }

    public T commaIf(boolean c) {
        if (c) {
            return comma();
        } else {
            return self();
        }
    }

    public T commaSpaceIf(boolean c) {
        if (c) {
            return comma().space();
        } else {
            return self();
        }
    }

    public T nlIf(boolean c) {
        if (c) {
            return nl();
        } else {
            return self();
        }
    }

    public T comma() {
        return symbol(",");
    }


    public T dot() {
        return symbol(".");
    }


    public T equals() {
        return symbol("=");
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
        return symbol("//").space().commented(line).nl();
    }


    public T blockComment(String block) {
        return symbol("/*").nl().commented(block).nl().symbol("*/").nl();
    }

    public T blockInlineComment(String block) {
        return symbol("/*").space().commented(block).space().symbol("*/").space();
    }

    public T newKeyword() {
        return keyword("new");
    }


    public T staticKeyword() {
        return keyword("static");
    }


    public T constKeyword() {
        return keyword("const");
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


    public T nullKeyword() {
        return symbol("NULL");
    }


    public T elseKeyword() {
        return keyword("else");
    }


    public T returnKeyword() {
        return keyword("return");
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

    public T asterisk() {
        return symbol("*");
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

    public T dqattr(String name, String value) {
        return append(name).equals().dquote(value);

    }

    public T sqattr(String name, String value) {
        return append(name).equals().squote(value);

    }

    public T attr(String name, String value) {
        return append(name).equals().append(value);
    }

    public T attr(String name, Integer value) {
        return append(name).equals().append(value.toString());
    }

    public T attr(String name, Float value) {
        return append(name).equals().append(value.toString());
    }

    public T oparen() {
        return symbol("(");
    }

    public final T paren(Consumer<T> consumer) {
        return oparen().accept(consumer).cparen();
    }

    public T parenWhen(boolean value, Consumer<T> consumer) {
        if (value) {
            oparen().accept(consumer).cparen();
        } else {
            accept(consumer);
        }
        return self();
    }

    public T line(Consumer<T> consumer) {
        return accept(consumer).nl();
    }

    public T semicolonTerminatedLine(Consumer<T> consumer) {
        return semicolonTerminatedLineNoNl(consumer).nl();
    }

    public T semicolonTerminatedLineNoNl(Consumer<T> consumer) {
        return accept(consumer).semicolon();
    }

    public T obrace() {
        return symbol("{");
    }

    public T braceNlIndented(Consumer<T> ct) {
        return obrace().nl().indent(ct).nl().cbrace();
    }

    public T parenNlIndented(Consumer<T> ct) {
        return oparen().nl().indent(ct).nl().cparen();
    }

    public T brace(Consumer<T> ct) {
        return obrace().indent(ct).cbrace();
    }

    public T sbrace(Consumer<T> ct) {
        return osbrace().accept(ct).csbrace();
    }

    public T accept(Consumer<T> ct) {
        ct.accept(self());
        return self();
    }

    public T indent(Consumer<T> ct) {
        return in().accept(ct).out();
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
        return osquote().append(txt).csquote();
    }

    public T rarrow() {
        return symbol("->");
    }

    public T larrow() {
        return symbol("<-");
    }


    public T u08_t() {
        return typeName("u08_t");
    }

    public T s08_t() {
        return typeName("s08_t");
    }

    public T s32_t() {
        return typeName("s32_t");
    }

    public T s16_t() {
        return typeName("s16_t");
    }

    public T z8_t() {
        return typeName("z8_t");
    }

    public T u32_t() {
        return typeName("u32_t");
    }

    public T u16_t() {
        return typeName("u16_t");
    }

    public T f32_t() {
        return typeName("f32_t");
    }

    public T f64_t() {
        return typeName("f64_t");
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


    public <I> T zeroOrOneOrMore(Collection<I> collection, Consumer<T> zero, Consumer<I> one, Consumer<Iterable<I>> more) {
        if (collection == null || collection.isEmpty()) {
            zero.accept(self());
        } else if (collection.size() == 1) {
            one.accept(collection.iterator().next());
        } else {
            more.accept(collection);
        }
        return self();
    }


    public <I> T commaSeparated(Iterable<I> iterable, Consumer<I> c) {
        StreamCounter.of(iterable, (counter, t) -> {
            if (counter.isNotFirst()) {
                comma().space();
            }
            c.accept(t);
        });
        return self();
    }

    public <I> T nlSeparated(Iterable<I> iterable, Consumer<I> c) {
        StreamCounter.of(iterable, (countStream, t) -> {
            if (countStream.isNotFirst()) {
                nl();
            }
            c.accept(t);
        });
        return self();
    }

    public static class ConcreteCodeBuilder extends CodeBuilder<ConcreteCodeBuilder> {
    }

    public static ConcreteCodeBuilder concreteCodeBuilder() {
        return new ConcreteCodeBuilder();
    }

    public final T intType() {
        return append("int");
    }

    public final T intZero() {
        return append("0");
    }


    public final T voidType() {
        return typeName("void");
    }

    public final T charType() {
        return typeName("char");
    }


    public final T floatType() {
        return typeName("float");
    }

    public final T longType() {
        return typeName("long");
    }

    public final T doubleType() {
        return typeName("double");
    }

    public final T booleanType() {
        return typeName("char");
    }


    public final T shortType() {
        return typeName("short");
    }
}
