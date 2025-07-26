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
package hat.tools.textmodel.tokens;

import hat.tools.textmodel.Cursor;

import java.util.ArrayList;
import java.util.List;
import java.util.Optional;
import java.util.function.Consumer;
import java.util.stream.Stream;

public interface Parent extends Token {
    List<Token> children();

    default Token child(int i) {
        return children().get(i);
    }

    default Optional<Token> nextSiblingOf(Token token) {
        int nextIdx = children().indexOf(token) + 1;
        if (nextIdx < children().size()) {
            return Optional.of(children().get(nextIdx));
        } else {
            return Optional.empty();
        }

    }

    default Optional<Token> prevSiblingOf(Token token) {
        int prevIdx = children().indexOf(token) - 1;
        if (prevIdx >= 0) {
            return Optional.of(children().get(prevIdx));
        } else {
            return Optional.empty();
        }
    }

    interface TokenConsumer<T extends Token> {
    }

    interface TokenReplacer<T extends Token> {
    }

    interface TokenPredicate<T extends Token> {
    }

    interface TknPredicate1<T extends Token> extends TokenPredicate<T> {
        boolean test(T l);
    }

    interface TknReplacer1<T extends Token> extends TokenReplacer<T> {
        T replace(T l);
    }

    interface TknConsumer1<T extends Token> extends TokenConsumer<T> {
        void accept(T l);
    }

    interface TknPredicate3<T extends Token> extends TokenPredicate<T> {
        boolean test(T l, T m, T r);
    }

    interface TknReplacer3<T extends Token> extends TokenReplacer<T> {
        T replace(T l, T m, T r);
    }

    interface TknConsumer3<T extends Token> extends TokenConsumer<T> {
        void accept(T l, T m, T r);
    }

    interface TknPredicate2<T extends Token> extends TokenPredicate<T> {
        boolean test(T l, T r);
    }

    interface TknReplacer2<T extends Token> extends TokenReplacer<T> {
        T replace(T l, T r);
    }

    interface TknConsumer2<T extends Token> extends TokenConsumer<T> {
        void accept(T l, T r);
    }


    interface TknPredicate4<T extends Token> extends TokenPredicate<T> {
        boolean test(T l, T lm, T rm, T r);
    }

    interface TknReplacer4<T extends Token> extends TokenReplacer<T> {
        T replace(T l, T lm, T rm, T r);
    }

    interface TknConsumer4<T extends Token> extends TokenConsumer<T> {
        void accept(T l, T lm, T rm, T r);
    }

    default Parent find(boolean recurse, TokenPredicate<Token> predicate, TokenConsumer<Token> consumer, int count) {
        for (int i = 0; i < (children().size() - count); i++) {
            if (switch (predicate) {
                case TknPredicate1<Token> p -> p.test(child(i));
                case TknPredicate2<Token> p -> p.test(child(i), child(i + 1));
                case TknPredicate3<Token> p -> p.test(child(i), child(i + 1), child(i + 2));
                case TknPredicate4<Token> p -> p.test(child(i), child(i + 1), child(i + 2), child(i + 3));
                default -> throw new IllegalStateException("Unexpected value: " + predicate);
            }) {
                switch (consumer) {
                    case TknConsumer1<Token> p -> p.accept(child(i));
                    case TknConsumer2<Token> p -> p.accept(child(i), child(i + 1));
                    case TknConsumer3<Token> p -> p.accept(child(i), child(i + 1), child(i + 2));
                    case TknConsumer4<Token> p -> p.accept(child(i), child(i + 1), child(i + 2), child(i + 3));
                    default -> throw new IllegalStateException("Unexpected value: " + predicate);
                }
            }
        }
        if (recurse) {
            children().stream().filter(p -> p instanceof Parent).map(Parent.class::cast).forEach(child ->
                    child.find(recurse, predicate, consumer, count)
            );
        }
        return this;
    }
    default Stream<Token> find( TknPredicate1<Token> predicate) {
        var collected = new ArrayList<Token>();
        visit(c -> {
            if (predicate.test(c)){
                collected.add(c);
            }
        });
        return collected.stream();
    }
    default void find(boolean recurse, TknPredicate1<Token> predicate, TknConsumer1<Token> consumer) {
        find(recurse, predicate, consumer, 1);
    }

    default void find(boolean recurse, TknPredicate2<Token> predicate, TknConsumer2<Token> consumer) {
        find(recurse, predicate, consumer, 2);
    }

    default void find(boolean recurse, TknPredicate3<Token> predicate, TknConsumer3<Token> consumer) {
        find(recurse, predicate, consumer, 3);
    }

    default void find(boolean recurse, TknPredicate4<Token> predicate, TknConsumer4<Token> consumer) {
        find(recurse, predicate, consumer, 4);
    }

    default Parent replace(boolean recurse, TokenPredicate<Token> predicate, TokenReplacer<Token> replacement, int count) {
        int i = 0;
        while (i < (children().size() - count)) {
            if (switch (predicate) {
                case TknPredicate1<Token> p -> p.test(child(i));
                case TknPredicate2<Token> p -> p.test(child(i), child(i + 1));
                case TknPredicate3<Token> p -> p.test(child(i), child(i + 1), child(i + 2));
                case TknPredicate4<Token> p -> p.test(child(i), child(i + 1), child(i + 2), child(i + 3));
                default -> throw new IllegalStateException("Unexpected value: " + predicate);
            }) {
                children().set(i, switch (replacement) {
                    case TknReplacer1<Token> p -> p.replace(child(i));
                    case TknReplacer2<Token> p -> p.replace(child(i), child(i + 1));
                    case TknReplacer3<Token> p -> p.replace(child(i), child(i + 1), child(i + 2));
                    case TknReplacer4<Token> p -> p.replace(child(i), child(i + 1), child(i + 2), child(i + 3));
                    default -> throw new IllegalStateException("Unexpected value: " + predicate);
                });
                for (int n = 0; n < (count - 1); n++) {
                    children().remove(i + 1);
                }
            } else {
                i++;
            }
        }
        if (recurse) {
            children().stream().filter(p -> p instanceof Parent).map(Parent.class::cast).forEach(child ->
                    child.replace(recurse, predicate, replacement, count)
            );
        }
        return this;
    }

    default void replace(boolean recurse, TknPredicate1<Token> predicate, TknReplacer1<Token> replacement) {
        replace(recurse, predicate, replacement, 1);
    }

    default void replace(boolean recurse, TknPredicate2<Token> predicate, TknReplacer2<Token> replacement) {
        replace(recurse, predicate, replacement, 2);
    }

    default void replace(boolean recurse, TknPredicate3<Token> predicate, TknReplacer3<Token> replacement) {
        replace(recurse, predicate, replacement, 3);
    }

    default void replace(boolean recurse, TknPredicate4<Token> predicate, TknReplacer4<Token> replacement) {
        replace(recurse, predicate, replacement, 4);
    }


    default Token add(Token token) {
        children().add(token);
        return token;
    }


    @Override
    default void visit(Consumer<Token> visitor) {
        children().forEach(c ->
                c.visit(visitor)
        );
    }

    @Override
    default int len() {
        if (children().isEmpty()) {
            return 0;
        } else {
            var last = children().getLast();
            return last.startOffset() + last.len() - pos().textOffset();
        }
    }

    default Factory factory() {
        if (this instanceof Factory) {
            return (Factory) this;
        } else {
            return parent().factory();
        }
    }

    default Root root() {
        if (this instanceof Root) {
            return (Root) this;
        } else {
            return parent().root();
        }
    }

    default void parse(Cursor c) {
        while (c.next() instanceof Cursor.Loc loc && loc.ch() instanceof Character) {
            if (loc.ch() == '\n') {
                add(factory().nl(this, loc.pos()));
            } else if (loc.ch() == ' ') {
                Cursor.Loc start = loc;
                while (loc.peek() instanceof Character peeked && peeked == ' ') {
                    loc = c.next();
                }
                int len = loc.delta(start) + 1;
                add(factory().ws(this, start.pos(), len));
            } else if (root().opensWith(loc.ch()) instanceof Parenthesis.OpenClose openClose) {
                Parenthesis parenthesis = (Parenthesis) add(factory().parenthesis(this, loc.pos(), openClose));
                parenthesis.add(factory().open(parenthesis, loc.pos(), loc.ch()));
                parenthesis.parse(c);
            } else if (this instanceof Parenthesis && root().closedBy(loc.ch()) instanceof Parenthesis.OpenClose) {
                add(factory().close(this, loc.pos(), loc.ch()));
                return;
            } else if (loc.ch() == '/' && loc.peek() instanceof Character peeked && peeked == '/') {
                while (c.next() instanceof Cursor.Loc next && next.ch() instanceof Character) {
                    if (next.ch() == '\n') {
                        add(factory().lineComment(this, loc.pos(), next.delta(loc) + 1));
                        break;
                    }
                }
            } else if (loc.ch() == '\'') {
                boolean escaping = false;
                boolean done = false;
                while (!done) {
                    var slitLoc = c.next();
                    if (slitLoc != null) {
                        if (slitLoc.ch() == '\'' && !escaping) {
                            add(factory().charLiteral(this, loc.pos(), (slitLoc.textOffset() - loc.textOffset()) + 1));
                            done = true;
                        } else escaping = slitLoc.ch() == '\\' && !escaping;
                    } else {
                        throw new IllegalStateException(" eof before end of literal");
                    }
                }
            } else if (loc.ch() == '"') {
                boolean escaping = false;
                boolean done = false;
                while (!done) {
                    var slitLoc = c.next();
                    if (slitLoc != null) {
                        if (slitLoc.ch() == '"' && !escaping) {
                            add(factory().stringLiteral(this, loc.pos(), (slitLoc.textOffset() - loc.textOffset()) + 1));
                            done = true;
                        } else escaping = slitLoc.ch() == '\\' && !escaping;
                    } else {
                        throw new IllegalStateException(" eof before end of literal");
                    }
                }
            } else if (loc.ch() == '/' && loc.peek() instanceof Character peeked && peeked == '*') {
                boolean done = false;
                while (!done) {
                    while (c.next() instanceof Cursor.Loc asteriskFinder && asteriskFinder.ch() != '*') {
                    }
                    if (c.next() instanceof Cursor.Loc slashFinder && slashFinder.ch() == '/') {
                        done = true;
                        add(factory().multiLineComment(this, loc.pos(), (slashFinder.textOffset() - loc.textOffset()) + 1));
                    }
                }
            } else if (Character.isJavaIdentifierPart(loc.ch())) {  // we don't use isJavaItendifierStart because we can slurp ints here too.
                Cursor.Loc start = loc;
                while (loc.peek() instanceof Character peeked && Character.isJavaIdentifierPart(peeked)) {
                    loc = c.next();
                }
                add(factory().seq(this, start.pos(), loc.delta(start) + 1));
            } else {
                add(factory().ch(this, loc.pos()));
            }
        }
    }
}
