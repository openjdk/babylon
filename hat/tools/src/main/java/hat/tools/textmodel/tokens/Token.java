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

import java.util.Optional;
import java.util.function.Consumer;

public interface Token extends Span {

    Parent parent();

    Pos pos();

    default int depth() {
        return (parent() != null) ? parent().depth() + 1 : 0;
    }

    int len();
    @Override
    default int startOffset() {
        return (pos().textOffset());
    }

    @Override
    default int endOffset() {
        return startOffset() + len();
    }

    default String asString() {
        return new String(pos().text(), pos().textOffset(), Math.min(len(), pos().text().length - pos().textOffset()));
    }
    default boolean next(Parent.TknPredicate1<Token> predicate) {
        var opt = next();
        if (opt.isPresent()) {
            var o= opt.get();
            return predicate.test(o);
        }
        return false;
    }
    default boolean next2(Parent.TknPredicate2<Token> predicate) {
        var opt1 = next();
        if (opt1.isPresent()) {
            var opt2 = opt1.get().next();
            if (opt2.isPresent()) {
                return   predicate.test(opt1.get(), opt2.get());
            }
        }
        return false;
    }
    default boolean prev(Parent.TknPredicate1 predicate) {
        var opt = prev();
        if (opt.isPresent()) {
            return predicate.test(opt.get());
        }else  {
            return false;
        }
    }
    default Optional<Token> next(){
        return parent().nextSiblingOf(this);
    }
    default Optional<Token> prev(){
        return parent().prevSiblingOf(this);
    }

    default boolean is(String s) {
        var s1= asString();
        return s1.equals(s);
    }

    void visit(Consumer<Token> visitor);

}
