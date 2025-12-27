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
package optkl.util;

import java.util.function.BiFunction;
import java.util.function.Function;
import java.util.function.Predicate;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

public record Regex(Function<Match,Match> factory,Pattern pattern) {
    public interface Result {
    }

    public interface Match extends Result {
        Regex regex();
        Matcher matcher();
        default  String stringOf(int idx) {
            return matcher().group(idx);
        }

        default float floatOf(int idx) {
            return Float.parseFloat(stringOf(idx));
        }

        default  int intOf(int idx) {
            return Integer.parseInt(stringOf(idx));
        }
        default  int[] intArrayOf(int from, int count) {
            int[] ints = new int[count];
            for (int i = 0; i<count; i++) {
                ints[i]= Integer.parseInt(stringOf(from + i));
            }
            return ints;
        }
        default  float[] floatArrayOf(int from, int count) {
            float[] floats = new float[count];
            for (int i = 0; i<count; i++) {
                floats[i]=Float.parseFloat(stringOf(from + i));
            }
            return floats;
        }
        default int count(){
            return matcher().groupCount();
        }
    }

    interface FAIL extends Result {
        static FAIL of() {
            return new FAIL(){};
        }
    }

    public static Regex of(String... strings) {
        return of(m->m, strings);
    }
    public static Regex of(Function<Match,Match> factory, String... strings) {
        return new Regex(factory,Pattern.compile(String.join("", strings)));
    }
    public static Result any(String line, Regex... regexes) {
        for (Regex r : regexes) {
            if (r.is(line) instanceof Match match) {
                return match;
            }
        }
        return FAIL.of();
    }
    public Result is(String s, Predicate<Matcher> matcherPredicate, BiFunction<Regex,Matcher, Match> factory) {
        if (pattern.matcher(s) instanceof Matcher matcher && matcher.matches() && matcherPredicate.test(matcher)) {
            return factory.apply(this, matcher);
        } else {
            return FAIL.of();
        }
    }

    public Result is(String s, BiFunction<Regex,Matcher, Match> factory) {
        return is(s, _->true,factory);
    }
    public Result is(String s, Predicate<Matcher> matcherPredicate) {
        record DefaultMatch(Regex regex, Matcher matcher) implements Match { }
        return is(s, matcherPredicate, DefaultMatch::new);
    }
    public Result is(String s) {
        return is(s, _->true);
    }
    public boolean matches(String s, Predicate<Matcher> matcherPredicate) {
        return is(s, matcherPredicate) instanceof Match;
    }
    public boolean matches(String s) {
        return is(s) instanceof Match;
    }
    public boolean matchesOrThrow(String s) {
        if(is(s) instanceof Match) {
            return true;
        }
        throw new RuntimeException("failed expected match");
    }
}
