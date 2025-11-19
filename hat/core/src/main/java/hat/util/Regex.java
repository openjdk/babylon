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
package hat.util;

import java.util.function.BiFunction;
import java.util.function.Predicate;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

public record Regex(Pattern pattern) {
    public interface Match {
        boolean matched();
    }

    public interface OK extends Match {
        Regex regex();
        Matcher matcher();
        default  String string(int idx) {
            return matcher().group(idx);
        }

        default float asFloat(int idx) {
            return Float.parseFloat(string(idx));
        }

        default  int asInt(int idx) {
            return Integer.parseInt(string(idx));
        }
        default  int[] asInts(int from, int count) {
            int[] ints = new int[count];
            for (int i = 0; i<count; i++) {
                ints[i]= Integer.parseInt(string(from + i));
            }
            return ints;
        }
        default  float[] asFloats(int from, int count) {
            float[] floats = new float[count];
            for (int i = 0; i<count; i++) {
                floats[i]=Float.parseFloat(string(from + i));
            }
            return floats;
        }
    }


    public record DefaultOk(Regex regex, Matcher matcher, boolean matched) implements OK {
        public static OK of(Regex regex, Matcher matcher) {
            return new DefaultOk(regex, matcher, true);
        }
    }

    record FAIL(boolean matched) implements Match {
        public static FAIL of() {
            return new FAIL(false);
        }
    }

    public static Regex of(String... strings) {
        return new Regex(Pattern.compile(String.join("", strings)));
    }

    public static Match any(String line, Regex... regexes) {
        for (Regex r : regexes) {
            if (r.is(line) instanceof OK ok) {
                return ok;
            }
        }
        return FAIL.of();
    }
    public Match is(String s, Predicate<Matcher> matcherPredicate, BiFunction<Regex,Matcher,OK> factory) {
        if (pattern.matcher(s) instanceof Matcher matcher && matcher.matches() && matcherPredicate.test(matcher)) {
            return factory.apply(this, matcher);
        } else {
            return FAIL.of();
        }
    }
    public Match is(String s, BiFunction<Regex,Matcher,OK> factory) {
        return is(s, _->true,factory);
    }
    public Match is(String s, Predicate<Matcher> matcherPredicate) {
        return is(s, matcherPredicate,(r,m)->new DefaultOk(r,m, true));
    }
    public Match is(String s) {
        return is(s, _->true);
    }
    public boolean matches(String s, Predicate<Matcher> matcherPredicate) {
        return is(s, matcherPredicate).matched();
    }
    public boolean matches(String s) {
        return is(s).matched();
    }
    public boolean matchesOrThrow(String s) {
        if(!is(s).matched()){
            throw new RuntimeException("failed expected match");
        }else{
            return true;
        }
    }
}
