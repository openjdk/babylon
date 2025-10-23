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
package view;

import java.util.regex.Matcher;
import java.util.regex.Pattern;

record Regex(Pattern pattern) {
    interface Match {
        boolean matched();
    }

    record OK(Regex regex, Matcher matcher, boolean matched) implements Match {
        public static OK of(Regex regex, Matcher matcher) {
            return new OK(regex, matcher, true);
        }
        float hex2Float(String s) {
            return (s.startsWith("-"))? (-Integer.parseInt(s.substring(2), 16) / 64f): (Integer.parseInt(s.substring(1), 16) / 64f);
        }
        public float f(int idx) {
            return hex2Float(string(idx));
        }

        public int i(int idx) {
            return Integer.parseInt(string(idx));
        }

        public String string(int idx) {
            return matcher.group(idx);
        }
    }

    record FAIL(boolean matched) implements Match {
        public static FAIL of() {
            return new FAIL(false);
        }
    }

    static Regex of(String... strings) {
        return new Regex(Pattern.compile(String.join("", strings)));
    }

    static Match any(String line, Regex... regexes) {
        for (Regex r : regexes) {
            if (r.is(line) instanceof OK ok) {
                return ok;
            }
        }
        return FAIL.of();
    }

    Match is(String s) {
        if (pattern.matcher(s) instanceof Matcher matcher && matcher.matches()) {
            return OK.of(this, matcher);
        } else {
            return FAIL.of();
        }
    }
}
