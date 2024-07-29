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


import java.util.function.BiConsumer;
import java.util.function.Consumer;
import java.util.stream.Stream;

public class StreamCounter<E> {
    private int value;

    public static <E> void of(Iterable<E> iterable, BiConsumer<StreamCounter<E>, E> counterConsumer) {
        StreamCounter<E> sc = new StreamCounter<>();
        iterable.spliterator();
        iterable.forEach((e) -> {
            sc.convey = e;
            counterConsumer.accept(sc, e);
            sc.inc();
        });

    }

    E convey;

    public static <E> void of(Stream<E> stream, BiConsumer<StreamCounter<E>, E> counterConsumer) {
        StreamCounter<E> sc = new StreamCounter<>();
        stream.forEach((e) -> {
            sc.convey = e;
            counterConsumer.accept(sc, e);
            sc.inc();
        });

    }

    public static void of(Consumer<StreamCounter> counterConsumer) {
        counterConsumer.accept(new StreamCounter());
    }

    public int value() {
        return value;
    }

    public boolean isFirst() {
        return value == 0;
    }

    public boolean isNotFirst() {
        return value != 0;
    }

    public boolean onFirst(Consumer<E> consumer) {
        if (isFirst()) {
            consumer.accept(convey);
            return true;
        }
        return false;
    }

    public boolean onNotFirst(Consumer<E> consumer) {
        if (!isFirst()) {
            consumer.accept(convey);
            return true;
        }
        return false;
    }

    public void inc() {
        value++;
    }

    public StreamCounter() {
        this.value = 0;
    }

}
