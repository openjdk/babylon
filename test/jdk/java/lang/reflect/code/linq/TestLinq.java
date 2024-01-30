/*
 * Copyright (c) 2024, Oracle and/or its affiliates. All rights reserved.
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * This code is free software; you can redistribute it and/or modify it
 * under the terms of the GNU General Public License version 2 only, as
 * published by the Free Software Foundation.
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

import org.testng.Assert;
import org.testng.annotations.Test;

import java.lang.invoke.MethodHandles;
import java.lang.reflect.code.interpreter.Interpreter;
import java.util.stream.Stream;

/*
 * @test
 * @enablePreview
 * @run testng TestLinq
 */

public class TestLinq {

    // A record modeling a table with three columns, one for each component
    record Customer(String contactName, String phone, String city) {
    }

    @Test
    public void testSimpleQuery() {
        QueryProvider qp = new TestQueryProvider();

        // Query all customers based in London, and return their names
        QueryResult<Stream<String>> qr = qp.query(Customer.class)
                .where(c -> c.city.equals("London"))
                .select(c -> c.contactName)
                .elements();

        System.out.println(qr.expression().toText());

        @SuppressWarnings("unchecked")
        QueryResult<Stream<String>> qr2 = (QueryResult<Stream<String>>) Interpreter.invoke(MethodHandles.lookup(),
                qr.expression(), qp.query(Customer.class));
        System.out.println(qr2.expression().toText());

        Assert.assertEquals(qr.expression().toText(), qr2.expression().toText());
    }
}
