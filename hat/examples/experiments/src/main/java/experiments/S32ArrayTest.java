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
package experiments;

import hat.Schema;
import hat.buffer.Buffer;

public class S32ArrayTest implements Buffer {

    public static void main(String[] args) {
        hat.buffer.S32Array os32  = hat.buffer.S32Array.create(Schema.GlobalArenaAllocator, 100);
        System.out.println("Layout from hat S32Array "+ Buffer.getLayout(os32));

        var s32Array = S32Array.schema.allocate( 100);
       // Schema.BoundSchema boundSchema = (Schema.BoundSchema)Buffer.getHatData(s32Array);
        int s23ArrayLen = s32Array.length();
        System.out.println(s23ArrayLen);

        System.out.println("Layout from schema "+Buffer.getLayout(s32Array));
        ResultTable.schema.toText(t->System.out.print(t));
    }

}
