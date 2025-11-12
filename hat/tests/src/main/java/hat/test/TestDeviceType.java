/*
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
 * questions.
 */
package hat.test;

import hat.buffer.F16;
import hat.device.DeviceSchema;
import hat.device.DeviceType;
import hat.test.annotation.HatTest;
import hat.test.engine.HATAsserts;

public class TestDeviceType {

    public interface MyDeviceArray extends DeviceType {
        F16 array(int index);
        void array(int index, F16 value);

        void x(float x);
        float x();

        DeviceSchema<MyDeviceArray> schema = DeviceSchema.of(MyDeviceArray.class, builder ->
                builder.withArray("array", 2048)
                        .withDeps(F16.class, half -> half.withField("value"))
                        .withField("x"));

        static MyDeviceArray create() {
            return null;
        }
    }

    // This test checks the representation of the prev. array
    @HatTest
    public void testdevice_type_01() {
        MyDeviceArray myDeviceArray = MyDeviceArray.create();
        String text = MyDeviceArray.schema.toText();
        boolean isEquals = text.equals("<hat.buffer.F16:s:half:value;><hat.test.TestDeviceType$MyDeviceArray:[:hat.buffer.F16:array:2048;s:float:x;>");
        HATAsserts.assertTrue(isEquals);
    }


    // A way to construct 2D arrays
    public interface MyNDRAnge extends DeviceType {
        SubRange array(int index);
        void array(int index, SubRange value);

        interface SubRange {
            int range();
            void range(int index, int val);
        }

        DeviceSchema<MyNDRAnge> schema = DeviceSchema.of(MyNDRAnge.class, builder ->
                builder.withArray("array", 2048)
                        .withDeps(SubRange.class, subrange -> subrange.withArray("range", 64)));

        static MyNDRAnge create() {
            return null;
        }
    }

    @HatTest
    public void testdevice_type_02() {
        MyNDRAnge myDeviceArray = MyNDRAnge.create();
        String text = MyNDRAnge.schema.toText();
        boolean isEquals = text.equals("<hat.test.TestDeviceType$MyNDRAnge$SubRange:[:int:range:64;><hat.test.TestDeviceType$MyNDRAnge:[:hat.test.TestDeviceType$MyNDRAnge$SubRange:array:2048;>");
        HATAsserts.assertTrue(isEquals);
    }

    // A way to construct 3D arrays
    public interface MultiDim extends DeviceType {
        _2D array(int index);
        void array(int index, _2D value);

        interface _2D {
            _3D _range2(int index);
            void _range2(int index, _3D val);

            interface _3D {
                int value(int index);
                void value(int index, int val);
            }
        }

        DeviceSchema<MultiDim> schema = DeviceSchema.of(MultiDim.class, builder ->
                builder.withArray("array", 2048)
                        .withDeps(_2D.class,subrange -> subrange.withArray("range2", 64)
                                                                                       .withDeps(_2D._3D.class, f -> f.withArray("value", 32))));

        static MultiDim create() {
            return null;
        }
    }

    @HatTest
    public void testdevice_type_03() {
        // This test is expected to fail. It request a member called "range2" from the _2D class.
        // However, the method name is "_range2". Thus the requested method doen't exits.
        try {
            MultiDim myDeviceArray = MultiDim.create();
            String text = MultiDim.schema.toText();
            // If we request the correct method, the result should be as follows:
            boolean isEquals = text.equals("<hat.test.TestDeviceType$MultiDim$_2D$_3D:[:int:value:32;><hat.test.TestDeviceType$MultiDim$_2D:[:hat.test.TestDeviceType$MultiDim$_2D$_3D:_range2:64;><hat.test.TestDeviceType$MultiDim:[:hat.test.TestDeviceType$MultiDim$_2D:array:2048;>");
            HATAsserts.assertFalse(isEquals);
        } catch (ExceptionInInitializerError e) {
            HATAsserts.assertTrue(true);
        }
    }

}
