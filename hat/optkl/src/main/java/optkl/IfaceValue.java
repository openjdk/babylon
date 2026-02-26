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
package optkl;


import jdk.incubator.code.TypeElement;

import java.util.List;

public interface IfaceValue {

    interface Union extends IfaceValue {
    }

    interface Struct extends IfaceValue {
    }

     interface Vector {
        interface Shape {
            // This is the type that best describes the usage from a Java POV
            TypeElement typeElement();
            // This is how we expect to represent the type
            // So an FP16x4 might be
            //    typeElement=FLOAT (we will see float accessors)
            //    representedBy=SHORT (the data is 'packed into')
            TypeElement representedBy();
            int lanes();
            static Shape of(TypeElement typeElement, TypeElement representedBy, int lanes) {
                record Impl(TypeElement typeElement, TypeElement representedBy, int lanes) implements Shape {
                    @Override public String toString(){
                        return typeElement.toString() + Impl.this.lanes;
                    }
                }
                return new Impl(typeElement,representedBy, lanes);
            }
            static Shape of(TypeElement typeElement,  int lanes) {
               return of (typeElement,typeElement,lanes);
            }
            default  List<String> laneNames(){
                return switch (lanes()){
                    case 2 -> List.of("x","y");
                    case 3 -> List.of("x","y","z");
                    case 4 -> List.of("x","y","z","w");
                    default -> throw new RuntimeException("We only support 2,3 or 4 lanes");
                };
            }
        }
    }

    interface vec {
        interface Shape {
            // This is the type that best describes the usage from a Java POV
            TypeElement typeElement();
            // This is how we expect to represent the type
            // So an FP16x4 might be
            //    typeElement=FLOAT (we will see float accessors)
            //    representedBy=SHORT (the data is 'packed into')
            TypeElement representedBy();
            int lanes();
            static Shape of(TypeElement typeElement, TypeElement representedBy, int lanes) {
                record Impl(TypeElement typeElement, TypeElement representedBy, int lanes) implements Shape {
                    @Override public String toString(){
                        return typeElement.toString() + Impl.this.lanes;
                    }
                }
                return new Impl(typeElement,representedBy, lanes);
            }
            static Shape of(TypeElement typeElement, int lanes) {
                return of (typeElement,typeElement,lanes);
            }
            default  List<String> laneNames(){
                return switch (lanes()){
                    case 2 -> List.of("x","y");
                    case 3 -> List.of("x","y","z");
                    case 4 -> List.of("x","y","z","w");
                    default -> throw new RuntimeException("We only support 2,3 or 4 lanes");
                };
            }
        }
    }
    // Experimental ... considering for any Struct acting as an aggregate containing a length field.
    interface mat {
        interface Shape {
            TypeElement typeElement();
            int rows();
            int cols();

            static Shape of(TypeElement typeElement, int rows, int cols) {
                record Impl(TypeElement typeElement, int rows, int cols) implements Shape {
                    @Override public String toString(){
                        return typeElement.toString() + Impl.this.rows+"x"+Impl.this.cols;
                    }
                }
                return new Impl(typeElement, rows,cols);
            }
        }
    }

    // Experimental ... considering for any Struct acting as an aggregate containing a length field.
    interface Array extends IfaceValue {
    }
}
