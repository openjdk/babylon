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

#include "shared.h"


int main(int argc, char **argv) {
    char *mandelSchema = (char *) "args:[5:?:8+S32Array2D:{width:s32,height:s32,array:[*:?:s32]},?:4+S32Array:{length:s32,array:[*:?:s32]},?:f32,?:f32,?:f32]";
    char *squaresSchema = (char *) "args:[1:?:4+S32Array:{length:s32,array:[*:?:s32]}]";
    char *colIntegralSchema = (char *)
            "args:[2:"
            "?:8+F32Array2D:{"
            "width:s32,"
            "height:s32,"
            "array:[*:"
            "?:f32"
            "]"
            "},"
            "?:8+F32Array2D:{width:s32,height:s32,array:[*:?:f32]}"
            "]";
    char *rowIntegralSchema = (char *) "args:[3:?:8+F32Array2D:{width:s32,height:s32,array:[*:?:f32]},?:8+F32Array2D:{width:s32,height:s32,array:[*:?:f32]},?:8+F32Array2D:{width:s32,height:s32,array:[*:?:f32]}]";

    char *cascadeSchema = (char *) "args:[5:!:163448!Cascade:{width:s32,height:s32,featureCount:s32,feature:[2913:Feature:{id:s32,threshold:f32,left:{hasValue:z8,?:x3,anon:<featureId:s32|value:f32>},right:{hasValue:z8,?:x3,anon:<featureId:s32|value:f32>},rect:[3:Rect:{x:s8,y:s8,width:s8,height:s8,weight:f32}]}],stageCount:s32,stage:[25:Stage:{id:s32,threshold:f32,firstTreeId:s16,treeCount:s16}],treeCount:s32,tree:[2913:Tree:{id:s32,firstFeatureId:s16,featureCount:s16}]},?:8+F32Array2D:{width:s32,height:s32,array:[*:?:f32]},?:8+F32Array2D:{width:s32,height:s32,array:[*:?:f32]},?:8+ScaleTable:{length:s32,multiScaleAccumulativeRange:s32,scale:[*:Scale:{scaleValue:f32,scaledXInc:f32,scaledYInc:f32,invArea:f32,scaledFeatureWidth:s32,scaledFeatureHeight:s32,gridWidth:s32,gridHeight:s32,gridSize:s32,accumGridSizeMin:s32,accumGridSizeMax:s32}]},?:8+ResultTable:{length:s32,atomicResultTableCount:s32,result:[*:Result:{x:f32,y:f32,width:f32,height:f32}]}]";
    char *schema = cascadeSchema;
    std::cout << "schema = '" << schema << "'" << std::endl;
    Schema::dumpSchema(std::cout, schema);

}

