/*
 * Copyright (c) 2026, Oracle and/or its affiliates. All rights reserved.
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

import java.lang.classfile.ClassFile;
import java.lang.invoke.MethodHandles;
import jdk.incubator.code.Op;
import jdk.incubator.code.Reflect;
import jdk.incubator.code.dialect.core.CoreOp.FuncOp;
import jdk.incubator.code.interpreter.Interpreter;
import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.Test;

/*
 * @test
 * @modules jdk.incubator.code
 * @run junit TestOpBuilderLimits
 * @run main Unreflect TestOpBuilderLimits
 * @run junit TestOpBuilderLimits
 */
public class TestOpBuilderLimits {

    @Reflect
    static int bigModelBuilder(int i) {
        i++;i++;i++;i++;i++;i++;i++;i++;i++;i++;i++;i++;i++;i++;i++;i++;i++;i++;i++;i++;i++;i++;i++;i++;i++;
        i++;i++;i++;i++;i++;i++;i++;i++;i++;i++;i++;i++;i++;i++;i++;i++;i++;i++;i++;i++;i++;i++;i++;i++;i++;
        i++;i++;i++;i++;i++;i++;i++;i++;i++;i++;i++;i++;i++;i++;i++;i++;i++;i++;i++;i++;i++;i++;i++;i++;i++;
        i++;i++;i++;i++;i++;i++;i++;i++;i++;i++;i++;i++;i++;i++;i++;i++;i++;i++;i++;i++;i++;i++;i++;i++;i++;
        i++;i++;i++;i++;i++;i++;i++;i++;i++;i++;i++;i++;i++;i++;i++;i++;i++;i++;i++;i++;i++;i++;i++;i++;i++;
        i++;i++;i++;i++;i++;i++;i++;i++;i++;i++;i++;i++;i++;i++;i++;i++;i++;i++;i++;i++;i++;i++;i++;i++;i++;
        i++;i++;i++;i++;i++;i++;i++;i++;i++;i++;i++;i++;i++;i++;i++;i++;i++;i++;i++;i++;i++;i++;i++;i++;i++;
        i++;i++;i++;i++;i++;i++;i++;i++;i++;i++;i++;i++;i++;i++;i++;i++;i++;i++;i++;i++;i++;i++;i++;i++;i++;
        i++;i++;i++;i++;i++;i++;i++;i++;i++;i++;i++;i++;i++;i++;i++;i++;i++;i++;i++;i++;i++;i++;i++;i++;i++;
        i++;i++;i++;i++;i++;i++;i++;i++;i++;i++;i++;i++;i++;i++;i++;i++;i++;i++;i++;i++;i++;i++;i++;i++;i++;
        return i;
    }

    @Test
    public void testOpLimit() throws Exception {
        FuncOp fop = Op.ofMethod(TestOpBuilderLimits.class.getDeclaredMethod("bigModelBuilder", int.class)).orElseThrow();
        Assertions.assertEquals(bigModelBuilder(0), (int)Interpreter.invoke(MethodHandles.lookup(), fop, 0));
    }

    interface T<T0, T1, T2, T3, T4, T5, T6, T7> {}
    interface A {}
    interface B {}

    @Reflect
    static int manyTypes(int i) {
        T<A,A,A,A,A,A,A,A> t0; T<A,A,A,A,A,A,A,B> t1; T<A,A,A,A,A,A,B,A> t2; T<A,A,A,A,A,A,B,B> t3; T<A,A,A,A,A,B,A,A> t4;
        T<A,A,A,A,A,B,A,B> t5; T<A,A,A,A,A,B,B,A> t6; T<A,A,A,A,A,B,B,B> t7; T<A,A,A,A,B,A,A,A> t8; T<A,A,A,A,B,A,A,B> t9;
        T<A,A,A,A,B,A,B,A> t10; T<A,A,A,A,B,A,B,B> t11; T<A,A,A,A,B,B,A,A> t12; T<A,A,A,A,B,B,A,B> t13; T<A,A,A,A,B,B,B,A> t14;
        T<A,A,A,A,B,B,B,B> t15; T<A,A,A,B,A,A,A,A> t16; T<A,A,A,B,A,A,A,B> t17; T<A,A,A,B,A,A,B,A> t18; T<A,A,A,B,A,A,B,B> t19;
        T<A,A,A,B,A,B,A,A> t20; T<A,A,A,B,A,B,A,B> t21; T<A,A,A,B,A,B,B,A> t22; T<A,A,A,B,A,B,B,B> t23; T<A,A,A,B,B,A,A,A> t24;
        T<A,A,A,B,B,A,A,B> t25; T<A,A,A,B,B,A,B,A> t26; T<A,A,A,B,B,A,B,B> t27; T<A,A,A,B,B,B,A,A> t28; T<A,A,A,B,B,B,A,B> t29;
        T<A,A,A,B,B,B,B,A> t30; T<A,A,A,B,B,B,B,B> t31; T<A,A,B,A,A,A,A,A> t32; T<A,A,B,A,A,A,A,B> t33; T<A,A,B,A,A,A,B,A> t34;
        T<A,A,B,A,A,A,B,B> t35; T<A,A,B,A,A,B,A,A> t36; T<A,A,B,A,A,B,A,B> t37; T<A,A,B,A,A,B,B,A> t38; T<A,A,B,A,A,B,B,B> t39;
        T<A,A,B,A,B,A,A,A> t40; T<A,A,B,A,B,A,A,B> t41; T<A,A,B,A,B,A,B,A> t42; T<A,A,B,A,B,A,B,B> t43; T<A,A,B,A,B,B,A,A> t44;
        T<A,A,B,A,B,B,A,B> t45; T<A,A,B,A,B,B,B,A> t46; T<A,A,B,A,B,B,B,B> t47; T<A,A,B,B,A,A,A,A> t48; T<A,A,B,B,A,A,A,B> t49;
        T<A,A,B,B,A,A,B,A> t50; T<A,A,B,B,A,A,B,B> t51; T<A,A,B,B,A,B,A,A> t52; T<A,A,B,B,A,B,A,B> t53; T<A,A,B,B,A,B,B,A> t54;
        T<A,A,B,B,A,B,B,B> t55; T<A,A,B,B,B,A,A,A> t56; T<A,A,B,B,B,A,A,B> t57; T<A,A,B,B,B,A,B,A> t58; T<A,A,B,B,B,A,B,B> t59;
        T<A,A,B,B,B,B,A,A> t60; T<A,A,B,B,B,B,A,B> t61; T<A,A,B,B,B,B,B,A> t62; T<A,A,B,B,B,B,B,B> t63; T<A,B,A,A,A,A,A,A> t64;
        T<A,B,A,A,A,A,A,B> t65; T<A,B,A,A,A,A,B,A> t66; T<A,B,A,A,A,A,B,B> t67; T<A,B,A,A,A,B,A,A> t68; T<A,B,A,A,A,B,A,B> t69;
        T<A,B,A,A,A,B,B,A> t70; T<A,B,A,A,A,B,B,B> t71; T<A,B,A,A,B,A,A,A> t72; T<A,B,A,A,B,A,A,B> t73; T<A,B,A,A,B,A,B,A> t74;
        T<A,B,A,A,B,A,B,B> t75; T<A,B,A,A,B,B,A,A> t76; T<A,B,A,A,B,B,A,B> t77; T<A,B,A,A,B,B,B,A> t78; T<A,B,A,A,B,B,B,B> t79;
        T<A,B,A,B,A,A,A,A> t80; T<A,B,A,B,A,A,A,B> t81; T<A,B,A,B,A,A,B,A> t82; T<A,B,A,B,A,A,B,B> t83; T<A,B,A,B,A,B,A,A> t84;
        T<A,B,A,B,A,B,A,B> t85; T<A,B,A,B,A,B,B,A> t86; T<A,B,A,B,A,B,B,B> t87; T<A,B,A,B,B,A,A,A> t88; T<A,B,A,B,B,A,A,B> t89;
        T<A,B,A,B,B,A,B,A> t90; T<A,B,A,B,B,A,B,B> t91; T<A,B,A,B,B,B,A,A> t92; T<A,B,A,B,B,B,A,B> t93; T<A,B,A,B,B,B,B,A> t94;
        T<A,B,A,B,B,B,B,B> t95; T<A,B,B,A,A,A,A,A> t96; T<A,B,B,A,A,A,A,B> t97; T<A,B,B,A,A,A,B,A> t98; T<A,B,B,A,A,A,B,B> t99;
        T<A,B,B,A,A,B,A,A> t100; T<A,B,B,A,A,B,A,B> t101; T<A,B,B,A,A,B,B,A> t102; T<A,B,B,A,A,B,B,B> t103; T<A,B,B,A,B,A,A,A> t104;
        T<A,B,B,A,B,A,A,B> t105; T<A,B,B,A,B,A,B,A> t106; T<A,B,B,A,B,A,B,B> t107; T<A,B,B,A,B,B,A,A> t108; T<A,B,B,A,B,B,A,B> t109;
        T<A,B,B,A,B,B,B,A> t110; T<A,B,B,A,B,B,B,B> t111; T<A,B,B,B,A,A,A,A> t112; T<A,B,B,B,A,A,A,B> t113; T<A,B,B,B,A,A,B,A> t114;
        T<A,B,B,B,A,A,B,B> t115; T<A,B,B,B,A,B,A,A> t116; T<A,B,B,B,A,B,A,B> t117; T<A,B,B,B,A,B,B,A> t118; T<A,B,B,B,A,B,B,B> t119;
        T<A,B,B,B,B,A,A,A> t120; T<A,B,B,B,B,A,A,B> t121; T<A,B,B,B,B,A,B,A> t122; T<A,B,B,B,B,A,B,B> t123; T<A,B,B,B,B,B,A,A> t124;
        T<A,B,B,B,B,B,A,B> t125; T<A,B,B,B,B,B,B,A> t126; T<A,B,B,B,B,B,B,B> t127; T<B,A,A,A,A,A,A,A> t128; T<B,A,A,A,A,A,A,B> t129;
        T<B,A,A,A,A,A,B,A> t130; T<B,A,A,A,A,A,B,B> t131; T<B,A,A,A,A,B,A,A> t132; T<B,A,A,A,A,B,A,B> t133; T<B,A,A,A,A,B,B,A> t134;
        T<B,A,A,A,A,B,B,B> t135; T<B,A,A,A,B,A,A,A> t136; T<B,A,A,A,B,A,A,B> t137; T<B,A,A,A,B,A,B,A> t138; T<B,A,A,A,B,A,B,B> t139;
        T<B,A,A,A,B,B,A,A> t140; T<B,A,A,A,B,B,A,B> t141; T<B,A,A,A,B,B,B,A> t142; T<B,A,A,A,B,B,B,B> t143; T<B,A,A,B,A,A,A,A> t144;
        T<B,A,A,B,A,A,A,B> t145; T<B,A,A,B,A,A,B,A> t146; T<B,A,A,B,A,A,B,B> t147; T<B,A,A,B,A,B,A,A> t148; T<B,A,A,B,A,B,A,B> t149;
        T<B,A,A,B,A,B,B,A> t150; T<B,A,A,B,A,B,B,B> t151; T<B,A,A,B,B,A,A,A> t152; T<B,A,A,B,B,A,A,B> t153; T<B,A,A,B,B,A,B,A> t154;
        T<B,A,A,B,B,A,B,B> t155; T<B,A,A,B,B,B,A,A> t156; T<B,A,A,B,B,B,A,B> t157; T<B,A,A,B,B,B,B,A> t158; T<B,A,A,B,B,B,B,B> t159;
        T<B,A,B,A,A,A,A,A> t160; T<B,A,B,A,A,A,A,B> t161; T<B,A,B,A,A,A,B,A> t162; T<B,A,B,A,A,A,B,B> t163; T<B,A,B,A,A,B,A,A> t164;
        T<B,A,B,A,A,B,A,B> t165; T<B,A,B,A,A,B,B,A> t166; T<B,A,B,A,A,B,B,B> t167; T<B,A,B,A,B,A,A,A> t168; T<B,A,B,A,B,A,A,B> t169;
        T<B,A,B,A,B,A,B,A> t170; T<B,A,B,A,B,A,B,B> t171; T<B,A,B,A,B,B,A,A> t172; T<B,A,B,A,B,B,A,B> t173; T<B,A,B,A,B,B,B,A> t174;
        T<B,A,B,A,B,B,B,B> t175; T<B,A,B,B,A,A,A,A> t176; T<B,A,B,B,A,A,A,B> t177; T<B,A,B,B,A,A,B,A> t178; T<B,A,B,B,A,A,B,B> t179;
        T<B,A,B,B,A,B,A,A> t180; T<B,A,B,B,A,B,A,B> t181; T<B,A,B,B,A,B,B,A> t182; T<B,A,B,B,A,B,B,B> t183; T<B,A,B,B,B,A,A,A> t184;
        T<B,A,B,B,B,A,A,B> t185; T<B,A,B,B,B,A,B,A> t186; T<B,A,B,B,B,A,B,B> t187; T<B,A,B,B,B,B,A,A> t188; T<B,A,B,B,B,B,A,B> t189;
        T<B,A,B,B,B,B,B,A> t190; T<B,A,B,B,B,B,B,B> t191; T<B,B,A,A,A,A,A,A> t192; T<B,B,A,A,A,A,A,B> t193; T<B,B,A,A,A,A,B,A> t194;
        T<B,B,A,A,A,A,B,B> t195; T<B,B,A,A,A,B,A,A> t196; T<B,B,A,A,A,B,A,B> t197; T<B,B,A,A,A,B,B,A> t198; T<B,B,A,A,A,B,B,B> t199;
        T<B,B,A,A,B,A,A,A> t200; T<B,B,A,A,B,A,A,B> t201; T<B,B,A,A,B,A,B,A> t202; T<B,B,A,A,B,A,B,B> t203; T<B,B,A,A,B,B,A,A> t204;
        T<B,B,A,A,B,B,A,B> t205; T<B,B,A,A,B,B,B,A> t206; T<B,B,A,A,B,B,B,B> t207; T<B,B,A,B,A,A,A,A> t208; T<B,B,A,B,A,A,A,B> t209;
        T<B,B,A,B,A,A,B,A> t210; T<B,B,A,B,A,A,B,B> t211; T<B,B,A,B,A,B,A,A> t212; T<B,B,A,B,A,B,A,B> t213; T<B,B,A,B,A,B,B,A> t214;
        T<B,B,A,B,A,B,B,B> t215; T<B,B,A,B,B,A,A,A> t216; T<B,B,A,B,B,A,A,B> t217; T<B,B,A,B,B,A,B,A> t218; T<B,B,A,B,B,A,B,B> t219;
        T<B,B,A,B,B,B,A,A> t220; T<B,B,A,B,B,B,A,B> t221; T<B,B,A,B,B,B,B,A> t222; T<B,B,A,B,B,B,B,B> t223; T<B,B,B,A,A,A,A,A> t224;
        T<B,B,B,A,A,A,A,B> t225; T<B,B,B,A,A,A,B,A> t226; T<B,B,B,A,A,A,B,B> t227; T<B,B,B,A,A,B,A,A> t228; T<B,B,B,A,A,B,A,B> t229;
        T<B,B,B,A,A,B,B,A> t230; T<B,B,B,A,A,B,B,B> t231; T<B,B,B,A,B,A,A,A> t232; T<B,B,B,A,B,A,A,B> t233; T<B,B,B,A,B,A,B,A> t234;
        T<B,B,B,A,B,A,B,B> t235; T<B,B,B,A,B,B,A,A> t236; T<B,B,B,A,B,B,A,B> t237; T<B,B,B,A,B,B,B,A> t238; T<B,B,B,A,B,B,B,B> t239;
        T<B,B,B,B,A,A,A,A> t240; T<B,B,B,B,A,A,A,B> t241; T<B,B,B,B,A,A,B,A> t242; T<B,B,B,B,A,A,B,B> t243; T<B,B,B,B,A,B,A,A> t244;
        T<B,B,B,B,A,B,A,B> t245; T<B,B,B,B,A,B,B,A> t246; T<B,B,B,B,A,B,B,B> t247; T<B,B,B,B,B,A,A,A> t248; T<B,B,B,B,B,A,A,B> t249;
        T<B,B,B,B,B,A,B,A> t250; T<B,B,B,B,B,A,B,B> t251; T<B,B,B,B,B,B,A,A> t252; T<B,B,B,B,B,B,A,B> t253; T<B,B,B,B,B,B,B,A> t254;
        T<B,B,B,B,B,B,B,B> t255;
        return i;
    }

    @Test
    public void testTypeLimit() throws Exception {
        FuncOp fop = Op.ofMethod(TestOpBuilderLimits.class.getDeclaredMethod("manyTypes", int.class)).orElseThrow();
        Assertions.assertEquals(manyTypes(42), (int)Interpreter.invoke(MethodHandles.lookup(), fop, 42));
    }

    @Test
    public void testLimitsExceeded() throws Exception {
        var methods = ClassFile.of().parse(TestOpBuilderLimits.class.getResourceAsStream("TestOpBuilderLimits$$CM.class").readAllBytes()).methods();
        Assertions.assertTrue(methods.stream().anyMatch(m -> m.methodName().equalsString("lambda$0")), "ops limit exceeded");
        Assertions.assertTrue(methods.stream().anyMatch(m -> m.methodName().equalsString("$exterType1")), "types limit exceeded");
    }
}
