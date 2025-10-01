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
package experiments;

import hat.Accelerator;
import hat.ComputeContext;
import hat.ComputeRange;
import hat.GlobalMesh1D;
import hat.KernelContext;
import hat.LocalMesh1D;
import hat.backend.Backend;
import hat.buffer.Buffer;
import hat.buffer.S32Array;
import hat.ifacemapper.MappableIface.RO;
import hat.ifacemapper.MappableIface.RW;
import hat.ifacemapper.Schema;
import jdk.incubator.code.CodeReflection;

import java.lang.invoke.MethodHandles;

/**
 * How to test?
 * <code>
 *     HAT=SHOW_CODE java -cp job.jar hat.java exp ffi-opencl PrefixSum
 *     HAT=SHOW_CODE java -cp job.jar hat.java exp ffi-cuda PrefixSum
 * </code>
 */
public class PrefixSum {
    private interface SharedS32x32Array extends Buffer {
        void array(long index, int value);
        int array(long index);

        Schema<SharedS32x32Array> schema = Schema.of(SharedS32x32Array.class,
                $ -> $
                        // It is a bound schema, so we fix the size here
                        .array("array", 256));

        static SharedS32x32Array create(Accelerator accelerator) {
            return schema.allocate(accelerator);
        }

       // static SharedS32x32Array createLocal(Accelerator accelerator) {
         //   return schema.allocate(accelerator);
       // }

        static SharedS32x32Array createLocal() {
            return schema.allocate(new Accelerator(MethodHandles.lookup(), Backend.FIRST));
        }
    }




//  4    2    3    2    6    1    2    3
//   \   |     \   |     \   |     \   |
//    \  |      \  |      \  |      \  |
//     \ |       \ |       \ |       \ |
//      \|        \|        \|        \|
//       +         +         +         +
//  4    6    3    5    6    7    2    5
//        \        |          \        |
//         \       |           \       |
//          \      |            \      |
//           \     |             \     |
//            \    |              \    |
//             \   |               \   |
//              \  |                \  |
//               \ |                 \ |
//                \|                  \|
//                 +                   +
//  4    6    3   11    6    7    2   12
//                  \                  |
//                   \                 |
//                    \                |
//                     \               |
//                      \              |
//                       \             |
//                        \            |
//                         \           |
//                          \          |   This last pass can be ommitted!
//                           \         |
//                            \        |
//                             \       |
//                              \      |
//                               \     |
//                                \    |
//                                 \   |
//                                  \  |
//                                   \ |
//                                    \|
//                                     +
//  4    6    3   11    6    7    2   23
//                                     |
//                             overwrite with 0
//                                     |
//                                     V
//  4    6    3   11    6    7    2    0
//                  \                 /|
//                   \               / |
//                    \             /  |
//                     \           /   |
//                      \         /    |
//                       \       /     |
//                        \     /      |
//                         \   /       |
//                          \ /        |
//                           /         |
//                          / \        |
//                         /   \       |
//                        /     \      |
//                       /       \     |
//                      /         \    |
//                     /           \   |
//                    /             \  |
//                   /               \ |
//                  /                 \|
//                 V                   +
//  4    6    3    0    6    7    2    11
//         \      /|          \      / |
//          \    / |           \    /  |
//           \  /  |            \  /   |
//            \/   |             \/    |
//            /\   |             /\    |
//           /  \  |            /  \   |
//          /    \ |           /    \  |
//         /      \|          /      \ |
//        V        +         V         +
//  4     0   3    6    6   11    2    18
//    \  /|    \  /|     \  /|     \  /|
//     \/ |     \/ |      \/ |      \/ |
//     /\ |     /\ |      /\ |      /\ |
//    /  \|    /  \|     /  \|     /  \|
//   V    +   V    +     V   +    V    +
//  0     4   6    9    11  17    18   20


// Gather consecutive global values into local
// if local_size(0) were 8 and global_size(0) was 40
//      group 0      |    group 1      |    group 2      |    group 3      |     group 4      |
// | 0,1,2,3,4,5,6,7 | 0,1,2,3,4,5,6,7 | 0,1,2,3,4,5,6,7 | 0,1,2,3,4,5,6,7 |  0,1,2,3,4,5,6,7 |
//
// then sum each group

    @CodeReflection
     static void groupScan(@RO KernelContext kc,@RW S32Array dataBuf){
        var scratchBuf = SharedS32x32Array.createLocal();
        int[] data = dataBuf.arrayView();
        // int[] scratch=scratchBuf.arrayView(); one day
        //  scratch[kc.lix]=data[kc.gix];
        scratchBuf.array(kc.lix,data[kc.gix]); // copy into local scratch for the reduction
        kc.barrier();

        for (int step=2; step <= kc.lsx; step<<=1){
            if (((kc.lix+1)%step) == 0){
                //  one day scratch[kc.lix]+=scratch[kc.lix-(step>>1)];
                scratchBuf.array(kc.lix, scratchBuf.array(kc.lix)+scratchBuf.array(kc.lix-(step>>1)));
            }
            kc.barrier();
        }
        int sum=0;
        if ((kc.lix+1) == kc.lsx){
           // one day  sum = scratch[kc.lix];
            sum = scratchBuf.array(kc.lix);
           // one day  scratch[kc.lix]=0;
            scratchBuf.array(kc.lix,0);
        }
        kc.barrier();
        for (int step=kc.lsx; step >1 ; step>>=1){
            if (((kc.lix+1)%step) == 0){
               // int prev = scratch[kc.lix-(step>>1)];
                int prev = scratchBuf.array(kc.lix-(step>>1));
               // scratch[kc.lix-(step>>1)]=scratch[kc.lix];
                scratchBuf.array(kc.lix-(step>>1),scratchBuf.array(kc.lix));
                //  scratch[kc.lix]+= prev;
                scratchBuf.array(kc.lix, scratchBuf.array(kc.lix)+prev);
            }
            kc.barrier();
        }

        if ((kc.lix+1) == kc.lsx){
            data[ kc.gix] = sum;
        }else{
           // data[ kc.gix] = scratch[kc.lix+1];
            data[ kc.gix] = scratchBuf.array(kc.lix+1);
        }
        kc.barrier();

    }


// Sum 'group_width(0)-1' for all groups in place
// if local_size(0) were 8 and global_size(0) was 40
//      group 0      |    group 1      |    group 2      |    group 3      |     group 4      |
// | 0,1,2,3,4,5,6,7 | 0,1,2,3,4,5,6,7 | 0,1,2,3,4,5,6,7 | 0,1,2,3,4,5,6,7 |  0,1,2,3,4,5,6,7 |
//                 ^                 ^                 ^                 ^                  ^
//                 s0                s1                s2                s3                 s4

    @CodeReflection static void crossGroupScan(@RO KernelContext kc, @RW S32Array  dataBuf){
       var scratchBuf = SharedS32x32Array.createLocal();
        int[] data = dataBuf.arrayView();
       // int[] scratch=scratchBuf.arrayView();

        int gid = (kc.gix*(kc.gsx))-1; // 0-> -1?  hence the >0 checks below.
        //scratch[kc.lix]= (gid>0)?data[gid]:0;
        scratchBuf.array(kc.lix, (gid>0)?data[gid]:0); // copy into local scratch for the reduction
      kc.barrier(); // make sure all of scratch is populated
        for (int step=2; step <= kc.gsx; step<<=1){
            if (((kc.lix+1)%step) == 0){
               // scratch[kc.lix]+=scratch[kc.lix-(step>>1)];
                scratchBuf.array(kc.lix, scratchBuf.array(kc.lix)+scratchBuf.array(kc.lix-(step>>1)));
            }
          kc.barrier();
        }
        int sum=0;
        if ((kc.lix+1) == kc.gsx){
           // sum = scratch[kc.lix];
            sum = scratchBuf.array(kc.lix);

            //scratch[kc.lix]=0;
            scratchBuf.array(kc.lix, 0);
        }
      kc.barrier();
        for (int step=kc.gsx; step >1 ; step>>=1){
            if (((kc.lix+1)%step) == 0){
              //  int prev = scratch[kc.lix-(step>>1)];
                int prev = scratchBuf.array(kc.lix-(step>>1));
              //  scratch[kc.lix-(step>>1)]=scratch[kc.lix];
                scratchBuf.array(kc.lix-(step>>1), scratchBuf.array(kc.lix));
               // scratch[kc.lix]+= prev;
                scratchBuf.array(kc.lix, scratchBuf.array(kc.lix)+prev);
            }
          kc.barrier();
        }

        if ((kc.lix+1) == kc.gsx){
            data[ gid] = sum;
        }else if (gid>0){
          //  data[ gid] = scratch[kc.lix+1];
            data[ gid] = scratchBuf.array(kc.lix+1);
        }
      kc.barrier();
    }


// add s[?] to each element in group[?+1]
// if local_size(0) were 8 and global_size(0) was 40
//      group 0      |    group 1      |    group 2      |    group 3      |     group 4      |
// | 0,1,2,3,4,5,6,7 | 0,1,2,3,4,5,6,7 | 0,1,2,3,4,5,6,7 | 0,1,2,3,4,5,6,7 |  0,1,2,3,4,5,6,7 |
//                 ^                 ^                 ^                 ^                  ^
//                 s0                s1                s2                s3                 s4
//                     0+s0, 1+s0, ....| 0+s1, 1+s1, ....| 0+s2, 1+s2, ....| 0+s3, 1+s4, ....
@CodeReflection
   static  void sumKernel(@RO KernelContext kc, @RW S32Array  dataBuf){
       var scratchBuf = SharedS32x32Array.createLocal();
        int[] data = dataBuf.arrayView();
      //  int[] scratch=scratchBuf.arrayView(); one day
       // scratch[kc.lix] = data[kc.gix];
        scratchBuf.array(kc.lix, data[kc.gix]); // copy into local scratch
      kc.barrier();
        if ((kc.lix+1)!=kc.gsx && kc.gix>0){// don't do this for last in group
           // scratch[kc.lix]+= data[(kc.gix*kc.gsx)-1];
            scratchBuf.array(kc.lix, scratchBuf.array(kc.lix)+ data[(kc.gix*kc.gsx)-1]);
        }
      kc.barrier();
       //data[kc.gix]=scratch[kc.lix];
        data[kc.gix]=scratchBuf.array(kc.lix);
    }


    private static final int GROUP_SIZE = 32;

    @CodeReflection
    private static void compute(ComputeContext cc,  @RW S32Array data) {
        cc.dispatchKernel(data.length(),kc-> groupScan(kc,  data));

        int groupCount = data.length() / GROUP_SIZE; // we assume 32 bit groups
        int log2=1;
        while (log2<groupCount){
            log2<<=1;
        }
        cc.dispatchKernel(data.length(),kc-> crossGroupScan(kc, data));
        cc.dispatchKernel(data.length(),kc-> sumKernel(kc,  data));

    }

    public static void main(String[] args) {
        Accelerator accelerator = new Accelerator(MethodHandles.lookup(), Backend.FIRST);

        S32Array input = S32Array.create(accelerator, GROUP_SIZE *GROUP_SIZE);

        int result = 0;
        for (int i = 0; i < input.length(); i++) {
            var randInt = (int)Math.round( Math.random() );
            result+=randInt;
            input.array(i,randInt);
        }


        // Compute on the accelerator
        accelerator.compute( cc -> PrefixSum.compute(cc, input));


    }

}
