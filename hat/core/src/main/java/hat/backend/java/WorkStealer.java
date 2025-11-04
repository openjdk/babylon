
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
package hat.backend.java;

import hat.Accelerator;
import hat.KernelContext;
import hat.NDRange;

import java.util.concurrent.BrokenBarrierException;
import java.util.concurrent.CyclicBarrier;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.function.Consumer;

public class WorkStealer {
    final int threadCount;
    AtomicInteger taskCount;
    Thread[] threads;

    CyclicBarrier setupBarrier;
    CyclicBarrier startBarrier;
    CyclicBarrier doneBarrier;
    Consumer<KernelContext> rangeConsumer;

    volatile int range;
    volatile boolean running = true;
    final int chunkSize = 1024;
    private KernelContext[] ranges;

    public WorkStealer(int threadCount) {
        super();
      //  System.out.println("new workstealer!");
        this.threadCount = threadCount;
        if (threadCount > 1) {
            taskCount = new AtomicInteger(0);
            threads = new Thread[threadCount];
            this.ranges = new KernelContext[threadCount];
            setupBarrier = new CyclicBarrier(threadCount + 1);
            startBarrier = new CyclicBarrier(threadCount + 1);
            doneBarrier = new CyclicBarrier(threadCount + 1);
            for (int i = 0; i < threadCount; i++) {
                final int fini = i;
                ranges[fini] = new KernelContext(range);
                threads[fini] = new Thread(() -> {

                    hat.KernelContext kernelContext = ranges[fini];
                    while (running) {
                        rendezvous(setupBarrier);
                        //  System.out.println("Thread #"+Thread.currentThread()+" waiting start");
                        rendezvous(startBarrier);
                        //  System.out.println("Thread #"+Thread.currentThread()+" started");

                        int myChunk;
                        while ((myChunk = taskCount.getAndIncrement()) < (range / chunkSize) + 1) {
                            for (kernelContext.gix = myChunk * chunkSize; kernelContext.gix < (myChunk + 1) * chunkSize && kernelContext.gix < range; kernelContext.gix++) {
                                rangeConsumer.accept(kernelContext);
                            }
                        }
                        //  System.out.println("Thread #"+Thread.currentThread()+" done");
                        rendezvous(doneBarrier);
                        //  System.out.println("Thread #"+Thread.currentThread()+" pastDone");
                    }
                });
                threads[i].setDaemon(true);
                threads[i].start();
            }
        }
        // After this all threads should be parked waiting for us to
        // set max
        // set work
        // rendezvous on startBarrier
        // rendezvous on endBarrier

    }

    public static WorkStealer of(int threads) {
        return new WorkStealer(threads);
    }

    public static WorkStealer usingAllProcessors() {
        return WorkStealer.of(Runtime.getRuntime().availableProcessors());
    }

    public void forEachInRange(KernelContext kernelContext, Consumer<KernelContext> rangeConsumer) {
        if (threadCount > 1) {
            rendezvous(setupBarrier);
            this.taskCount.set(0);
            this.range = kernelContext.gsx;
            this.rangeConsumer = rangeConsumer;

            rendezvous(startBarrier);
            // This should start all threads
            rendezvous(doneBarrier);
        } else {
            for (kernelContext.gix = 0; kernelContext.gix < range; kernelContext.gix++) {
                rangeConsumer.accept(kernelContext);
            }
        }
    }

    void rendezvous(CyclicBarrier barrier) {
        try {
            barrier.await();
        } catch (BrokenBarrierException | InterruptedException e) {
            throw new IllegalStateException(e);
        }
    }


}
