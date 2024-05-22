
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
package hat.backend;

import hat.Accelerator;
import hat.KernelContext;
import hat.NDRange;

import java.util.concurrent.BrokenBarrierException;
import java.util.concurrent.CyclicBarrier;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.function.Consumer;

public class WorkStealer {
    final  int threadCount;
     AtomicInteger taskCount;
    Thread[] threads;

    CyclicBarrier setupBarrier;
    CyclicBarrier startBarrier;
    CyclicBarrier doneBarrier;
    Consumer<NDRange> rangeConsumer;

    volatile int range;
    volatile boolean running = true;
    final int chunkSize = 1024;
    private NDRange[] ranges;

    public WorkStealer(Accelerator accelerator, int threadCount) {
        super();
        System.out.println("new workstealer!");
        this.threadCount = threadCount;
        if (threadCount > 1) {
            taskCount = new AtomicInteger(0);
            threads = new Thread[threadCount];
            this.ranges = new NDRange[threadCount];
            setupBarrier = new CyclicBarrier(threadCount + 1);
            startBarrier = new CyclicBarrier(threadCount + 1);
            doneBarrier = new CyclicBarrier(threadCount + 1);
            for (int i = 0; i < threadCount; i++) {
                final int fini = i;
                ranges[fini] = new NDRange(accelerator);
                threads[fini] = new Thread(() -> {

                    NDRange ndRange = ranges[fini];
                    while (running) {
                        rendezvous(setupBarrier);
                        //  System.out.println("Thread #"+Thread.currentThread()+" waiting start");
                        rendezvous(startBarrier);
                        //  System.out.println("Thread #"+Thread.currentThread()+" started");

                        int myChunk;
                        ndRange.kid = new KernelContext(ndRange,range,0);

                        while ((myChunk = taskCount.getAndIncrement()) < (range / chunkSize) + 1) {

                            for (ndRange.kid.x = myChunk * chunkSize; ndRange.kid.x < (myChunk + 1) * chunkSize && ndRange.kid.x < range; ndRange.kid.x++) {

                                rangeConsumer.accept(ndRange);
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

    public static WorkStealer of(Accelerator accelerator, int threads) {
        return new WorkStealer(accelerator, threads);
    }

    public static WorkStealer usingAllProcessors(Accelerator accelerator) {
        return WorkStealer.of(accelerator, Runtime.getRuntime().availableProcessors());
    }

    public void forEachInRange(NDRange ndRange, Consumer<NDRange> rangeConsumer) {
        if (threadCount > 1) {
            rendezvous(setupBarrier);
            this.taskCount.set(0);
            this.range = ndRange.kid.maxX;
            this.rangeConsumer = rangeConsumer;

            rendezvous(startBarrier);
            // This should start all threads
            rendezvous(doneBarrier);
        } else {
            for (ndRange.kid.x = 0; ndRange.kid.x < range; ndRange.kid.x++) {
                rangeConsumer.accept(ndRange);
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
