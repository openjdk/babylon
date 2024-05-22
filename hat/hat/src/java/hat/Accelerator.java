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
package hat;


import hat.backend.Backend;
import hat.optools.LambdaOpWrapper;
import hat.optools.OpWrapper;
import java.lang.invoke.MethodHandles;
import java.lang.reflect.Method;
import java.lang.reflect.code.Quotable;
import java.lang.reflect.code.Quoted;
import java.lang.reflect.code.op.CoreOp;
import java.util.HashMap;
import java.util.Map;
import java.util.ServiceLoader;
import java.util.function.Consumer;
import java.util.function.Predicate;
/**
 * This class provides the developer facing view of HAT, and wraps a <a href="backend/Backend.html">Backend</a> capable of
 * executing <b>NDRange</b> style execution.
 * <p/>
 * An Accelerator is provided a <a href="java/lang/invoke/MethodHandles.Lookup.html">MethodHandles.Lookup</a> with visibility to the
 * compute to be performed.
 * <p/>
 * As well we either a <a href="backend/Backend.html">Backend</a> directly
 * <pre>
 * Accelerator accelerator =
 *    new Accelerator(MethodHandles.lookup(),
 *       new JavaMultiThreadedBackend());
 * </pre>
 * or a {@code java.util.function.Predicate<Backend>} which can be used to select the required {@code Backend}
 * loaded via Javas ServiceLoader mechanism
 * {@code}
 * <pre>
 * Accelerator accelerator =
 *    new Accelerator(MethodHandles.lookup(),
 *        be -> be.name().startsWith("OpenCL));
 * </pre>}
 *
 * @author Gary Frost
 */
public class Accelerator  {
    public final MethodHandles.Lookup lookup;
    public  final Backend backend;
    private final Map<Method, hat.ComputeContext> cache= new HashMap<>();

    public NDRange range(int max) {
        NDRange ndRange = new NDRange(this);
        ndRange.kid = new KernelContext(ndRange, max, 0);
        return ndRange;
    }
    protected Accelerator(MethodHandles.Lookup lookup, ServiceLoader.Provider<Backend> provider) {
        this(lookup,provider.get());
    }
    /**
     * @param lookup
     * @param backend
     */
    public Accelerator(MethodHandles.Lookup lookup, Backend backend) {
        this.lookup = lookup;
        this.backend = backend;
    }

    /**
     * @param lookup
     * @param backendPredicate
     */
    public Accelerator(MethodHandles.Lookup lookup, Predicate<Backend> backendPredicate) {
        this(lookup, Backend.getBackend(backendPredicate));
    }

    /**
     * An interface used for wrapping the compute entrypoint of work to be performed by the Accelerator.
     * <p/>
     * So given a ComputeClass such as...
     * <pre>
     *  public class MyComputeClass {
     *    @ CodeReflection
     *    public static void addDeltaKernel(KernelContext kc, S32Array arrayOfInt, int delta) {
     *        arrayOfInt.array(kc.x, arrayOfInt.array(kc.x)+delta);
     *    }
     *
     *    @ CodeReflection
     *    static public void doSomeWork(final ComputeContext cc, S32Array arrayOfInt) {
     *    }
     *  }
     *  </pre>
     *  The accelerator will be passed the doSomeWork entrypoint, wrapped in a {@code QuotableComputeContextConsumer}
     *  <pre>
     *  accelerator.compute(cc ->
     *     MyCompute.doSomeWork(cc, arrayOfInt)
     *  );
     *  </pre>
     */
    public interface QuotableComputeContextConsumer extends Quotable, Consumer<ComputeContext> {
    }

    /**
     * This method provides the Accelerator with the {@code Compute Entrypoint} from a Compute class.
     *
     * The entrypoint is wrapped in a <a href="QuotableComputeContextConsumer.html">QuotableComputeContextConsumer</a> lambda.
     *
     * <pre>
     * accelerator.compute(cc -&gt;
     *     MyCompute.doSomeWork(cc, intArray)
     * )
     * </pre>
     */

    public void compute(QuotableComputeContextConsumer quotableComputeContextConsumer) {
        Quoted quoted = quotableComputeContextConsumer.quoted();
        LambdaOpWrapper lambda = OpWrapper.wrap((CoreOp.LambdaOp)quoted.op());
        Method method = lambda.getQuotableTargetMethod();

        // Create (or get cached) a compute context which closes over compute entryppint and reachable kernels.
        // The models of all compute and kernel methods are passed to the backend during creation
        // The backend may well mutate the models.
        // It will also use this opportunity to generate ISA specific code for the kernels.
        ComputeContext computeContext = cache.computeIfAbsent(method,(_)->
                new ComputeContext(this,method)
        );
        // Here we get the captured values  from the Quotable
        Object[] args = lambda.getQuotableCapturedValues(quoted, method);
        args[0]=computeContext;

        // now ask the backend to execute
        backend.dispatchCompute(computeContext, args);
    }
}
