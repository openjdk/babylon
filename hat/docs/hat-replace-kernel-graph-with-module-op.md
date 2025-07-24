
### Refactor the existing code for closing over kernel+compute call chains with ModuleOp
----

* [Contents](hat-00.md)
* House Keeping
    * [Project Layout](hat-01-01-project-layout.md)
    * [Building Babylon](hat-01-02-building-babylon.md)
    * [Building HAT](hat-01-03-building-hat.md)
* Programming Model
    * [Programming Model](hat-03-programming-model.md)
* Interface Mapping
    * [Interface Mapping Overview](hat-04-01-interface-mapping.md)
    * [Cascade Interface Mapping](hat-04-02-cascade-interface-mapping.md)
* Implementation Detail
    * [Walkthrough Of Accelerator.compute()](hat-accelerator-compute.md)
    * [How we minimize buffer transfers](hat-minimizing-buffer-transfers.md)

----

At present, we take the CodeModel (FuncOp rooted tree) of the kernel entrypoint, traverse it to
locate calls. We recursively traverse those calls and end up closing over the complete 'graph' of reachable calls.

We have a test case which demonstrates a much more elegant way of doing this and exposing the whole 'closure' as a ModuleOp.

See [test/jdk/java/lang/reflect/code/TestTransitiveInvokeModule.java](https://github.com/openjdk/babylon/blob/code-reflection/test/jdk/java/lang/reflect/code/TestTransitiveInvokeModule.java)

In my defen[sc]e ;) The hat code predated ModuleOp...

So this task would involve taking the test code for generating this closure and use it rather than
the clunky existing version.


See [hat/core/src/main/java/hat/callgraph/CallGraph.java](https://github.com/openjdk/babylon/blob/code-reflection/hat/core/src/main/java/hat/callgraph/CallGraph.java)

And other code in [hat/core/src/main/java/hat/callgraph](https://github.com/openjdk/babylon/tree/code-reflection/hat/core/src/main/java/hat/callgraph)

The existing HAT code does also do some tests on the code (especially kernel code) to determine if the model is valid. We

We check for allocations, exceptions, we assert that all method parameters are mapped byte buffers or primitives.

There are other checks which we might add BTW.

As a followup to this 'chore' .. For bonus points as it were

Me might Consider inlining trivial methods (the life `var()` example above is an interesting candidate ;)

We need to determine which buffers are mutated or just accessed from the kernel call graph.
This is especially important for minimising buffer transfers.
I planned to do this when I implemented the prototype, but it was tougher than I first thought.

Tracing buffer aliasing across calls hurt my head a little.

At present, we rely on annotations @RO,@RW and @WO on kernel args (and trust them) we might still need these for
compute calls which are not code reflectable. But we should not need them for kernel graphs.



