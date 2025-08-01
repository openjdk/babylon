
### FuncOpViewer and CodeModelFormatter
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


We have a few 'tools' in [hat/tools/src/main/java/hat/tools](https://github.com/openjdk/babylon/blob/code-reflection/hat/tools)
which might be useful to others

Two in particular

#### FuncOpViewer
This is a swing app which takes a text file containing text form of a CodeModel (as dumped by the OopWriter)


We can generate text for kernels using
```
HAT=SHOW_KERNEL_MODEL java @hat/run ffi-opencl mandel
```

From the standard out copy the text satrting with `func @loc....1` to the closing `}` and paste it into a file (say mandel.cr)

Then

```
java -cp build/hat-tools-1.0.jar hat.tools.textmodel.ui.FuncOpViewer mandle.cr
```

### CodeModelFormatter

Is a terminal based tool (useful if you are accessing a machine via ssh - and can't launch ui)

With it you can dump to the terminal a colorized version of the code model, slightly prettied up with java source for each line inserted as a comment.

```
java -cp build/hat-tools-1.0.jar hat.tools.textmodel.terminal.CodeModelFormatter code.cr
```

