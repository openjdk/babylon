### FuncOpViewer and CodeModelFormatter
[Back to Index ../](../index.md)

We have a few 'tools' in which might be useful to others

Two in particular

#### FuncOpViewer
This is a swing app which takes a text file containing text form of a CodeModel (as dumped by the OopWriter)

We can generate text for kernels using
```bash
HAT=SHOW_KERNEL_MODEL java @.ffi-opencl-example mandel.Main
```

From the standard out copy the text starting with `func @loc....1` to the closing `}` and paste it into a file (say mandel.cr)
Then
```
java -cp build/hat-optkl-1.0.jar:build/hat-tools-1.0.jar hat.tools.textmodel.ui.FuncOpViewer mandle.cr
```

### CodeModelFormatter

Is a terminal based tool (useful if you are accessing a machine via ssh - and can't launch ui)

With it you can dump to the terminal a colorized version of the code model, slightly prettied up with java source for each line inserted as a comment.

```
java -cp build/hat-optkl-1.0.jar:build/hat-tools-1.0.jar  hat.tools.textmodel.terminal.CodeModelFormatter mandle.cr
```

