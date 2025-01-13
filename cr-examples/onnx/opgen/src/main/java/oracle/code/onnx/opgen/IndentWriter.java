package oracle.code.onnx.opgen;

import java.io.IOException;
import java.io.Writer;

final class IndentWriter extends Writer {
    static final int INDENT = 4;

    final Writer w;
    int indent;
    boolean writeIndent = true;

    IndentWriter(Writer w) {
        this(w, 0);
    }

    IndentWriter(Writer w, int indent) {
        this.w = w;
        this.indent = indent;
    }

    @Override
    public void write(char[] cbuf, int off, int len) throws IOException {
        if (writeIndent) {
            w.write(" ".repeat(indent));
            writeIndent = false;
        }

        int end = off + len;
        for (int i = off; i < end; i++) {
            if (cbuf[i] != '\n') {
                continue;
            }

            if (writeIndent) {
                w.write(" ".repeat(indent));
            }

            w.write(cbuf, off, i - off + 1);
            writeIndent = true;
            off = i + 1;
        }
        if (off < end) {
            w.write(cbuf, off, end - off);
        }
    }

    @Override
    public void flush() throws IOException {
        w.flush();
    }

    @Override
    public void close() throws IOException {
        w.close();
    }

    void in() {
        in(INDENT);
    }

    void in(int i) {
        indent += i;
    }

    void out() {
        out(INDENT);
    }

    void out(int i) {
        indent -= i;
    }
}
