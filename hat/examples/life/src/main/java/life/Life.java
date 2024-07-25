package life;

import hat.Accelerator;
import hat.ComputeContext;
import hat.KernelContext;
import hat.backend.Backend;
import hat.buffer.Buffer;
import hat.ifacemapper.Schema;
import io.github.robertograham.rleparser.RleParser;
import io.github.robertograham.rleparser.domain.PatternData;

import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.lang.invoke.MethodHandles;
import java.lang.runtime.CodeReflection;

import static java.lang.foreign.ValueLayout.JAVA_BYTE;
import static java.lang.foreign.ValueLayout.JAVA_INT;

public class Life {

    public interface CellGrid extends Buffer {
        int width();

        int height();

        byte cell(long idx);

        void cell(long idx, byte b);

        Schema<CellGrid> schema = Schema.of(CellGrid.class, lifeData -> lifeData
                .arrayLen("width", "height").stride(2).array("cell")
        );

        static CellGrid create(Accelerator accelerator, int width, int height) {
            return schema.allocate(accelerator, width,height);
        }

        ValueLayout valueLayout = JAVA_BYTE;
        long headerOffset = JAVA_INT.byteOffset() * 2;

        default CellGrid copySliceTo(byte[] bytes, int to) {
            long offset = headerOffset + to * valueLayout.byteOffset();
            MemorySegment.copy(Buffer.getMemorySegment(this), valueLayout, offset, bytes, 0, width()*height());
            return this;
        }
    }

    public interface Control extends Buffer {
        int from();

        void from(int from);

        int to();

        void to(int to);

        Schema<Control> schema = Schema.of(Control.class, lifeSupport -> lifeSupport.fields("from", "to"));

        static Control create(Accelerator accelerator, CellGrid cellGrid) {
            var instance = schema.allocate(accelerator);
            instance.to(cellGrid.width()*cellGrid.height());
            instance.from(0);
            return instance;
        }
    }


    public final static byte ALIVE = (byte) 0xff;
    public final static byte DEAD = 0x00;

    public static class Compute {
        @CodeReflection
        public static void life(KernelContext kc, Control control, CellGrid cellGrid) {
            if (kc.x < kc.maxX) {
                int w = cellGrid.width();
                int h = cellGrid.height();
                int from = control.from();
                int x = kc.x % w;
                int y = kc.x / w;
                byte cell = cellGrid.cell(kc.x + from);
                if (x > 0 && x < (w - 1) && y > 0 && y < (h - 1)) { // passports please
                    int count = 0;
                    for (int dx = -1; dx <= 1; dx++) {
                        for (int dy = -1; dy <= 1; dy++) {
                            if (!(dx == 0 && dy == 0)) {
                                int offset = from + y * w + dy * w + x + dx;
                                count += cellGrid.cell(offset) & 1;
                            }
                        }
                    }
                    cell = ((count == 3) || ((count == 2) && (cell == ALIVE))) ? ALIVE : DEAD;// B3/S23.
                }
                cellGrid.cell(kc.x + control.to(), cell);
            }
        }

        @CodeReflection
        static public void compute(final ComputeContext computeContext, Control control, CellGrid cellGrid) {
            computeContext.dispatchKernel(cellGrid.width()*cellGrid.height(), kc -> Compute.life(kc, control, cellGrid));
        }
    }


    public static void main(String[] args) {
        boolean headless = Boolean.getBoolean("headless") || (args.length > 0 && args[0].equals("--headless"));

        Accelerator accelerator = new Accelerator(MethodHandles.lookup(), Backend.FIRST);

        PatternData patternData = RleParser.readPatternData(
                Life.class.getClassLoader().getResourceAsStream("orig.rle")
        );
        CellGrid cellGrid = CellGrid.create(accelerator,
                (((patternData.getMetaData().getWidth() + 2) / 16) + 1) * 16,
                (((patternData.getMetaData().getHeight() + 2) / 16) + 1) * 16
        );
        patternData.getLiveCells().getCoordinates().stream().forEach(c ->
                cellGrid.cell((1 + c.getX()) + (1 + c.getY()) * cellGrid.width(), ALIVE)
        );

        Control control = Control.create(accelerator, cellGrid);
        final Viewer viewer = new Viewer("Life", control, cellGrid);
        viewer.update();
        viewer.waitForStart();

        final long startMillis = System.currentTimeMillis();

        for (int generation = 0; generation < Integer.MAX_VALUE; generation++) {
            accelerator.compute(cc -> Compute.compute(cc, control, cellGrid));
            //swap from/to
            int tmp = control.from();
            control.from(control.to());
            control.to(tmp);
            long elapsedMs = System.currentTimeMillis() - startMillis;
            viewer.setGeneration(generation, ((generation * 1000f) / elapsedMs));
        }
    }
}
