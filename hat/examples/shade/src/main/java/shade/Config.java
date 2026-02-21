package shade;

import hat.Accelerator;
import hat.util.ui.Menu;
import hat.util.ui.SevenSegmentDisplay;

import javax.swing.JMenuBar;
import javax.swing.JToggleButton;

public class Config {
    private final Accelerator accelerator;

    private final boolean showAllocations;

    private final int width;
    private final int height;
    private final int targetFps;
    private final String shaderName;
    private final Shader shader;


    public Accelerator accelerator() {
        return accelerator;
    }

    public boolean showAllocations() {
        return showAllocations;
    }

    public int width() {
        return width;
    }


    public int height() {
        return height;
    }


    public int targetFps() {
        return targetFps;
    }


    public Shader shader() {
        return shader;
    }


    public String shaderName() {
        return shaderName;
    }


    public boolean running() {
        return running != null && running.isSelected();
    }


    private final Menu menu;

    public Menu menu() {
        return menu;
    }

    private final boolean showTargetFps;

    private boolean showTargetFps() {
        return showTargetFps;
    }

    private final boolean showActualFps;

    private boolean showFps() {
        return showActualFps;
    }

    private final boolean showShaderTimeUs;

    private boolean showShaderTimeUs() {
        return showShaderTimeUs;
    }


    private final boolean showElapsedMs;

    private boolean showElapsedMs() {
        return showElapsedMs;
    }

    private final boolean showFrameNumber;

    private boolean showFrameCount() {
        return showFrameNumber;
    }

    private SevenSegmentDisplay allocations7Seg;
    private SevenSegmentDisplay shaderTimeUs7Seg;
    private SevenSegmentDisplay targetFps7Seg;
    private SevenSegmentDisplay actualFps7Seg;
    private SevenSegmentDisplay frameCount7Seg;
    private SevenSegmentDisplay elapsedMs7Seg;
    private JToggleButton running;

    public Config(
            Accelerator accelerator,
            int width,
            int height,
            int targetFps,
            String shaderName,
            Shader shader,
            boolean showTargetFps,
            boolean showActualFps,
            boolean showShaderTimeUs,
            boolean showAllocations,
            boolean showElapsedMs,
            boolean showFrameNumber
    ) {
        this.accelerator = accelerator;
        this.width = width;
        this.height = height;
        this.targetFps = targetFps;
        this.shaderName = shaderName;
        this.shader = shader;
        this.showTargetFps = showTargetFps;
        this.showActualFps = showActualFps;
        this.showShaderTimeUs = showShaderTimeUs;
        this.showAllocations = showAllocations;
        this.showElapsedMs = showElapsedMs;

        this.showFrameNumber = showFrameNumber;
        this.menu = new Menu(new JMenuBar())
                .exit();
        if (showAllocations) {
            this.menu
                    .label("Vectors + Mats")
                    .sevenSegment(10, 15, $ -> allocations7Seg = $).space(20);
        }

        if (showShaderTimeUs) {
            this.menu
                    .label("Shader Time (us)")
                    .sevenSegment(6, 15, $ -> shaderTimeUs7Seg = $).space(20);
        }
        if (showFrameNumber) {
            this.menu
                    .label("Frame ").sevenSegment(6, 15, $ -> frameCount7Seg = $).space(20);
        }
        if (showElapsedMs) {
            this.menu
                    .label(showElapsedMs, "Elapsed (ms)")
                    .sevenSegment(6, 15, $ -> elapsedMs7Seg = $).space(20);
        }
        if (showTargetFps) {
            this.menu.label("Target Frames (per sec)").sevenSegment(4, 15, $ -> {
                targetFps7Seg = $;
                targetFps7Seg.set(targetFps);
            }).space(20);

        }
        if (showActualFps) {
            this.menu.label("Actual Frames (per sec)").sevenSegment(4, 15, $ -> actualFps7Seg = $).space(20);
        }
        this.menu
                .toggle("Stop", "Go", true, $ -> running = $, _ -> {
                })
                .space(40);
    }

    Config allocations(int v) {
        if (showAllocations) {
            allocations7Seg.set(v);
        }
        return this;
    }

    Config shaderTimeUs(int v) {
        if (showShaderTimeUs) {
            shaderTimeUs7Seg.set(v);
        }
        return this;
    }


    Config actualFps(int v) {
        if (showActualFps) {
            actualFps7Seg.set(v);
        }
        return this;
    }

    Config frameNumber(int v) {
        if (showFrameNumber) {
            frameCount7Seg.set(v);
        }
        return this;
    }

    Config elapsedMs(int v) {
        if (showElapsedMs) {
            elapsedMs7Seg.set(v);
        }
        return this;
    }

    public static Config of(
            Accelerator accelerator,
            int width,
            int height,
            int targetFps,
            String shaderName,
            Shader shader,
            boolean showTargetFps,
            boolean showActualFps,
            boolean showShaderTimeUs,
            boolean showAllocations,
            boolean showElapsedMs,
            boolean showFrameNumber) {
        return new Config(accelerator, width, height, targetFps, shaderName, shader, showTargetFps, showActualFps, showShaderTimeUs, showAllocations, showElapsedMs, showFrameNumber);
    }


    public static Config of(Accelerator accelerator, int width, int height, int targetFps, String name, Shader shader) {
        return new Config(accelerator, width, height, targetFps, name, shader, false, false, false, false, false, false);
    }

    public static Config of(Accelerator accelerator, int width, int height, int targetFps, Shader shader) {
        return new Config(accelerator, width, height, targetFps, shader.getClass().getSimpleName(), shader,
                Boolean.parseBoolean(System.getProperty("showTargetFps", "true")),
                Boolean.parseBoolean(System.getProperty("showActualFps", "true")),
                Boolean.parseBoolean(System.getProperty("showShaderTimeUs", "true")),
                Boolean.parseBoolean(System.getProperty("showAllocations", "false")),
                Boolean.parseBoolean(System.getProperty("showElapsedMs", "false")),
                Boolean.parseBoolean(System.getProperty("showFrameCount", "false")));
    }
}
