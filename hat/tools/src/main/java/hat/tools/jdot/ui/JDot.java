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
package hat.tools.jdot.ui;

import hat.tools.jdot.DotBuilder;
import hat.tools.json.Json;
import hat.tools.json.JsonArray;
import hat.tools.json.JsonNumber;
import hat.tools.json.JsonObject;
import hat.tools.json.JsonString;
import hat.tools.json.JsonValue;

import javax.swing.JComponent;
import javax.swing.JFrame;
import javax.swing.JScrollPane;
import javax.swing.SwingUtilities;
import java.awt.BorderLayout;
import java.awt.Color;
import java.awt.Dimension;
import java.awt.Font;
import java.awt.FontMetrics;
import java.awt.Graphics;
import java.awt.Graphics2D;
import java.awt.RenderingHints;
import java.awt.Shape;
import java.awt.event.MouseAdapter;
import java.awt.event.MouseEvent;
import java.awt.font.TextAttribute;
import java.awt.geom.CubicCurve2D;
import java.awt.geom.Line2D;
import java.awt.geom.Path2D;
import java.awt.geom.Point2D;
import java.awt.geom.Rectangle2D;
import java.awt.geom.RectangularShape;
import java.awt.geom.RoundRectangle2D;
import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.regex.Matcher;
import java.util.regex.Pattern;
import java.util.stream.Stream;

public class JDot {
    public static final String FloatRegex = "([0-9]+|[0-9]*\\.[0-9]*)"; // not completely foolproof but seems to work for  json0 DOT
    public static final String FloatPointRegex = FloatRegex + "," + FloatRegex;
    public static final Pattern RectRegex = Pattern.compile(FloatPointRegex + "," + FloatPointRegex);
    public static final Pattern PointRegex = Pattern.compile(FloatPointRegex);

    public static class JsonQuery {
        public static JsonValue query(JsonValue jsonValue, String path) {
            return jsonValue instanceof JsonObject jsonObject && jsonObject.members().get(path) instanceof JsonValue v ? v : null;
        }

        public static String str(JsonValue jsonValue, String path) {
            return query(jsonValue, path) instanceof JsonString jsonString ? jsonString.value() : null;
        }

        public static Number num(JsonValue jsonValue, String path) {
            return (query(jsonValue, path) instanceof JsonNumber jsonNumber) ? jsonNumber.toNumber() : null;
        }

        public static Float floatOr(JsonValue jsonValue, String path, float defaultValue) {
            return JsonQuery.num(jsonValue, path) instanceof JsonNumber n ? n.toNumber().floatValue() : defaultValue;
        }

        public static Float floatStrOr(JsonValue jsonValue, String path, float defaultValue) {
            return JsonQuery.str(jsonValue, path) instanceof String s ? Float.parseFloat(s) : defaultValue;
        }

        public static List<JsonValue> arr(JsonValue jsonValue, String path) {
            return (jsonValue instanceof JsonObject jsonObject
                    && jsonObject.members().get(path) instanceof JsonArray jsonArray) ? jsonArray.values() : null;
        }

        public static Number obj(JsonValue jsonValue, String path) {
            return (jsonValue instanceof JsonObject jsonObject
                    && jsonObject.members().get(path) instanceof JsonNumber jsonNumber) ? jsonNumber.toNumber() : null;
        }

        public static Color colorOr(JsonValue jsonValue, String path, Color defaultColor) {
            return JsonQuery.str(jsonValue, path) instanceof String s ? Color.getColor(s) : defaultColor;
        }

    }

    public interface Label {
        String value();
    }

    public static class SimpleLabel implements Label {
        public String value;

        @Override
        public String value() {
            return value;
        }

        SimpleLabel(String value) {
            this.value = value;
        }
    }


    abstract static class Renderable {
        final JsonValue jsonValue;
        public Color color;
        public Color fillColor;
        public Color fontColor;
        final int id;


        public abstract void render(Graphics2D g2d);

        Renderable(JsonValue jsonValue) {
            this.jsonValue = jsonValue;
            this.id = JsonQuery.num(jsonValue, "_gvid").intValue();
            this.color = JsonQuery.colorOr(jsonValue, "color", Color.BLACK);
            this.fillColor = JsonQuery.colorOr(jsonValue, "fillcolor", Color.WHITE);
            this.fontColor = JsonQuery.colorOr(jsonValue, "fontcolor", Color.BLACK);
        }
    }

    abstract static class RenderableShape<T extends Shape> extends Renderable {
        final String name;
        final float width;
        final float height;
        T shape;

        Label label;

        RenderableShape(JsonValue jsonValue) {
            super(jsonValue);
            this.name = JsonQuery.str(jsonValue, "name");
            // for reasons inexplicable.  width and height are inches so need to be *72 or *60 ?
            this.height = JsonQuery.floatStrOr(jsonValue, "height", 0f) * 60;
            this.width = JsonQuery.floatStrOr(jsonValue, "width", 0f) * 60;
        }
    }

    static class Line extends RenderableShape<Line2D.Float> {
        Line(JsonValue jsonValue, Point2D.Float start, Point2D.Float end) {
            super(jsonValue);
            shape = new Line2D.Float(start, end);
        }

        public void render(Graphics2D g2d) {
            g2d.setColor(color);
            g2d.draw(this.shape);
        }
    }

    abstract static class TextBox<T extends RectangularShape> extends RenderableShape<T> {

        TextBox(JsonValue jsonValue) {
            super(jsonValue);
        }

        @Override
        public void render(Graphics2D g2d) {
            g2d.setColor(this.fillColor);
            g2d.fill(this.shape);
            g2d.setColor(this.color);
            g2d.draw(this.shape);
            g2d.setColor(this.fontColor);
            FontMetrics metrics = g2d.getFontMetrics(g2d.getFont());
            float xStr = (float) this.shape.getCenterX() - ((float) metrics.stringWidth(label.value()) / 2);
            float yStr = (float) this.shape.getCenterY() - ((float) metrics.getHeight() / 2) + metrics.getAscent();
            g2d.drawString(label.value(), xStr, yStr);
        }
    }

    static class RoundedTextBox extends TextBox<RoundRectangle2D.Float> {
        RoundedTextBox(JsonValue jsonValue, List<Point2D.Float> points) {
            super(jsonValue); // we need the name
            this.shape = new RoundRectangle2D.Float(
                    points.getFirst().x - width / 2,
                    points.getFirst().y - height / 2,
                    width, height,
                    20,
                    20);
            this.label = new SimpleLabel(name);
        }

    }

    abstract static class SquareTextBox extends TextBox<Rectangle2D.Float> {
        SquareTextBox(JsonValue jsonValue, Point2D.Float[] points, SimpleLabel label) {
            super(jsonValue);
            this.shape = new Rectangle2D.Float(points[0].x, points[0].y, points[1].x - points[0].x, points[1].y - points[0].y);
            this.label = label;
        }
    }

    abstract static class RecordShape<T extends RectangularShape> extends RenderableShape<T> {

        /*
                           " One | Two "                    [One|Two]
                           " One | Two | Three "            [One|Two|Three]
                           " One | Two | Three | Four "     [One|Two|Three|Four]
                                                            +---+-------+----+
                                                            |   | Two   |    |
                           " One |{ Two | Three }| Four "   |One+-------+Four+
                                                            |   | Three |    |
                                                            +---+-------+----+
        */
        private final RecordLabel recordLabel;

        public static class RecordLabel implements Label {
            public static class Box implements Label {
                public static class Port {
                    String name = "";

                    void append(char ch) {
                        name += ch;
                    }
                }

                public Box parent;
                public boolean v;
                public Port port;
                private String value;

                @Override
                public String value() {
                    return value;
                }

                Rectangle2D.Float rect;

                public Box(Box parent, boolean v, Rectangle2D.Float rect) {
                    this.parent = parent;
                    this.v = v;
                    this.rect = rect;
                    this.value = "";
                    this.port = null;
                }

                @Override
                public String toString() {
                    return (v ? "V" : "H") + "Box " + (port == null ? "" : "<" + port.name + ">") + value + " " +
                            "x1=" + rect.getMinX() + " y1=" + rect.getMinY() + "x2=" + rect.getMaxX() + " y2=" + rect.getMaxY() + " w=" + rect.getWidth() + " h=" + rect.getHeight();
                }

                public void append(char ch) {
                    value += ch;
                }

                public void portAppend(char ch) {
                    port.name += ch;
                }
            }

            private final String value;

            @Override
            public String value() {
                return value;
            }

            List<Box> boxes = new ArrayList<>();

            RecordLabel(String value,
                        List<Rectangle2D.Float> rects) {
                this.value = value;
                // this.points = points;
                int boxIdx = 0;

                /*
                 *  <name> port
                 *  | is  used to separate record 'boxes'
                 *                            +---------------------+
                 *  |here|there|everywhere -> |here|there|everywhere|
                 *                            +---------------------+
                 *  {} used (with bar) to change dir
                 *                            +---------------+
                 *                            |    |there     |
                 *  |here|{there|everywhere}  |here+----------+
                 *                            |    |everywhere|
                 *                            +---------------+
                 */

                int len = value.length();
                enum STATE {
                    NORMAL,
                    INPORT,
                    ESCAPING,
                    ERR
                }
                STATE state = STATE.NORMAL;
                boxes.add(new Box(null, false, rects.get(boxIdx++)));
                boolean v = false;
                for (int idx = 0; idx < len && state != STATE.ERR; idx++) {
                    char ch = value.charAt(idx);

                    switch (state) {
                        case NORMAL -> {
                            switch (ch) {
                                case '\\' -> state = STATE.ESCAPING;
                                case '<' -> {
                                    if (boxes.getLast().port == null) {
                                        state = STATE.INPORT;
                                        boxes.getLast().port = new Box.Port();
                                    } else {
                                        state = STATE.ERR;
                                        System.out.println("one port per box!");
                                    }
                                }
                                case '|' -> {
                                    if ((idx + 1 < len) && (value.charAt(idx + 1) == '{')) {
                                        v = !v;
                                        idx++;
                                    }
                                    this.boxes.add(new Box(this.boxes.getLast(), v, rects.get(boxIdx++)));
                                }
                                case '{' -> {
                                    v = !v;
                                    this.boxes.add(new Box(this.boxes.getLast(), v, rects.get(boxIdx++)));
                                }
                                case '}' -> v = !v;
                                default -> boxes.getLast().append(ch);

                            }
                        }
                        case INPORT -> {
                            switch (ch) {
                                case '>' -> state = STATE.NORMAL;
                                default -> {
                                    if (Character.isAlphabetic(ch) || Character.isDigit(ch)) {
                                        boxes.getLast().port.append(ch);
                                    } else {
                                        state = STATE.ERR;
                                        System.out.println("invalid char '" + ch + "' in port");
                                    }
                                }
                            }
                        }
                        case ESCAPING -> {
                            boxes.getLast().append(ch);
                            state = STATE.NORMAL;
                        }
                    }
                }
                if (state == STATE.ERR) {
                    throw new IllegalStateException("err!");
                }
            }


            @Override
            public String toString() {
                StringBuilder sb = new StringBuilder();
                boxes.forEach(b -> sb.append("    ").append(b).append('\n'));
                sb.append("}").append('\n');
                return sb.toString();
            }
        }

        public RecordShape(JsonValue jsonValue, List<Rectangle2D.Float> rects) {
            super(jsonValue);
            this.recordLabel = new RecordLabel(JsonQuery.str(jsonValue, "label"), rects);
        }

        @Override
        public void render(Graphics2D g2d) {
            Map<TextAttribute, Object> attributes = new HashMap<>();
            // The Text was too close to the edge of the box, so this hack rescales the font 96% so we can avoid the edges.
            var currentFont = g2d.getFont();
            attributes.put(TextAttribute.FAMILY, currentFont.getFamily());
            attributes.put(TextAttribute.WEIGHT, TextAttribute.WEIGHT_SEMIBOLD);
            attributes.put(TextAttribute.SIZE, (int) (currentFont.getSize() * .96));
            var myFont = Font.getFont(attributes);
            g2d.setFont(myFont);
            FontMetrics metrics = g2d.getFontMetrics(g2d.getFont());
            // These are the boxes that comprise the record
            recordLabel.boxes.forEach(box -> {
                g2d.setColor(this.fillColor);
                g2d.fill(box.rect);
                g2d.setColor(this.color);
                g2d.draw(box.rect);
                g2d.setColor(this.fontColor);
                Rectangle2D stringBounds = metrics.getStringBounds(box.value(), g2d);
                g2d.drawString(box.value(), (float) (box.rect.getCenterX() - stringBounds.getCenterX()), (float) (box.rect.getCenterY() - stringBounds.getCenterY()));

            });
            //g2d.setColor(Color.RED); //useful for debugging
            //g2d.fill(this.shape);

            //switch back to the original sized font
            g2d.setFont(currentFont);
        }
    }

    static Rectangle2D.Float sqRect(List<Point2D.Float> points) {
        return new Rectangle2D.Float(points.getFirst().x - 1, points.getFirst().y - 1, 2f, 2f);
    }

    static RoundRectangle2D.Float roundRect(List<Point2D.Float> points, float radius) {
        var sqr = sqRect(points);
        return new RoundRectangle2D.Float(sqr.x, sqr.y, sqr.width, sqr.height, radius, radius);
    }

    static class RoundedRecordShape extends RecordShape<RoundRectangle2D.Float> {
        public RoundedRecordShape(JsonValue jsonValue, List<Point2D.Float> points, List<Rectangle2D.Float> rects) {
            super(jsonValue, rects);
            this.shape = roundRect(points, 20f);
        }
    }

    static class SqRecordShape extends RecordShape<Rectangle2D.Float> {
        public SqRecordShape(JsonValue jsonValue, List<Point2D.Float> points, List<Rectangle2D.Float> rects) {
            super(jsonValue, rects);
            this.shape = sqRect(points);
        }
    }

    static class Curve extends Renderable {
        List<Point2D.Float> pos;
        List<Shape> shapesToDraw = new ArrayList<>();
        List<Shape> tailShapesToFill = new ArrayList<>();
        List<Shape> headShapesToFill = new ArrayList<>();
        Line2D.Float headLine;
        Line2D.Float tailLine;
        Path2D.Float tailPath;
        long head;
        long tail;
        float weight;
        char posType;

        /*
         * From the DOT docs.
         * spline = (endp)? (startp)? point (triple)+
         * and triple = point point point
         * and endp = "e,%f,%f"
         * and startp = "s,%f,%f"
         * If a spline has points p₁ p₂ p₃ ... pₙ, (n = 1 (mod 3)), the points correspond to the
         * control points of a cubic B-spline from p₁ to pₙ. If startp is given, it touches one node
         * of the edge, and the arrowhead goes from p₁ to startp. If startp is not given, p₁ touches a node.
         *  Similarly for pₙ and endp.
         */

        public Curve(JsonValue jasonValue, char posType, List<Point2D.Float> pos) {
                    /*
                     https://stackoverflow.com/questions/71744148/graphviz-dot-output-what-are-the-4-6-pos-coordinates-for-an-edge
                     https://forum.graphviz.org/t/fun-with-edges/888/4
                     https://stackoverflow.com/questions/3162645/convert-a-quadratic-bezier-to-a-cubic-one
                     https://stackoverflow.com/questions/65410883/bezier-curve-forcing-a-curve-of-4-points-to-pass-through-control-points-in-3d
                    */

            super(jasonValue);
            this.head = JsonQuery.obj(jsonValue, "head").longValue();
            this.tail = JsonQuery.obj(jsonValue, "tail").longValue();
            this.weight = JsonQuery.floatOr(jsonValue, "weight", 0f);
            this.posType = posType;
            this.pos = pos;
            if (this.posType == 'e') {
                /*
                 * triple = (point point point)
                 * curve = (endp/pos[0])? (startp/pos[1])? point/pos[2] triple+
                 */

                this.headLine = new Line2D.Float(pos.get(1).x, pos.get(1).y, pos.get(2).x, pos.get(2).y);

                int n = this.pos.size();
                for (int posIdx = 1; posIdx < (n - 1); posIdx += 3) {
                    this.shapesToDraw.add(
                            new CubicCurve2D.Float(
                                    this.pos.get(posIdx).x,
                                    this.pos.get(posIdx).y,
                                    this.pos.get(posIdx + 1).x,
                                    this.pos.get(posIdx + 1).y,
                                    this.pos.get(posIdx + 2).x,
                                    this.pos.get(posIdx + 2).y,
                                    this.pos.get(posIdx + 3).x,
                                    this.pos.get(posIdx + 3).y));
                }
                this.tailLine = new Line2D.Float(this.pos.get(n - 1).x, this.pos.get(n - 1).y, this.pos.getFirst().x, this.pos.getFirst().y);
                float hypot = (float) Math.hypot(this.tailLine.x1 - this.tailLine.x2, this.tailLine.y1 - this.tailLine.y2);
                this.tailPath = new Path2D.Float();
                tailPath.moveTo(hypot / 2, 0);
                tailPath.lineTo(0, hypot);
                tailPath.lineTo(-hypot / 2, 0);
                tailPath.closePath();
                this.tailShapesToFill.add(tailPath);
            } else {
                throw new IllegalStateException("no support for " + posType);
            }
        }

        @Override
        public void render(Graphics2D g2d) {
            g2d.setColor(color);
            shapesToDraw.forEach(g2d::draw);
            // we create our own g2d copy
            // Translate to the end of the tail
            // Rotate perpendicular to the tail line segment
            final Graphics2D tail2d = (Graphics2D) g2d.create();

            tail2d.translate(tailLine.x1, tailLine.y1);
            tail2d.rotate(-Math.atan2(tailLine.x2 - tailLine.x1, tailLine.y2 - tailLine.y1));
            tailShapesToFill.forEach(tail2d::fill);
            // we create our own g2d copy
            // Translate to the end of the head
            // Rotate perpendicular to the head line segment
            final Graphics2D head2d = (Graphics2D) g2d.create();
            head2d.translate(headLine.x1, headLine.y1);
            head2d.rotate(-Math.atan2(headLine.x2 - headLine.x1, headLine.y2 - headLine.y1));
            headShapesToFill.forEach(head2d::fill);
        }
    }


    double scale = 1;

    public JScrollPane pane;

    JComponent viewer;

    final JsonValue jsonValue;
    final Rectangle2D.Float bounds;


    public JDot(JsonValue jsonValue) {
        this.jsonValue = jsonValue;
        String bb = JsonQuery.str(jsonValue, "bb");
        this.bounds = RectRegex.matcher(bb) instanceof Matcher m && m.matches() && m.groupCount() == 4
                ? new Rectangle2D.Float(Float.parseFloat(m.group(1)), Float.parseFloat(m.group(2)),
                Float.parseFloat(m.group(3)), Float.parseFloat(m.group(4)))
                : null;

        List<Renderable> renderables = new ArrayList<>();


        JsonQuery.arr(jsonValue, "edges").forEach(edgeJsonValue -> {
            var posStr = JsonQuery.str(edgeJsonValue, "pos");

            var points = Arrays.stream(posStr.substring(2).split(" "))
                    .map(s -> (PointRegex.matcher(s) instanceof Matcher m && m.matches() && m.groupCount() == 2)
                            ? new Point2D.Float(Float.parseFloat(m.group(1)), bounds.height - Float.parseFloat(m.group(2)))
                            : null).toList();
            var curve = new Curve(edgeJsonValue, posStr.charAt(0), points);
            renderables.add(curve);
        });
        JsonQuery.arr(jsonValue, "objects").forEach(objectJsonValue -> {
            var shape = JsonQuery.str(objectJsonValue, "shape");
            var posStr = JsonQuery.str(objectJsonValue, "pos");
            var points = Arrays.stream(posStr.split(" "))
                    .map(s -> (PointRegex.matcher(s) instanceof Matcher m && m.matches() && m.groupCount() == 2)
                            ? new Point2D.Float(Float.parseFloat(m.group(1)), bounds.height - Float.parseFloat(m.group(2)))
                            : null).toList();

            if (shape != null && shape.equals("record")) {
                var rectString = JsonQuery.str(objectJsonValue, "rects").split(" ");
                var recordRects = Arrays.stream(rectString).map(s -> {
                    if (RectRegex.matcher(s) instanceof Matcher m && m.matches() && m.groupCount() == 4) {
                        var x1 = Float.parseFloat(m.group(1));
                        var y1 = bounds.height - Float.parseFloat(m.group(4));
                        var x2 = Float.parseFloat(m.group(3));
                        var y2 = bounds.height - Float.parseFloat(m.group(2));
                        return new Rectangle2D.Float(x1, y1, x2 - x1, y2 - y1);
                    } else {
                        return null;
                    }
                }).toList();
                renderables.add(new RoundedRecordShape(objectJsonValue, points, recordRects));
            } else {
                renderables.add(new RoundedTextBox(objectJsonValue, points));
            }
        });

        this.viewer =
                new JComponent() {
                    @Override
                    public void paintComponent(Graphics g1d) {
                        super.paintComponent(g1d);
                        if (g1d instanceof Graphics2D g2d) {
                            g2d.scale(scale, scale);
                            g2d.setRenderingHints(new RenderingHints(RenderingHints.KEY_ANTIALIASING, RenderingHints.VALUE_ANTIALIAS_ON));
                            renderables.forEach(e -> e.render(g2d));
                        }
                    }
                };

        viewer.addMouseListener(
                new MouseAdapter() {
                    @Override
                    public void mouseClicked(MouseEvent e) {
                        // System.out.println(e.getPoint());
                        // synchronized (doorBell) {
                        //   doorBell.notify();
                        // }
                    }
                });
        viewer.addMouseWheelListener(
                e -> {
                    scale *= (e.getWheelRotation() < 0) ? 1.01 : 1 / 1.01;
                    viewer.repaint();
                });

        this.pane = new JScrollPane(this.viewer);
        pane.setPreferredSize(new Dimension((int) (bounds.width * scale), (int) (bounds.height * scale)));
    }

    static JsonValue dotToJson(Path dotFile) {
        final Path dotExecutable =
                Stream.of(
                                System.getProperty("DOT_PATH"),
                                "/usr/bin/dot",
                                "/opt/homebrew/bin/dot"
                        )
                        .filter(Objects::nonNull) // incase we hve no var
                        .map(Path::of)
                        .filter(Files::isExecutable)
                        .findFirst()
                        .orElse(null);
        try {
            Process process = new ProcessBuilder()
                    .command(dotExecutable.toString(), "-Tjson0", dotFile.toString())
                   // .redirectErrorStream(true)
                    .start();

            String jsonText = String.join("\n", new BufferedReader(new InputStreamReader(process.getInputStream())).readAllLines());

            process.waitFor();

              boolean success = (process.exitValue() == 0);
            if (!success) {
                throw new RuntimeException("DOT  exited with code " + process.exitValue());
            }
            return Json.parse(jsonText);

        } catch (IOException e) {
            throw new RuntimeException(e);
        } catch (InterruptedException e) {
            throw new RuntimeException(e);
        }
    }

    static JsonValue dotToJson(String dotText) {
        try {
            Path tmp = Files.createTempFile("", ".dot");
            tmp.toFile().deleteOnExit();
            Files.writeString(tmp, dotText);
            return dotToJson(tmp);

        } catch (IOException e) {
            throw new RuntimeException(e);
        }
    }

    public JDot(Path dotFile) {
        this(dotToJson(dotFile));
    }

    public JDot(String dotText) {
        this(dotToJson(dotText));
    }


    public static void main(String[] args) throws IOException {
        SwingUtilities.invokeLater(() -> {
            JDot jDot = new JDot(DotBuilder.dotDigraph("cmake",db->db
                    .assign("rankdir", "RL")
                    .nodeShape("record")
                    .record("backend-ffi-opencl", "backend|{ffi|extracted}|opencl")
                    .record("backend-ffi-shared", "backend|ffi|<in>shared")
                    .record("backend-ffi", "backend|ffi")
                    .record("core",  "core")
                    .record("cmake-info-opencl", "<in>cmake|info|opencl")
                    .edge("backend-ffi-opencl","backend-ffi-shared:se")
                    .edge("backend-ffi-shared","backend-ffi:n")
                    .edge("backend-ffi","core:s")
                    .edge("backend-ffi-opencl","cmake-info-opencl:ne")
            ));
            var frame = new JFrame();
            frame.setLayout(new BorderLayout());
            frame.getContentPane().add(jDot.pane);
            frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
            frame.pack();
            frame.setVisible(true);
        });
    }

}
