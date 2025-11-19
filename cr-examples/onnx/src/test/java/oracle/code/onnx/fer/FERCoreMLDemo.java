/*
 * Copyright (c) 2025, Oracle and/or its affiliates. All rights reserved.
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * This code is free software; you can redistribute it and/or modify it
 * under the terms of the GNU General Public License version 2 only, as
 * published by the Free Software Foundation.
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

package oracle.code.onnx.fer;

import javax.imageio.ImageIO;
import javax.swing.*;
import java.awt.*;
import java.awt.event.MouseAdapter;
import java.awt.event.MouseEvent;
import java.awt.image.BufferedImage;
import java.io.IOException;
import java.lang.foreign.Arena;
import java.net.URISyntaxException;
import java.net.URL;
import java.nio.file.*;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;
import java.util.Map;
import java.util.logging.Level;
import java.util.logging.Logger;
import java.util.stream.IntStream;

import oracle.code.onnx.OnnxProvider;
import oracle.code.onnx.OnnxRuntime;

public class FERCoreMLDemo {

    static final int IMAGE_SIZE = 64;
    static final String[] EMOTIONS = {
            "neutral", "happiness", "surprise", "sadness",
            "anger", "disgust", "fear", "contempt"
    };

    private static final Logger logger = Logger.getLogger(FERCoreMLDemo.class.getName());

    private static final int MAX_SELECTIONS = 6;
    private static final int MAX_THUMBNAILS = 12;
    private static final String BASE_PATH = "/oracle/code/onnx/fer/";
    public static final String EMPTY_STRING = "";
    public static final String RED_ERROR_SPAN = "<span style='color:red'>Error!</span>";
    private final List<URL> selectedUrls = new ArrayList<>();
    private final boolean useCondensedModel;
    private JFrame frame;
    private JLabel[] imageLabels;
    private JLabel[] resultLabels;
    private final FERInference inference;

    private FERCoreMLDemo(boolean useCondensedModel) {
        this.inference = new FERInference();
        this.useCondensedModel = useCondensedModel;
    }

    public static void main(String[] args) throws IOException, URISyntaxException {
        boolean useArgModel = args.length > 0 && Boolean.parseBoolean(args[0]);
        new FERCoreMLDemo(useArgModel).buildGUI();
    }

    private void buildGUI() throws IOException, URISyntaxException {
        frame = new JFrame("Facial Expression Recognition - CoreML Demo");
        frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
        frame.setLayout(new BorderLayout(10, 10));

        JPanel bigPanel = new JPanel(new GridLayout(2, 3, 10, 10));
        imageLabels = new JLabel[MAX_SELECTIONS];
        resultLabels = new JLabel[MAX_SELECTIONS];
        for (int i = 0; i < MAX_SELECTIONS; i++) {
            JPanel slot = new JPanel(new BorderLayout());
            imageLabels[i] = new JLabel("Empty", SwingConstants.CENTER);
            imageLabels[i].setPreferredSize(new Dimension(300, 300));
            imageLabels[i].setBorder(BorderFactory.createLineBorder(Color.LIGHT_GRAY));
            resultLabels[i] = new JLabel(EMPTY_STRING, SwingConstants.CENTER);
            slot.add(imageLabels[i], BorderLayout.CENTER);
            slot.add(resultLabels[i], BorderLayout.SOUTH);
            bigPanel.add(slot);
        }

        JPanel thumbPanel = new JPanel(new FlowLayout());
        thumbPanel.setBorder(BorderFactory.createTitledBorder("Select memes (max 6)"));

        List<URL> pngUrls = loadResourcePNGs();
        for (int i = 0; i < Math.min(MAX_THUMBNAILS, pngUrls.size()); i++) {
            URL url = pngUrls.get(i);
            BufferedImage img = ImageIO.read(url);
            Image scaled = img.getScaledInstance(80, 80, Image.SCALE_SMOOTH);
            JLabel thumb = retrieveLabel(scaled, url, img);
            thumbPanel.add(thumb);
        }

        JButton analyzeBtn = new JButton("Analyze");

        analyzeBtn.addActionListener(_ -> {
            if (analyzeBtn.getText().equals("Analyze")) {
                if (selectedUrls.isEmpty()) {
                    JOptionPane.showMessageDialog(frame, "Please select at least one meme!");
                    return;
                }
                analyzeSelection(analyzeBtn);
            } else if (analyzeBtn.getText().equals("Restart")) {
                restartAnalysis(analyzeBtn);
            }
        });

        JPanel southPanel = new JPanel();
        southPanel.add(analyzeBtn);

        frame.add(thumbPanel, BorderLayout.NORTH);
        frame.add(bigPanel, BorderLayout.CENTER);
        frame.add(southPanel, BorderLayout.SOUTH);

        frame.pack();
        frame.setLocationRelativeTo(null);
        frame.setVisible(true);

        frame.getRootPane().setDefaultButton(analyzeBtn);
    }

    private JLabel retrieveLabel(Image scaled, URL url, BufferedImage img) {
        JLabel thumb = new JLabel(new ImageIcon(scaled));
        thumb.setBorder(BorderFactory.createLineBorder(Color.GRAY));
        thumb.setCursor(Cursor.getPredefinedCursor(Cursor.HAND_CURSOR));
        thumb.addMouseListener(new MouseAdapter() {
            @Override
            public void mousePressed(MouseEvent e) {
                if (selectedUrls.size() < MAX_SELECTIONS && !selectedUrls.contains(url)) {
                    selectedUrls.add(url);
                    int idx = selectedUrls.size() - 1;
                    imageLabels[idx].setIcon(new ImageIcon(
                            img.getScaledInstance(300, 300, Image.SCALE_SMOOTH)));
                    imageLabels[idx].setText(EMPTY_STRING);
                    resultLabels[idx].setText(EMPTY_STRING);
                    logger.info("Thumbnail selected: %s".formatted(url));
                }
            }
        });
        return thumb;
    }

    private List<URL> loadResourcePNGs() throws IOException, URISyntaxException {
        List<URL> urls = new ArrayList<>();
        URL dirUrl = FERCoreMLDemo.class.getResource(BASE_PATH);
        if (dirUrl == null) return urls;

        if (dirUrl.getProtocol().equals("file")) {
            Path dirPath = Paths.get(dirUrl.toURI());
            try (DirectoryStream<Path> stream = Files.newDirectoryStream(dirPath, "*.png")) {
                for (Path p : stream) {
                    urls.add(p.toUri().toURL());
                }
            }
        } else if (dirUrl.getProtocol().equals("jar")) {
            String jarPath = dirUrl.getPath().substring(5, dirUrl.getPath().indexOf("!"));
            try (FileSystem fs = FileSystems.newFileSystem(Paths.get(jarPath), (ClassLoader) null)) {
                Path dirPath = fs.getPath(BASE_PATH);
                try (DirectoryStream<Path> stream = Files.newDirectoryStream(dirPath, "*.png")) {
                    for (Path p : stream) {
                        urls.add(p.toUri().toURL());
                    }
                }
            }
        }
        urls.sort(Comparator.comparing(URL::toString));
        logger.info("Loaded %s PNG thumbnails from resources.".formatted(urls.size()));
        return urls;
    }

    private void analyzeSelection(JButton analyzeBtn) {
        int size = selectedUrls.size();
        analyzeBtn.setEnabled(false);

        Map<String, String> options = Map.of("ModelFormat", "MLProgram",
                "MLComputeUnits", "CPUAndGPU", "EnableOnSubgraphs", "1",
                "AllowLowPrecisionAccumulationOnGPU", "1",
                "ModelCacheDirectory", FERCoreMLDemo.class.getResource(BASE_PATH).getPath());
        OnnxProvider provider = new OnnxProvider("CoreML", options);

        long initStartTime = System.nanoTime();
        long initTime = 0;
        long totalInferenceTime = 0;

        try (var arena = Arena.ofConfined()) {
            OnnxRuntime.SessionOptions sessionOptions = inference.prepareSessionOptions(arena, provider);

            long initEndTime = System.nanoTime();
            initTime = (initEndTime - initStartTime) / 1000000;

            for (int i = 0; i < size; i++) {
                URL url = selectedUrls.get(i);
                String result = "<html>%s</html>";
                try {
                    long inferenceStart = System.nanoTime();
                    float[] probs = inference.analyzeImage(arena, sessionOptions, url, useCondensedModel);
                    long inferenceEnd = System.nanoTime();
                    long inferenceTime = (inferenceEnd - inferenceStart) / 1000000;
                    totalInferenceTime += inferenceTime;
                    logger.info("Finished inference for image %d in %d ms".formatted(i + 1, inferenceTime));
                    String top3 = formatTopK(probs);
                    resultLabels[i].setText(result.formatted(top3 ));
                    frame.repaint();
                } catch (Exception ex) {
                    logger.log(Level.SEVERE, "Error occurred when evaluating images", ex);
                    resultLabels[i].setText(result.formatted(result.formatted(RED_ERROR_SPAN)));
                }
            }
        } catch (Exception initEx) {
            logger.log(Level.SEVERE, "Failed to initialize inference resources", initEx);
        } finally {
            logger.info("Total time initializing ORT: %d ms".formatted(initTime));
            logger.info("Total inference time: %d ms for %d images".formatted(totalInferenceTime, size));
            analyzeBtn.setEnabled(true);
            analyzeBtn.setText("Restart");
            logger.info("=== FER analysis complete ===");
        }
    }

    private void restartAnalysis(JButton analyzeBtn) {
        selectedUrls.clear();

        for (int i = 0; i < MAX_SELECTIONS; i++) {
            imageLabels[i].setIcon(null);
            imageLabels[i].setText("Empty");
            resultLabels[i].setText(EMPTY_STRING);
        }

        analyzeBtn.setText("Analyze");
        frame.repaint();
    }

    private String formatTopK(float[] probs) {
        int[] idxs = topK(probs, 3);
        StringBuilder sb = new StringBuilder();
        for (int idx : idxs) {
            sb.append(String.format("%s : %.1f%%<br>", FERCoreMLDemo.EMOTIONS[idx], probs[idx] * 100));
        }
        return sb.toString();
    }

    private int[] topK(float[] arr, int k) {
        return IntStream.range(0, arr.length)
                .boxed()
                .sorted((i, j) -> Float.compare(arr[j], arr[i]))
                .limit(k)
                .mapToInt(i -> i)
                .toArray();
    }

}
