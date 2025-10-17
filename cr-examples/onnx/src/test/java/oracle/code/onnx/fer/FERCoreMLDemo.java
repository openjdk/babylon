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

import oracle.code.onnx.provider.CoreMLProvider;

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
import java.util.logging.Level;
import java.util.logging.Logger;
import java.util.stream.IntStream;

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
    private final List<URL> selectedUrls = new ArrayList<>();
    private final boolean useCondensedModel;
    private JFrame frame;
    private JLabel[] imageLabels;
    private JLabel[] resultLabels;
    private final FERInference inference;

    private FERCoreMLDemo(boolean useCondensedModel) throws IOException {
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
            resultLabels[i] = new JLabel("", SwingConstants.CENTER);
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
        JProgressBar progressBar = new JProgressBar(0, MAX_SELECTIONS);
        progressBar.setStringPainted(true);
        progressBar.setVisible(false);

        analyzeBtn.addActionListener(_ -> {
            if (analyzeBtn.getText().equals("Analyze")) {
                if (selectedUrls.isEmpty()) {
                    JOptionPane.showMessageDialog(frame, "Please select at least one meme!");
                    return;
                }
                analyzeSelection(progressBar, analyzeBtn);
            } else if (analyzeBtn.getText().equals("Restart")) {
                restartAnalysis(analyzeBtn);
            }
        });

        JPanel southPanel = new JPanel(new BorderLayout());
        southPanel.add(analyzeBtn, BorderLayout.CENTER);
        southPanel.add(progressBar, BorderLayout.SOUTH);

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
        thumb.addMouseListener(new MouseAdapter() {
            @Override
            public void mouseClicked(MouseEvent e) {
                if (selectedUrls.size() < MAX_SELECTIONS) {
                    selectedUrls.add(url);
                    int idx = selectedUrls.size() - 1;
                    imageLabels[idx].setIcon(new ImageIcon(
                            img.getScaledInstance(300, 300, Image.SCALE_SMOOTH)));
                    imageLabels[idx].setText("");
                    resultLabels[idx].setText("");
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

    private void analyzeSelection(JProgressBar progressBar, JButton analyzeBtn) {
        progressBar.setValue(0);
        progressBar.setMaximum(selectedUrls.size());
        progressBar.setVisible(true);
        analyzeBtn.setEnabled(false);

        long startTime = System.nanoTime();

        for (int i = 0; i < selectedUrls.size(); i++) {
            URL url = selectedUrls.get(i);

            try (var arena = Arena.ofConfined()) {
                float[] probs = inference.analyzeImage(arena, new CoreMLProvider(), url, useCondensedModel);
                String top3 = formatTopK(probs);

                resultLabels[i].setText("<html>" + top3 + "</html>");
                progressBar.setValue(i + 1);
                progressBar.setString("Processed " + (i + 1) + "/" + selectedUrls.size());

                frame.repaint();
            } catch (Exception ex) {
                logger.log(Level.SEVERE, "Error occurred when evaluating images", ex);
                resultLabels[i].setText("<html><span style='color:red'>Error!</span></html>");
            }
        }
        long endTime = System.nanoTime();
        logger.info("Total time spent in evaluation %s ms".formatted((endTime - startTime)/1000000));
        analyzeBtn.setEnabled(true);
        analyzeBtn.setText("Restart");
        progressBar.setString("Analysis complete!");
        logger.info("=== FER analysis complete ===");
    }

    private void restartAnalysis(JButton analyzeBtn) {
        selectedUrls.clear();

        for (int i = 0; i < MAX_SELECTIONS; i++) {
            imageLabels[i].setIcon(null);
            imageLabels[i].setText("Empty");
            resultLabels[i].setText("");
        }

        analyzeBtn.setText("Analyze");

        JProgressBar progressBar = (JProgressBar) analyzeBtn.getParent().getComponent(1);
        progressBar.setVisible(false);

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
