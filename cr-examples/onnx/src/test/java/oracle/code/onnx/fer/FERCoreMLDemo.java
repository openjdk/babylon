package oracle.code.onnx.fer;

import oracle.code.onnx.OnnxRuntime;
import oracle.code.onnx.Tensor;

import javax.imageio.ImageIO;
import javax.swing.*;
import java.awt.*;
import java.awt.event.*;
import java.awt.image.BufferedImage;
import java.io.IOException;
import java.lang.foreign.Arena;
import java.net.URL;
import java.net.URISyntaxException;
import java.nio.file.*;
import java.util.*;
import java.util.List;

import static oracle.code.onnx.fer.FERFFMAppleSilicon.*;

/**
 * GUI demo for FER (Facial Expression Recognition) using CoreML EP.
 * - Automatically loads first 10–12 PNGs from resource folder
 * - User can click thumbnails to enlarge in slots (max 6)
 * - ENTER or "Analyze" runs FER inference
 * - Displays top 3 emotions under each selected meme
 */
public class FERCoreMLDemo {

    private static final int MAX_SELECTIONS = 6;
    private static final int MAX_THUMBNAILS = 12;
    private static final String BASE_PATH = "/oracle/code/onnx/fer/";
    private static final String MODEL_PATH = BASE_PATH + "emotion-ferplus-8.onnx";
    private static final int IMAGE_SIZE = 64;

    private JFrame frame;
    private JPanel bigPanel;
    private JLabel[] imageLabels;
    private JLabel[] resultLabels;
    private List<URL> selectedUrls = new ArrayList<>();

    // ONNX Runtime stuff
    private OnnxRuntime runtime;
    private OnnxRuntime.Session session;
    private Arena arena;

    public static void main(String[] args) {
        SwingUtilities.invokeLater(() -> {
            try {
                new FERCoreMLDemo().start();
            } catch (Exception e) {
                e.printStackTrace(System.out);
            }
        });
    }

    public void start() throws Exception {
        initOnnx();
        buildGUI();
    }

    private void initOnnx() throws Exception {
        arena = Arena.ofConfined();
        runtime = OnnxRuntime.getInstance();

        // Load model bytes
        URL modelUrl = FERCoreMLDemo.class.getResource(MODEL_PATH);
        if (modelUrl == null) {
            throw new RuntimeException("Model not found: " + MODEL_PATH);
        }
        byte[] modelBytes = modelUrl.openStream().readAllBytes();

        var sessionOptions = runtime.createSessionOptions(arena);

        // 🔑 bring in the CLI setup
        FERFFMAppleSilicon.checkAvailableProviders(runtime, arena);
        FERFFMAppleSilicon.enableVerboseLogging(sessionOptions, arena);
        FERFFMAppleSilicon.enableCoreML(sessionOptions, arena);

        session = runtime.createSession(arena, modelBytes, sessionOptions);

        System.out.println("=== ONNX Runtime session initialized with model: " + modelUrl + " ===");
    }

    private void buildGUI() throws IOException, URISyntaxException {
        frame = new JFrame("Facial Expression Recognition - CoreML Demo");
        frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
        frame.setLayout(new BorderLayout(10, 10));

        // Big panel for 6 selected images
        bigPanel = new JPanel(new GridLayout(2, 3, 10, 10));
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

        // Thumbnail panel
        JPanel thumbPanel = new JPanel(new FlowLayout());
        thumbPanel.setBorder(BorderFactory.createTitledBorder("Select memes (max 6)"));

        List<URL> pngUrls = loadResourcePNGs();
        for (int i = 0; i < Math.min(MAX_THUMBNAILS, pngUrls.size()); i++) {
            URL url = pngUrls.get(i);
            BufferedImage img = ImageIO.read(url);
            Image scaled = img.getScaledInstance(80, 80, Image.SCALE_SMOOTH);
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
                        System.out.println("Thumbnail selected: " + url);
                    }
                }
            });
            thumbPanel.add(thumb);
        }

        // Analyze/Restart button + progress bar
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

        // ENTER key triggers analyze
        frame.getRootPane().setDefaultButton(analyzeBtn);
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
        Collections.sort(urls, Comparator.comparing(URL::toString));
        System.out.println("Loaded " + urls.size() + " PNG thumbnails from resources.");
        return urls;
    }

    private void analyzeSelection(JProgressBar progressBar, JButton analyzeBtn) {
        progressBar.setValue(0);
        progressBar.setMaximum(selectedUrls.size());
        progressBar.setVisible(true);
        analyzeBtn.setEnabled(false);

        // Run everything on the main thread sequentially
        System.out.println("=== Starting FER analysis on " + selectedUrls.size() + " images ===");
        
        for (int i = 0; i < selectedUrls.size(); i++) {
            URL url = selectedUrls.get(i);
            System.out.println("Processing image " + (i + 1) + ": " + url);
            
            try {
                System.out.println("  [MainThread] Calling runFER for image " + (i + 1));
                float[] probs = runFER(url);
                System.out.println("  [MainThread] runFER returned probabilities array of length: " + probs.length);
                String top3 = formatTop3(probs);
                System.out.println("  [MainThread] Formatted top3: " + top3);
                
                // Update GUI immediately
                resultLabels[i].setText("<html>" + top3 + "</html>");
                progressBar.setValue(i + 1);
                progressBar.setString("Processed " + (i + 1) + "/" + selectedUrls.size());
                
                // Force GUI update
                frame.repaint();
                
                System.out.println("  [MainThread] Updated GUI for image " + (i + 1));
                
            } catch (Exception ex) {
                System.out.println("  [ERROR] Failed on image: " + url);
                ex.printStackTrace(System.out);
                resultLabels[i].setText("<html><span style='color:red'>Error!</span></html>");
            }
            
            // Small delay to see progress
            try {
                Thread.sleep(500);
            } catch (InterruptedException ignored) {}
        }
        
        analyzeBtn.setEnabled(true);
        analyzeBtn.setText("Restart");
        progressBar.setString("Analysis complete!");
        System.out.println("=== FER analysis complete ===");
    }

    private void restartAnalysis(JButton analyzeBtn) {
        System.out.println("=== Restarting analysis ===");
        
        // Clear selected URLs
        selectedUrls.clear();
        
        // Clear all image slots
        for (int i = 0; i < MAX_SELECTIONS; i++) {
            imageLabels[i].setIcon(null);
            imageLabels[i].setText("Empty");
            resultLabels[i].setText("");
        }
        
        // Reset button
        analyzeBtn.setText("Analyze");
        
        // Hide progress bar
        JProgressBar progressBar = (JProgressBar) ((JPanel) analyzeBtn.getParent()).getComponent(1);
        progressBar.setVisible(false);
        
        // Force GUI update
        frame.repaint();
        
        System.out.println("=== Analysis restarted - ready for new selections ===");
    }

    private float[] runFER(URL url) {
        System.out.println("  [runFER] Loading image: " + url);
        try {
            float[] imageData = loadImageAsFloatArray(url);
            System.out.println("  [runFER] Image loaded, size=" + imageData.length);

            // Use the existing session and arena from main thread
            System.out.println("  [runFER] Using existing session and arena");
            
            // Prepare tensor
            long[] shape = {1, 1, IMAGE_SIZE, IMAGE_SIZE};
            System.out.println("  [runFER] Creating input tensor with shape: " + java.util.Arrays.toString(shape));
            var inputTensor = Tensor.ofShape(arena, shape, imageData);
            System.out.println("  [runFER] Input tensor created successfully");

            System.out.println("  [runFER] Starting inference...");
            long start = System.nanoTime();
            List<Tensor> outputs = session.run(arena, List.of(inputTensor));
            long end = System.nanoTime();

            double ms = (end - start) / 1_000_000.0;
            System.out.printf("  [runFER] Inference completed in %.2f ms%n", ms);

            System.out.println("  [runFER] Processing outputs...");
            float[] rawScores = outputs.get(0)
                    .data().toArray(java.lang.foreign.ValueLayout.JAVA_FLOAT);
            System.out.println("  [runFER] Raw scores length: " + rawScores.length);
            
            float[] probs = softmax(rawScores);
            System.out.println("  [runFER] Probabilities computed");

            // Print all emotion probabilities like CLI
            System.out.println("\nEmotion probabilities:");
            for (int i = 0; i < probs.length; i++) {
                System.out.printf("  %-10s : %.2f%%%n", FERFFMAppleSilicon.EMOTIONS[i], probs[i] * 100);
            }

            int pred = argMax(probs);
            System.out.println("Predicted emotion: " + FERFFMAppleSilicon.EMOTIONS[pred] + "\n");

            return probs;
        } catch (Exception e) {
            System.out.println("  [runFER] ERROR while processing " + url + ": " + e.getMessage());
            e.printStackTrace(System.out);
            throw new RuntimeException("Failed to run FER on " + url, e);
        }
    }


    private String formatTop3(float[] probs) {
        int[] idxs = topK(probs, 3);
        StringBuilder sb = new StringBuilder();
        for (int idx : idxs) {
            sb.append(String.format("%s : %.1f%%<br>", EMOTIONS[idx], probs[idx] * 100));
        }
        return sb.toString();
    }

    private int[] topK(float[] arr, int k) {
        return java.util.stream.IntStream.range(0, arr.length)
                .boxed()
                .sorted((i, j) -> Float.compare(arr[j], arr[i]))
                .limit(k)
                .mapToInt(i -> i)
                .toArray();
    }

    private static float[] loadImageAsFloatArray(URL imgUrl) throws IOException {
        System.out.println("    [loadImage] Attempting to read image: " + imgUrl);
        BufferedImage src = ImageIO.read(imgUrl);
        if (src == null) {
            throw new IOException("Unsupported or corrupt image: " + imgUrl);
        }
        System.out.println("    [loadImage] ImageIO.read succeeded: "
                + src.getWidth() + "x" + src.getHeight());

        BufferedImage graySrc = new BufferedImage(src.getWidth(), src.getHeight(), BufferedImage.TYPE_BYTE_GRAY);
        Graphics2D g0 = graySrc.createGraphics();
        g0.drawImage(src, 0, 0, null);
        g0.dispose();

        BufferedImage gray = new BufferedImage(IMAGE_SIZE, IMAGE_SIZE, BufferedImage.TYPE_BYTE_GRAY);
        Graphics2D g = gray.createGraphics();
        g.setRenderingHint(RenderingHints.KEY_INTERPOLATION, RenderingHints.VALUE_INTERPOLATION_BILINEAR);
        g.drawImage(graySrc, 0, 0, IMAGE_SIZE, IMAGE_SIZE, null);
        g.dispose();

        float[] data = new float[IMAGE_SIZE * IMAGE_SIZE];
        gray.getData().getSamples(0, 0, IMAGE_SIZE, IMAGE_SIZE, 0, data);

        System.out.println("    [loadImage] Returning float array of length " + data.length);
        return data;
    }


    public static float[] softmax(float[] scores) {
        float max = Float.NEGATIVE_INFINITY;
        for (float s : scores) if (s > max) max = s;
        double sum = 0.0;
        double[] exps = new double[scores.length];
        for (int i = 0; i < scores.length; i++) {
            exps[i] = Math.exp(scores[i] - max);
            sum += exps[i];
        }
        float[] out = new float[scores.length];
        for (int i = 0; i < scores.length; i++) {
            out[i] = (float)(exps[i] / sum);
        }
        return out;
    }

    public static int argMax(float[] arr) {
        int idx = 0;
        float max = arr[0];
        for (int i = 1; i < arr.length; i++) {
            if (arr[i] > max) {
                max = arr[i];
                idx = i;
            }
        }
        return idx;
    }
}
