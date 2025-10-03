package oracle.code.onnx.fer;

import javax.imageio.ImageIO;
import javax.swing.*;
import java.awt.*;
import java.awt.event.*;
import java.awt.image.BufferedImage;
import java.io.IOException;
import java.net.URL;
import java.net.URISyntaxException;
import java.nio.file.*;
import java.util.*;
import java.util.List;

public class FERCoreMLDemo {

    private static final int MAX_SELECTIONS = 6;
    private static final int MAX_THUMBNAILS = 12;
    private static final String BASE_PATH = "/oracle/code/onnx/fer/";

    private JFrame frame;
    private JPanel bigPanel;
    private JLabel[] imageLabels;
    private JLabel[] resultLabels;
    private List<URL> selectedUrls = new ArrayList<>();

    private FERInference ferInference;

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
        initFER();
        buildGUI();
    }

    private void initFER() throws Exception {
        ferInference = new FERInference();
    }

    private void buildGUI() throws IOException, URISyntaxException {
        frame = new JFrame("Facial Expression Recognition - CoreML Demo");
        frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
        frame.setLayout(new BorderLayout(10, 10));

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

        for (int i = 0; i < selectedUrls.size(); i++) {
            URL url = selectedUrls.get(i);
            
            try {
                float[] probs = ferInference.analyzeImage(url);
                String top3 = FERUtils.formatTopK(probs, 3, FERInference.getEmotions());
                
                resultLabels[i].setText("<html>" + top3 + "</html>");
                progressBar.setValue(i + 1);
                progressBar.setString("Processed " + (i + 1) + "/" + selectedUrls.size());
                
                frame.repaint();
                
            } catch (Exception ex) {
                resultLabels[i].setText("<html><span style='color:red'>Error!</span></html>");
            }
            
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
        selectedUrls.clear();
        
        for (int i = 0; i < MAX_SELECTIONS; i++) {
            imageLabels[i].setIcon(null);
            imageLabels[i].setText("Empty");
            resultLabels[i].setText("");
        }
        
        analyzeBtn.setText("Analyze");
        
        JProgressBar progressBar = (JProgressBar) ((JPanel) analyzeBtn.getParent()).getComponent(1);
        progressBar.setVisible(false);
        
        frame.repaint();
    }

}
