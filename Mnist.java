import javax.swing.*;
import java.awt.*;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import java.awt.event.MouseEvent;
import java.awt.event.MouseMotionListener;
import java.io.ByteArrayOutputStream;
import java.io.RandomAccessFile;
import java.nio.ByteBuffer;
import java.nio.channels.FileChannel;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;

public class Mnist extends JPanel implements ActionListener, MouseMotionListener {

    private final NeuralNetwork net;

    private final List<Integer> labels;
    private final List<float[][]> images;
    private final int pixelSize;

    public int current;
    private final ArrayList<Integer> shuffleIndex;

    private final JLabel currentIndexLabel, networkOutput, currentLabel, currentGuess, canvasGuess;

    private float[][] canvas;

    public Mnist(int pixelSize, NeuralNetwork net) {
        this.net = net;

        this.labels = getLabels();
        this.images = getImages();
        this.pixelSize = pixelSize;

        //addStaticData(labels.size()/10);

        setSize(new Dimension(pixelSize*28,pixelSize*28*2+20));
        setMinimumSize(new Dimension(pixelSize*28,pixelSize*28*2+20));
        setMaximumSize(new Dimension(pixelSize*28,pixelSize*28*2+20));
        setPreferredSize(new Dimension(pixelSize*28,pixelSize*28*2+20));

        shuffleIndex = new ArrayList<>();
        for (int i = 0;i<images.size();i++) shuffleIndex.add(i);

        canvas = new float[784][1];

        currentIndexLabel = new JLabel();
        networkOutput = new JLabel();
        currentLabel = new JLabel();
        currentGuess = new JLabel();
        canvasGuess = new JLabel();
        setImage(0);

        makeDisplay();

        this.addMouseMotionListener(this);
    }

    public void nextPicture() {
        setImage(current+1);
    }
    public void prevPicture() {
        setImage(current-1);
    }
    public void setImage(int n) {
        if (n>=0&n<images.size()) {
            current = n;
            update();
        }
    }
    public void update() {
        currentIndexLabel.setText("Current Image: " + (shuffleIndex.get(current) +1) + " / " + (this.images.size()));
        networkOutput.setText("Network Output: " + Arrays.toString(net.networkOutput(currentImage())));
        currentLabel.setText("Label: " + labels.get(shuffleIndex.get(current)));
        currentGuess.setText("Network Guess: " + net.networkGuess(currentImage()));
        canvasGuess.setText("Canvas Guess: " + net.networkGuess(canvas));
        repaint();
    }
    public float[][] currentImage() {
        return images.get(shuffleIndex.get(current));
    }
    public float[][] currentLabel() {
        float[][] label = new float[10][1];
        if (labels.get(shuffleIndex.get(current))==-1) return label;
        label[labels.get(shuffleIndex.get(current))][0] = 1;
        return label;
    }
    public int currentIntLabel() {
        return labels.get(shuffleIndex.get(current));
    }
    public void shuffle() {
        Collections.shuffle(shuffleIndex);
        setImage(0);
    }
    public void removeShuffle() {
        Collections.sort(shuffleIndex);
        setImage(0);
    }

    public int length() {
        return images.size();
    }
    public boolean hasNext() {
        return current+1<images.size();
    }

    private void addStaticData(int n) {
        while (n>0) {
            float[][] image = new float[784][1];
            for (int i = 0; i < 784; i++) {
                image[i][0] = (float) Math.random();
            }
            images.add(image);
            labels.add(-1);

            n--;
        }
    }
    // GRAPHICS
    public void paint(Graphics g) {
        float[][] image = images.get(shuffleIndex.get(current));
        for (int y = 0;y<28;y++) {
            for (int x = 0;x<28;x++) {
                g.setColor(new Color(image[y*28 + x][0],image[y*28 + x][0],image[y*28 + x][0]));
                g.fillRect(pixelSize*x,pixelSize*y,pixelSize,pixelSize);
            }
        }

        for (int y = 0;y<28;y++) {
            for (int x = 0;x<28;x++) {
                g.setColor(new Color(canvas[y*28 + x][0],canvas[y*28 + x][0],canvas[y*28 + x][0]));
                g.fillRect(pixelSize*x,28*pixelSize+10+pixelSize*y,pixelSize,pixelSize);
            }
        }
    }
    private void makeDisplay() {
        JFrame frame = new JFrame("MNIST");
        frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);

        JPanel mainPanel = new JPanel();
        mainPanel.setLayout(new BoxLayout(mainPanel,BoxLayout.Y_AXIS));

        JPanel mnistPanel = new JPanel();
        mnistPanel.setLayout(new BoxLayout(mnistPanel,BoxLayout.X_AXIS));

        mnistPanel.add(this);

        //MNIST OPTIONS PANEL
        JPanel mnistOptionsPanel = new JPanel();
        mnistOptionsPanel.setLayout(new BoxLayout(mnistOptionsPanel,BoxLayout.Y_AXIS));
        mnistOptionsPanel.add(currentIndexLabel);
        mnistOptionsPanel.add(networkOutput);
        mnistOptionsPanel.add(currentLabel);
        mnistOptionsPanel.add(currentGuess);
        mnistOptionsPanel.add(canvasGuess);

        mnistPanel.add(mnistOptionsPanel);
        //BUTTON PANEL
        JPanel buttonPanel = new JPanel();
        buttonPanel.setLayout(new BoxLayout(buttonPanel,BoxLayout.X_AXIS));

        JButton previous_picture = new JButton("Prev Picture");
        previous_picture.addActionListener(this);
        previous_picture.setActionCommand("previousPicture");
        buttonPanel.add(previous_picture);

        JButton next_picture = new JButton("Next Picture");
        next_picture.addActionListener(this);
        next_picture.setActionCommand("nextPicture");
        buttonPanel.add(next_picture);

        JButton clear_canvas = new JButton("Clear Canvas");
        clear_canvas.addActionListener(this);
        clear_canvas.setActionCommand("clearCanvas");
        buttonPanel.add(clear_canvas);

        JButton shuffle = new JButton("Shuffle");
        shuffle.addActionListener(this);
        shuffle.setActionCommand("shuffle");
        buttonPanel.add(shuffle);

        //FINAL ADDS
        mainPanel.add(mnistPanel);
        mainPanel.add(buttonPanel);

        frame.add(mainPanel);
        frame.pack();
        frame.setVisible(true);
    }
    public void actionPerformed(ActionEvent e) {
        if (e.getActionCommand().equals("nextPicture")) nextPicture();
        if (e.getActionCommand().equals("previousPicture")) prevPicture();
        if (e.getActionCommand().equals("clearCanvas")) {
            canvas = new float[784][1];
            update();
        }
        if (e.getActionCommand().equals("shuffle")) shuffle();
    }

    //MNIST_READER----------------------------------------------------------------------------------
    public static final int LABEL_FILE_MAGIC_NUMBER = 2049;
    public static final int IMAGE_FILE_MAGIC_NUMBER = 2051;
    private static ArrayList<Integer> getLabels() {
        ByteBuffer bb = loadFileToByteBuffer("src/train-labels-idx1-ubyte");
        bb.getInt();
        int numLabels = bb.getInt();
        ArrayList<Integer> labels = new ArrayList<>();
        for (int i = 0;i<numLabels;i++) labels.add(bb.get() & 0xFF);
        return labels;
    }
    private static List<float[][]> getImages() {
        ByteBuffer bb = loadFileToByteBuffer("src/train-images.idx3-ubyte");
        bb.getInt();
        int numImages = bb.getInt();
        int numRows = bb.getInt();
        int numCols = bb.getInt();
        List<float[][]> images = new ArrayList<>();
        for (int i = 0; i<numImages;i++) images.add(readImage(numRows*numCols,bb));
        return images;
    }
    private static float[][] readImage(int size, ByteBuffer bb) {
        float[][] image = new float[size][1];
        for (int y = 0; y< size;y++) image[y][0] = (bb.get() & 0xFF)/255f;
        return image;
    }
    private static ByteBuffer loadFileToByteBuffer(String file) {
        try {
            RandomAccessFile f = new RandomAccessFile(file, "r");
            FileChannel c = f.getChannel();
            long fileSize = c.size();
            ByteBuffer bb = ByteBuffer.allocate((int)fileSize);
            c.read(bb);
            bb.flip();
            ByteArrayOutputStream baos = new ByteArrayOutputStream();
            for (int i = 0;i<fileSize;i++)
                baos.write(bb.get());
            c.close();
            f.close();
            return ByteBuffer.wrap(baos.toByteArray());
        }
        catch (Exception e) {
            throw new RuntimeException(e);
        }
    }

    public void mouseDragged(MouseEvent e) {
        int x = e.getX() / pixelSize;
        int y = (e.getY() - pixelSize * 28 - 10) / pixelSize;
        if (x>0&x<27&y>0&y<27) {//e.getX()>pixelSize && e.getX()<pixelSize*27 && e.getY()>pixelSize*28-14 && e.getY()<pixelSize*55+10) {


            if (canvas[y * 28 + x + 28][0] == 0) canvas[y * 28 + x + 28][0] = (float) Math.random()/2 + .25f; //TOP
            if (canvas[y * 28 + x - 28][0] == 0) canvas[y * 28 + x - 28][0] = (float) Math.random()/2 + .25f; //BOTTOM
            if (canvas[y * 28 + x + 1][0] == 0) canvas[y * 28 + x + 1][0] = (float) Math.random()/2 + .25f; //RIGHT
            if (canvas[y * 28 + x - 1][0] == 0) canvas[y * 28 + x - 1][0] = (float) Math.random()/2 + .25f; //LEFT

            canvas[y * 28 + x][0] = (float) Math.random()/10 + .9f; //CENTER
        }
        update();
    }

    public void mouseMoved(MouseEvent e) {

    }
}
