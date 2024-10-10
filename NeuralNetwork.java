import java.util.Random;

public class NeuralNetwork {

    private final Mnist mnist;

    private final int[] sizes;
    private float[][][] weights;
    private float[][][] biases;

    public static void main(String[] args) {
        NeuralNetwork net = new NeuralNetwork(new int[]{784,25,25,10});

        net.stochastic_gradient_descent(10,15,3f);

        net.testNetwork();
    }

    public NeuralNetwork(int[] sizes) {
        this.sizes = sizes;
        initParameters();

        this.mnist = new Mnist(10, this);
    }
    private void initParameters() {
        Random r = new Random();
        weights =  new float[sizes.length-1][][];
        biases = new float[sizes.length-1][][];
        for (int i = 1;i<sizes.length;i++) {
            weights[i-1] = new float[sizes[i]][sizes[i-1]];
            biases[i-1] = new float[sizes[i]][1];
            for (int j = 0;j<sizes[i];j++) {
                biases[i-1][j][0] = (float) r.nextGaussian();
                for (int k = 0;k<sizes[i-1];k++) {
                    weights[i-1][j][k] = (float) r.nextGaussian();
                }
            }
        }
    }
    public int[] networkOutput(float[][] input) {
        float[][] output = feedForward(input);
        int[] roundedOutput = new int[output.length];
        for (int i = 0;i<roundedOutput.length;i++) {
            roundedOutput[i] = (int)(output[i][0]*100);
        }
        return roundedOutput;
    }
    public int networkGuess(float[][] input) {
        float[][] results = feedForward(input);
        int guess = maxIndex(results);
        if (results[guess][0]<.5) return -1;
        return guess;
    }
    private int maxIndex(float[][] output) {
        float max = output[0][0];
        int maxIndex = 0;
        for (int i = 1;i<output.length;i++) {
            if (output[i][0]>max) {
                max = output[i][0];
                maxIndex = i;
            }
        }
        return maxIndex;
    }
    public float[][] feedForward(float[][] input) {
        for (int l = 1;l<sizes.length;l++) {
            input = sigmoid(vecAdd(dot(weights[l-1], input), biases[l-1]));
        }
        return input;
    }
    public void stochastic_gradient_descent(int epochs, int batchSize, float learningRate) {
        for (int i = 0;i<epochs;i++) { //EPOCH LOOP
            mnist.shuffle();
            while (mnist.current + batchSize < mnist.length()) { // BATCH LOOP
                float[][][] nabla_b = zeroCopy(biases);
                float[][][] nabla_w = zeroCopy(weights);
                for (int k = 0; k < batchSize; k++) { //TRAINING DATA LOOP
                    //BACKPROPAGATION
                    float[][] x = mnist.currentImage();
                    float[][] y = mnist.currentLabel();

                    float[][][] delta_nb = zeroCopy(biases);
                    float[][][] delta_nw = zeroCopy(weights);

                    //Feedforward
                    float[][][] a = new float[sizes.length][][];
                    a[0] = x;
                    float[][][] z = new float[sizes.length-1][][];
                    for (int l = 1;l<sizes.length;l++) {
                        z[l-1] = vecAdd(dot(weights[l-1], a[l-1]), biases[l-1]);
                        a[l] = sigmoid(z[l-1]);
                    }
                    //Backpropagation
                    delta_nb[delta_nb.length-1] = hadamardProduct(vecSub(a[a.length-1],y), sigmoidPrime(z[z.length-1]));
                    delta_nw[delta_nw.length-1] = dot(delta_nb[delta_nb.length-1], transpose(a[a.length-2])); //GOT RID OF TRANSPOSE? Maybe put it back?
                    for (int l = 2;l<sizes.length;l++) {
                        delta_nb[delta_nb.length-l] = hadamardProduct(dot(transpose(weights[weights.length-l+1]), delta_nb[delta_nb.length-l+1]), sigmoidPrime(z[z.length-l]));
                        delta_nw[delta_nw.length-l] = dot(delta_nb[delta_nb.length-l], transpose(a[a.length-l-1]));
                    }
                    //ADD TO GRADIENT
                    for (int l = 1;l<sizes.length;l++) {
                        //System.out.println(dim(nabla_w[l-1]) + dim(delta_nw[l-1]));
                        nabla_w[l-1] = vecAdd(nabla_w[l-1], delta_nw[l-1]);
                        nabla_b[l-1] = vecAdd(nabla_b[l-1], delta_nb[l-1]);
                    }
                    //NEXT IMAGE
                    mnist.nextPicture();
                }
                for (int l = 1;l<sizes.length;l++) {
                    weights[l-1] = vecSub(weights[l-1], scalarProduct(nabla_w[l-1], learningRate/batchSize));
                    biases[l-1] = vecSub(biases[l-1], scalarProduct(nabla_b[l-1], learningRate/batchSize));
                }
            }
            System.out.println("EPOCH " + (i + 1) + " COMPLETED");
            //testNetwork();
        }
        mnist.removeShuffle();
    }
    public void testNetwork() {
        mnist.removeShuffle();
        int count = 0;
        while (mnist.hasNext()) {
            if (mnist.currentIntLabel()==networkGuess(mnist.currentImage())) count++;
            mnist.nextPicture();
        }
        System.out.println(count + "/" + mnist.length() + " | " + ((int)(10000*(count/ (double)mnist.length()))/100.) + "%");
    }

    //UTILITIES
    private String dim(float[][] m) {
        return "(" + m.length  + ", " + m[0].length + ")";
    }
    private float[][] transpose(float[][] m) { //CHECK
        float[][] result = new float[m[0].length][m.length];
        for (int i = 0;i<m.length;i++) {
            for (int j = 0;j<m[0].length;j++) {
                result[j][i] = m[i][j];
            }
        }
        return result;
    }
    private float[][] hadamardProduct(float[][] v1, float[][] v2) {
        float[][] product = new float[v1.length][v1[0].length];
        for (int i = 0;i<v1.length;i++) {
            for (int j = 0;j<v1[i].length;j++) {
                product[i][j] = v1[i][j] * v2[i][j];
            }
        }
        return product;
    }
    private float[][][] zeroCopy(float[][][] arr) {
        float[][][] result = new float[arr.length][][];
        for (int i = 0;i<arr.length;i++) {
            result[i] = new float[arr[i].length][arr[i][0].length];
            for (int j = 0;j<arr[i].length;j++) {
                for (int k = 0;k<arr[i][j].length;k++) {
                    result[i][j][k] = 0;
                }
            }
        }
        return result;
    }
    private float[] col(float[][] m, int x) {
        float[] arr = new float[m.length];
        for (int i = 0;i<arr.length;i++) {
            arr[i] = m[i][x];
        }
        return arr;
    }
    private float[][] vecAdd(float[][] v1, float[][] v2) {
        float[][] arr = new float[v1.length][v1[0].length];
        for (int i = 0;i<v1.length;i++) {
            for (int j = 0;j<v1[0].length;j++) {
                arr[i][j] = v1[i][j] + v2[i][j];
            }
        }
        return arr;
    }
    private float[][] vecSub(float[][] v1, float[][] v2) {
        float[][] arr = new float[v1.length][v1[0].length];
        for (int i = 0;i<v1.length;i++) {
            for (int j = 0;j<v1[0].length;j++) {
                arr[i][j] = v1[i][j] - v2[i][j];
            }
        }
        return arr;
    }
    private float[][] scalarProduct(float[][] m, float n) {
        float[][] product = new float[m.length][m[0].length];
        for (int i = 0;i<m.length;i++) {
            for (int j = 0;j<m[i].length;j++) {
                product[i][j] = m[i][j] * n;
            }
        }
        return product;
    }
    private float[][] dot(float[][] m, float[][] v) { //FIX TO SOMETHING MAYBE
        v = transpose(v);
        float[][] arr = new float[m.length][v.length];
        for (int i = 0;i<m.length;i++) {
            for (int j = 0;j<v.length;j++) {
                arr[i][j] = dot(m[i], v[j]);
            }
        }
        return arr;
    }
    private float dot(float[] v1, float[] v2) {
        float sum = 0;
        for (int i = 0;i<v1.length;i++) {
            sum += v1[i] * v2[i];
        }
        return sum;
    }
    private float[][] sigmoidPrime(float[][] v) {
        float[][] arr = new float[v.length][1];
        for (int i = 0;i<v.length;i++) {
            arr[i][0] = sigmoidPrime(v[i][0]);
        }
        return arr;
    }
    private float sigmoidPrime(float z) {
        return sigmoid(z) * (1 - sigmoid(z));
    }
    private float[][] sigmoid(float[][] v) {
        float[][] arr = new float[v.length][1];
        for (int i = 0;i<v.length;i++) {
            arr[i][0] = sigmoid(v[i][0]);
        }
        return arr;
    }
    private float sigmoid(float z) {
        return (float) (1/(1 + Math.exp(-z)));
    }
}
