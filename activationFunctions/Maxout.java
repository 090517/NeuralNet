package activationFunctions;

import java.util.Arrays;

public class Maxout extends ActivationFunction {
	private static final long serialVersionUID = 1L;

	public Maxout() {

	}

	public double[] activationOutputVector(double[] inputs) {
		double maxNum = inputs[0];
		int maxIndex = 0;

		for (int i = 1; i < inputs.length; i++) {
			if (inputs[i] > maxNum) {
				maxNum = inputs[i];
				maxIndex = i;
			}
		}

		double[] output = new double[inputs.length];
		Arrays.fill(output, 0.0);
		output[maxIndex] = maxNum;
		return output;
	}

	public double[] derivativeVector(double outputHolder[], double[] inputHolder) {
		double[] output = new double[outputHolder.length];
		for (int i = 0; i < outputHolder.length; i++) {
			if (outputHolder[i] != 0) {
				output[i] = 1;
			} else
				output[i] = 1;
		}
		return output;
	}

	public boolean returnVectorOutput() {
		return true;
	}
}
