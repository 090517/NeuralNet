package activationFunctions;

public class Maxout extends ActivationFunction {
	public Maxout() {
	}

	public double[] activationOutputArray(double[] inputs) {
		double maxNum = inputs[0];
		int maxIndex = 0;

		for (int i = 1; i < inputs.length; i++) {
			if (inputs[i] > maxNum) {
				maxNum = inputs[i];
				maxIndex = i;
			}
		}

		double[] output = new double[inputs.length];
		for (int i = 0; i < inputs.length; i++) {
			if (i == maxIndex) {
				output[i] = maxNum;
			} else
				output[i] = 0;
		}
		return output;
	}

	public double[] derivativeArray(double outputHolder[], double[] inputHolder) {
		double[] output = new double[outputHolder.length];
		for (int i = 0; i < outputHolder.length; i++) {
			if (outputHolder[i]!=0) {
				output[i] = 1;				
			}
			else output[i] = 0;
		}
		return output;
	}
}
