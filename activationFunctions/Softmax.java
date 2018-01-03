package activationFunctions;


public class Softmax extends ActivationFunction{
	private static final long serialVersionUID = 1L;
	public final boolean VectorOutput = true;

	public Softmax() {
	}

	public double[] activationOutputVector(double[] inputs) {
		double[] expInput = new double[inputs.length];

		double sum = 0;

		for (int i = 0; i < inputs.length; i++) {
			expInput[i] = Math.exp(inputs[i]);
			sum += Math.exp(inputs[i]);
		}

		double[] output = new double[inputs.length];
		for (int i = 0; i < inputs.length; i++) {
			output[i]=expInput[i]/sum;
		}
		return output;
	}

	public double[] derivativeVector(double outputHolder[], double[] inputHolder) {
			//todo
		return null;
	}

}