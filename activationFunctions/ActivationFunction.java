package activationFunctions;

import java.io.Serializable;

public abstract class ActivationFunction implements Serializable {
	private static final long serialVersionUID = 1L;

	public double activationOutput(double[] inputs) {
		return 0.0;
	}

	public double[] activationOutputVector(double[] inputs) {
		return null;
	}

	public double derivative(double output, double[] inputHolder) {
		return 0.0;
	}

	public double[] derivativeVector(double[] outputHolder, double[] inputHolder) {
		return null;
	}

	public double sumInputs(double[] inputs) {
		double sum = 0;
		for (int i = 0; i < inputs.length; i++) {
			sum += inputs[i];
		}
		return sum;
	}

	public void forwardPropMega(double[] inputs) {
		// TODO Auto-generated method stub

	}

	public boolean returnVectorOutput() {
		return false;
	}
}
