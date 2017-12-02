package activationFunctions;

import java.io.Serializable;

public class ActivationFunction implements Serializable{
	public double activationOutput(double[] inputs) {
		return 0.0;
	}
	
	public double[] activationOutputArray(double[] inputs) {
		return null;
	}
	
	public double derivative(double output, double[] inputHolder) {
		return 0.0;
	}
	
	public double[] derivativeArray(double outputHolder, double[] inputHolder) {
		return null;
	}
	
	public double sumInputs(double[] inputs) {
		double sum=0;
		for (int i=0; i< inputs.length; i++) {
			sum+=inputs[i];
		}
		return sum;
	}
}
