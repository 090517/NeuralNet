package activationFunctions;

public class LeakyRelu extends ActivationFunction{

	private static final long serialVersionUID = 1L;
	double leakyScale;
	
	public LeakyRelu(double scale) {
		leakyScale=scale;
	}
	
	public double activationOutput(double[] inputs) {
		double input=sumInputs(inputs);
		if (input<0) {
			return .01;
		}
		else return input*leakyScale;
	}
	
	public double derivative(double output, double[] inputHolder) {
		if (output>=0) {
			return 1;
		}
		return leakyScale;
	}
}