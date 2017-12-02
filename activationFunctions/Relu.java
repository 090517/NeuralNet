package activationFunctions;

public class Relu extends ActivationFunction{
	
	public Relu() {}
	
	public double activationOutput(double[] inputs) {
		double input=sumInputs(inputs);
		if (input<0) {
			return 0;
		}
		else return input;
	}
	
	public double derivative(double output, double[] inputHolder) {
		if (output>0) {
			return 1;
		}
		return 0;
	}
}
