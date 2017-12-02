package activationFunctions;

public class Sigmoid extends ActivationFunction{
	
	public Sigmoid() {}
	
	public double activationOutput(double[] inputs) {
		double input=sumInputs(inputs);
		return 1/(1+Math.exp(-1.0*input));
	}
	
	public double derivative(double output, double[] inputHolder) {
		return output*(1-output);
	}
}
