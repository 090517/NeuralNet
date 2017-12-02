package activationFunctions;

public class TanH extends ActivationFunction{
	
	public TanH() {}
	
	public double activationOutput(double inputs[]) {
		double input=sumInputs(inputs);
		return 2/(1+Math.exp(-2.0*input))-1;
	}
	
	public double derivative(double output, double[] inputHolder) {
		return 1-Math.pow(output, 2);
	}
}
