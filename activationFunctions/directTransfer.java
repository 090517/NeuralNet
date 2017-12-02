package activationFunctions;

public class directTransfer extends ActivationFunction{
	
	public directTransfer() {
	}
	
	public double[] activationOutputArray(double[] input) {
		return input;
	}
	
	public double[] derivativeArray(double[] outputHolder, double[] inputHolder) {
		double[] output=new double[outputHolder.length];
		for (int i=0; i<outputHolder.length; i++) {
		output[i]=1;	
		}
		return output;
	}
}

