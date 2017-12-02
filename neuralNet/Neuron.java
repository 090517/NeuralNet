package neuralNet;

import java.io.Serializable;

import activationFunctions.ActivationFunction;

public class Neuron implements Serializable{
	/**
	 * 
	 */
	private static final long serialVersionUID = 1L;
	
	public double[] weights;
	double[] newWeights;
	double[] inputHolder;
	double outputHolder;
	double outputError;
	double backPropDeriv;

	public Neuron(int inputs, boolean bias) {
		if (bias) {
			inputs++;
		}
		
		weights = new double[inputs];
		for (int i = 0; i < weights.length; i++) {
			// numerator returns number random from +-1.
			weights[i] = ((Math.random() - 1) * 2) / Math.sqrt(inputs);
		}
	}

	public void setWeights(double[] newWeights) {
		weights = newWeights;
	}

	public double forwardProp(double[] input, ActivationFunction function) {
		inputHolder=input.clone();
		double[] functionInput = new double[input.length+1];
		for (int i = 0; i < input.length; i++) {
			functionInput[i] = input[i] * weights[i];
			
		}
		// bias term
		if (input.length < weights.length) {
			functionInput[functionInput.length-1] = weights[weights.length - 1];
		}
		
		double output = function.activationOutput(functionInput);
		outputHolder = output;
		return output;
	}

	public void backProp(double ideal, ActivationFunction AF, double learningRate, boolean outputLayerTrue) {
		if (outputLayerTrue)
			backPropDeriv = outputHolder - ideal;
		else
			backPropDeriv = ideal;
		
		backPropDeriv = backPropDeriv * AF.derivative(outputHolder, inputHolder);
		
		//if (AF.getClass() == activationFunctions.Sigmoid.class) {	
		 
		newWeights = weights.clone();
		
		for (int i=0; i<weights.length; i++) {
		}

		for (int i = 0; i < inputHolder.length; i++) {
			double weightError = backPropDeriv * inputHolder[i];
			newWeights[i] = weights[i] - weightError * learningRate;
		}
		
		if (weights.length!=inputHolder.length) {
			newWeights[newWeights.length-1]= weights[newWeights.length-1]-backPropDeriv;
		}
	}

	public double errorPassThrough(int prevNeuronIndex) {
		return backPropDeriv * weights[prevNeuronIndex];
	}

	public void updateWeights() {
		weights = newWeights;
	}
}
