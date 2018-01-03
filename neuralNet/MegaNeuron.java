package neuralNet;

import java.util.ArrayList;
import java.util.Collections;

import activationFunctions.ActivationFunction;

public class MegaNeuron extends Neuron {

	private static final long serialVersionUID = 1L;
	double[] outputHolderVector;
	double[] backPropDerivVector;

	public MegaNeuron(int numInputs, boolean bias) {
		super( numInputs,  bias, null);
		if (bias)
			numInputs++;
		this.weightDeltaHolder = new ArrayList<>(Collections.nCopies(numInputs, 0.0));
		newRandomWeights(numInputs);
	}

	//
	public double[] forwardPropMega(double[] inputs, ActivationFunction AF) {
		inputHolder = inputs.clone();
		double[] functionInput = new double[weights.size()];
		for (int i = 0; i < inputs.length; i++) {
			functionInput[i] = inputs[i] * weights.get(i);
		}
		if (inputs.length < weights.size()) // bias term
			functionInput[functionInput.length - 1] = weights.get(weights.size() - 1);	
		return outputHolderVector = AF.activationOutputVector(functionInput);
	}

	public void backPropMega(double[] ideal, double learningRate, double momentum, boolean outputLayerTrue, ActivationFunction LayerAF) {
		double[] LayerDerivative = LayerAF.derivativeVector(outputHolderVector, inputHolder);
		backPropDerivVector= new double[ideal.length];
		if (outputLayerTrue) {
			for (int i = 0; i < ideal.length; i++) {			
				backPropDerivVector[i] = (outputHolderVector[i] - ideal[i])*LayerDerivative[i];
			}
		}
		else
			for (int i = 0; i < ideal.length; i++) {		
				backPropDerivVector[i] = (ideal[i])*LayerDerivative[i];
			}
		
		for (int i = 0; i < inputHolder.length; i++) {
			weightDeltaHolder.set(i,
					backPropDerivVector[i] * inputHolder[i] * learningRate + momentum * weightDeltaHolder.get(i));	
		}

		if (weights.size() != inputHolder.length) {
			weightDeltaHolder.set(weightDeltaHolder.size() - 1,
					backPropDerivVector[weightDeltaHolder.size()-1] * learningRate + momentum * weightDeltaHolder.get(weightDeltaHolder.size() - 1));
		}
	}
	
	public void backPropSoftmax(double[] ideal, double learningRate, boolean outputLayerTrue) {
		
	}
	
	public double errorPassThrough(int prevNeuronIndex) {
		return backPropDerivVector[prevNeuronIndex] * weights.get(prevNeuronIndex);
	}
	
	/**
	 * Back prop single output
	 */

}
