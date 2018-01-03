package neuralNet;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.Collections;

import activationFunctions.ActivationFunction;

public class Neuron implements Serializable {
	private static final long serialVersionUID = 1L;

	ArrayList<Double> weights;
	ArrayList<Double> weightDeltaHolder;

	double[] inputHolder;
	double outputHolder;
	double backPropDeriv;
	ActivationFunction NeuronLevelAF;

	/**
	 * CONSTRUCTOR
	 */

	public Neuron(int numInputs, boolean prevLayerBias, ActivationFunction newFunction) {
		if (prevLayerBias)
			numInputs++;
		this.NeuronLevelAF = newFunction;
		this.weightDeltaHolder = new ArrayList<>(Collections.nCopies(numInputs, 0.0));
		newRandomWeights(numInputs);
	}
	
	public Neuron(int numInputs, boolean prevLayerBias) {
		this(numInputs, prevLayerBias, null);
	}
	
	/**
	 * Forward prop single output
	 */

	public double forwardProp(double[] input, ActivationFunction LayerAF) {
		inputHolder = input.clone();
		double[] functionInput = new double[weights.size()];
		for (int i = 0; i < input.length; i++) {
			functionInput[i] = input[i] * weights.get(i);
		}
		if (input.length < weights.size()) // bias term
			functionInput[functionInput.length - 1] = weights.get(weights.size() - 1);
		if (LayerAF == null)
			outputHolder = NeuronLevelAF.activationOutput(functionInput);
		else
			outputHolder = LayerAF.activationOutput(functionInput);
		return outputHolder;
	}

	public double forwardProp(double[] input) {
		return forwardProp(input, null);
	}

	/**
	 * Back prop single output
	 */

	public void backProp(double ideal, double learningRate, double momentum, boolean outputLayerTrue,
			ActivationFunction LayerAF) {
		if (outputLayerTrue)
			backPropDeriv = outputHolder - ideal;
		else
			backPropDeriv = ideal;

		if (LayerAF == null)
			backPropDeriv = backPropDeriv * NeuronLevelAF.derivative(outputHolder, inputHolder);
		else
			backPropDeriv = backPropDeriv * LayerAF.derivative(outputHolder, inputHolder);

		for (int i = 0; i < inputHolder.length; i++) {
			weightDeltaHolder.set(i,
					backPropDeriv * inputHolder[i] * learningRate + momentum * weightDeltaHolder.get(i));
		}

		if (weightDeltaHolder.size() == (inputHolder.length + 1)) {// bias term
			weightDeltaHolder.set(weightDeltaHolder.size() - 1,
					backPropDeriv * learningRate + momentum * weightDeltaHolder.get(weightDeltaHolder.size() - 1));
		}
	}

	/**
	 * Update weights after backprop
	 */

	public void updateWeights() {
		for (int i = 0; i < weights.size(); i++) {
			weights.set(i, weights.get(i) - weightDeltaHolder.get(i));
		}
	}

	/**
	 * Other Methods
	 */

	public double errorPassThrough(int prevNeuronIndex) {
		return backPropDeriv * weights.get(prevNeuronIndex);
	}

	public void setWeights(ArrayList<Double> newWeights) {
		weights = newWeights;
	}

	public void resetNeuron(int inputs, boolean bias) {
		if (bias)
			inputs++;
		newRandomWeights(inputs);
		inputHolder = null;
		outputHolder = backPropDeriv = 0;
		this.weightDeltaHolder = new ArrayList<>(Collections.nCopies(inputs, 0.0));
		Collections.fill(weightDeltaHolder, 0.0);
	}
	public void resetNeuron() {
		resetNeuron(weights.size(), false);
	}

	public double returnOutput() {
		return outputHolder;
	}
	
	/**
	 * Mutation Methods
	 * 
	 * @return
	 */

	public void removeWeight(int weightIndex) {
		weights.remove(weightIndex);
	}

	public Neuron copyNeuron() {
		try {
			return (Neuron) this.clone();
		} catch (CloneNotSupportedException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
		System.out.println("error in copying neuron");
		return null;
	}

	// private Methods

	// Random weight algo from
	// https://stats.stackexchange.com/questions/47590/what-are-good-initial-weights-in-a-neural-network

	public void newRandomWeights(int inputs) {
		this.weights = new ArrayList<Double>(inputs);
		for (int i = 0; i < inputs; i++) {
			this.weights.add(((Math.random() * inputs) * Math.sqrt(2.0 / inputs)) / Math.sqrt(inputs));
		}
	}

	public void newRandomWeights() {
		newRandomWeights(inputHolder.length);
	}

	public void printNeuron() {
		for (int i = 0; i < weights.size(); i++) {
			System.out.print(weights.get(i) + "\t");
		}
		System.out.println();
	}
	
	// Multithreading functions copy of forward prop and backprop functions
	public void setFPInputs(double[] newInput) {
		inputHolder = newInput;
	}
	
	public double forwardPropMultiThread() {
		double[] functionInput = new double[inputHolder.length + 1];
		for (int i = 0; i < inputHolder.length; i++) {
			functionInput[i] = inputHolder[i] * weights.get(i);
		}
		// bias term
		if (inputHolder.length < weights.size()) {
			functionInput[functionInput.length - 1] = weights.get(weights.size() - 1);
		}

		outputHolder = NeuronLevelAF.activationOutput(functionInput);
		return outputHolder;
	}



	// Empty submethods for override in mega neruon

	public double[] forwardPropMega(double[] inputs, ActivationFunction AF) {
		// TODO Auto-generated method stub
		return null;
	}
}
