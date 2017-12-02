package neuralNet;

import java.io.Serializable;
import java.util.ArrayList;

import activationFunctions.ActivationFunction;

public class Layer implements Serializable{
	/**
	 * 
	 */
	private static final long serialVersionUID = 1L;
	
	public ArrayList<Neuron> neuronArray;
	ActivationFunction AF;
	boolean bias;
	double learningRate;

	public Layer(int numNeurons, int inputs, ActivationFunction newAF, boolean prevLayerBias, boolean thisLayerBias,
			double newLearningRate) {
		bias = thisLayerBias;
		neuronArray = new ArrayList<Neuron>();
		for (int i = 0; i < numNeurons; i++) {
			neuronArray.add(new Neuron(inputs, prevLayerBias));
		}
		AF = newAF;
		learningRate = newLearningRate;
	}

	public double[] forwardProp(double[] inputs) {
		double[] output = new double[neuronArray.size()];
				
		for (int i = 0; i < neuronArray.size(); i++) {
			output[i] = neuronArray.get(i).forwardProp(inputs, AF);
		}
		return output;
	}

	public void backPropOutputLayer(double[] ideal) {
		for (int i = 0; i < ideal.length; i++) {
			neuronArray.get(i).backProp(ideal[i], AF, learningRate, true);
		}
	}

	public void backPropHiddenLayer(Layer prevLayer) {
		for (int i = 0; i < neuronArray.size(); i++) {
			double prevLayerErrors=0;
			for (int j = 0; j < prevLayer.neuronArray.size(); j++) {
				prevLayerErrors+=prevLayer.neuronArray.get(j).errorPassThrough(i);
			}
			neuronArray.get(i).backProp(prevLayerErrors, AF, learningRate, false);
		}
	}

	public void setWeights(double[][] weights) {
		for (int i = 0; i < weights.length; i++) {
			neuronArray.get(i).setWeights(weights[i]);
		}
	}

	public void updateWeights() {
		for (Neuron n: neuronArray) {
			n.updateWeights();
		}
	}
}
