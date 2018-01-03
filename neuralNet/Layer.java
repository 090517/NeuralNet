package neuralNet;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import activationFunctions.ActivationFunction;

public class Layer implements Serializable {
	private static final long serialVersionUID = 1L;

	public ArrayList<Neuron> neuronArray;
	ActivationFunction AF;
	boolean bias;
	double learningRate;
	double momentum;

	/**
	 * Layer Constructor
	 */

	public Layer(int numNeurons, int inputs, ActivationFunction newAF, boolean prevLayerBias, boolean thisLayerBias,
			double newLearningRate, double momentum) {

		if (newAF.returnVectorOutput()) {
			this.MegaNeuronLayer(numNeurons, inputs, newAF, prevLayerBias, thisLayerBias, newLearningRate);
		} else {
			this.AF = newAF;
			this.bias = thisLayerBias;
			this.learningRate = newLearningRate;
			this.momentum= momentum;
			neuronArray = new ArrayList<Neuron>();
			for (int i = 0; i < numNeurons; i++) {
				neuronArray.add(new Neuron(inputs, prevLayerBias));
			}
		}
	}

	private void MegaNeuronLayer(int numNeurons, int inputs, ActivationFunction newAF, boolean prevLayerBias,
			boolean thisLayerBias, double newLearningRate) {
		this.AF = newAF;
		this.bias = thisLayerBias;
		this.learningRate = newLearningRate;
		neuronArray = new ArrayList<Neuron>();
		neuronArray.add(new MegaNeuron(inputs, prevLayerBias));
	}

	public Layer(int inputs, ActivationFunction[] AFArray, boolean prevLayerBias, boolean thisLayerBias,
			double newLearningRate) {
		this.bias = thisLayerBias;
		neuronArray = new ArrayList<Neuron>(AFArray.length);
		for (int i = 0; i < AFArray.length; i++) {
			neuronArray.add(new Neuron(inputs, prevLayerBias, AFArray[i]));
		}
		learningRate = newLearningRate;
	}

	// for forwardprop, need to do one layer at a time

	public double[] forwardProp(double[] inputs) {
		double[] output = new double[neuronArray.size()];
		if (AF.returnVectorOutput()) {
			output = neuronArray.get(0).forwardPropMega(inputs, AF);

		} else {
			for (int i = 0; i < neuronArray.size(); i++) {
				output[i] = neuronArray.get(i).forwardProp(inputs, AF);
			}
		}
		return output;
	}

	public void backPropOutputLayer(double[] ideal) {
		if (AF.returnVectorOutput()) {
			((MegaNeuron) neuronArray.get(0)).backPropMega(ideal, learningRate, momentum, true, AF);
		} else {
			for (int i = 0; i < ideal.length; i++) {
				neuronArray.get(i).backProp(ideal[i], learningRate, momentum, true, AF);
			}
		}
	}

	public void backPropHiddenLayer(Layer prevLayer) {
		// get error vector from prevLayer
		if (AF.returnVectorOutput()) {
			if (prevLayer.AF.returnVectorOutput()) {
				
				//todo

			} else {
				double[] prevLayerErrors = new double[((MegaNeuron) neuronArray.get(0)).outputHolderVector.length];
				int b = 0;
				if (bias)
					b = 1;

				for (int i = 0; i < prevLayerErrors.length - b; i++) {
					double errorSum = 0;
					for (int j = 0; j < prevLayer.neuronArray.size(); j++) {
						errorSum += prevLayer.neuronArray.get(j).errorPassThrough(i);
					}
					prevLayerErrors[i] = errorSum;
				}
				((MegaNeuron) neuronArray.get(0)).backPropMega(prevLayerErrors, learningRate, momentum, false, AF);
			}
		} else {
			for (int i = 0; i < neuronArray.size(); i++) {
				double prevLayerErrors = 0;
				for (int j = 0; j < prevLayer.neuronArray.size(); j++) {
					prevLayerErrors += prevLayer.neuronArray.get(j).errorPassThrough(i);
				}
				neuronArray.get(i).backProp(prevLayerErrors, learningRate, momentum, false, AF);
			}
		}
	}
	
	//mutation methods
	
	public void removeNeuron(int neuronIndex) {
		neuronArray.remove(neuronIndex);
	}
	
	public void removeWeights(int weightIndex) {
		for (Neuron n: neuronArray) {
			n.removeWeight(weightIndex);
		}
	}

	public void addNeuron(int position) {
		neuronArray.add(position, neuronArray.get(0).copyNeuron());
	}
	
	public void addWeights(int weightIndex, double weight) {
		for (Neuron n: neuronArray) {
			n.weights.add(weightIndex, weight);;
		}
	}
	
	public void mutateWeights() {
		
	}
	
	//Layer Adjustments
	
	public void setWeights(double[][] weights) {
		for (int i = 0; i < weights.length; i++) {
			ArrayList<Double> newWeights = new ArrayList<Double>();
			for (double d : weights[i])
				newWeights.add(d);
			neuronArray.get(i).setWeights(newWeights);
		}
	}

	public void updateWeights() {
		for (Neuron n : neuronArray) {
			n.updateWeights();
		}
	}

	public void reset() {
		for (Neuron n : neuronArray) {
			n.resetNeuron();
		}
	}

	public void printLayer() {
		for (int i = 0; i < neuronArray.size(); i++) {
			System.out.print("Neuron " + i + " weights: ");
			neuronArray.get(i).printNeuron();
		}
	}

	// multithread methods

	// Load data concurrently,and get all to calculate concurrently, return double
	// array output;

	public double[] forwardPropMulti(double[] newInput) throws Exception {
		ExecutorService executorService = Executors.newCachedThreadPool();
		List<Future<Double>> futureList = new ArrayList<Future<Double>>();
		double[] output = new double[neuronArray.size()];

		for (int i = 0; i < neuronArray.size(); i++) {
			neuronArray.get(i).setFPInputs(newInput.clone());
			futureList.add(executorService.submit(neuronArray.get(i)::forwardPropMultiThread));
		}

		for (int i = 0; i < neuronArray.size(); i++) {
			if (futureList.get(i).isDone()) {
				output[i] = futureList.get(i).get();
			}
		}

		executorService.shutdown();
		return output;
	}
}
