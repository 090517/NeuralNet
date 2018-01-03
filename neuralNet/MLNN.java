package neuralNet;

import java.io.*;
import java.util.ArrayList;

import org.jfree.data.xy.XYSeries;

import java.awt.Color;

import javax.swing.JFrame;
import javax.swing.WindowConstants;

import org.jfree.chart.ChartFactory;
import org.jfree.chart.ChartPanel;
import org.jfree.chart.JFreeChart;
import org.jfree.chart.plot.XYPlot;
import org.jfree.data.xy.XYSeriesCollection;
import activationFunctions.ActivationFunction;

public class MLNN extends JFrame implements Serializable {
	/**
	 * 
	 */
	private static final long serialVersionUID = 1L;

	double error;
	public int rounds;
	public int totalRounds;
	ArrayList<Layer> hiddenLayers;
	Layer outputLayer;
	XYSeries errorHistory;

	ArrayList<Layer> bestHiddenLayers;
	Layer bestOutPutLayer;
	double bestError;
	boolean bestLoaded;
	double learningRate;
	double momentum;

	public MLNN(int[] nnHiddenArrangment, int inputs, int outputs, ActivationFunction[] AFArray, boolean[] bias,
			double newLearningRate, double newMometum) {
		
		if (!checkValidMLNN(nnHiddenArrangment, inputs, outputs, AFArray, bias)) {
			System.out.println("INVALID NN ARRANGMENT, CHECK");
		};
		
		error = 0;
		bestError = Double.MAX_VALUE;
		rounds = 0;
		totalRounds = 0;
		errorHistory = new XYSeries("Error History");
		hiddenLayers = new ArrayList<Layer>();
		bestLoaded = false;

		// Creation of hidden Layer arraylist
		int prevInputs = inputs;
		for (int i = 0; i < nnHiddenArrangment.length; i++) {
			hiddenLayers.add(
					new Layer(nnHiddenArrangment[i], prevInputs, AFArray[i], bias[i], bias[i + 1], newLearningRate, newMometum));
			if (!AFArray[i].returnVectorOutput()) {// need to change to passthrough check
				prevInputs = nnHiddenArrangment[i];
			}
		}
		
		if (AFArray[AFArray.length - 1].returnVectorOutput()) {
			outputLayer = new Layer(1, prevInputs, AFArray[AFArray.length - 1], bias[AFArray.length - 1], false,
					newLearningRate, newMometum);
		} else {
			outputLayer = new Layer(outputs, prevInputs, AFArray[AFArray.length - 1], bias[AFArray.length - 1], false,
					newLearningRate, newMometum);
		}
	}
	
	public MLNN(int[] nnHiddenArrangment, int inputs, int outputs, ActivationFunction[] AFArray, boolean[] bias,
			double newLearningRate) {
		this( nnHiddenArrangment,  inputs,  outputs,  AFArray,  bias, newLearningRate, 0);
	}
	
	/**what layers have requirments
	 * Maxout Layer - input must equal output size - i.e at output layer
	 */
	
	private boolean checkValidMLNN(int[] nnHiddenArrangment, int inputs, int outputs, ActivationFunction[] aFArray,
			boolean[] bias) {	
		if (aFArray[aFArray.length-1].returnVectorOutput()){
			if (outputs!=nnHiddenArrangment[nnHiddenArrangment.length-1]) {
				return false;
			}
		}
		return true;
	}

	// regulars

	public MLNN() {//empty constructor for testing purposes
	}

	public double[] forwardPropWError(double[] input, double[] ideal) {
		for (int i = 0; i < hiddenLayers.size(); i++) {
			input = hiddenLayers.get(i).forwardProp(input);
		}
		 input = outputLayer.forwardProp(input);
		// errorcalc
		
			for (int i = 0; i < ideal.length; i++) {
				this.error += .5 * Math.pow(ideal[i] - input[i],2);
			}
		rounds++;
		totalRounds++;
		return input;
	}

	public double[] forwardProp(double[] input) {
		for (int i = 0; i < hiddenLayers.size(); i++) {
			input = hiddenLayers.get(i).forwardProp(input);
		}
		return outputLayer.forwardProp(input);
	}

	public double getError() {
		double output = this.error / this.rounds;
		errorHistory.add(totalRounds, output);
		if (output < bestError) {
			bestHiddenLayers = hiddenLayers;
			bestOutPutLayer = outputLayer;
			bestError = output;
		}

		error = 0;
		rounds = 0;
		return output;
	}

	public void backProp(double[] ideal) {
		outputLayer.backPropOutputLayer(ideal);
		Layer prevLayer = outputLayer;
		for (int i = hiddenLayers.size() - 1; i >= 0; i--) {
			hiddenLayers.get(i).backPropHiddenLayer(prevLayer);
			prevLayer = hiddenLayers.get(i);
		}
		outputLayer.updateWeights();
		for (int i = 0; i < hiddenLayers.size(); i++) {
			hiddenLayers.get(i).updateWeights();
		}
	}

	public void setWeights(int layerIdx, double[][] weights) {
		if (layerIdx > hiddenLayers.size() - 1) {
			outputLayer.setWeights(weights);
		} else {
			hiddenLayers.get(layerIdx).setWeights(weights);
		}
	}

	public void resetNN() {
		outputLayer.reset();
		for (Layer l : hiddenLayers) {
			l.reset();
		}
		error = rounds = totalRounds = 0;
		bestError = Double.MAX_VALUE;
		errorHistory = new XYSeries("Error History");
		bestLoaded = false;
	}

	public void loadBest() {
		if (!bestLoaded) {
			Layer tempLayer = outputLayer;
			ArrayList<Layer> tempHiddenLayers = hiddenLayers;
			outputLayer = bestOutPutLayer;
			hiddenLayers = bestHiddenLayers;
			bestOutPutLayer = tempLayer;
			bestHiddenLayers = tempHiddenLayers;
			bestLoaded = true;
		}
	}

	public void loadCurrent() {
		if (bestLoaded) {
			Layer tempLayer = outputLayer;
			ArrayList<Layer> tempHiddenLayers = hiddenLayers;
			outputLayer = bestOutPutLayer;
			hiddenLayers = bestHiddenLayers;
			bestOutPutLayer = tempLayer;
			bestHiddenLayers = tempHiddenLayers;
			bestLoaded = false;
		}
	}

	// Neuron Methods

	public void addNeuron(int layer, int index, Neuron newNeuron) {
		
		

	}

	public void removeNeuron(int layer, int index) {
		hiddenLayers.get(layer).removeNeuron(index);
		hiddenLayers.get(layer+1).removeWeights(index);;
	}

	// Layer Methods
	
	//todo need to update weights for other layers besides inserted ones.
	public void addLayer(int index, int numNeurons, ActivationFunction AF, boolean thisLayerBias) {
		//if last layer, need to replace
		if ((index-1)>hiddenLayers.size()) {
			hiddenLayers.add(outputLayer);
			outputLayer=new Layer(numNeurons, hiddenLayers.get(index-1).neuronArray.size(), AF, hiddenLayers.get(index-1).bias, thisLayerBias, learningRate, momentum);
		}
		else {
		//else need to insert
		hiddenLayers.add(index, new Layer(numNeurons, hiddenLayers.get(index-1).neuronArray.size(), AF, hiddenLayers.get(index-1).bias, thisLayerBias, learningRate, momentum));
		}
	}

	public void removeLayer(int index) {}
	
	// Mutation Methods TODO
	public void trimNeurons(double threshold) {
		// removes neurons are weighed small in the next layer. TODO
		ArrayList<Integer> neuronsToDelete = new ArrayList<Integer>();
		for (int i = 0; i < hiddenLayers.get(hiddenLayers.size() - 1).neuronArray.size(); i++) {
			for (int j = 0; j < outputLayer.neuronArray.size(); j++) {
				if (outputLayer.neuronArray.get(j).weights.get(i) > threshold) {
					break;
				}
				if (j == outputLayer.neuronArray.size() - 1) {
					System.out.println("neruon added" + i);
					neuronsToDelete.add(i);
				}
			}
		}

		// todo delete neruosn. Maybe need to rewright everything as arraylist

		neuronsToDelete = new ArrayList<Integer>();
		for (int layer = hiddenLayers.size() - 1; layer > 1; layer--) {
			for (int i = 0; i < hiddenLayers.get(layer - 1).neuronArray.size(); i++) {
				for (int j = 0; j < hiddenLayers.get(layer).neuronArray.size(); j++) {
					if (hiddenLayers.get(layer).neuronArray.get(j).weights.get(i) > threshold) {
						break;
					}
					System.out.print(hiddenLayers.get(layer).neuronArray.get(j).weights.get(i) + "\t");
					if (j == hiddenLayers.get(layer).neuronArray.size() - 1) {
						System.out.println("neruon added" + i);
						neuronsToDelete.add(i);
					}
				}
			}
			// todo delete neruosn. Maybe need to rewright everything as arraylist
		}
	}

	// Utility Functions

	public void saveNN(String filename) {
		try {
			ObjectOutputStream oos = new ObjectOutputStream(new FileOutputStream(filename));
			oos.writeObject(this);
			oos.close();
		} catch (FileNotFoundException e) {
			System.out.println("File not found exception. Save Method.");
		} catch (IOException e) {
			e.printStackTrace();
			System.out.println("IO exception.  Save method.");
		}
	}

	public static MLNN readNN(String filename) {
		ObjectInputStream ois;
		try {
			ois = new ObjectInputStream(new FileInputStream(filename));
			try {
				MLNN output = (MLNN) ois.readObject();
				ois.close();
				return output;
			} catch (ClassNotFoundException e) {
				System.out.println("Class not found exception.  Read Method.");
			}
			ois.close();
		} catch (FileNotFoundException e) {
			System.out.println("File not found exception.  Read Method.");
		} catch (IOException e) {
			System.out.println("IO exception.  Read method.");
		}
		System.out.println("Error, Returned Null");
		return null;
	}

	public void plotErrors() {
		XYSeriesCollection dataset = new XYSeriesCollection();
		dataset.addSeries(errorHistory);
		JFreeChart chart = ChartFactory.createScatterPlot("Errors Vs Epochs", "Training Rounds", "Errors", dataset);
		// Changes background color
		XYPlot plot = (XYPlot) chart.getPlot();
		plot.setBackgroundPaint(new Color(255, 228, 196));

		// Create Panel
		ChartPanel panel = new ChartPanel(chart);
		setContentPane(panel);
		this.setSize(1200, 1000);
		this.setLocationRelativeTo(null);
		this.setDefaultCloseOperation(WindowConstants.EXIT_ON_CLOSE);
		this.setVisible(true);
	}

	public void printNN() {
		for (int i = 0; i < hiddenLayers.size(); i++) {
			hiddenLayers.get(i).printLayer();
		}
		outputLayer.printLayer();
	}

	// Multithreading code TODO

	public double[] forwardPropMulti(double[] input, double[] ideal) throws Exception {
		for (int i = 0; i < hiddenLayers.size(); i++) {
			input = hiddenLayers.get(i).forwardProp(input);
		}
		// errorcalc
		for (int i = 0; i < ideal.length; i++) {
			this.error += .5 * Math.pow(ideal[i] - outputLayer.neuronArray.get(i).outputHolder, 2);
		}
		rounds++;
		totalRounds++;
		return outputLayer.forwardProp(input);
	}

	public double[] forwardPropMulti(double[] input) throws Exception {
		for (int i = 0; i < hiddenLayers.size(); i++) {
			input = hiddenLayers.get(i).forwardProp(input);
		}
		return outputLayer.forwardProp(input);
	}



}