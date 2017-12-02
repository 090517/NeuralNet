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
	Layer[] hiddenLayers;
	Layer outputLayer;
	XYSeries errorHistory;

	Layer[] bestHiddenLayers;
	Layer bestOutPutLayer;
	double bestError;
	boolean bestLoaded;

	public MLNN(int[] nnHiddenArrangment, int inputs, int outputs, ActivationFunction[] AFArray, boolean[] bias,
			double newLearningRate) {
		error = 0;
		bestError = Double.MAX_VALUE;
		rounds = 0;
		totalRounds = 0;
		errorHistory = new XYSeries("Error History");
		hiddenLayers = new Layer[nnHiddenArrangment.length];
		bestLoaded = false;
		int prevInputs = inputs;

		for (int i = 0; i < nnHiddenArrangment.length; i++) {
			hiddenLayers[i] = new Layer(nnHiddenArrangment[i], prevInputs, AFArray[i], bias[i], bias[i + 1],
					newLearningRate);
			prevInputs = nnHiddenArrangment[i];
		}
		outputLayer = new Layer(outputs, prevInputs, AFArray[AFArray.length - 1], bias[AFArray.length - 1], false,
				newLearningRate);

	}

	public MLNN() {
		// TODO Auto-generated constructor stub
	}

	public double[] forwardProp(double[] input, double[] ideal) {
		for (int i = 0; i < hiddenLayers.length; i++) {
			input = hiddenLayers[i].forwardProp(input);
		}
		// errorcalc
		for (int i = 0; i < ideal.length; i++) {
			this.error += .5 * Math.pow(ideal[i] - outputLayer.neuronArray.get(i).outputHolder, 2);
		}
		rounds++;
		totalRounds++;
		return outputLayer.forwardProp(input);
	}

	public double[] forwardProp(double[] input) {
		for (int i = 0; i < hiddenLayers.length; i++) {
			input = hiddenLayers[i].forwardProp(input);
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
		for (int i = hiddenLayers.length - 1; i >= 0; i--) {
			hiddenLayers[i].backPropHiddenLayer(prevLayer);
			prevLayer = hiddenLayers[i];
		}
		outputLayer.updateWeights();
		for (int i = 0; i < hiddenLayers.length; i++) {
			hiddenLayers[i].updateWeights();
		}
	}

	public void setWeights(int layer, double[][] weights) {
		if (layer > hiddenLayers.length) {
			outputLayer.setWeights(weights);
		} else {
			hiddenLayers[layer - 1].setWeights(weights);
		}
	}

	public void loadBest() {
		if (!bestLoaded) {
			Layer tempLayer = outputLayer;
			Layer[] tempHiddenLayers = hiddenLayers;
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
			Layer[] tempHiddenLayers = hiddenLayers;
			outputLayer = bestOutPutLayer;
			hiddenLayers = bestHiddenLayers;
			bestOutPutLayer = tempLayer;
			bestHiddenLayers = tempHiddenLayers;
			bestLoaded = false;
		}
	}

	// removes neurons are weighed small in the next layer.
	public void trimNeurons(double threshold) {
		ArrayList<Integer> neuronsToDelete = new ArrayList<Integer>();
		for (int i = 0; i < hiddenLayers[hiddenLayers.length-1].neuronArray.size(); i++) {
			for (int j = 0; j < outputLayer.neuronArray.size(); j++) {
				if (outputLayer.neuronArray.get(j).weights[i] > threshold) {
					break;
				}
				if (j==outputLayer.neuronArray.size()-1) {
					System.out.println("neruon added" + i);
					neuronsToDelete.add(i);
				}
			}			
		}

		// todo delete neruosn. Maybe need to rewright everything as arraylist

		neuronsToDelete = new ArrayList<Integer>();
		for (int layer = hiddenLayers.length-1; layer > 1; layer--) {
			for (int i = 0; i < hiddenLayers[layer-1].neuronArray.size(); i++) {
				for (int j = 0; j < hiddenLayers[layer].neuronArray.size(); j++) {
					if (hiddenLayers[layer].neuronArray.get(j).weights[i] > threshold) {
						break;
					}
					System.out.print(hiddenLayers[layer].neuronArray.get(j).weights[i]+"\t");
					if (j==hiddenLayers[layer].neuronArray.size()-1) {
						System.out.println("neruon added" + i);
						neuronsToDelete.add(i);
					}
				}
			}
			// todo delete neruosn. Maybe need to rewright everything as arraylist			
		}


	}

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
		// ScatterPlotExample example = new ScatterPlotExample("Scatter Chart Example |
		// BORAJI.COM");
		this.setSize(1200, 1000);
		this.setLocationRelativeTo(null);
		this.setDefaultCloseOperation(WindowConstants.EXIT_ON_CLOSE);
		this.setVisible(true);
	}
}