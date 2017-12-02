package neuralNet;

import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.io.*;

public class BJNeuralNet implements Serializable {
	/**
	 * 
	 */
	private static final long serialVersionUID = 1L;
	
	int inputNeurons;
	int outputNeurons;
	int hiddenNeurons;
	double learningRate;
	double momentum;

	double errors;

	int totalNeurons;
	int weights;

	double outputResults[];
	double resultsMatrix[];
	double lastErrors[];
	double changes[];
	double thresholds[];
	double weightChanges[];
	double allThresholds[];
	double threshChanges[];

	double errorChanges[];

	public BJNeuralNet(int inputCount, int hiddenCount, int outputCount, double learnRate, double newMonetum) {
		learningRate = learnRate;
		momentum = newMonetum;
		inputNeurons = inputCount;
		hiddenNeurons = hiddenCount;
		outputNeurons = outputCount;
		totalNeurons = inputCount + hiddenCount + outputCount;
		weights = (inputCount * hiddenCount) + (hiddenCount * outputCount);
		outputResults = new double[totalNeurons];
		resultsMatrix = new double[weights];
		weightChanges = new double[weights];
		thresholds = new double[totalNeurons];
		errorChanges = new double[totalNeurons];
		lastErrors = new double[totalNeurons];
		allThresholds = new double[totalNeurons];
		changes = new double[weights];
		threshChanges = new double[totalNeurons];
		reset();
	}

	public void reset() {
		for (int loc = 0; loc < totalNeurons; loc++) {
			thresholds[loc] = 0.5 - (Math.random());
			threshChanges[loc] = 0;
			allThresholds[loc] = 0;
		}
		for (int loc = 0; loc < resultsMatrix.length; loc++) {
			resultsMatrix[loc] = 0.5 - (Math.random());
			weightChanges[loc] = 0;
			changes[loc] = 0;
		}
	}

	public double threshold(double sum) {
		return (1.0 / (1 + Math.exp(-1.0 * sum)));
	}

	public double[] calOuput(double input[]) {
		int loc, pos;
		final int hiddenIndex = inputNeurons;
		final int outIndex = inputNeurons + hiddenNeurons;

		for (loc = 0; loc < inputNeurons; loc++) {
			outputResults[loc] = input[loc];
		}

		// hidden layer
		int rLoc = 0;
		for (loc = hiddenIndex; loc < outIndex; loc++) {
			double sum = thresholds[loc];
			for (pos = 0; pos < inputNeurons; pos++) {
				sum += outputResults[pos] * resultsMatrix[rLoc++];
			}
			outputResults[loc] = threshold(sum);
		}

		// final layer
		double results[] = new double[outputNeurons];
		for (loc = outIndex; loc < totalNeurons; loc++) {
			double sum = thresholds[loc];

			for (pos = hiddenIndex; pos < outIndex; pos++) {
				sum += outputResults[pos] * resultsMatrix[rLoc++];
			}
			outputResults[loc] = threshold(sum);
			results[loc - outIndex] = outputResults[loc];
		}

		return results;
	}

	public void calcError(double ideal[]) {
		int loc, pos;
		final int hiddenIndex = inputNeurons;
		final int outputIndex = inputNeurons + hiddenNeurons;

		for (loc = inputNeurons; loc < totalNeurons; loc++) {
			lastErrors[loc] = 0;
		}

		for (loc = outputIndex; loc < totalNeurons; loc++) {
			lastErrors[loc] = ideal[loc - outputIndex] - outputResults[loc];
			errors += lastErrors[loc] * lastErrors[loc];
			errorChanges[loc] = lastErrors[loc] * outputResults[loc] * (1 - outputResults[loc]);
		}
		int locx = inputNeurons * hiddenNeurons;
		for (loc = outputIndex; loc < totalNeurons; loc++) {
			for (pos = hiddenIndex; pos < outputIndex; pos++) {
				changes[locx] += errorChanges[loc] * outputResults[pos];
				lastErrors[pos] += resultsMatrix[locx] * errorChanges[loc];
				locx++;
			}
			allThresholds[loc] += errorChanges[loc];
		}

		// hidden layer deltas
		for (loc = hiddenIndex; loc < outputIndex; loc++) {
			errorChanges[loc] = lastErrors[loc] * outputResults[loc] * (1 - outputResults[loc]);
		}

		// input layer errors
		locx = 0; // offset into weight array
		for (loc = hiddenIndex; loc < outputIndex; loc++) {
			for (pos = 0; pos < hiddenIndex; pos++) {
				changes[locx] += errorChanges[loc] * outputResults[pos];
				lastErrors[pos] += resultsMatrix[locx] * errorChanges[loc];
				locx++;
			}
			allThresholds[loc] += errorChanges[loc];
		}
	}

	public double getError() {
		double err = Math.sqrt(errors / (inputNeurons * outputNeurons));
		errors = 0;
		return err;
	}

	public void train() {
		int loc;
		for (loc = 0; loc < resultsMatrix.length; loc++) {
			weightChanges[loc] = (learningRate * changes[loc]) + (momentum * weightChanges[loc]);
			resultsMatrix[loc] += weightChanges[loc];
			changes[loc] = 0;
		}
		for (loc = inputNeurons; loc < totalNeurons; loc++) {
			threshChanges[loc] = learningRate * allThresholds[loc] + (momentum * threshChanges[loc]);
			thresholds[loc] += threshChanges[loc];
			allThresholds[loc] = 0;
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
			System.out.println("IO exception.  Save method.");
		}
	}

	public static BJNeuralNet readNN(String filename) {
		ObjectInputStream ois;
		try {
			ois = new ObjectInputStream(new FileInputStream(filename));
			try {
				BJNeuralNet output = (BJNeuralNet) ois.readObject();
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
}
