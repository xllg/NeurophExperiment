package cn.neuron;

import java.util.Arrays;

import org.neuroph.core.NeuralNetwork;
import org.neuroph.core.learning.DataSet;
import org.neuroph.core.learning.DataSetRow;
import org.neuroph.nnet.MultiLayerPerceptron;
import org.neuroph.nnet.learning.LMS;

public class StockPrediction {
	public static void main(String[] args) {
		new StockPrediction().Calculate();
	}
	
	private void Calculate() {
		int maxIterations = 10000;
		NeuralNetwork neuralNet = new MultiLayerPerceptron(4,9,1);
		((LMS)neuralNet.getLearningRule()).setMaxError(0.001);//0-1
		((LMS)neuralNet.getLearningRule()).setLearningRate(0.7);//0-1
		((LMS)neuralNet.getLearningRule()).setMaxIterations(maxIterations);
		DataSet trainingSet = new DataSet(4,1);
		double daxmax = 10000.0;
		trainingSet.addRow(new double[]{3710.0D / daxmax,3690.0D / daxmax,3890.0D / daxmax,3695.0D / daxmax}, new double[]{3666.0D / daxmax});
		trainingSet.addRow(new double[]{3690.0D / daxmax,3890.0D / daxmax,3695.0D / daxmax,3666.0D / daxmax}, new double[]{3692.0D / daxmax});
		trainingSet.addRow(new double[]{3890.0D / daxmax,3695.0D / daxmax,3666.0D / daxmax,3692.0D / daxmax}, new double[]{3886.0D / daxmax});
		trainingSet.addRow(new double[]{3695.0D / daxmax,3666.0D / daxmax,3692.0D / daxmax,3886.0D / daxmax}, new double[]{3914.0D / daxmax});
		trainingSet.addRow(new double[]{3666.0D / daxmax,3692.0D / daxmax,3886.0D / daxmax,3914.0D / daxmax}, new double[]{3956.0D / daxmax});
		trainingSet.addRow(new double[]{3692.0D / daxmax,3886.0D / daxmax,3914.0D / daxmax,3956.0D / daxmax}, new double[]{3953.0D / daxmax});
		trainingSet.addRow(new double[]{3886.0D / daxmax,3914.0D / daxmax,3956.0D / daxmax,3953.0D / daxmax}, new double[]{4044.0D / daxmax});
		trainingSet.addRow(new double[]{3914.0D / daxmax,3956.0D / daxmax,3953.0D / daxmax,4044.0D / daxmax}, new double[]{3987.0D / daxmax});
		trainingSet.addRow(new double[]{3956.0D / daxmax,3953.0D / daxmax,4044.0D / daxmax,3987.0D / daxmax}, new double[]{3996.0D / daxmax});
		trainingSet.addRow(new double[]{3953.0D / daxmax,4044.0D / daxmax,3987.0D / daxmax,3996.0D / daxmax}, new double[]{4043.0D / daxmax});
		trainingSet.addRow(new double[]{4044.0D / daxmax,3987.0D / daxmax,3996.0D / daxmax,4043.0D / daxmax}, new double[]{4068.0D / daxmax});
		trainingSet.addRow(new double[]{3987.0D / daxmax,3996.0D / daxmax,4043.0D / daxmax,4068.0D / daxmax}, new double[]{4176.0D / daxmax});
		trainingSet.addRow(new double[]{3996.0D / daxmax,4043.0D / daxmax,4068.0D / daxmax,4176.0D / daxmax}, new double[]{4187.0D / daxmax});
		trainingSet.addRow(new double[]{4043.0D / daxmax,4068.0D / daxmax,4176.0D / daxmax,4187.0D / daxmax}, new double[]{4223.0D / daxmax});
		trainingSet.addRow(new double[]{4068.0D / daxmax,4176.0D / daxmax,4187.0D / daxmax,4223.0D / daxmax}, new double[]{4259.0D / daxmax});
		trainingSet.addRow(new double[]{4176.0D / daxmax,4187.0D / daxmax,4223.0D / daxmax,4259.0D / daxmax}, new double[]{4203.0D / daxmax});
		trainingSet.addRow(new double[]{4187.0D / daxmax,4223.0D / daxmax,4259.0D / daxmax,4203.0D / daxmax}, new double[]{3989.0D / daxmax});
		neuralNet.learn(trainingSet);
		DataSet testSet = new DataSet(4);
		testSet.addRow(new double[]{4223.0D / daxmax,4259.0D / daxmax,4203.0D / daxmax,3989.0D / daxmax});
		testSet.addRow(new double[]{4223.0D / daxmax,4259.0D / daxmax,4203.0D / daxmax,3989.0D / daxmax});
		testSet.addRow(new double[]{4223.0D / daxmax,4259.0D / daxmax,4203.0D / daxmax,3989.0D / daxmax});
		testSet.addRow(new double[]{4223.0D / daxmax,4259.0D / daxmax,4203.0D / daxmax,3989.0D / daxmax});
		
		for (DataSetRow dataRow : testSet.getRows()) {
			neuralNet.setInput(dataRow.getInput());
			neuralNet.calculate();
			double[] networkOutput = neuralNet.getOutput();
			System.out.print("Input: " + Arrays.toString(dataRow.getInput()));
			System.out.println(" Output: " + Arrays.toString(networkOutput));
		}
	}
}
