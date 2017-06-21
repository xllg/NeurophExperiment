package cn.neuron;

import java.util.Arrays;

import org.neuroph.core.NeuralNetwork;
import org.neuroph.core.events.LearningEvent;
import org.neuroph.core.events.LearningEventListener;
import org.neuroph.core.learning.DataSet;
import org.neuroph.core.learning.DataSetRow;
import org.neuroph.nnet.MultiLayerPerceptron;
import org.neuroph.nnet.learning.BackPropagation;
import org.neuroph.nnet.learning.MomentumBackpropagation;

public class AnimalClassic implements LearningEventListener {
	public static void main(String[] args) {
		new AnimalClassic().Calculate();
	}
	
	public void Calculate() {
		//定义animal训练数据
		String trainingSetFileName = "data_sets/animals_data.txt";		
		//定义两个变量分别为输入层神经元个数20，输出神经元个数7个
		int inputsCount = 20;
		int outputsCount = 7;
		DataSet dataSet = DataSet.createFromFile(trainingSetFileName, inputsCount, outputsCount, "\t");
		// 建立训练集
		System.out.println("Creating neural network...");
		// 建立一个神经网络，定义一个隐层，神经元设为22
		MultiLayerPerceptron neuralNet = new MultiLayerPerceptron(inputsCount,16,outputsCount);
		// 定义学习规则、BP
		MomentumBackpropagation learningRule = (MomentumBackpropagation)neuralNet.getLearningRule();
		learningRule.addListener(this);
		//设置最大误差、学习速度
		learningRule.setLearningRate(0.2);
		learningRule.setMaxError(0.01);
		System.out.println("Training network...");
		// 开始学习训练集
		neuralNet.learn(dataSet);
		System.out.println("Training completed.");
		// 测试感知机是否正确输出，打印
		System.out.println("Testing trained network...");
		testNeralNetwork(neuralNet, dataSet);
	}
	
	/***
	 * 结果测试
	 * @param myPerceptron
	 * @param trainingSet
	 */
	public static void testNeralNetwork(NeuralNetwork myPerceptron,
			DataSet trainingSet) {
		for (DataSetRow dataRow : trainingSet.getRows()) {
			myPerceptron.setInput(dataRow.getInput());
			myPerceptron.calculate();
			double[] networkOutput = myPerceptron.getOutput();
			System.out.print("Input: " + Arrays.toString(dataRow.getInput()));
			System.out.println(" Output: " + Arrays.toString(networkOutput));
		}
	}
	
	/***
	 * 事件监听
	 */
	@Override
	public void handleLearningEvent(LearningEvent event) {
		BackPropagation bp = (BackPropagation)event.getSource();
		System.out.println(bp.getCurrentIteration() + ". iteration : "+ bp.getTotalNetworkError());
	}
}
