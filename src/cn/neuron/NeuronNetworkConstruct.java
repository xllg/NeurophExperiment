package cn.neuron;

import java.util.Arrays;

import org.neuroph.core.NeuralNetwork;
import org.neuroph.core.events.LearningEvent;
import org.neuroph.core.events.LearningEventListener;
import org.neuroph.core.learning.DataSet;
import org.neuroph.core.learning.DataSetRow;
import org.neuroph.core.learning.LearningRule;
import org.neuroph.nnet.MultiLayerPerceptron;
import org.neuroph.nnet.Perceptron;
import org.neuroph.nnet.learning.BackPropagation;
import org.neuroph.util.TransferFunctionType;

public class NeuronNetworkConstruct implements LearningEventListener {
	
	public static void main(String[] args) {
		new NeuronNetworkConstruct().Calculate();
	}

	/***
	 * 单层神经网络AND和OR逻辑运算
	 */
	public void Calculate() {
		// 建立训练集，有两个输入一个输出
		DataSet trainingSet = new DataSet(2, 1);
		trainingSet.addRow(new DataSetRow(new double[] { 0, 0 }, new double[] { 0 }));
		trainingSet.addRow(new DataSetRow(new double[] { 0, 1 }, new double[] { 1 }));
		trainingSet.addRow(new DataSetRow(new double[] { 1, 0 }, new double[] { 1 }));
		trainingSet.addRow(new DataSetRow(new double[] { 1, 1 }, new double[] { 1 }));
		// 建立一个感知机，定义输入刺激是2个，感知机输出是1个，这里调用Neuroph提供的Perceptron类
		NeuralNetwork myPerceptron = new Perceptron(2, 1);
		// 反向误差传播
		myPerceptron.setLearningRule(new BackPropagation());		
		LearningRule lr = myPerceptron.getLearningRule();
		lr.addListener(this);
		// 开始学习训练集
		myPerceptron.learn(trainingSet);
		// 测试感知机是否正确输出，打印
		System.out.println("Testing trained perceptron");
		testNeralNetwork(myPerceptron, trainingSet);
	}

	/***
	 * 双层神经网络XOR逻辑运算
	 */
	public void XORCalculate() {
		// 建立训练集，有两个输入一个输出
		DataSet trainingSet = new DataSet(2, 1);
		trainingSet.addRow(new DataSetRow(new double[] { 0, 0 }, new double[] { 0 }));
		trainingSet.addRow(new DataSetRow(new double[] { 0, 1 }, new double[] { 1 }));
		trainingSet.addRow(new DataSetRow(new double[] { 1, 0 }, new double[] { 1 }));
		trainingSet.addRow(new DataSetRow(new double[] { 1, 1 }, new double[] { 0 }));
		//创建多个感知机，输入层2个神经元，隐含层3个神经元，最后输出层为1个隐含神经元，我们使用TANH传输函数用于最后的格式化输出
		MultiLayerPerceptron myMlPerceptron = new MultiLayerPerceptron(TransferFunctionType.TANH,2,2,1);
		//反向误差传播
		myMlPerceptron.setLearningRule(new BackPropagation());
		LearningRule learningRule = myMlPerceptron.getLearningRule();
        learningRule.addListener(this);
		//开始训练
		myMlPerceptron.learn(trainingSet);
		// 测试感知机是否正确输出，打印
		System.out.println("Testing trained perceptron");
		testNeralNetwork(myMlPerceptron, trainingSet);
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
