package jp.masazdream.ml.inf;

public interface MachineLearning {
	/**
	 * 学習
	 * 
	 * @param cls クラス
	 * @param data double配列
	 */
	public void train(int entryCnt, int[] clz, double[][] data);
	
	/**
	 * 精度測定
	 * 
	 * @param data double配列
	 * @return 分類クラス
	 */
	public int evaluate(double[] data);
}
