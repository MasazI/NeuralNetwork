package jp.masazdream.ml.graph;

import java.awt.Color;
import java.awt.Graphics;
import java.awt.GridLayout;
import java.awt.Image;
import java.awt.image.BufferedImage;

import javax.swing.ImageIcon;
import javax.swing.JFrame;
import javax.swing.JLabel;

import jp.masazdream.ml.inf.MachineLearning;
import jp.masazdream.ml.nn.NeuralNetworkParameter;
import jp.masazdream.ml.param.MachineLearningParameter;

public class Graph {
	/**
	 * 機械学習器を作成する（オーバーライドすること）
	 * @return
	 */
	public MachineLearning createLearningMachine() {
		return null;
	}
	
	/**
	 * パラメータを指定して機械学習器を作成する（オーバーライドすること）
	 * 
	 * @param param
	 * @return
	 */
	public MachineLearning createLearningMachine(MachineLearningParameter param){
		return null;
	}
	
	/**
	 * コンストラクタ
	 * 
	 * @param title
	 */
	public Graph(String title){
		JFrame f = new JFrame(title);
	    f.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
	    f.setSize(840, 300);
	    f.setLayout(new GridLayout(1, 2));
	    
	    //線形分離可能
	    double[] linear1X = {0.15, 0.3, 0.35, 0.4, 0.55};
	    double[] linear1Y = {0.3,  0.6, 0.25, 0.5, 0.4};
	    double[] linear2X = {0.4,  0.7, 0.7, 0.85, 0.9};
	    double[] linear2Y = {0.85, 0.9, 0.8, 0.7,  0.6};
	    f.add(createGraph("線形分離可能", 
	            linear1X, linear1Y, linear2X, linear2Y));
	    //線形分離不可能
	    double[] nonlinear1X = {0.15, 0.45, 0.6, 0.3, 0.75, 0.9};
	    double[] nonlinear1Y = {0.5,  0.85, 0.75,  0.75, 0.7, 0.55};
	    double[] nonlinear2X = {0.2,  0.55, 0.4,  0.6, 0.8, 0.85};
	    double[] nonlinear2Y = {0.3,  0.6,  0.55, 0.4, 0.55, 0.2};
	    
	    f.add(createGraph("線形分離不可能",
	            nonlinear1X, nonlinear1Y, nonlinear2X, nonlinear2Y));
	    
	    // 線形分類不可能2
	    double[] nonlinear1X2 = {0.2, 0.15, 0.45, 0.6, 0.3, 0.75, 0.9, 0.85};
	    double[] nonlinear1Y2 = {0.3, 0.5,  0.85, 0.75,  0.75, 0.7, 0.55, 0.2};
	    double[] nonlinear2X2 = {0.55, 0.4,  0.6, 0.8};
	    double[] nonlinear2Y2 = {0.6,  0.55, 0.4, 0.55};
	    
	    f.add(createGraph("線形分離不可能2",
	            nonlinear1X2, nonlinear1Y2, nonlinear2X2, nonlinear2Y2));

	    
	    // 線形分類不可能3(3クラス分類)
	    double[] nonlinear1X3 = {0.2, 0.15, 0.45, 0.6, 0.3, 0.75, 0.9, 0.85};
	    double[] nonlinear1Y3 = {0.3, 0.5,  0.85, 0.75,  0.75, 0.7, 0.55, 0.2};
	    
	    double[] nonlinear2X3 = {0.55, 0.4,  0.6, 0.8};
	    double[] nonlinear2Y3 = {0.6,  0.55, 0.4, 0.55};
	    
	    double[] nonlinear3X3 = {0.1, 0.2, 0.3, 0.4};
	    double[] nonlinear3Y3 = {0.8, 0.85, 0.9, 0.9};
	    
	    f.add(createGraph("線形分離不可能3",
	            nonlinear1X3, nonlinear1Y3, nonlinear2X3, nonlinear2Y3, nonlinear3X3, nonlinear3Y3));
	   
	    f.setVisible(true);
	}
	
	/**
	 * グラフの作成(2クラス分類)
	 * 
	 * @param title
	 * @param linear1X
	 * @param linear1Y
	 * @param linear2X
	 * @param linear2Y
	 * @return
	 */
    JLabel createGraph(String title, double[] linear1X, double[] linear1Y, double[] linear2X, double[] linear2Y) {
        MachineLearning ml = createLearningMachine();
        if(ml == null){
        	return null;
        }
        
        //学習
        int[][] clz = new int[linear1X.length + linear2X.length][2];
        double[][] examples = new double[linear1X.length + linear2X.length][2];
        for(int i = 0; i < linear1X.length; ++i){
        	clz[i][0] = 1; // 出力層の第1ユニットが1
        	clz[i][1] = 0;
            examples[i] = new double[]{linear1X[i], linear1Y[i]};
        }
        for(int i = 0; i < linear2X.length; ++i){
           clz[i + linear1X.length][0] = 0;
           clz[i + linear1X.length][1] = 1; // 出力層の第2ユニットが1
           examples[i + linear1X.length] = new double[]{linear2X[i], linear2Y[i]};
        }        
        ml.train(linear1X.length + linear2X.length, clz, examples);

        Image img = new BufferedImage(200, 200, BufferedImage.TYPE_INT_RGB);
        Graphics g = img.getGraphics();
        g.setColor(Color.WHITE);
        g.fillRect(0, 0, 200, 200);

        //判定結果
        for (int x = 0; x < 180; x += 2) {
            for (int y = 0; y < 180; y += 2) {
                double[] outs = ml.evaluate(new double[]{x / 180., y / 180.});
                
        		double maxout = 0;
        		int maxoutIndex = 0;
        		for(int k = 0; k < outs.length; ++k){
        			if(outs[k] >= maxout){
        				maxout = outs[k];
        				maxoutIndex = k;
        			}
        		}
                
                g.setColor(maxoutIndex == 1 ? new Color(192, 192, 255) : new Color(255, 192, 192));
                g.fillRect(x + 10, y + 10, 5, 5);
            }
        }
        //学習パターン
        for (int i = 0; i < linear1X.length; ++i) {
            int x = (int) (linear1X[i] * 180) + 10;
            int y = (int) (linear1Y[i] * 180) + 10;
            g.setColor(Color.RED);
            g.fillOval(x - 3, y - 3, 7, 7);
        }
        for (int i = 0; i < linear2X.length; ++i) {
            int x = (int) (linear2X[i] * 180) + 10;
            int y = (int) (linear2Y[i] * 180) + 10;
            g.setColor(Color.BLUE);
            g.fillOval(x - 3, y - 3, 7, 7);
        }
        //ラベル作成
        JLabel l = new JLabel(title, new ImageIcon(img), JLabel.CENTER);
        l.setVerticalTextPosition(JLabel.BOTTOM);
        l.setHorizontalTextPosition(JLabel.CENTER);
        return l;
    }
	

    /**
     * グラフの作成(3クラス分類)
     * 
     * @param title
     * @param linear1X
     * @param linear1Y
     * @param linear2X
     * @param linear2Y
     * @param linear3X
     * @param linear3Y
     * @return
     */
    JLabel createGraph(String title, double[] linear1X, double[] linear1Y, double[] linear2X, double[] linear2Y, double[] linear3X, double[] linear3Y) {
        NeuralNetworkParameter nnParam = new NeuralNetworkParameter(2, 10, 3, 10000, .3d);
    	MachineLearning ml = createLearningMachine(nnParam);
        if(ml == null){
        	return null;
        }
        
        //学習
        int[][] clz = new int[linear1X.length + linear2X.length + linear3X.length][3];
        double[][] examples = new double[linear1X.length + linear2X.length + linear3X.length][2];
        for(int i = 0; i < linear1X.length; ++i){
        	clz[i][0] = 1; // 出力層の第1ユニットが1
        	clz[i][1] = 0;
        	clz[i][2] = 0;
            examples[i] = new double[]{linear1X[i], linear1Y[i]};
        }
        for(int i = 0; i < linear2X.length; ++i){
           clz[i + linear1X.length][0] = 0;
           clz[i + linear1X.length][1] = 1; // 出力層の第2ユニットが1
           clz[i + linear1X.length][2] = 0;
           examples[i + linear1X.length] = new double[]{linear2X[i], linear2Y[i]};
        }        
        for(int i = 0; i < linear3X.length; ++i){
            clz[i + linear1X.length + linear2X.length][0] = 0;
            clz[i + linear1X.length + linear2X.length][1] = 0;
            clz[i + linear1X.length + linear2X.length][2] = 1; // 出力層の第3ユニットが1
            examples[i + linear1X.length + linear2X.length] = new double[]{linear3X[i], linear3Y[i]};
         }        

        ml.train(linear1X.length + linear2X.length + linear3X.length, clz, examples);

        Image img = new BufferedImage(200, 200, BufferedImage.TYPE_INT_RGB);
        Graphics g = img.getGraphics();
        g.setColor(Color.WHITE);
        g.fillRect(0, 0, 200, 200);

        //判定結果
        for (int x = 0; x < 180; x += 2) {
            for (int y = 0; y < 180; y += 2) {
                double[] outs = ml.evaluate(new double[]{x / 180., y / 180.});
                
        		double maxout = 0;
        		int maxoutIndex = 0;
        		for(int k = 0; k < outs.length; ++k){
        			if(outs[k] >= maxout){
        				maxout = outs[k];
        				maxoutIndex = k;
        			}
        		}
                Color color = null;
                switch (maxoutIndex) {
				case 0:
					color = new Color(255, 192, 192);
					break;
				case 1:
					color = new Color(192, 192, 255);
					break;
				case 2:
					color = new Color(192, 255, 192);
					break;
				default:
					color = new Color(255, 255, 255);
					break;
				}
                g.setColor(color);
                g.fillRect(x + 10, y + 10, 5, 5);
            }
        }
        //学習パターン
        for (int i = 0; i < linear1X.length; ++i) {
            int x = (int) (linear1X[i] * 180) + 10;
            int y = (int) (linear1Y[i] * 180) + 10;
            g.setColor(Color.RED);
            g.fillOval(x - 3, y - 3, 7, 7);
        }
        for (int i = 0; i < linear2X.length; ++i) {
            int x = (int) (linear2X[i] * 180) + 10;
            int y = (int) (linear2Y[i] * 180) + 10;
            g.setColor(Color.BLUE);
            g.fillOval(x - 3, y - 3, 7, 7);
        }
        for (int i = 0; i < linear3X.length; ++i) {
            int x = (int) (linear3X[i] * 180) + 10;
            int y = (int) (linear3Y[i] * 180) + 10;
            g.setColor(Color.GREEN);
            g.fillOval(x - 3, y - 3, 7, 7);
        }
        //ラベル作成
        JLabel l = new JLabel(title, new ImageIcon(img), JLabel.CENTER);
        l.setVerticalTextPosition(JLabel.BOTTOM);
        l.setHorizontalTextPosition(JLabel.CENTER);
        return l;
    }
}
