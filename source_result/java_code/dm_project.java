import java.io.BufferedReader;
import java.io.IOException;
import java.io.*;
import java.util.List;
import java.lang.String;
import java.nio.file.*;
import java.util.Arrays;
import java.nio.charset.Charset;
import java.util.Random;

public class dm_project {
    public static void main(String[] args) {
        int Uc = 943, Oc = 1682, Kc = 3;
        int total = Uc * Oc;
        double[][] RRM = new double[Uc][Oc];
        double[][] PM = new double[Uc][Kc];
        double[][] QM = new double[Kc][Oc];
        double[][] testRM = new double[Uc][Oc];
        double[][] trainRM = new double[Uc][Oc];

        //init RM[][], fullfill 0.
        for (int i = 0; i < Uc; i++) {
            for (int j = 0; j < Oc; j++) {
                RRM[i][j] = 0;
                testRM[i][j] = 0;
                trainRM[i][j] = 0;
            }
        }
        //init PM[][]
        for (int i = 0; i < Uc; i++) {
            for (int j = 0; j < Kc; j++) {
                PM[i][j] = Math.random() % 9; 
            }//need to be improved
        }
        //init QM[][]
        for (int i = 0; i < Kc; i++) {
            for (int j = 0; j < Oc; j++) {
                QM[i][j] = Math.random() % 9; 
            }//need to be improved
        }

        try{
            //read input file.
            String path = "train_all_txt.txt";
            File file = new File(path);  
            BufferedReader br = new BufferedReader(new FileReader(file));
            String[] strBuffer = new String[3];
            int user = 0;
            int item = 0;
            int rate = 0;
            String combineLine = null;
            //readLine，once per line, input the rate in RM[][]
            while((combineLine = br.readLine())!=null){
                strBuffer[0] = null;
                strBuffer[1] = null;
                strBuffer[2] = null;
                strBuffer = combineLine.split(" ");
                user = Integer.parseInt(strBuffer[0]);
                item = Integer.parseInt(strBuffer[1]);
                rate = Integer.parseInt(strBuffer[2]);
                RRM[user-1][item-1] = rate;
                trainRM[user-1][item-1] = rate;

            }

            //valid value in input file
            int countNonZero = 0;
            for (int i = 0; i < Uc; i++) {
                for (int j = 0; j < Oc; j++) {
                    if (RRM[i][j] > 0)
                        countNonZero++;
                }
            }   

            //set capacity
            int testData = countNonZero * 2/10;  // 20% test
            int trainData = countNonZero * 8/10; // 80% training

            //ramdon pick value to create test/training set.
            int loopCounter = 0;
            int ri = 0, rj = 0;
            Random rand = new Random();
            while(loopCounter < testData){
                ri = rand.nextInt(Uc);
                rj = rand.nextInt(Oc);
                if(trainRM[ri][rj] != 0 && testRM[ri][rj] == 0){
                    testRM[ri][rj] = trainRM[ri][rj];
                    trainRM[ri][rj] = 0;
                    loopCounter++;
                }   
            }

            System.out.println("train set >> "+ trainData);
            System.out.println("test set >> "+ testData);
            System.out.println("K >> "+ Kc);


            // Run 5000 steps，Alpha 0.002，Beta 0.04
            NNmf nmf = new NNmf(trainRM, PM, QM, Kc, Uc, Oc, 5000, 0.002, 0.04);
            nmf.run();
            double temp = 0;
            for (int i = 0; i < Uc; i++) {
                for (int j = 0; j < Oc; j++) {
                    temp = 0;
                    for (int k = 0; k < Kc; k++) {
                        temp += PM[i][k] * QM[k][j];
                    }
                    trainRM[i][j] = temp;
                }
            }

            //RMSE part
            double errorSum = 0;
            double RMSE = 0;
            for (int i = 0; i < Uc; i++) {
                for (int j = 0; j < Oc; j++) {
                    if(testRM[i][j] != 0){
                        errorSum = errorSum + Math.pow(testRM[i][j] - trainRM[i][j], 2);
                    }
                }
            }
            RMSE = Math.sqrt(errorSum/testData);
            System.out.println("RMSE >> "+ RMSE);

            //trainRM: merge(trainRM, RRM)
            for(int i = 0; i<Uc; i++){
                for(int j = 0; j<Oc; j++){
                    if(RRM[i][j] != 0){
                        trainRM[i][j] = RRM[i][j];
                    }
                }
            }

            for(int i = 0; i<Uc; i++){
                for(int j = 0; j<Oc; j++){
                    if(trainRM[i][j] < 1){
                        trainRM[i][j] = 1;
                    }else if(trainRM[i][j] > 5){
                        trainRM[i][j] = 5;
                    }
                }
            }


            
            //output to txt file
            String[] totalOutput = new String[total];
            int n = 0;
            int realUser = 0;
            int realItem = 0;
            for(int i = 0; i<Uc; i++){
                for(int j = 0; j<Oc; j++){
                    realUser = i + 1;
                    realItem = j + 1;
                    totalOutput[n] = realUser + " " + realItem + " " + trainRM[i][j];
                    n++;
                } 
            }
            List <String> lines = Arrays.asList(totalOutput);   
            Path filePath = Paths.get("output.txt");
            Files.write(filePath, lines, Charset.forName("UTF-8"));
            br.close();
        }catch(Exception e){
            e.printStackTrace();
        } 
    }
}





class NNmf {
    public double[][] RM, PM, QM;
    public int Kc, Uc, Oc;
    public int steps;
    public double Alpha, Beta;

    public void run() {
        double loss = 0;

        for (int s = 0; s < steps; s++) {
            for (int i = 0; i < Uc; i++) {
                for (int j = 0; j < Oc; j++) {
                    if (RM[i][j] > 0) {
                        //eij
                        double e = 0, pq = 0;
                        for (int k = 0; k < Kc; k++) {
                            pq += PM[i][k] * QM[k][j];
                        }
                        e = RM[i][j] - pq;
                        // Update Pik, Qkj
                        for (int k = 0; k < Kc; k++) {
                            PM[i][k] += Alpha
                                    * (2 * e * QM[k][j] - Beta * PM[i][k]);
                            PM[i][k] = PM[i][k] > 0 ? PM[i][k] : 0;
                            QM[k][j] += Alpha
                                    * (2 * e * PM[i][k] - Beta * QM[k][j]);
                            QM[k][j] = QM[k][j] > 0 ? QM[k][j] : 0;
                        }
                    }
                }
            }

            
            //loss function, training cost etc..
            loss = 0;
            for (int i = 0; i < Uc; i++) {
                for (int j = 0; j < Oc; j++) {
                    if (RM[i][j] > 0) {
                        //eij^2
                        double e2 = 0, pq = 0;
                        for (int k = 0; k < Kc; k++) {
                            pq += PM[i][k] * QM[k][j];
                        }
                        e2 = Math.pow(RM[i][j] - pq, 2);
                        for (int k = 0; k < Kc; k++) {
                            e2 += Beta / 2 * (Math.pow(PM[i][k], 2) + Math.pow(QM[k][j], 2));
                        }
                        loss += e2;
                    }
                }
            }  
        }
        System.out.println("loss > " + loss);  
    }

    public NNmf(double[][] RM, double[][] PM, double[][] QM, int Kc, int Uc,
            int Oc, int steps, double Alpha, double Beta) {
        this.RM = RM;
        this.PM = PM;
        this.QM = QM;
        this.Kc = Kc;
        this.Uc = Uc;
        this.Oc = Oc;
        this.steps = steps;
        this.Alpha = Alpha;
        this.Beta = Beta;
    }
}