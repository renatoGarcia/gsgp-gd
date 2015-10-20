/*
 * To change this template, choose Tools | Templates
 * and open the template in the editor.
 */
package sgp.data;

import sgp.MersenneTwister;
import java.io.File;
import java.util.ArrayList;
import java.util.Iterator;

/**
 * HoldoutHandler.java
 * Copyright (C) 20014, Federal University of Minas Gerais, Belo Horizonte, Brazil
 */
public class HoldoutHandler implements DataProducer{
    /** Random number generator. */
    private MersenneTwister rnd;
    
    /** Defines if absolute quatities are used */
    private boolean useAbsoluteQuantity;
    
    /** Input dataset. Used when the algorithm samples the instances.*/
    private Dataset dataset;
    
    /** Input parted datasets. Used when sampled data are provided by user files.*/
    private ExperimentDataset[] datasetFromFiles;
        
    /** Percentage of data used for test. */
    private double testPercentage;
    
    /** Boolean to indicates if we should sample or read the folds from files **/
    private boolean useFiles = false;
    
    /** Current experiment. Used to sinalize the current train/test files to be used.*/
    private int currentExperiment;
    
    private int numInputs;
        
    /**
     * Constructor declaration.
     * @param dataset Input dataset
     * @param rnd Random number generator used to shuffle data. If it is null, data is not shuffled.
     * @param testPercentage Percentage used for test
     */
//    public HoldoutHandler(Dataset dataset, MersenneTwisterFast rnd, double testPercentage) {
//        this.rnd = rnd;
//        this.dataset = dataset;
//        this.testPercentage = testPercentage;
//    }

    /**
     * Constructor declaration.
     * @param testPercentage Percentage used for test
     */
    public HoldoutHandler(double testPercentage) {
        this.testPercentage = testPercentage;
        if(this.testPercentage > 1){
            useAbsoluteQuantity = true;
        }
        else{
            useAbsoluteQuantity = false;
        }
    }

    public HoldoutHandler() {
        useFiles = true;
        currentExperiment = 0;
    }
    
    /**
     * Gets a training/test set based on holdout validation. A percentage of data
     * is used for test and remaining for training.
     * @return Array with training and test data in first and second positions, respectvely
     */
    @Override
    public ExperimentDataset getExperimentDataset(){
        // If a list of files is provided, alternate the current dataset at each method call
        if(useFiles){
            ExperimentDataset data = datasetFromFiles[currentExperiment];
            if(currentExperiment < datasetFromFiles.length - 1) currentExperiment++;
            else currentExperiment = 0;
            return data;
        }
        
        ExperimentDataset data = new ExperimentDataset();
        ArrayList<Instance> dataCopy = dataset.softClone();
        
        
        // Set the size of the subsets
        int testSize = 0;
        if(!useAbsoluteQuantity)
            testSize = (int)Math.round(testPercentage * dataCopy.size());
        else
            testSize = (int)Math.round(testPercentage);
        int trainingSize = dataCopy.size() - testSize;
        
        // If rnd is null, we don't resample the data
        if(rnd == null){
            Iterator<Instance> it = dataCopy.iterator();
            for(int i = 0; i < trainingSize; i++){
                data.training.add(it.next());
                it.remove();
            }
            while(it.hasNext()){
                data.test.add(it.next());
                it.remove();
            }
        }
        else{
            for(int i = 0; i < testSize; i++){
                data.test.add(dataCopy.remove(rnd.nextInt(dataCopy.size())));
            }
            while(!dataCopy.isEmpty()){
                data.training.add(dataCopy.remove(rnd.nextInt(dataCopy.size())));
            }
        }
        return data;
    }

    @Override
    public void setDataset(String dataPath) throws Exception{
        if(!useFiles){
            Dataset data = DataReader.readInputDataFile(dataPath);
            this.dataset = data;
            numInputs = data.getInputNumber();
        }
        else{
            getFromFile(dataPath);
        }
    }

    @Override
    public void setRandomGenerator(MersenneTwister rnd) {
        this.rnd = rnd;
    }

    @Override
    public boolean isValid() {
        if(testPercentage >= 0)
            return true;
        return false;
    }

    /**
     * Get a list of partitions test/training from a file pattern for holdout
     * @param dataPath Full path to the file with the pattern prename#posname.
     * @return The number of inputs of the dataset
     * @throws Exception Error while reading the dataset within a file.
     * @throws SSRException Error in the file path/pattern.
     */
    private void getFromFile(String dataPath) throws Exception{
        int lastFileSeparator = dataPath.lastIndexOf(File.separator);
        String filePattern = dataPath.substring(lastFileSeparator + 1);
        String folderName = dataPath.substring(0, lastFileSeparator);
        String[] aux = filePattern.split("#");
        if(aux.length != 2)
            throw new Exception("The file pattern must have one and only one # symbol as fold index.");
        ArrayList<File> trainFiles = new ArrayList<File>();
        ArrayList<File> testFiles = new ArrayList<File>();
        int index = 0;
        File newTrain = new File(folderName + File.separator + aux[0] + "train-" + index + aux[1]);
        File newTest = new File(folderName + File.separator + aux[0] + "test-" + index + aux[1]);
        while(newTrain.isFile() && newTest.isFile()){
            trainFiles.add(newTrain);
            testFiles.add(newTest);
            index++;
            newTrain = new File(folderName + File.separator + aux[0] + "train-" + index + aux[1]);
            newTest = new File(folderName + File.separator + aux[0] + "test-" + index + aux[1]);
        }
        if(trainFiles.isEmpty() || testFiles.isEmpty()) 
            throw new Exception("No files found for this file pattern/path: \"" + newTrain.getAbsolutePath() + "\" and \"" + newTest.getAbsolutePath() +  "\"\nUsing HOLDOUT.\n");
        if(trainFiles.size() != testFiles.size())
            throw new Exception("The number of test and training files is different. Check if the names are correct.\n");
        
        datasetFromFiles = new ExperimentDataset[trainFiles.size()];
        for(int i = 0; i < datasetFromFiles.length; i++){
            datasetFromFiles[i] = new ExperimentDataset(DataReader.readInputDataFile(trainFiles.get(i)), DataReader.readInputDataFile(testFiles.get(i)));
//            datasetFromFiles[i].training = DataReader.readInputDataFile(trainFiles.get(i));
//            datasetFromFiles[i].test = DataReader.readInputDataFile(testFiles.get(i));
        }
        numInputs = datasetFromFiles[0].training.getInputNumber();
    }

    @Override
    public int getNumInputs() {
        return numInputs;
    }
}
