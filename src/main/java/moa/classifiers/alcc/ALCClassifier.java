package moa.classifiers.alcc;

import java.util.*;

import com.github.javacliparser.FlagOption;
import moa.MOAObject;
import moa.classifiers.AbstractClassifier;
import moa.classifiers.Classifier;

import com.yahoo.labs.samoa.instances.Instance;
import moa.classifiers.MultiClassClassifier;
import moa.cluster.Cluster;
import moa.cluster.Clustering;
import moa.clusterers.AbstractClusterer;
import moa.clusterers.Clusterer;

import moa.core.Measurement;
import moa.options.ClassOption;
import com.github.javacliparser.FloatOption;
import com.github.javacliparser.IntOption;

public class ALCClassifier extends AbstractClassifier implements MultiClassClassifier {

    private static final long serialVersionUID = 1L;

    @Override
    public String getPurposeString() {
        return "Active learning classifier for evolving data streams";
    }

    public ClassOption baseLearnerOption = new ClassOption("baseLearner", 'l',
            "Classifier to train.", Classifier.class, "bayes.NaiveBayes");

    public ClassOption clustererOption = new ClassOption("clusterer", 'C',
            "Clusterer to perform clustering.", AbstractClusterer.class, "clustree.ClusTree");

    public FloatOption budgetOption = new FloatOption("budget",
            'b', "Budget to use.",
            0.3, 0.0, 1.0);

    public IntOption chunkSizeOption = new IntOption("chunkSize",
            'c', "Number of instances creating one chunk.",
            1000, 0, Integer.MAX_VALUE);

    public FlagOption checkPointsOutsideOfClusteringsOption = new FlagOption("checkPointsOutsideOfClusterings",
            'p', "Check points outside of clusterings");

    public Classifier classifier;

    public Clusterer clusterer;

    private List<Instance> chunk;

    @Override
    public void resetLearningImpl() {
        this.classifier = ((Classifier) getPreparedClassOption(this.baseLearnerOption)).copy();
        this.classifier.resetLearning();
        this.clusterer = ((Clusterer) getPreparedClassOption(this.clustererOption)).copy();
        this.clusterer.resetLearning();

        this.chunk = new ArrayList<>();
    }

    private Instance stripClassFromInstance(Instance instance) {
        Instance instanceWithoutClass = instance.copy();
        instanceWithoutClass.deleteAttributeAt(instance.classIndex());
        return instanceWithoutClass;
    }

    private Clustering extractClusteringsFromClusterer() {
        // now this gets tricky... we can extract clusterings, but not points which created them
        // so we need to fit all samples from chunk to clusters
        // micro / macro clustering based on code in moa.gui.visualization.RunVisualizer
        Clustering macroClustering = this.clusterer.getClusteringResult();
        Clustering microClustering;
        Clustering clustering = null;
        if(this.clusterer.implementsMicroClusterer()) {
            microClustering = this.clusterer.getMicroClusteringResult();
            if(macroClustering == null && microClustering != null) {
                Clustering gtPoints = new Clustering(this.chunk);
                macroClustering = moa.clusterers.KMeans.gaussianMeans(gtPoints, microClustering);
            }
            if(((AbstractClusterer)this.clusterer).evaluateMicroClusteringOption.isSet()) {
                clustering = microClustering;
            } else {
                clustering = macroClustering;
            }
        } else {
            clustering = macroClustering;
        }
        // okay, if at this point clustering is still null, then something is wrong with used clusterer
        return clustering;
    }

    private ArrayList<ArrayList<Instance>> fitPointsToClusterings(Clustering clustering) {
        ArrayList<ArrayList<Instance>> pointsFittingToClusters = new ArrayList<>();
        for(int i = 0; i < clustering.size() + 1; ++i) {
            pointsFittingToClusters.add(new ArrayList<>());
        }
        for(Instance sample: chunk) {
            Instance sampleWithoutClass = stripClassFromInstance(sample);
            boolean sampleAdded = false;
            for(int i = 0; i < clustering.size() - 1; ++i) {
                Cluster cluster = clustering.get(i);
                if(cluster.getInclusionProbability(sampleWithoutClass) > 0.8) {
                    pointsFittingToClusters.get(i).add(sample);
                    sampleAdded = true;
                    break;
                }
            }
            if(!sampleAdded) {
                pointsFittingToClusters.get(clustering.size() - 1).add(sample);
            }
        }
        return pointsFittingToClusters;
    }

    private void trainFittedPointsWithRegardOfBudget(ArrayList<ArrayList<Instance>> pointsFittingToClusters) {
        // samples have been fitted, so now for every cluster, we are training classifier number of samples,
        // according to budget
        int clusteringIndex = 0;
        for(ArrayList<Instance> samples: pointsFittingToClusters) {
            if(clusteringIndex == pointsFittingToClusters.size() - 1 &&
                    !checkPointsOutsideOfClusteringsOption.isSet()) {
                break;
            }
            Collections.shuffle(samples);
            for(int i = 0; i < this.budgetOption.getValue() * samples.size(); ++i) {
                this.classifier.trainOnInstance(samples.get(i));
            }
            clusteringIndex++;
        }
    }

    @Override
    public void trainOnInstanceImpl(Instance inst) {
        this.chunk.add(inst);
        Instance instWithoutClass = stripClassFromInstance(inst);
        this.clusterer.trainOnInstance(instWithoutClass);

        if(chunk.size() >= chunkSizeOption.getValue()) {
            Clustering clustering = extractClusteringsFromClusterer();
            ArrayList<ArrayList<Instance>> pointsFittingToClusters = fitPointsToClusterings(clustering);
            trainFittedPointsWithRegardOfBudget(pointsFittingToClusters);
            chunk.clear();
        }
    }

    @Override
    public double[] getVotesForInstance(Instance inst) {
        return this.classifier.getVotesForInstance(inst);
    }

    @Override
    public boolean isRandomizable() {
        return true;
    }

    @Override
    public void getModelDescription(StringBuilder out, int indent) {
        ((AbstractClassifier) this.classifier).getModelDescription(out, indent);
    }

    @Override
    protected Measurement[] getModelMeasurementsImpl() {
        List<Measurement> measurementList = new LinkedList<>();
        for(MOAObject object: new MOAObject[]{this.classifier, this.clusterer}) {
            Measurement[] modelMeasurements = null;
            if(object instanceof Classifier) {
                try {
                    modelMeasurements = ((Classifier) object).getModelMeasurements();
                } catch(UnsupportedOperationException e) {}
            } else if(object instanceof Clusterer) {
                try {
                    modelMeasurements = ((Clusterer) object).getModelMeasurements();
                } catch(UnsupportedOperationException e) {}
            }
            if (modelMeasurements != null) {
                Collections.addAll(measurementList, modelMeasurements);
            }
        }
        return measurementList.toArray(new Measurement[measurementList.size()]);
    }
}
