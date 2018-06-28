package moa.classifiers.alcc;

import java.util.*;

//import javafx.util.Pair;
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
import com.github.javacliparser.FlagOption;
import com.github.javacliparser.FloatOption;
import com.github.javacliparser.IntOption;

public class ALCClassifier extends AbstractClassifier implements MultiClassClassifier {

    private class Pair<T1, T2> {
        private T1 first;
        private T2 last;

        Pair(T1 first, T2 last) {
            this.first = first;
            this.last = last;
        }

        public T1 getFirst() {
            return first;
        }

        public T2 getLast() {
            return last;
        }
    }

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

    public FlagOption computeDistancesBetweenPointsInClusterOption = new FlagOption(
            "computeDistancesBetweenPointsInCluster", 'd',
            "Compute distances between points in cluster");

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
        Clustering macroClustering = clusterer.getClusteringResult();
        Clustering microClustering;
        Clustering clustering = macroClustering;
        if(clusterer.implementsMicroClusterer()) {
            microClustering = clusterer.getMicroClusteringResult();
            if(macroClustering == null && microClustering != null) {
                Clustering gtPoints = new Clustering(chunk);
                macroClustering = moa.clusterers.KMeans.gaussianMeans(gtPoints, microClustering);
            }
            if(((AbstractClusterer)clusterer).evaluateMicroClusteringOption.isSet()) {
                clustering = microClustering;
            } else {
                clustering = macroClustering;
            }
        }
        // okay, if at this point clustering is still null, then something is wrong with used clusterer
        return clustering;
    }

    private Pair<ArrayList<ArrayList<Instance>>, ArrayList<Double>> fitPointsToClusterings(Clustering clustering) {
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
        ArrayList<Double> distances;
        if(computeDistancesBetweenPointsInClusterOption.isSet()) {
            distances = computeDistances(pointsFittingToClusters);
            distances = normalizeDistances(distances);
            distances.add(1.0); // points outside of clustering
        } else {
            distances = equalDistances(pointsFittingToClusters.size());
        }
        return new Pair<>(pointsFittingToClusters, distances);
    }

    private ArrayList<Double> computeDistances(ArrayList<ArrayList<Instance>> pointsFittingToClusters) {
        ArrayList<Double> distances = new ArrayList<>();
        // we don't care for last "cluster" - they're points outside of any clustering
        for(int k = 0; k < pointsFittingToClusters.size() - 1; ++k) {
            int connections = 0;
            double distance = 0;
            for(int i = 0; i < pointsFittingToClusters.get(k).size() - 1; ++i) {
                for(int j = i + 1; j < pointsFittingToClusters.get(k).size(); ++j) {
                    // distance between (i, j)
                    Instance instanceI = stripClassFromInstance(pointsFittingToClusters.get(k).get(i));
                    Instance instanceJ = stripClassFromInstance(pointsFittingToClusters.get(k).get(j));
                    double ijDistance = 0;
                    for(int m = 0; m < instanceI.numAttributes(); ++m) {
                        ijDistance += Math.pow(instanceI.value(m) - instanceJ.value(m), 2);
                    }
                    ijDistance = Math.sqrt(ijDistance);
                    distance += ijDistance;
                    connections++;
                }
            }
            distance /= connections;
            distances.add(distance);
        }
        return distances;
    }

    private ArrayList<Double> normalizeDistances(ArrayList<Double> distances) {
        ArrayList<Double> normalizedDistances = new ArrayList<>(distances);
        double maxDistance = Collections.max(distances);
        for(int i = 0; i < normalizedDistances.size(); ++i) {
            normalizedDistances.set(i, normalizedDistances.get(i) / maxDistance);
        }
        return normalizedDistances;
    }

    private ArrayList<Double> equalDistances(int n) {
        return new ArrayList<>(Collections.nCopies(n, 1.0));
    }

    private void trainFittedPointsWithRegardOfBudget(
            Pair<ArrayList<ArrayList<Instance>>, ArrayList<Double>> pointsFittingToClusters) {
        // samples have been fitted, so now for every cluster, we are training classifier number of samples,
        // according to budget
        int clusteringIndex = 0;
        for(int i = 0; i < pointsFittingToClusters.getFirst().size(); ++i) {
            ArrayList<Instance> samples = pointsFittingToClusters.getFirst().get(i);
            if(clusteringIndex == pointsFittingToClusters.getFirst().size() - 1 &&
                    !checkPointsOutsideOfClusteringsOption.isSet()) {
                break;
            }

            Collections.shuffle(samples);

            for(int j = 0; j < samples.size()
                    * budgetOption.getValue() * pointsFittingToClusters.getLast().get(i); ++j) {
                classifier.trainOnInstance(samples.get(j));
            }
            clusteringIndex++;
        }
    }

    @Override
    public void trainOnInstanceImpl(Instance inst) {
        chunk.add(inst);
        Instance instWithoutClass = stripClassFromInstance(inst);
        clusterer.trainOnInstance(instWithoutClass);

        if(chunk.size() >= chunkSizeOption.getValue()) {
            Clustering clustering = extractClusteringsFromClusterer();
            Pair<ArrayList<ArrayList<Instance>>, ArrayList<Double>> pointsFittingToClusters =
                    fitPointsToClusterings(clustering);
            trainFittedPointsWithRegardOfBudget(pointsFittingToClusters);
            chunk.clear();
        }
    }

    @Override
    public double[] getVotesForInstance(Instance inst) {
        return classifier.getVotesForInstance(inst);
    }

    @Override
    public boolean isRandomizable() {
        return true;
    }

    @Override
    public void getModelDescription(StringBuilder out, int indent) {
        ((AbstractClassifier)classifier).getModelDescription(out, indent);
    }

    @Override
    protected Measurement[] getModelMeasurementsImpl() {
        List<Measurement> measurementList = new LinkedList<>();
        for(MOAObject object: new MOAObject[]{classifier, clusterer}) {
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
