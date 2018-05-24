package moa.clusterers;

import com.yahoo.labs.samoa.instances.Instance;
import moa.cluster.Clustering;
import moa.cluster.SphereCluster;
import moa.core.Measurement;

import java.util.ArrayList;

public class DummyClusterer extends AbstractClusterer {
    private ArrayList<Instance> allInstances = new ArrayList<>();

    @Override
    public void resetLearningImpl() {
        allInstances.clear();
    }

    @Override
    public void trainOnInstanceImpl(Instance inst) {
        allInstances.add(inst);
    }

    @Override
    protected Measurement[] getModelMeasurementsImpl() {
        return new Measurement[0];
    }

    @Override
    public void getModelDescription(StringBuilder out, int indent) {
    }

    @Override
    public boolean isRandomizable() {
        return false;
    }

    @Override
    public double[] getVotesForInstance(Instance inst) {
        return null;
    }

    @Override
    public Clustering getClusteringResult() {
        int numAttributes = -1;
        if(allInstances.size() > 0) {
            numAttributes = allInstances.get(0).numAttributes();
        }
        Clustering clustering = new Clustering();
        clustering.add(new SphereCluster(allInstances, numAttributes));
        return clustering;
    }
}
