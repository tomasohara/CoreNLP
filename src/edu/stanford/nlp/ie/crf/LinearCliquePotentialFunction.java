package edu.stanford.nlp.ie.crf;
import edu.stanford.nlp.util.logging.Redwood;
import java.util.Arrays;
import java.util.function.Supplier;

/**
 * @author Mengqiu Wang
 */
public class LinearCliquePotentialFunction implements CliquePotentialFunction {

  private final float[][] fWeights;
  private final double[][] dWeights;

  /** A logger for this class */
  private static final Redwood.RedwoodChannels log = Redwood.channels(LinearCliquePotentialFunction.class);
    
  LinearCliquePotentialFunction(float[][] weights) {
    log.debug((Supplier<String>) () -> "in LinearCliquePotentialFunction/1: " + Arrays.deepToString(weights));
    this.fWeights = weights;
    this.dWeights = null;
  }

  LinearCliquePotentialFunction(double[][] weights) {
    log.debug((Supplier<String>) () -> "in LinearCliquePotentialFunction/2: " + Arrays.deepToString(weights));
    this.fWeights = null;
    this.dWeights = weights;
  }

  @Override
  public double computeCliquePotential(int cliqueSize, int labelIndex,
                                       int[] cliqueFeatures, double[] featureVal, int posInSent) {
    log.debug((Supplier<String>) () -> "in computeCliquePotential: " + cliqueSize + "," + labelIndex +
	      "," + Arrays.toString(cliqueFeatures) + "," + Arrays.toString(featureVal) + "," + posInSent);
    double output = 0.0;
    for (int m = 0; m < cliqueFeatures.length; m++) {
      double dotProd = fWeights == null ? dWeights[cliqueFeatures[m]][labelIndex] : fWeights[cliqueFeatures[m]][labelIndex];
      if (featureVal != null) {
        dotProd *= featureVal[m];
      }
      output += dotProd;
    }
    log.debug("computeCliquePotential() => " + output);
    return output;
  }

}
