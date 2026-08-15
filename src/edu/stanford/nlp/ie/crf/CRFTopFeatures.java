package edu.stanford.nlp.ie.crf;

import edu.stanford.nlp.ling.CoreLabel;
import edu.stanford.nlp.util.Index;

import java.io.OutputStreamWriter;
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

/**
 * Prints the highest- and lowest-weighted node (state) and edge (transition)
 * features of a trained {@link CRFClassifier}, similar in spirit to
 * sklearn-crfsuite's {@code state_features_}/{@code transition_features_}
 * inspection utilities. Node features (clique size 1) score a single label;
 * edge features (clique size 2+) score a label tuple, e.g. "O -&gt; B-LOC".
 *
 * <p>Usage: {@code java edu.stanford.nlp.ie.crf.CRFTopFeatures <serializedClassifier> [topN]}
 *
 * <p>Change facilitated by Claude (Sonnet 5).
 *
 * @author Claude (Sonnet 5)
 */
public class CRFTopFeatures {

  private CRFTopFeatures() {} // static methods only

  private static final class Weighted {
    final double weight;
    final String label;
    final String feature;

    Weighted(double weight, String label, String feature) {
      this.weight = weight;
      this.label = label;
      this.feature = feature;
    }
  }

  /**
   * Prints, for each clique type (node, edge, ...) the classifier uses, the
   * {@code topN} most positive and most negative (feature, label) weights.
   */
  public static void printTopFeatures(CRFClassifier<?> crf, int topN, PrintWriter out) {
    Index<String> featureIndex = crf.featureIndex;
    Index<String> classIndex = crf.classIndex;
    float[][] weights = crf.weights;
    List<Index<CRFLabel>> labelIndices = crf.labelIndices;

    // crf.map is only populated during training; a freshly loadClassifier()-ed
    // model has map == null, so clique type is recovered from the feature
    // name's suffix convention instead (see CRFClassifier.getFeatureTypeIndex).
    for (int cliqueType = 0; cliqueType < labelIndices.size(); cliqueType++) {
      List<Weighted> entries = new ArrayList<>();
      for (int f = 0; f < featureIndex.size(); f++) {
        if (CRFClassifier.getFeatureTypeIndex(featureIndex.get(f)) != cliqueType) {
          continue;
        }
        String featureName = featureIndex.get(f);
        float[] w = weights[f];
        for (int l = 0; l < w.length; l++) {
          CRFLabel crfLabel = labelIndices.get(cliqueType).get(l);
          entries.add(new Weighted(w[l], labelString(crfLabel, classIndex), featureName));
        }
      }
      entries.sort((a, b) -> Double.compare(b.weight, a.weight));

      String kind = (cliqueType == 0) ? "node (state)"
          : "edge (transition), clique size " + (cliqueType + 1);
      out.println("=== " + kind + " features: top " + topN + " positive ===");
      printSlice(entries, topN, out);
      out.println();

      out.println("=== " + kind + " features: top " + topN + " negative ===");
      List<Weighted> reversed = new ArrayList<>(entries);
      Collections.reverse(reversed);
      printSlice(reversed, topN, out);
      out.println();
    }
    out.flush();
  }

  private static void printSlice(List<Weighted> entries, int n, PrintWriter out) {
    int to = Math.min(entries.size(), n);
    for (int i = 0; i < to; i++) {
      Weighted e = entries.get(i);
      out.printf("%10.6f  %-24s  %s%n", e.weight, e.label, e.feature);
    }
  }

  private static String labelString(CRFLabel label, Index<String> classIndex) {
    int[] indices = label.getLabel();
    StringBuilder sb = new StringBuilder();
    for (int i = 0; i < indices.length; i++) {
      if (i > 0) {
        sb.append(" -> ");
      }
      sb.append(classIndex.get(indices[i]));
    }
    return sb.toString();
  }

  public static void main(String[] args) throws Exception {
    if (args.length < 1) {
      System.err.println("Usage: java edu.stanford.nlp.ie.crf.CRFTopFeatures <serializedClassifier> [topN]");
      System.exit(1);
    }
    String path = args[0];
    int topN = (args.length > 1) ? Integer.parseInt(args[1]) : 30;
    CRFClassifier<CoreLabel> crf = CRFClassifier.getClassifier(path);
    PrintWriter out = new PrintWriter(new OutputStreamWriter(System.out), true);
    printTopFeatures(crf, topN, out);
  }

}
