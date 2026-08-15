package edu.stanford.nlp.ie.crf;

import edu.stanford.nlp.ling.CoreAnnotations;
import edu.stanford.nlp.ling.CoreLabel;
import edu.stanford.nlp.objectbank.ObjectBank;
import edu.stanford.nlp.sequences.DocumentReaderAndWriter;
import edu.stanford.nlp.stats.ClassicCounter;
import edu.stanford.nlp.stats.Counter;
import edu.stanford.nlp.util.ConfusionMatrix;

import java.io.OutputStreamWriter;
import java.io.PrintWriter;
import java.util.Collections;
import java.util.List;
import java.util.Set;
import java.util.TreeSet;

/**
 * Prints a per-tag precision/recall/F1 report for a trained {@link CRFClassifier}
 * against a gold test file -- e.g. separate rows for "B-LOC" and "I-LOC" -- as
 * opposed to the entity-level scoring {@code CRFClassifier}'s own {@code -testFile}
 * evaluation prints by default. Comparable to scikit-learn's
 * {@code flat_classification_report} over BIO tags.
 *
 * <p>The test file's gold labels must already use the same tag scheme the
 * classifier was trained with (e.g. both using {@code entitySubclassification=iob2},
 * or -- as with the CoNLL 2002 data -- BIO tags already baked into the file), since
 * this compares raw {@code AnswerAnnotation} strings without any recoding.
 *
 * <p>Usage: {@code java edu.stanford.nlp.ie.crf.CRFPerTagReport <serializedClassifier> <testFile> [omitLabels]}
 * {@code omitLabels} is an optional comma-separated list of labels to leave out of the
 * printed table (e.g. "O"), matching the common practice (see
 * {@code sklearn.metrics.flat_classification_report(labels=sorted_labels)} in
 * sklearn-crfsuite tutorials) of excluding the background label from the report.
 * Omitted labels still count toward the weighted-average "total" row's support.
 *
 * <p>Change facilitated by Claude (Sonnet 5).
 *
 * @author Claude (Sonnet 5)
 */
public class CRFPerTagReport {

  private CRFPerTagReport() {} // static methods only

  public static void printReport(CRFClassifier<CoreLabel> crf, String testFile, PrintWriter out)
      throws Exception {
    printReport(crf, testFile, Collections.emptySet(), out);
  }

  public static void printReport(CRFClassifier<CoreLabel> crf, String testFile,
                                  Set<String> omitLabels, PrintWriter out)
      throws Exception {
    // This is a read-only evaluation tool; don't let a debug flag left over in the
    // serialized model (e.g. -printFeatures from an earlier training run) make
    // classify() try to dump a feature file as a side effect.
    crf.flags.printFeatures = null;

    DocumentReaderAndWriter<CoreLabel> readerAndWriter = crf.defaultReaderAndWriter();
    ObjectBank<List<CoreLabel>> documents = crf.makeObjectBankFromFile(testFile, readerAndWriter);

    ConfusionMatrix<String> confusion = new ConfusionMatrix<>();
    Counter<String> support = new ClassicCounter<>();

    for (List<CoreLabel> doc : documents) {
      crf.classify(doc);
      for (CoreLabel token : doc) {
        String guess = token.get(CoreAnnotations.AnswerAnnotation.class);
        String gold = token.get(CoreAnnotations.GoldAnswerAnnotation.class);
        confusion.add(guess, gold);
        support.incrementCount(gold);
      }
    }

    out.printf("%-16s %10s %10s %10s %10s%n", "tag", "precision", "recall", "f1", "support");
    double weightedP = 0.0, weightedR = 0.0, weightedF1 = 0.0;
    double totalSupport = 0.0;
    for (String label : new TreeSet<>(confusion.uniqueLabels())) {
      double labelSupport = support.getCount(label);
      if (omitLabels.contains(label)) {
        continue;
      }
      ConfusionMatrix<String>.Contingency c = confusion.getContingency(label);
      out.printf("%-16s %10.3f %10.3f %10.3f %10.0f%n",
          label, c.precision(), c.recall(), c.f1(), labelSupport);
      weightedP += c.precision() * labelSupport;
      weightedR += c.recall() * labelSupport;
      weightedF1 += c.f1() * labelSupport;
      totalSupport += labelSupport;
    }
    if (totalSupport > 0) {
      out.printf("%-16s %10.3f %10.3f %10.3f %10.0f%n", "avg / total",
          weightedP / totalSupport, weightedR / totalSupport, weightedF1 / totalSupport, totalSupport);
    }
    out.flush();
  }

  public static void main(String[] args) throws Exception {
    if (args.length < 2) {
      System.err.println("Usage: java edu.stanford.nlp.ie.crf.CRFPerTagReport "
          + "<serializedClassifier> <testFile> [omitLabels]");
      System.err.println("  omitLabels: optional comma-separated labels to leave out, e.g. O");
      System.exit(1);
    }
    CRFClassifier<CoreLabel> crf = CRFClassifier.getClassifier(args[0]);
    Set<String> omitLabels = (args.length > 2)
        ? new TreeSet<>(java.util.Arrays.asList(args[2].split(",")))
        : Collections.emptySet();
    PrintWriter out = new PrintWriter(new OutputStreamWriter(System.out), true);
    printReport(crf, args[1], omitLabels, out);
  }

}
