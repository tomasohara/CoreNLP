package edu.stanford.nlp.util;

import java.util.Arrays;
import java.util.HashSet;
import java.util.Set;

import org.junit.Test;

import static org.junit.Assert.*;

/**
 * @author Christopher Manning
 * @author John Bauer
 */
public class ArrayUtilsTest {

  private int[] sampleGaps = {1, 5, 6, 10, 17, 22, 29, 33, 100, 1000, 10000, 9999999};
  private int[] sampleBadGaps = {1, 6, 5, 10, 17};

  @Test
  public void testEqualContentsInt() {
    org.junit.Assert.assertTrue(ArrayUtils.equalContents(sampleGaps, sampleGaps));
    org.junit.Assert.assertTrue(ArrayUtils.equalContents(sampleBadGaps, sampleBadGaps));
    org.junit.Assert.assertFalse(ArrayUtils.equalContents(sampleGaps, sampleBadGaps));
  }

  @Test
  public void testGaps() {
    byte[] encoded = ArrayUtils.gapEncode(sampleGaps);
    int[] decoded = ArrayUtils.gapDecode(encoded);
    org.junit.Assert.assertTrue(ArrayUtils.equalContents(decoded, sampleGaps));

    try {
      ArrayUtils.gapEncode(sampleBadGaps);
      throw new RuntimeException("Expected an IllegalArgumentException");
    } catch(IllegalArgumentException e) {
      // yay, we passed
    }
  }

  @Test
  public void testDelta() {
    byte[] encoded = ArrayUtils.deltaEncode(sampleGaps);
    int[] decoded = ArrayUtils.deltaDecode(encoded);
    org.junit.Assert.assertTrue(ArrayUtils.equalContents(decoded, sampleGaps));

    try {
      ArrayUtils.deltaEncode(sampleBadGaps);
      throw new RuntimeException("Expected an IllegalArgumentException");
    } catch(IllegalArgumentException e) {
      // yay, we passed
    }
  }

  @Test
  public void testRemoveAt() {
    String[] strings = new String[]{"a", "b", "c"};
    strings = (String[]) ArrayUtils.removeAt(strings, 2);
    int i = 0;
    for (String string : strings) {
      if (i == 0) {
        assertEquals("a", string);
      } else if (i == 1) {
        assertEquals("b", string);
      } else {
        org.junit.Assert.fail("Array is too big!");
      }
      i++;
    }
  }

  @Test
  public void testAsSet() {
    String[] items = {"larry", "moe", "curly"};
    Set<String> set = new HashSet<>(Arrays.asList(items));
    assertEquals(set, ArrayUtils.asSet(items));
  }


  @Test
  public void testgetSubListIndex() {
    String[] t1 = {"this", "is", "test"};
    String[] t2 = {"well","this","is","not","this","is","test","also"};
    assertEquals(4,(ArrayUtils.getSubListIndex(t1, t2).get(0).intValue()));
    String[] t3 = {"cough","increased"};
    String[] t4 = {"i","dont","really","cough"};
    assertEquals(0, ArrayUtils.getSubListIndex(t3, t4).size());
    String[] t5 = {"cough","increased"};
    String[] t6 = {"cough","aggravated"};
    assertEquals(0, ArrayUtils.getSubListIndex(t5, t6).size());
    String[] t7 = {"cough","increased"};
    String[] t8 = {"cough","aggravated","cough","increased","and","cough", "increased","and","cough","and","increased"};
    assertEquals(2, ArrayUtils.getSubListIndex(t7, t8).get(0).intValue());
    assertEquals(5, ArrayUtils.getSubListIndex(t7, t8).get(1).intValue());
  }

  @Test
  public void testToString() {
      // Make sure the primitive overloads and the generic method all work.
      final String EXPECTED_INT_MATRIX_STR = "[[1, 2, 3],[4, 5, 6],[7, 8, 9]]";
      final String EXPECTED_DOUBLE_MATRIX_STR = "[[1.1, 2.1, 3.1],[4.1, 5.1, 6.1],[7.1, 8.1, 9.1]]";
      int[][] int_matrix = {{1, 2, 3}, {4, 5, 6}, {7, 8, 9}};
      final int size = 3;

      // Guard against a bad revision where int_matrix's actual dimensions
      // drift from size (e.g., someone edits the literal but not size, or
      // vice versa), which would silently invalidate the assertions below.
      int expected_int_bytes = size * size * Integer.BYTES;
      int actual_int_bytes = int_matrix.length * int_matrix[0].length * Integer.BYTES;
      assertEquals(expected_int_bytes, actual_int_bytes);

      // Separately confirm the row count itself is size: the byte check above
      // is a product, so it alone wouldn't catch a non-square drift (e.g., 1x9 vs 3x3).
      // Note that EXPECTED_INT_MATRIX_STR, etc. implicitly assumes this exact size.
      assertEquals(size, int_matrix.length);

      // Derive other matrices from the integer one
      double[][] double_matrix = new double[size][size];
      double[][] doubled_int_matrix = new double[size][size];
      Integer[][] boxed_int_matrix = new Integer[size][size];
      Double[][] boxed_double_matrix = new Double[size][size];
      for (int r = 0; r < size; r++) {
	  for (int c = 0; c < size; c++) {
	      doubled_int_matrix[r][c] = int_matrix[r][c];
	      double_matrix[r][c] = int_matrix[r][c] + 0.1;
	      boxed_int_matrix[r][c] = int_matrix[r][c];
	      boxed_double_matrix[r][c] = double_matrix[r][c];
	  }
      }

      // Check the primtive versions
      assertEquals(EXPECTED_INT_MATRIX_STR,
		   ArrayUtils.toString(int_matrix));
      assertNotEquals(EXPECTED_INT_MATRIX_STR,
		      ArrayUtils.toString(doubled_int_matrix));
      assertEquals(EXPECTED_DOUBLE_MATRIX_STR.replace(".1", ".0"),
		   ArrayUtils.toString(doubled_int_matrix));
      assertEquals(EXPECTED_DOUBLE_MATRIX_STR,
		   ArrayUtils.toString(double_matrix));

      // Exercise the generic <T> toString(T[][]) overload with reference-type arrays.
      assertEquals(EXPECTED_INT_MATRIX_STR,
		   ArrayUtils.toString(boxed_int_matrix));
      assertEquals(EXPECTED_DOUBLE_MATRIX_STR,
		   ArrayUtils.toString(boxed_double_matrix));
  }
}
