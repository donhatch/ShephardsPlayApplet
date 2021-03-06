/*

===================
Q: why are java.math.BigInteger.gcd and modInverse so slow?

I'm trying to use java.math.BigInteger
for some exact integer matrix computations in which the scalar values get up to millions of digits.
I've noticed that some of the builtin BigInteger operations are unexpectedly very slow--
particularly some cases of gcd, and many more cases of modInverse.
It seems I can implement my own versions of these functions that are much faster.

Below is a program that prints timings for calculating
gcd(10^n-3, 10^n) for increasing values of n up to a million or so,
using either the builtin gcd or my own simple alternative implementation.

I ran it using java 8 under ubuntu linux,
runtime version 1.8.0_111-8u111-b14-2ubuntu0.16.04.2-b14.
Timings are roughly similar, relatively, on a macbook with java runtime 1.8.0_92.

Builtin gcd is roughly quadratic:

    # numDigits seconds
    1 0.000005626
    2 0.000008172
    4 0.000002852
    8 0.000003097
    16 0.000019158
    32 0.000026365
    64 0.000058330
    128 0.000488692
    256 0.000148674
    512 0.007579581
    1024 0.001199623
    2048 0.001296036
    4096 0.021341193
    8192 0.024193484
    16384 0.093183709
    32768 0.233919912
    65536 1.165671857
    131072 4.169629967
    262144 16.280159394
    524288 67.685927438
    1048576 259.500887989

Mine is roughly linear:

    # numDigits seconds
    1 0.000002845
    2 0.000002667
    4 0.000001644
    8 0.000001743
    16 0.000032751
    32 0.000008616
    64 0.000014859
    128 0.000009440
    256 0.000011083
    512 0.000014031
    1024 0.000021142
    2048 0.000036936
    4096 0.000071258
    8192 0.000145553
    16384 0.000243337
    32768 0.000475620
    65536 0.000956935
    131072 0.002290251
    262144 0.003492482
    524288 0.009635206
    1048576 0.022034768

Notice that, for a million digits, the builtin gcd takes more than 10000
times as long as mine: 259 seconds vs. .0220 seconds.

Is the builtin gcd function doing something other than the euclidean algorithm?  Why?

I get similar timings for the builtin modInverse vs. my own implementation
using the extended euclidean algorithm (not shown here).
The builtin modInverse does poorly in even more cases than the builtin gcd does,
e.g. when a is a small number like 2,3,4,... and b is large.

Here are three plots of the above data (two different linear scales and then log scale):
[OUT0.gcd.png]
[OUT1.gcd.png]
[OUT2.gcd.png]

Here's the program listing:

*/

/*
  Benchmark builtin java.math.BigInteger.gcd vs. a simple alternative implementation.
  To run:
    javac BigIntegerBenchmarkGcd.java
    java BigIntegerBenchmarkGcd mine > OUT.gcd.mine
    java BigIntegerBenchmarkGcd theirs > OUT.gcd.theirs

    gnuplot
      set title "Timing gcd(a=10^n-3, b=10^n)"
      set ylabel "Seconds"
      set xlabel "Number of digits"
      unset log
      set yrange [0:.5]
      #set terminal png size 512,384 enhanced font "Helvetica,10"
      #set output 'OUT0.gcd.png'
      plot [1:2**20] "OUT.gcd.theirs" with linespoints title "a.gcd(b)", "OUT.gcd.mine" with linespoints title "myGcd(a,b)"
      #set output 'OUT1.gcd.png'
      unset yrange; replot
      #set output 'OUT2.gcd.png'
      set log; replot
*/
class BigIntegerBenchmarkGcd
{
    // Simple alternative implementation of gcd.
    // More than 10000 times faster than the builtin gcd for a=10^1000000-3, b=10^1000000.
    private static java.math.BigInteger myGcd(java.math.BigInteger a, java.math.BigInteger b)
    {
        a = a.abs();
        b = b.abs();
        while (true)
        {
            if (b.signum() == 0) return a;
            a = a.mod(b);
            if (a.signum() == 0) return b;
            b = b.mod(a);
        }
    } // myGcd

    private static java.math.BigInteger myGcd2(java.math.BigInteger a, java.math.BigInteger b)
    {
        a = a.abs();
        b = b.abs();
        if (a.compareTo(b) < 0)
        {
            // swap
            java.math.BigInteger temp = a; a = b; b = temp;
        }
        while (b.signum() != 0)
        {
            // avoid mod if we can do it in just a handful of subtractions.
            // hmm, seems to speed up worst case by a factor of 2.  hooray!  still not as good as builtin on worst case though.
            if (a.bitLength()-b.bitLength() < 3) // 3 seems to be the sweet spot for this
            {
                while (a.compareTo(b) >= 0)
                {
                    if (a.bitLength()-b.bitLength() > 2) // it's more than four times as big
                    {
                        a = a.subtract(b.shiftLeft(2));
                    }
                    else if (a.bitLength()-b.bitLength() > 1) // it's more than twice as big
                    {
                        a = a.subtract(b.shiftLeft(1));
                    }
                    else
                        a = a.subtract(b);
                }
            }
            else
            {
                a = a.mod(b);
            }
            // swap
            assert(a.signum() >= 0);
            assert(a.compareTo(b) < 0);
            java.math.BigInteger temp = a; a = b; b = temp;
        }
        return a;
    } // myGcd2

    private static java.math.BigInteger oddPart(java.math.BigInteger a)
    {
        int lowestSetBit = a.getLowestSetBit();
        return lowestSetBit >= 1 ? a.shiftRight(lowestSetBit) : a;
    }

    private static java.math.BigInteger myGcd3(java.math.BigInteger a, java.math.BigInteger b)
    {
        a = a.abs();
        b = b.abs();
        if (a.compareTo(b) < 0)
        {
            // swap
            java.math.BigInteger temp = a; a = b; b = temp;
        }

        //System.out.println("===========");
        //System.out.println("a = "+a);
        //System.out.println("b = "+b);
        int aLowestSetBit = a.getLowestSetBit();
        int bLowestSetBit = b.getLowestSetBit();
        int initialShift = aLowestSetBit <= bLowestSetBit ? aLowestSetBit : bLowestSetBit; // min
        if (initialShift >= 0) // i.e. if neither a nor b is 0
        {
            // downshift a and b til both odd. can be by different amounts.
            if (aLowestSetBit > 0) a = a.shiftRight(aLowestSetBit);
            if (bLowestSetBit > 0) b = b.shiftRight(bLowestSetBit);
        }
        //System.out.println("a = "+a);
        //System.out.println("b = "+b);

        while (b.signum() != 0)
        {
            // avoid mod if we can do it in just a handful of subtractions.
            // hmm, seems to speed up worst case by a factor of 2.  hooray!  still not as good as builtin on worst case though.
            if (false) // definitely no good, even in random case. would be prohibitively bad in "best" case which becomes worst
            {
                while (a.compareTo(b) >= 0)
                    a = a.subtract(b);
            }
            else if (a.bitLength()-b.bitLength() < 3) // 3 seems to be the sweet spot for this
            {
                while (a.compareTo(b) >= 0)
                {
                    if (a.bitLength()-b.bitLength() > 2) // it's more than four times as big
                    {
                        a = a.subtract(b.shiftLeft(2));
                    }
                    else if (a.bitLength()-b.bitLength() > 1) // it's more than twice as big
                    {
                        a = a.subtract(b.shiftLeft(1));
                    }
                    else
                        a = a.subtract(b);
                }
            }
            else
            {
                a = a.mod(b);
            }
            assert(a.signum() >= 0);
            assert(a.compareTo(b) < 0);

            // If ever even numbers come up, can immediately kill them.
            // HOWEVER, should we perhaps detect the case when we can get more mileage by
            // *not* shifting here?  That's when b-a became relatively small.
            // (Maybe also when b-2*a became small, etc?
            // or even when 2*a-b became small?  Argh!)

            // Fact that may or may not be of use:
            // if we could see the actual highest order bits,
            // we could test very quickly which of a or b-a is (probably) better,
            // and we generally don't need an exact test.

            if (a.signum() > 0)
            {
                if (false)
                {
                    // Bleah! This didn't help.  BUT... I wonder if it would be more practical
                    // if we could do the tests using the high bits and short-circuiting early
                    // as suggested above.  Just walk along the 3 candidates
                    // until we get a definitive answer, or just stop when we've gone
                    // far enough so that it's close enough to a tie that it doesn't matter much.
                    // Although.. it seems like it's always worthwhile to keep going
                    // if it's close to a tie, because *if* it's close to a tie
                    // then every bit examined shaves off another bit in the computation, right?

                    // So in general... shift around until we get kind of a best match
                    // between a and b,
                    // and see if (odd part of) the difference is better than (odd part of) a.
                    // So let's see, there are 3 candidates:
                    //    (1) odd part of a
                    //    (2) odd part of b-(a<< til just < b)
                    //    (3) odd part of b-(a<< til just > b)
                    int aBitLength = a.bitLength();
                    int bBitLength = b.bitLength();
                    assert(aBitLength <= bBitLength);
                    java.math.BigInteger aShiftedToSameSizeAsB = (aBitLength==bBitLength ? a : a.shiftLeft(bBitLength-aBitLength));
                    java.math.BigInteger aShiftedToJustBelowB, aShiftedToJustAboveB;
                    if (aShiftedToSameSizeAsB.compareTo(b) <= 0)
                    {
                        aShiftedToJustBelowB = aShiftedToSameSizeAsB;
                        aShiftedToJustAboveB = aShiftedToJustBelowB.shiftLeft(1);
                    }
                    else
                    {
                        aShiftedToJustAboveB = aShiftedToSameSizeAsB;
                        aShiftedToJustBelowB = aShiftedToJustAboveB.shiftRight(1);
                    }

                    java.math.BigInteger firstCandidate = oddPart(a);
                    java.math.BigInteger secondCandidate = oddPart(b.subtract(aShiftedToJustBelowB));
                    java.math.BigInteger thirdCandidate = oddPart(aShiftedToJustAboveB.subtract(b));
                    a = firstCandidate;
                    if (secondCandidate.compareTo(a) < 0)
                        a = secondCandidate;
                    if (thirdCandidate.compareTo(a) < 0)
                        a = thirdCandidate;
                }
                else
                {
                    // nothing here panned out much, except just going to odd part of a seemed like a good idea

                    java.math.BigInteger aSaved = a;
                    if (false) // fooey, seems like a loss
                    {
                        if (a.shiftLeft(1).compareTo(b) >= 0)
                            a = b.subtract(a);
                    }
                    a = oddPart(a);
                    if (false) // fooey, seems like a loss
                    {
                        java.math.BigInteger aOtherPossibility = b.subtract(aSaved);
                        if (aOtherPossibility.compareTo(a) < 0)
                            a = aOtherPossibility;
                    }
                }
            }

            // swap
            java.math.BigInteger temp = a; a = b; b = temp;
        }
        //System.out.println("------");
        //System.out.println("a = "+a);
        if (initialShift > 0)
        {
            a = a.shiftLeft(initialShift);
        }
        //System.out.println("a = "+a);
        //System.out.println("===========");
        return a;
    } // myGcd3

    // IDEA: for randomized case, almost never need to look very far
    // in from the highest bits to see what's going to happen.
    // Can we build up a sequence of relatively small operations
    // and apply them all at once, instead of operating on a,b directly each time?
    // Maybe as long as the coeff sizes fit in a "word" (whatever that is?)
    // since multiplying by a word is probably faster than
    // general multiplying.


    // Make sure myGcd(a,b) and variations give the same answer as a.gcd(b) for small values.
    private static void myGcdConfidenceTest()
    {
        System.err.print("Running confidence test... ");
        System.err.flush();
        for (int i = -10; i < 10; ++i)
        for (int j = -10; j < 10; ++j)
        {
            java.math.BigInteger a = java.math.BigInteger.valueOf(i);
            java.math.BigInteger b = java.math.BigInteger.valueOf(j);
            java.math.BigInteger theirAnswer = a.gcd(b);
            java.math.BigInteger myAnswer = myGcd(a, b);
            java.math.BigInteger myAnswer2 = myGcd2(a, b);
            java.math.BigInteger myAnswer3 = myGcd3(a, b);
            if (!myAnswer.equals(theirAnswer)) {
                throw new AssertionError("they say gcd("+a+","+b+") is "+theirAnswer+", I say it's "+myAnswer);
            }
            if (!myAnswer2.equals(theirAnswer)) {
                throw new AssertionError("they say gcd("+a+","+b+") is "+theirAnswer+", I2 say it's "+myAnswer2);
            }
            if (!myAnswer3.equals(theirAnswer)) {
                throw new AssertionError("they say gcd("+a+","+b+") is "+theirAnswer+", I3 say it's "+myAnswer3);
            }
        }
        System.err.println("passed.");
    }

    public static void main(String args[])
    {
        boolean useMine = false;
        boolean useMine2 = false;
        boolean useMine3 = false;
        if (args.length==1 && args[0].equals("theirs"))
            ;
        else if (args.length==1 && args[0].equals("mine"))
            useMine = true;
        else if (args.length==1 && args[0].equals("mine2"))
            useMine2 = true;
        else if (args.length==1 && args[0].equals("mine3"))
            useMine3 = true;
        else
        {
            System.err.println("Usage: BigIntegerBenchmarkGcd theirs|mine");
            System.exit(1);
        }

        myGcdConfidenceTest();

        java.util.Random rng = new java.util.Random(0);

        System.out.println("# numDigits seconds");
        for (int numDigits = 1; numDigits <= (1<<20); numDigits *= 2)
        {
            java.math.BigInteger a, b;
            if (false)
            {
                // Euclidean: best case scenario.
                // Binary: atrocious (worst case?)
                b = java.math.BigInteger.TEN.pow(numDigits);
                a = b.subtract(java.math.BigInteger.valueOf(3));
            }
            else
            {
                // Euclidean: worst case scenario.
                // Binary: actually quite a bit better than the above for some reason.
                a = new java.math.BigInteger(numDigits, rng);
                b = new java.math.BigInteger(numDigits, rng);
            }

            System.out.print(numDigits+" ");
            System.out.flush();

            long t0nanos = System.nanoTime();
            java.math.BigInteger aInverse = useMine2 ? myGcd2(a, b)
                                          : useMine3 ? myGcd3(a, b)
                                          : useMine  ? myGcd(a, b)
                                                     : a.gcd(b);
            long t1nanos = System.nanoTime();

            double seconds = (t1nanos-t0nanos)/1e9;
            System.out.println(String.format("%.9f", seconds));
        }
    } // main
} // class BigIntegerBenchmarkGcd
