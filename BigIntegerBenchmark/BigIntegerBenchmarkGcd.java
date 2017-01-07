/*

===================
Q: why are java.math.BigInteger.gcd and modInverse so slow?

I'm trying to use java.math.BigInteger
for some exact integer matrix computations in which the scalar values get up to millions of digits.
I've noticed that some of the builtin BigInteger operations are unexpectedly very slow--
in particular some cases of gcd, and many more cases of modInverse.
Apparently I can implement my own versions of these functions that are much faster.

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

    // Make sure myGcd(a,b) gives the same answer as a.gcd(b) for small values.
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
            if (!myAnswer.equals(theirAnswer)) {
                throw new AssertionError("they say gcd("+a+","+b+") is "+theirAnswer+", I say it's "+myAnswer);
            }
        }
        System.err.println("passed.");
    }

    public static void main(String args[])
    {
        boolean useMine = false;
        if (args.length==1 && args[0].equals("theirs"))
            useMine = false;
        else if (args.length==1 && args[0].equals("mine"))
            useMine = true;
        else
        {
            System.err.println("Usage: BigIntegerBenchmarkGcd theirs|mine");
            System.exit(1);
        }

        myGcdConfidenceTest();

        System.out.println("# numDigits seconds");
        for (int numDigits = 1; numDigits <= (1<<20); numDigits *= 2)
        {
            java.math.BigInteger b = java.math.BigInteger.TEN.pow(numDigits);
            java.math.BigInteger a = b.subtract(java.math.BigInteger.valueOf(3));

            System.out.print(numDigits+" ");
            System.out.flush();

            long t0nanos = System.nanoTime();
            java.math.BigInteger aInverse = useMine ? myGcd(a, b)
                                                    : a.gcd(b);
            long t1nanos = System.nanoTime();

            double seconds = (t1nanos-t0nanos)/1e9;
            System.out.println(String.format("%.9f", seconds));
        }
    } // main
} // class BigIntegerBenchmarkGcd
