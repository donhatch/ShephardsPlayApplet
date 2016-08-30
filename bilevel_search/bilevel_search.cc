#include <assert.h>
#include <stdint.h>
#include <stdlib.h>

#include <iostream>
#include <sstream>
#include <vector>

static int64_t intPow(int64_t a, int b)
{
  int64_t answer = 1;
  for (int i = 0; i < b; ++i)
    answer *= a;
  return answer;
}

static void index_to_syndrome(int nVerts, int64_t iSyndrome,
                              std::vector<std::vector<int>> &syndrome)
{
  int64_t scratch = iSyndrome;
  for (int i1minusi0 = 0; i1minusi0 < nVerts; ++i1minusi0)
  {
    for (int i0 = 0; i0 < nVerts-i1minusi0; ++i0)
    {
      int i1 = i0 + i1minusi0;
      int nChoices = i0==i1 ? 2 : 3;
      syndrome[i0][i1] = scratch % nChoices;
      scratch /= nChoices;
    }
  }
  assert(scratch == (int64_t)0);
} // index_to_syndrome

static std::string syndrome2string(const std::vector<std::vector<int>> &syndrome)
{
  int nVerts = syndrome.size();
  std::stringstream ss;
  bool nonempty = false;
  for (int i1minusi0 = 0; i1minusi0 < nVerts; ++i1minusi0)
  {
    for (int i0 = 0; i0 < nVerts-i1minusi0; ++i0)
    {
      int i1 = i0 + i1minusi0;
      if (nonempty)
        ss << ' ';
      ss << "["<<i0<<","<<i1<<"]:"<<syndrome[i0][i1];
      nonempty = true;
    }
  }
  return ss.str();
}

static void set2(int answer[2], int a, int b)
{
  answer[0] = a;
  answer[1] = b;
}

static int infinity = 1000000; // effectively
static void syndrome2interval(const std::vector<std::vector<int>> &syndrome, int i0, int i1,
                              int answer[2])
{
  int syndromeEntry = syndrome[i0][i1];
  if (i0 == i1)
  {
    if (syndromeEntry == 0)
      set2(answer, -infinity, i0);
    else if (syndromeEntry == 1)
      set2(answer, i0, infinity);
    else
      assert(false);
  }
  else
  {
    if (syndromeEntry == 0)
      set2(answer, -infinity, i0);
    else if (syndromeEntry == 1)
      set2(answer, i0, i1);
    else if (syndromeEntry == 2)
      set2(answer, i1, infinity);
    else
      assert(false);
  }
} // syndrome2interval
                              

static bool calcIsPlausible(const std::vector<std::vector<int>> &syndrome)
{
  int verboseLevel = 0;
  int nVerts = syndrome.size();
  bool answer = true;
  for (int i0 = 0; i0 < nVerts; ++i0)
  for (int i1 = i0; i1 < nVerts; ++i1)
  {
    int i2 = i1+1;
    for (int i3 = i2; i3 < nVerts; ++i3)
    {
      // is [i0,i1],[i2,i3] a violation?
      if (verboseLevel >= 1) std::cout << "              checking ["<<i0<<","<<i1<<"]["<<i2<<","<<i3<<"]" << std::endl;
      int interval01[2], interval23[2], interval03[2];
      syndrome2interval(syndrome,i0,i1, interval01);
      syndrome2interval(syndrome,i2,i3, interval23);
      syndrome2interval(syndrome,i0,i3, interval03);
      int intervalUnion[2] = {
        std::min(interval01[0],interval23[0]),
        std::max(interval01[1],interval23[1]),
      };
      int intervalIntersection[2] = {
        std::max(interval03[0], intervalUnion[0]),
        std::min(interval03[1], intervalUnion[1]),
      };
      // intervalIntersection must have nonempty interior
      if (!(intervalIntersection[0] < intervalIntersection[1]))
      {
        if (verboseLevel >= 1) std::cout << "                      bad" << std::endl;
        answer = false;
        //assert(false); // coverage
        if (true) return answer;
      }
      else
      {
        if (verboseLevel >= 1) std::cout << "                      good" << std::endl;
      }
    }
  }
  return answer;
} // calcIsPlausible

static bool calcIsGood(const std::vector<std::vector<int>> &syndrome, int gap)
{
  int nVerts = syndrome.size();
  int nVertsBeforeGap = gap;
  int nVertsAfterGap = nVerts - nVertsBeforeGap;
  int iVertJustBeforeGap = gap-1; // ok if out of bounds
  int iVertJustAfterGap = gap; // ok if out of bounds
  for (int iVertBeforeGap = 0; iVertBeforeGap < nVertsBeforeGap; ++iVertBeforeGap)
  {
    int i0 = iVertBeforeGap;
    int i1 = iVertJustBeforeGap;
    int forbidden = 0;
    if (syndrome[i0][i1] == forbidden)
      return false;
  }
  for (int iVertAfterGap = 0; iVertAfterGap < nVertsAfterGap; ++iVertAfterGap)
  {
    int i0 = iVertJustAfterGap;
    int i1 = iVertAfterGap;
    int forbidden = (i0==i1 ? 1 : 2);
    if (syndrome[i0][i1] == forbidden)
      return false;
  }
  return true;
} // calcIsGood

static void doItInefficient(int minVerts, int maxVerts)
{
  int verboseLevel = 1;
  for (int nVerts = minVerts; nVerts <= maxVerts; ++nVerts)
  {
    int64_t nSyndromes = intPow(3, nVerts*(nVerts-1)/2) * intPow(2, nVerts);
    if (verboseLevel >= 1) std::cout << "      nVerts="<<nVerts<<" : "<<nSyndromes<<" syndromes" << std::endl;

    // scratch for loop. syndrome[<i][i] not used.
    std::vector<std::vector<int>> syndrome(nVerts);
    for (int i = 0; i < nVerts; ++i) syndrome[i].resize(nVerts, -1);

    int nPlausible = 0;
    int nImplausible = 0;
    int nPlausibleButImpossible = 0;
    int nPossible = 0;
    for (int64_t iSyndrome = 0; iSyndrome < nSyndromes; ++iSyndrome)
    {
        index_to_syndrome(nVerts, iSyndrome, syndrome);
        bool isPlausible = calcIsPlausible(syndrome);

        if (false) isPlausible = true; // to exercise alarm

        if (!isPlausible)
        {
          nImplausible++;
          if (verboseLevel >= 3) std::cout << "          nVerts="<<nVerts<<" syndrome "<<iSyndrome<<"/"<<nSyndromes<<": "<<syndrome2string(syndrome)<<" -> IMPLAUSIBLE" << std::endl;
          continue;
        }
        else
        {
          nPlausible++;
          if (verboseLevel >= 2) std::cout << "          nVerts="<<nVerts<<" syndrome "<<iSyndrome<<"/"<<nSyndromes<<": "<<syndrome2string(syndrome)<<" -> plausible";
          bool hasGoodGap = false;
          char possibleSyndrome[nVerts+2];
          for (int gap = 0; gap < nVerts+1; ++gap)
          {
            bool isGoodGap = calcIsGood(syndrome, gap);
            possibleSyndrome[gap] = isGoodGap ? '1' : '0';
            if (isGoodGap)
              hasGoodGap = true;
          }
          possibleSyndrome[nVerts+1] = '\0';
          if (verboseLevel >= 2 || !hasGoodGap) std::cout << "-> "<<possibleSyndrome<<(!hasGoodGap ? "!!!!!" : "") << std::endl;
          if (hasGoodGap)
            nPossible++;
          else
            nPlausibleButImpossible++;
        }
    }
    if (verboseLevel >= 1) std::cout << "          "<<nImplausible<<"/"<<nSyndromes<<" implausible ("<<(100.*nImplausible/nSyndromes)<<"%), "<<nPlausibleButImpossible<<" plausible but impossible, "<<nPossible<<" possible" << std::endl;
    assert(nPlausible == nPlausibleButImpossible + nPossible);
  }
} // doItInefficient

static void explore(const std::vector<std::vector<int>> &syndrome,
                    int *nImplausible,
                    int *nPlausible,
                    int *nPlausibleButImpossible,
                    int *nPossible)
{
} // explore

static void doItEfficient(int minVerts, int maxVerts)
{
  int verboseLevel = 1;
  for (int nVerts = minVerts; nVerts <= maxVerts; ++nVerts)
  {
    // scratch for loop. syndrome[<i][i] not used.
    std::vector<std::vector<int>> syndrome(nVerts);
    for (int i = 0; i < nVerts; ++i) syndrome[i].resize(nVerts, -1);

    int nImplausible = 0;
    int nPlausible = 0;
    int nPlausibleButImpossible = 0;
    int nPossible = 0;
    explore(
    if (verboseLevel >= 1) std::cout << "          "<<nImplausible<<"/"<<nSyndromes<<" implausible ("<<(100.*nImplausible/nSyndromes)<<"%), "<<nPlausibleButImpossible<<" plausible but impossible, "<<nPossible<<" possible" << std::endl;
    assert(nPlausible == nPlausibleButImpossible + nPossible);
  }
} // doItEfficient
             

int main(int argc, char **argv)
{
  int verboseLevel = 1;
  if (verboseLevel >= 1) std::cout << "in main" << std::endl;
  if (argc != 2 && argc != 3)
  {
      std::cerr << "Usage: bilevel_search [<minVerts>] <maxVerts>" << std::endl;
      return 1;
  }

  int minVerts = argc==2 ? 0 : atoi(argv[1]);
  int maxVerts = atoi(argv[argc-1]);

  doItInefficient(minVerts, maxVerts);

  if (verboseLevel >= 1) std::cout << "out main" << std::endl;
  return 0;
} // main
