#include <omp.h>
#include <cmath>
#include <climits>
#include <cstdlib>
#include <unistd.h>
#include <iostream>
#include <sys/mman.h>
#include <sys/time.h>

using namespace std;

#define USE_MMAP

int const LOOP_TIMES = 100000;
int const ARRAY_SIZE = 61 * 1024;
double const Pi = 3.1415926535897932384;
double const RISKFREE_RATE =0.05;
double const VOLATILITY = 0.5;
double const STOCKPRICE_MIN = 2;
double const STOCKPRICE_MAX = 20;
double const STRIKEPRICE_MIN = 1;
double const STRIKEPRICE_MAX = 100;
double const MATURITYYEARS_MIN = 0.2;
double const MATURITYYEARS_MAX = 10; 

double rand(double lower, double upper) {
  double ratio = rand() / (double)RAND_MAX;
  return ratio * (upper - lower) + lower; 
}

double dtime() {
  double tseconds = 0.0;
  struct timeval mytime;
  gettimeofday(&mytime, (struct timezone*)0);
  tseconds = (double)(mytime.tv_sec + mytime.tv_usec * 1.0e-6);
  return tseconds; 
}

/* double N(double x) {
  double const a1 = 0.31938153;
  double const a2 = -0.356563782;
  double const a3 = 1.781477937;
  double const a4 = -1.821255978;
  double const a5 = 1.330274429;
  double const gamma = 0.2316419; 

  if (x < 0)
    return 1 - N(-x); 

  double k = 1 / (1 + gamma * fabs(x));
  
  return 1 - 1 / sqrt(2 * Pi) * exp(-pow(x, 2.0) / 2)
    * (a1 * k + a2 * pow(k, 2.0) + a3 * pow(k, 3.0) + a4 * pow(k, 4.0) + a5 * pow(k, 5.0));
 
}
*/ 

inline double N(double x) {
  return 1 / 2.0 + 1 / 2.0 * erf(1 / sqrt(2) * x); 
}

#define d1 (log(S / X) + (r + v * v / 2) * T)/(v * sqrt(T))
#define d2 (d1 - v * sqrt(T))


/*
double BlackScholes_CallPrice(double &callPrice, double S, double X, double T, double r, double v) {
  callPrice =  S * N(d1) - X * exp(-r * T) * N(d2);
}

double BlackScholes_GetPrice(double &getPrice, double callPrice, double S, double X, double T, double r, double v) {
  getPrice = X * exp(-r * T) * N(-d2) - S * N(-d1);
}
*/

double BlackScholes(double &callPrice, double &getPrice, 
		    double S, double X, double T, double r, double v) {
  double Nd1 = N(d1); 
  double Nd2 = N(d2); 
  double MNd1 = 1 - Nd1;
  double MNd2 = 1 - Nd2; 
  double mul = X * exp(-r * T);
  callPrice = S * Nd1 - mul * Nd2;
  getPrice = mul * MNd2 - S * MNd1; 
}

#undef d1
#undef d2

int main() {

  double tstart, tstop, ttime;

  ios :: sync_with_stdio(false); 
  
  cout << "Initializing" << endl;

#ifdef USE_MALLOC
  double *callPrice = (double *)malloc(ARRAY_SIZE * sizeof(double));
  double *getPrice = (double *)malloc(ARRAY_SIZE * sizeof(double));
  double *stockPrice = (double *)malloc(ARRAY_SIZE * sizeof(double));
  double *strikePrice = (double *)malloc(ARRAY_SIZE * sizeof(double));
  double *maturityYears = (double *)malloc(ARRAY_SIZE * sizeof(double));
#endif

#ifdef USE_MM_MALLOC
  double *callPrice = (double *)_mm_malloc(ARRAY_SIZE * sizeof(double), 64);
  double *getPrice = (double *)_mm_malloc(ARRAY_SIZE * sizeof(double), 64);
  double *stockPrice = (double *)_mm_malloc(ARRAY_SIZE * sizeof(double), 64);
  double *strikePrice = (double *)_mm_malloc(ARRAY_SIZE * sizeof(double), 64);
  double *maturityYears = (double *)_mm_malloc(ARRAY_SIZE * sizeof(double), 64);
#endif

#ifdef USE_MMAP
  double *callPrice = (double *)mmap(0,
                                     ARRAY_SIZE * sizeof(double),
                                     PROT_READ | PROT_WRITE,
                                     MAP_ANONYMOUS | MAP_PRIVATE | MAP_HUGETLB,
                                     -1, 0
                                     );
  double *getPrice =  (double *)mmap(0,
                                     ARRAY_SIZE * sizeof(double),
                                     PROT_READ | PROT_WRITE,
                                     MAP_ANONYMOUS | MAP_PRIVATE | MAP_HUGETLB,
                                     -1, 0
                                     );
  double *stockPrice =  (double *)mmap(0,
                                       ARRAY_SIZE * sizeof(double),
                                       PROT_READ | PROT_WRITE,
                                       MAP_ANONYMOUS | MAP_PRIVATE | MAP_HUGETLB,
                                       -1, 0
                                       );
  double *strikePrice =  (double *)mmap(0,
                                        ARRAY_SIZE * sizeof(double),
                                        PROT_READ | PROT_WRITE,
                                        MAP_ANONYMOUS | MAP_PRIVATE | MAP_HUGETLB,
                                        -1, 0
                                        );
  double *maturityYears =  (double *)mmap(0,
                                          ARRAY_SIZE * sizeof(double),
                                          PROT_READ | PROT_WRITE,
                                          MAP_ANONYMOUS | MAP_PRIVATE | MAP_HUGETLB,
                                          -1, 0
                                          );
#endif

  srand(time(NULL)); 
  int threadID = omp_get_thread_num();
  if (threadID == 0)
    tstart = dtime();
  
#pragma omp parallel for 
  //#pragma omp parallel for
  for (int i = 0; i < ARRAY_SIZE; i ++) {
    // cout << "Generating information for " << i << "th calculation" << endl; 
    stockPrice[i] = rand(STOCKPRICE_MIN, STOCKPRICE_MAX);
    strikePrice[i] = rand(STRIKEPRICE_MIN, STRIKEPRICE_MAX);
    maturityYears[i] = rand(MATURITYYEARS_MIN, MATURITYYEARS_MAX);
  }
  
  cout << "Start Computing" << endl;

#pragma omp parallel
  for (int i = 0; i < LOOP_TIMES; i ++)
#pragma ivdep
#pragma omp for

#pragma noprefetch callPrice
#pragma noprefetch getPrice 
#pragma prefetch stockPrice:1:512
#pragma prefetch stockPrice:0:32
#pragma prefetch strikePrice:1:512
#pragma prefetch strikePrice:0:32
#pragma prefetch maturityYears:1:512
#pragma prefetch maturityYears:0:32
    for (int j = 0; j < ARRAY_SIZE; j ++) {
      // cout << "LOOP_TIME : " << i << " ARRAY_IDX : " << j << endl;
      //    BlackScholes_CallPrice(callPrice[j], stockPrice[j], strikePrice[j],
      //		     maturityYears[j], RISKFREE_RATE, VOLATILITY);
      //    BlackScholes_GetPrice(getPrice[j], stockPrice[j], strikePrice[j],
      //		     maturityYears[j], RISKFREE_RATE, VOLATILITY);
      
      BlackScholes(callPrice[j], getPrice[j], stockPrice[j], strikePrice[j],
		   maturityYears[j], RISKFREE_RATE, VOLATILITY); 
    }
  
#ifdef USE_MALLOC
  free(callPrice);
  free(getPrice);
  free(stockPrice);
  free(strikePrice);
  free(maturityYears);
#endif

#ifdef USE_MM_MALLOC
  _mm_free(callPrice);
  _mm_free(getPrice);
  _mm_free(stockPrice);
  _mm_free(strikePrice);
  _mm_free(maturityYears);
#endif

#ifdef USE_MMAP
  munmap(callPrice, ARRAY_SIZE * sizeof(double));
  munmap(getPrice, ARRAY_SIZE * sizeof(double));
  munmap(stockPrice, ARRAY_SIZE * sizeof(double));
  munmap(strikePrice, ARRAY_SIZE * sizeof(double));
  munmap(maturityYears, ARRAY_SIZE * sizeof(double));
#endif

  
  if (threadID == 0) {
    tstop = dtime();
    ttime = tstop - tstart;
  }

  cout << "Finish execution" << endl;
  cout << "Used time : " << ttime << endl; 
  cout << "Speed : " << (double)LOOP_TIMES * ARRAY_SIZE / ttime
       << " times/s" << endl;


  return 0; 
}
