/*
Copyright (c) 2013, Aljoscha Rheinwalt <aljoscha at pik-potsdam dot de>

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:
* Redistributions of source code must retain the above copyright
  notice, this list of conditions and the following disclaimer.
* Redistributions in binary form must reproduce the above copyright
  notice, this list of conditions and the following disclaimer in the
  documentation and/or other materials provided with the distribution.
* Neither the name of the copyright holder nor the names of its
  contributors may be used to endorse or promote products derived
  from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
"AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/
#include <stdio.h>
#include <stdlib.h>
#include <stdarg.h>
#include <string.h>
#include <math.h>
#include <omp.h>

int zscores;
int nodes;
int tlen;

void
die(const char *errstr, ...) {
	va_list ap;

	va_start(ap, errstr);
	vfprintf(stderr, errstr, ap);
	va_end(ap);
	exit(EXIT_FAILURE);
}

unsigned int
count(const int *ex, const int *ey, const int lx, const int ly) {
	int m, n, d;
	unsigned int c;

	c = 0;
#pragma omp parallel for reduction(+:c) private(m, n, d)
	for(m = 0; m < lx; m++) {
		for(n = 0; n < ly; n++) {
			d = ex[m] - ey[n];
			if(d > 0)
				continue;
			if(!d)
				c++;
			if(d < 0)
				break;
		}
	}
	return c;
}

double *
count_prob(const int lx, const int ly) {
	int i, s, c, d, b;
	int x, y;
	double p, t, *r;
	
	if(ly > lx) {
		x = ly;
		y = lx;
	} else {
		x = lx;
		y = ly;
	}
	x++;
	r = calloc((y + 1), sizeof(double));
#pragma omp parallel for private(i, s, c, d, b, t, p)
	for(c = 0; c <= y; c++) {
		s = c;
		if(s > y - c)
			s = y - c;
		b = tlen - x + 2;
		d = tlen;
		p = t = 1.0;
		for(i = 1; i <= s; i++) {
			if(p < 1E-300) {
				p *= 100;
				t *= 100;
				if(t > 1E300)
					die("error: we're beyond two doubles ... you're on your own now. bye.\n");
			}
			if(p > 1E300) {
				p /= 100;
				t /= 100;
				if(t < 1E-300)
					die("error: we're beyond two doubles ... you're on your own now. bye.\n");
			}
			p *= b - i;
			p /= d--;
			p *= x - i;
		   	p /= d--;
			p *= y - (s - i);
			p /= i;
		}
		for(i = s + 1; i <= y - c; i++) {
			if(p < 1E-300) {
				p *= 100;
				t *= 100;
			}
			if(p > 1E300) {
				p /= 100;
				t /= 100;
				if(t < 1E-300)
					die("error: we're beyond two doubles ... you're on your own now. bye.\n");
			}
			p *= b - i;
			p /= d--;
		}
		for(i = s + 1; i <= c; i++) {
			if(p > 1E300) {
				p /= 100;
				t /= 100;
				if(t < 1E-300)
					die("error: we're beyond two doubles ... you're on your own now. bye.\n");
			}
			p *= x - i;
		   	p /= d--;
		}
		r[c] = p / t;
	}
	return r;
}

int *
read_events(void) {
	int i, *e;

	e = malloc(nodes * tlen * sizeof(int));
	if(!e)
		die("error: malloc failed in read_events().\n");
	//fprintf(stderr, "reading events from stdin ..");
	for(i = 0; i < nodes * tlen; i++)
		if(scanf("%i", &e[i]) != 1)
			die("\nerror: read of STDIN failed at position %i.\n", i);
	//fprintf(stderr, ". done.\n");
	return e;
}

double *
meanstdr(const int lx, const int ly) {
	int t, s;
	double mean, stdr;
	double *r, *p, b;

	r = malloc(2 * sizeof(double));
	if(!r)
		die("error: malloc failed in meanstd()\n");

	p = count_prob(lx, ly);
	if(lx > ly)
		s = ly + 1;
	else
		s = lx + 1;
	mean = stdr = 0.0;
	for(t = 1; t < s; t++) {
		b = t * p[t];
		if(b > 1E-13) {
			mean += b;
			stdr += t * b;
		}
	}
	free(p);
	stdr -= mean * mean;
	stdr = sqrt(stdr);
	r[0] = mean;
	r[1] = stdr;

	return r;
}

double *
simple_esync(const int *e) {
	int lx, ly;
	int i, k, t;
	int *ex, *ey;
	double c, *p;
	double *syncs;
	//double wtime1, wtime2, wtime3, rtime1, rtime2, rtime3;

	//if(!omp_get_thread_num())
	//	wtime1 = omp_get_wtime();

	syncs = malloc(nodes * (nodes - 1) / 2 * sizeof(double));
	if(!syncs)
		die("error: malloc() failed in simple_esync().\n");

	//if(!omp_get_thread_num()) {
	//	rtime1 = omp_get_wtime() - wtime1;
	//	fprintf(stderr, "syncs malloc: %.5f seconds\n", rtime1);
	//	wtime1 = omp_get_wtime();
	//}

#pragma omp parallel private(i, k, t, lx, ly, ex, ey, c, p)
{
	//if(!omp_get_thread_num())
	//	wtime2 = omp_get_wtime();

	ex = malloc(tlen * sizeof(int));
	ey = malloc(tlen * sizeof(int));
	if(!ex || !ey)
		die("error: malloc() failed in simple_esync().\n");

	//if(!omp_get_thread_num()) {
	//	rtime2 = omp_get_wtime() - wtime2;
	//	fprintf(stderr, "ex ey malloc: %.5f seconds\n", rtime2);	
	//}

#pragma omp for schedule(dynamic)
	for(i = 0; i < nodes; i++) {
		//if(!omp_get_thread_num()) {
		//	if(i == 0)
		//		wtime3 = omp_get_wtime();
		//}

		lx = 0;
		for(t = 0; t < tlen; t++)
			if(e[i*tlen + t])
				ex[lx++] = t;
		//if(lx < 5)
		//	fprintf(stderr, "warning: event series for node %i has less than 5 events!\n", i);

		for(k = 0; k < i; k++) {
			ly = 0;
			for(t = 0; t < tlen; t++)
				if(e[k*tlen + t])
					ey[ly++] = t;

			/* brute force quality insurance */
			if(lx < 5 || ly < 5) {
				syncs[i * (i - 1) / 2 + k] = 0;
				continue;
			}

			/* count synchronizations */
			c = count(ex, ey, lx, ly);
			if(zscores) {
				p = meanstdr(lx, ly);
				/* p[0] = probability mean, p[1] = probability stderr */
				syncs[i * (i - 1) / 2 + k] = (c - p[0]) / p[1];
				free(p);
			} else
				syncs[i * (i - 1) / 2 + k] = c / sqrt(lx * ly);
		}

		//if(!omp_get_thread_num()) {
		//	rtime3 = omp_get_wtime() - wtime3;
		//	wtime3 = omp_get_wtime();
		//	fprintf(stderr, "%i %i %.5f\n", i, omp_get_thread_num(), rtime3);
		//}		
	}
	free(ex);
	free(ey);
} // end pragma omp parallel
	
	//if(!omp_get_thread_num()) {
	//	rtime1 = omp_get_wtime() - wtime1;
	//	fprintf(stderr, "omp total: %.5f seconds\n", rtime1);
	//}

	return syncs;
}

int main(int argc, char *argv[]) {
	int i, k, m;
	int *e;
	double *s;
	//double time1, time2;

	if(argc < 3)
		die("usage: %s num_nodes time_len [z-scores_bool] < binary_event_series\ndefault is z-scores_bool = 1 (True)\n", argv[0]);

	nodes = atoi(argv[1]);
	tlen = atoi(argv[2]);
	if(argc == 4)
		zscores = atoi(argv[3]);
	else
		zscores = 1;

	//if(!omp_get_thread_num())
	//	time1 = omp_get_wtime();

	e = read_events();

	//if(!omp_get_thread_num()) {
	//	time2 = omp_get_wtime() - time1;
	//	fprintf(stderr, "read events: %.5f seconds\n", time2);
	//}	

	//if(!omp_get_thread_num())
	//	time1 = omp_get_wtime();

	//omp_set_num_threads(3);
	s = simple_esync(e);

	//if(!omp_get_thread_num()) {
	//	time2 = omp_get_wtime() - time1;
	//	fprintf(stderr, "simple esync: %.5f seconds\n", time2);
	//}	

	//if(!omp_get_thread_num())
	//	time1 = omp_get_wtime();

	m = 0;
	for(i = 1; i < nodes; i++) {
		for(k = 0; k < i; k++) {
			printf("%.4lf ", s[m++]);
		}
		printf("\n");
	}

	//if(!omp_get_thread_num()) {
	//	time2 = omp_get_wtime() - time1;
	//	fprintf(stderr, "write events: %.5f seconds\n", time2);
	//}

	free(e);
	free(s);

	return EXIT_SUCCESS;
}
