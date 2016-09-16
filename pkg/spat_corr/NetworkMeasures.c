#include "Python.h"
#include <stdarg.h>
#include <string.h>
#include "numpy/arrayobject.h"
#include <math.h>
#include <omp.h>

#define VERSION "0.3"

unsigned long nodes;

typedef struct Stack Stack;
typedef struct Queue Queue;
typedef struct Node Node;

struct Node {
    Node *next;
    unsigned long i, d;
};

struct Queue {
    Node *first;
    Node *last;
};

struct Stack {
    Stack *head;
    unsigned long i;
};

int put(Queue *q, const unsigned long i, const unsigned long d);
int get(Queue *q, unsigned long *i, unsigned long *d);
void initNetworkMeasures(void);

void
memdie(const char *errstr) {
	PyErr_SetString(PyExc_MemoryError, errstr);
	exit(EXIT_FAILURE);
}

int
put(Queue *q, const unsigned long i, const unsigned long d) {
    Node *n;
    
    n = malloc(sizeof(Node));
    if(!n)
        return 1;
    n->i = i;
    n->d = d;
    if(!q->first) {
        q->first = q->last = n;
    } else {
        q->last->next = n;
        q->last = n;
    }
    n->next = NULL;
    return 0;
}

int
get(Queue *q, unsigned long *i, unsigned long *d) {
    Node *tmp;

    if(!q->first)
        return 1;
    *i = q->first->i;
    *d = q->first->d;
    tmp = q->first;
    q->first = q->first->next;
    free(tmp);
    return 0;
}

double *
weighted_link_probability(const unsigned long *d, const double *c) {
	unsigned long l, dmax, *grid;
	unsigned long i, in, k, m;
	double *link, *prob;

	dmax = 0;
	for(i = 0; i < nodes * (nodes - 1) / 2; i++)
		if(d[i] > dmax)
			dmax = d[i];
	dmax++;
	grid = calloc(dmax, sizeof(unsigned long));
	link = calloc(dmax, sizeof(double));
	prob = calloc(dmax, sizeof(double));
	if(!grid || !link || !prob)
		memdie("Cannot allocate enough memory for link probability.");
	m = 0;
	for(i = 1; i < nodes; i++) {
		in = i * nodes;
		for(k = 0; k < i; k++) {
			l = d[m++];
			grid[l] += 1;
			link[l] += c[in + k];
		}
	}
	for(m = 0; m < dmax; m++) {
		if(grid[m])
			prob[m] = link[m] / grid[m];
		//printf("%.2f\n", prob[m]);
	}
	free(grid);
	free(link);
	return prob;
}

double *
weighted_spatial_surrogate(const unsigned long *d, const double *p) {
	unsigned long i, in, k, kn, l, m;
	double *s;

	s = calloc(nodes * nodes, sizeof(double));
	if(!s)
		memdie("Cannot allocate enough memory for surrogate.");
	m = 0;
	for(i = 1; i < nodes; i++) {
		in = i * nodes;
		for(k = 0; k < i; k++) {
			kn = k * nodes;
			l = d[m++];
			s[in + k] = s[kn + i] = p[l];
		}
	}
	return s;
}

double *
strength(const double *w) {
	unsigned long i, k, in;
	double *s, sum;

	s = malloc(nodes * sizeof(double));
	if(!s)
		memdie("Cannot allocate enough memory for strength array.");
	for(i = 0; i < nodes; i++) {
		in = i * nodes;
		sum = 0;
		for(k = 0; k < nodes; k++)
			sum += (w[in + k] - sum) / (k + 1);
		s[i] = sum;
	}
	return s;
}

unsigned long *
spherical_distances(const double *lon, const double *lat, const double grain) {
	unsigned long i, k, sum, *csum, *d;
	double dik, cd, buf, loni;
	double *cla, *sla;
	double clai, slai;
	double conv;

	conv = M_PI / 180.0;
	cla = malloc(nodes * sizeof(double));
	sla = malloc(nodes * sizeof(double));
	d = malloc(nodes * (nodes - 1) / 2 * sizeof(unsigned long));
	csum = malloc((nodes + 1) * sizeof(unsigned long));
	if(!cla || !sla || !d || !csum)
		memdie("Cannot allocate enough memory for distance array.");
	sum = 0;
	for(i = 0; i < nodes; i++) {
		sum += i;
		csum[i + 1] = sum;
		buf = conv * lat[i];
		cla[i] = cos(buf);
		sla[i] = sin(buf);
	}

#pragma omp parallel for private(k, loni, clai, slai, dik, cd)
	for(i = 1; i < nodes; i++) {
		loni = conv * lon[i];
		clai = cla[i];
		slai = sla[i];
		for(k = 0; k < i; k++) {
			dik = loni - (conv * lon[k]);
			cd = cos(dik);
			d[csum[i] + k] = (unsigned long)(grain * atan2(sqrt(pow(cla[k]*sin(dik), 2) + pow(clai*sla[k] - slai*cla[k]*cd, 2)), (slai*sla[k] + clai*cla[k]*cd)));
		}
	}
	free(csum);
	free(sla);
	free(cla);
	return d;
}


void
directionality(double *rho, double *phi, double *err, const double *corr, const double *lon, const double *lat, const double e, const int g) {
	unsigned long i, in, k, l, len, *dist;
	double loni, lati;
	double x, y, a;
	double *u, *v, *w;
	double *surr, *prob;
	double sc, mu, mv;
	double pih = M_PI / 2.0;
	double sum, var;

	/* spatial surrogate */
	dist = spherical_distances(lon, lat, g);
	prob = weighted_link_probability(dist, corr);
	surr = weighted_spatial_surrogate(dist, prob);
	free(dist);
	free(prob);

	/* for each node ... */
#pragma omp parallel for private(i, in, k, l, len, loni, lati, a, x, y, u, v, w, mu, mv, sc, sum, var)
	for(i = 0; i < nodes; i++) {
		u = malloc(nodes * sizeof(double));
		v = malloc(nodes * sizeof(double));
		w = malloc(nodes * sizeof(double));
		if(!u || !v || !w)
			memdie("Cannot allocate workspace memory in directionality().");
		in = i * nodes;
		loni = lon[i];
		lati = lat[i];
		for(k = 0; k < nodes; k++)
			w[k] = -10;
		/* ... look at all neighbors */
		for(k = 0; k < nodes; k++) {
			x = corr[in + k];
			y = surr[in + k];
			a = atan2(lat[k] - lati, lon[k] - loni);
			if(a < 0)
				a += M_PI;
			/* angle binning */
			for(l = 0; l < nodes; l++) {
				if(w[l] < 0) {
					/* new bin */
					u[l] = x;
					v[l] = y;
					w[l] = a;
					len = l;
					break;
				}
				if(w[l] + e > a && a > w[l] - e) {
					/* this (fuzzy) angle occured already */
					if(x > 10E-13)
						u[l] += x;
					if(y > 10E-13)
						v[l] += y;
					break;
				}
			}
		}
		/* get mean u and mean v */
		mu = mv = 0;
		for(l = 0; l <= len; l++) {
			sc = (double)(l+1);
			mu += (u[l] - mu) / sc;
			mv += (v[l] - mv) / sc;
		}
		/* get max bin - angle mode */
		sc = mu / mv;
		x = 0;
		for(l = 0; l <= len; l++) {
			u[l] -= v[l] * sc;
			if(u[l] > x) {
				x = u[l];
				k = l;
			}
		}
		y = w[k];
		sum = var = 0;
		for(l = 0; l <= len; l++) {
			if(u[l] > 10E-13) {
				a = abs(y - w[l]);
				if(a > pih)
					a -= pih;
				var += a * a * u[l];
				sum += u[l];
			}
		}
		rho[i] = x;
		phi[i] = y;
		err[i] = var / sum;
		free(u);
		free(v);
		free(w);
	}
	free(surr);
}

unsigned long *
components(const unsigned long *x) {
    unsigned long d, i, j, k, l, mark, c, *cn, *ret;
    Queue *open;

    if(!x)
        memdie("error: component() received NULL pointer.\n");
    ret = calloc(nodes, sizeof(unsigned long));
    cn = calloc(nodes, sizeof(unsigned long));
    open = malloc(sizeof(Queue));
    if(!ret || !cn || !open)
        memdie("error: component() malloc() failed.\n");
	c = 0;
    for(j = 0; j < nodes; j++) {
		if(!ret[j]) {
			c++;
	        open->first = open->last = NULL;
    	    /* put root into queue */
        	if(put(open, j, 0))
            	memdie("error: cannot fill queue in component().\n");
	        mark = j + 1;
	        cn[j] = mark;
	        while(!get(open, &i, &d)) {
	            ret[i] = c;
	            /* all neighbors l of i */
	            for(k = x[i]; k < x[i+1]; k++) {
	                l = x[k];
	                if(cn[l] == mark)
	                    /* we have seen this neighbor */
	                    continue;
	                cn[l] = mark;
	                if(put(open, l, d))
	                    memdie("error: cannot fill queue in component().\n");
	            }
	        }
		}
    }
    free(open);
    free(cn);
    return ret;
}

int
main(int argc, char **argv) {
    Py_SetProgramName(argv[0]);
    Py_Initialize();
    initNetworkMeasures();
    Py_Exit(0);
    return 0;
}

static PyObject *
NetworkMeasures_CompressDenseBool(PyObject *self, PyObject* args) {
    PyObject *arg;
    PyArrayObject *mat, *idx;
	unsigned long i, k, in, n, len;
	unsigned long *x, *a;
	npy_intp *dim;

    if(!PyArg_ParseTuple(args, "O", &arg))
		return NULL;
    mat = (PyArrayObject *) PyArray_ContiguousFromObject(arg,
                            PyArray_ULONG, 1, 1);
    if(!mat)
        return NULL;
	n = sqrt(mat->dimensions[0]);
	x = malloc(n * n * sizeof(unsigned long));
	if(!x)
		memdie("Cannot allocate enough memory for output.");
	a = (unsigned long *) mat->data;
	len = n;
	x[0] = len + 1;
	for(i = 0; i < n; i++) {
		in = i * n;
		for(k = 0; k < n; k++) {
			if(a[in + k])
				x[++len] = k;
		}
		x[i + 1] = len + 1;
	}
	len++;
	dim = malloc(sizeof(npy_intp));
	dim[0] = len;
	idx = (PyArrayObject *) PyArray_ZEROS(1, dim, PyArray_ULONG, 0);
	free(dim);
	if(!idx)
		memdie("Cannot create PyArrayObject for output.");
	a = (unsigned long *) idx->data;
	for(i = 0; i < len; i++)
		a[i] = x[i];
	free(x);
	Py_DECREF(mat);
	return PyArray_Return(idx);
}

static PyObject *
NetworkMeasures_CompressEdgeList(PyObject *self, PyObject* args) {
    PyObject *arg;
    PyArrayObject *lst, *idx;
	unsigned long i, k, l, m, n, len;
	unsigned long *a, new, old;
	unsigned long *head, *tail;
	npy_intp *dim;

    if(!PyArg_ParseTuple(args, "Oi", &arg, &nodes))
		return NULL;
    lst = (PyArrayObject *) PyArray_ContiguousFromObject(arg,
                            PyArray_ULONG, 1, 1);
    if(!lst)
        return NULL;
	n = lst->dimensions[0] / 2;
	len = n;
	head = malloc(len * sizeof(unsigned long));
	tail = malloc(len * sizeof(unsigned long));
	if(!head || !tail)
		memdie("Cannot allocate enough memory for output.");
	a = (unsigned long *) lst->data;
	m = l = old = 0;
	head[l++] = m;
	for(i = 0; i < n * 2; i += 2) {
		new = a[i];
		tail[m] = a[i+1];
		for(k = 0; k < new - old; k++) {
			head[l++] = m;
			if(l >= len) {
				len += 1024;
				head = realloc(head, len * sizeof(unsigned long));
				if(!head)
					memdie("Out of memory! I wanna die ...");
			}
		}
		m++;
		old = new;
	}
	i = l;
	for(k = 0; k < nodes - i + 1; k++) {
		head[l++] = m;
		if(l >= len) {
			len += 1024;
			head = realloc(head, len * sizeof(unsigned long));
			if(!head)
				memdie("Out of memory! I wanna die ...");
		}
	}
	len = l;
	head = realloc(head, len * sizeof(unsigned long));
	if(!head)
		memdie("Out of memory! I wanna die ...");
	dim = malloc(sizeof(npy_intp));
	dim[0] = len + n;
	idx = (PyArrayObject *) PyArray_ZEROS(1, dim, PyArray_ULONG, 0);
	free(dim);
	if(!idx)
		memdie("Cannot create PyArrayObject for output.");
	a = (unsigned long *) idx->data;
	for(i = 0; i < len; i++)
		a[i] = head[i] + len;
	for(i = len; i < len + n; i++)
		a[i] = tail[i - len];
	free(head);
	free(tail);
	Py_DECREF(lst);
	return PyArray_Return(idx);
}

static PyObject *
NetworkMeasures_Components(PyObject *self, PyObject* args) {
    PyObject *arg;
    PyArrayObject *comps, *links;
	unsigned long *ix;
	npy_intp *dim;

    if(!PyArg_ParseTuple(args, "O", &arg))
		return NULL;
    links = (PyArrayObject *) PyArray_ContiguousFromObject(arg,
                              PyArray_ULONG, 1, 1);
    if(!links)
        return NULL;
	ix = (unsigned long *) links->data;
    nodes = ix[0] - 1;
	if(ix[nodes] != links->dimensions[0]) {
        PyErr_SetString(PyExc_TypeError,
        "Links array sparse matrix format seems to be corrupted.");
        return NULL;
    }
	dim = malloc(sizeof(npy_intp));
	dim[0] = nodes;
	comps = (PyArrayObject *) PyArray_ZEROS(1, dim, PyArray_ULONG, 0);
	free(dim);
	if(!comps)
		memdie("Cannot create PyArrayObject for output.");
	memcpy(comps->data, components(ix), nodes * sizeof(unsigned long));
	Py_DECREF(links);
	return PyArray_Return(comps);
}

static PyObject *
NetworkMeasures_Strength(PyObject *self, PyObject* args) {
    PyObject *arg1, *arg2, *arg3;
    PyArrayObject *weights, *lons, *lats;
    PyArrayObject *strgth;
	unsigned long i;
	unsigned long *d; /* tri distance array */
	int granularity; /* distance to int scale */
	double *lon, *lat, *w, *s, *r, *p;
	double *x; /* original measure */
	double *y; /* surrogate measure */

	if(!PyArg_ParseTuple(args, "OOOi", &arg1, &arg2, &arg3, &granularity))
		return NULL;
	weights = (PyArrayObject *) PyArray_ContiguousFromObject(arg1,
                                PyArray_DOUBLE, 1, 1);
	lons = (PyArrayObject *) PyArray_ContiguousFromObject(arg2,
                             PyArray_DOUBLE, 1, 1);
	lats = (PyArrayObject *) PyArray_ContiguousFromObject(arg3,
                             PyArray_DOUBLE, 1, 1);
	if(!weights || !lons || !lats)
		return NULL;
	nodes = lons->dimensions[0];
	if(nodes * nodes != weights->dimensions[0]) {
		PyErr_SetString(PyExc_TypeError,
		"Mismatch between number of link weights and number of nodes.");
		return NULL;
	}
	strgth = (PyArrayObject *) PyArray_SimpleNew(1, lons->dimensions,
							   PyArray_DOUBLE);
	if(!strgth)
		memdie("Cannot allocate enough memory for output.");
	lon = (double *) lons->data;
	lat = (double *) lats->data;
	w = (double *) weights->data;
	
	/* original measure */
	x = strength(w);
		
	/* spatial effects */
	if(granularity) {
		d = spherical_distances(lon, lat, granularity);
		p = weighted_link_probability(d, w);
		s = weighted_spatial_surrogate(d, p);
		free(d);
		free(p);
		y = strength(s);
		free(s);
	} else
		y = calloc(nodes, sizeof(double));

	/* write into Py object */
	r = (double *) strgth->data;
	for(i = 0; i < nodes; i++)
		r[i] = x[i] - y[i];

	free(x);
	free(y);
    Py_DECREF(weights);
	Py_DECREF(lons);
	Py_DECREF(lats);
	return PyArray_Return(strgth);
}

static PyObject *
NetworkMeasures_UndirectedDenseWeightedDirectionality(PyObject *self, PyObject* args) {
    PyObject *arg1, *arg2, *arg3;
    PyArrayObject *weights, *lons, *lats;
    PyArrayObject *rho, *phi, *err;
	double *rh, *ph, *er;
	double *lon, *lat, *wg;
	double fuzziness;
	int granularity; /* distance to int scale */

    if(!PyArg_ParseTuple(args, "OOOdi", &arg1, &arg2, &arg3, &fuzziness, &granularity))
		return NULL;
    weights = (PyArrayObject *) PyArray_ContiguousFromObject(arg1,
                                PyArray_DOUBLE, 1, 1);
    lons = (PyArrayObject *) PyArray_ContiguousFromObject(arg2,
                             PyArray_DOUBLE, 1, 1);
	lats = (PyArrayObject *) PyArray_ContiguousFromObject(arg3,
                             PyArray_DOUBLE, 1, 1);
    if(!weights || !lons || !lats)
        return NULL;
    nodes = lons->dimensions[0];
    if(nodes * nodes != weights->dimensions[0]) {
        PyErr_SetString(PyExc_TypeError,
        "Mismatch between number of link weights and number of nodes.");
        return NULL;
    }
    rho = (PyArrayObject *) PyArray_SimpleNew(1, lons->dimensions,
                            PyArray_DOUBLE);
	phi = (PyArrayObject *) PyArray_SimpleNew(1, lons->dimensions,
                            PyArray_DOUBLE);
	err = (PyArrayObject *) PyArray_SimpleNew(1, lons->dimensions,
                            PyArray_DOUBLE);
	if(!rho || !phi || !err)
        memdie("Cannot allocate enough memory for output.");
	lon = (double *) lons->data;
	lat = (double *) lats->data;
	rh = (double *) rho->data;
	ph = (double *) phi->data;
	er = (double *) err->data;
	wg = (double *) weights->data;
	
	directionality(rh, ph, er, wg, lon, lat, fuzziness, granularity);

    Py_DECREF(weights);
	Py_DECREF(lons);
	Py_DECREF(lats);
	return Py_BuildValue("(OOO)", rho, phi, err);
}

static PyMethodDef NetworkMeasures_methods[] = {
	{"CompressDenseBool", NetworkMeasures_CompressDenseBool, METH_VARARGS,
	 "links = CompressDenseBool(mat)\n\nCompresses a square, dense, boolean matrix into a sparse row format.\n\nParameters\n----------\nmat : array_like, bool\n      A flattened square adjacency matrix.\n\nReturns\n-------\nlinks : ndarray, int\n        A new flat integer array with the positions of Trues in the dense boolean matrix.\n"},
	{"CompressEdgeList", NetworkMeasures_CompressEdgeList, METH_VARARGS,
	 "links = CompressEdgeList(edgelist, nodes)\n\nCompresses a sorted edgelist of the form ((0, 1), (1, 1), (1, 3), ...) into a sparse row format.\n\nParameters\n----------\nedgelist : array_like, int\n           A flattened N x 2 dim input matrix, N is the number of links.\n   nodes : int\n           The number of nodes of the network.\n\nReturns\n-------\nlinks : ndarray, int\n        A new flat integer array with the positions of Trues in the dense boolean matrix.\n"},
	{"Components", NetworkMeasures_Components, METH_VARARGS,
	 "comps = Components(links)\n\nIdentifies components of a given network.\n\nParameters\n----------\nlinks : array_like, int\n        The network in the sparse row format as created by the compress functions.\n\nReturns\n-------\ncomps : ndarray, int\n        An array with a component number for each node.\n"},
	{"Strength", NetworkMeasures_Strength, METH_VARARGS,
	 "strengths = Strength(weights, lons, lats, granularity)\n\nMeasures the spatially corrected strength of a given weighted network.\n\nParameters\n----------\nweights : array_like, float\n        The network as a dense weighted adjacency matrix.\n\nReturns\n-------\nstrengths : ndarray, float\n        The strength value for each node.\n"},
    {"UndirectedDenseWeightedDirectionality", NetworkMeasures_UndirectedDenseWeightedDirectionality, METH_VARARGS,
     "rho, phi = UndirectedDenseWeightedDirectionality(weights, lons, lats, fuzziness, granularity)\n\nMeasures the directionality for undirected and weighted networks.\n\nParameters\n----------\nweights : array_like\n          The flattened square matrix of link weights.\n   lons : array_like\n          The longitude coordinates for all nodes.\n   lats : array_like\n          The latitude coordinates for all nodes.\nfuzziness : float\n            The angle fuzziness parameter in rad.\n          It defines a r-neighbourhood around an angle in\n          which all angles are considered to be the same.\ngranularity : int\n              Floating point spatial distances between nodes are scaled\n          by this parameter and then rounded to integers.\n"},
    {NULL, NULL, 0, NULL}
};

void
initNetworkMeasures(void) {
	PyObject *m;
	PyObject *v;

	v = Py_BuildValue("s", VERSION);
    PyImport_AddModule("NetworkMeasures");
    m = Py_InitModule3("NetworkMeasures", NetworkMeasures_methods,
    "This module provides various network measures.");
    PyModule_AddObject(m, "__version__", v);
    import_array();
}
