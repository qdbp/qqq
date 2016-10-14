#include <python3.5m/Python.h>
#include <numpy/arrayobject.h>
#include <stdlib.h>
#include <stdio.h>
#include <X11/X.h>
#include <X11/Xlib.h>

static char module_docstring[] =
    "C extensions of the qqq library.";

static PyObject *get_rgb_screen(PyObject*, PyObject*);
static char get_rgb_screen_doc[] =
    "Function that captures the screen to a numpy array.\n"\
    "Args:\n"\
    "   int x: x coordinate from which to capture\n"\
    "   int x: x coordinate from which to capture\n"\
    "   int dx: width of area to capture \n"\
    "   int dy: height of area to capture\n"\
    "Returns:\n"\
    "   arr: (dh, dw, 4) numpy int8 array. Last dimension is (b,g,r,a)";

static PyMethodDef qqc_methods[] = {
    { "get_rgb_screen", get_rgb_screen, METH_VARARGS, get_rgb_screen_doc},
    {NULL, NULL, 0, NULL} /* Sentinel */
};

static struct PyModuleDef qqc_module = {
   PyModuleDef_HEAD_INIT,
   "qqc",
   module_docstring,
   -1,
   qqc_methods
};

PyMODINIT_FUNC
PyInit_qqc(void) {
    import_array();
    return PyModule_Create(&qqc_module);
};

static PyObject*
get_rgb_screen(PyObject *self, PyObject *args) {

    int x, y, dx, dy;

    int good_args = PyArg_ParseTuple(args, "IIII", &x, &y, &dx, &dy);
    if (!good_args) {
        return NULL;
    }

    Display *display = XOpenDisplay(NULL);
    Window window = RootWindow(display, DefaultScreen(display));
	XImage *img = XGetImage(display, window, x, y, dx, dy, AllPlanes, ZPixmap);

    int w = img->width;
    int h = img->height;

    unsigned char *rgb_arr = (unsigned char*) malloc(4*w*h);
    if (rgb_arr == NULL) {
        PyErr_NoMemory();
        return NULL;
    }

    npy_intp dims[] = {h, w, 4};
    PyObject *np_arr = PyArray_SimpleNewFromData(3, dims, NPY_UINT8, img->data);
    return np_arr;
}
