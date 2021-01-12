#include <Python.h>

#define nullptr 0
#include <Python.h>


typedef struct
{
    int width;
    int height;
} MatrixXd;

typedef struct
{
    PyObject_HEAD
    MatrixXd *matrix;
} PyMatrixObject;

static PyObject *
PyMatrix_new(PyTypeObject *type, PyObject *args, PyObject *kwds)
{
    PyMatrixObject *self;
    self = (PyMatrixObject *)type->tp_alloc(type, 0);

    char *kwlist[] = {"width", "height", NULL};
    int width = 0;
    int height = 0;

    if (!PyArg_ParseTupleAndKeywords(args, kwds, "ii", kwlist,
                                     &width, &height))
    {
        Py_DECREF(self);
        return NULL;
    }
    
    self->matrix = malloc(sizeof(MatrixXd));
    return (PyObject *)self;
}

static void
*PyMatrix_dealloc(PyObject *obj)
{
    free(((PyMatrixObject *)obj)->matrix);
    Py_TYPE(obj)->tp_free(obj);

    Py_INCREF(Py_None);
    return Py_None;
}


inline MatrixXd *ParseMatrix(PyObject *obj){
    return ((PyMatrixObject *)obj)->matrix;
}

inline PyObject *ReturnMatrix(MatrixXd *m, PyTypeObject *type){
    PyMatrixObject *obj = PyObject_NEW(PyMatrixObject, type);
    obj->matrix = m;
    return (PyObject *)obj;
}


static PyObject *PyMatrix_str(PyObject *a)
{
    MatrixXd *matrix = ParseMatrix(a);
    char str[128];
    sprintf(str, "width:%d, height:%d", matrix->width, matrix->height);
    return Py_BuildValue("s", str);
}

static PyNumberMethods numberMethods = {
    nullptr,      //nb_add
    nullptr,    //nb_subtract;
    nullptr, //nb_multiply
    nullptr,           //nb_remainder;
    nullptr,           //nb_divmod;
    nullptr,           // nb_power;
    nullptr,           // nb_negative;
    nullptr,           // nb_positive;
    nullptr,           // nb_absolute;
    nullptr,           // nb_bool;
    nullptr,           // nb_invert;
    nullptr,           // nb_lshift;
    nullptr,           // nb_rshift;
    nullptr,           // nb_and;
    nullptr,           // nb_xor;
    nullptr,           // nb_or;
    nullptr,           // nb_int;
    nullptr,           // nb_reserved;
    nullptr,           // nb_float;

    nullptr, // nb_inplace_add;
    nullptr, // nb_inplace_subtract;
    nullptr, // nb_inplace_multiply;
    nullptr, // nb_inplace_remainder;
    nullptr, // nb_inplace_power;
    nullptr, // nb_inplace_lshift;
    nullptr, // nb_inplace_rshift;
    nullptr, // nb_inplace_and;
    nullptr, // nb_inplace_xor;
    nullptr, // nb_inplace_or;

    nullptr, // nb_floor_divide;
    nullptr, // nb_true_divide;
    nullptr, // nb_inplace_floor_divide;
    nullptr, // nb_inplace_true_divide;

    nullptr, // nb_index;

    nullptr, //nb_matrix_multiply;
    nullptr  //nb_inplace_matrix_multiply;

};

PyObject *PyMatrix_data(PyObject *self, void *closure)
{
    PyMatrixObject *obj = (PyMatrixObject *)self;
    Py_ssize_t width = obj->matrix->width;
    Py_ssize_t height = obj->matrix->height;

    PyObject *list = PyList_New((Py_ssize_t)2);

    PyObject *value = PyFloat_FromDouble((double)1.1);
    PyList_SetItem(list, 0, value);
    value = PyFloat_FromDouble((double)1.1);
    PyList_SetItem(list, 1, value);
    // for (int i = 0; i < height; i++)
    // {
    //     PyObject *internal = PyList_New(width);

    //     for (int j = 0; j < width; j++)
    //     {
    //         PyObject *value = PyFloat_FromDouble((*obj->matrix)(i, j));
    //         PyList_SetItem(internal, j, value);
    //     }

    //     PyList_SetItem(list, i, internal);
    // }
    return list;
}
// typedef PyObject *(*getter)(PyObject *, void *);
// typedef int (*setter)(PyObject *, PyObject *, void *);

static PyGetSetDef MatrixGetSet[] = {
    {"data", (getter)PyMatrix_data, nullptr, nullptr},
    {nullptr}};

PyObject *PyMatrix_tolist(PyObject *self, PyObject *args)
{
    return PyMatrix_data(self, nullptr);
}

static PyMethodDef MatrixMethods[] = {
    {"to_list", (PyCFunction)PyMatrix_tolist, METH_VARARGS, "Return the matrix data to a list object."},
    {nullptr}};

static PyTypeObject MatrixType = {
    PyVarObject_HEAD_INIT(nullptr, 0) "matrix.Matrix", /* tp_name */
    sizeof(PyMatrixObject),                            /* tp_basicsize */
    0,                                                 /* tp_itemsize */
    (destructor)PyMatrix_dealloc,                      /* tp_dealloc */
    nullptr,                                           /* tp_print */
    nullptr,                                           /* tp_getattr */
    nullptr,                                           /* tp_setattr */
    nullptr,                                           /* tp_reserved */
    nullptr,                                           /* tp_repr */
    &numberMethods,                                    /* tp_as_number */
    nullptr,                                           /* tp_as_sequence */
    nullptr,                                           /* tp_as_mapping */
    nullptr,                                           /* tp_hash  */
    nullptr,                                           /* tp_call */
    PyMatrix_str,                                      /* tp_str */
    nullptr,                                           /* tp_getattro */
    nullptr,                                           /* tp_setattro */
    nullptr,                                           /* tp_as_buffer */
    Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE,          /* tp_flags */
    "Coustom matrix class.",                           /* tp_doc */
    nullptr,                                           /* tp_traverse */
    nullptr,                                           /* tp_clear */
    nullptr,                                           /* tp_richcompare */
    0,                                                 /* tp_weaklistoffset */
    nullptr,                                           /* tp_iter */
    nullptr,                                           /* tp_iternext */
    MatrixMethods,                                     /* tp_methods */
    nullptr,                                           /* tp_members */
    MatrixGetSet,                                      /* tp_getset */
    nullptr,                                           /* tp_base */
    nullptr,                                           /* tp_dict */
    nullptr,                                           /* tp_descr_get */
    nullptr,                                           /* tp_descr_set */
    0,                                                 /* tp_dictoffset */
    nullptr,                                           /* tp_init */
    nullptr,                                           /* tp_alloc */
    PyMatrix_new                                       /* tp_new */
};

static PyObject *PyMatrix_ones(PyObject *self, PyObject *args, PyObject *kwargs)
{
    PyMatrixObject *m = (PyMatrixObject *)PyMatrix_new(&MatrixType, args, kwargs);
    m->matrix->width=1;
    m->matrix->height=1;
    return (PyObject *)m;
}

static PyObject *PyMatrix_zeros(PyObject *self, PyObject *args, PyObject *kwargs)
{
    PyMatrixObject *m = (PyMatrixObject *)PyMatrix_new(&MatrixType, args, kwargs);
    m->matrix->width=0;
    m->matrix->height=0;
    return (PyObject *)m;
}


static PyMethodDef matrixMethods[] = {
    {"ones", (PyCFunction)PyMatrix_ones, METH_VARARGS | METH_KEYWORDS, "Return a new matrix with initial values one."},
    {"zeros", (PyCFunction)PyMatrix_zeros, METH_VARARGS | METH_KEYWORDS, "Return a new matrix with initial values zero."},
    {nullptr}};

static struct PyModuleDef module = {
    PyModuleDef_HEAD_INIT,
    "matrix",
    "Python interface for Matrix calculation",
    -1,
    matrixMethods};

//PyInit_matrix
// PyObject *initModule(void)
PyMODINIT_FUNC PyInit_matrix(void)
{
    PyObject *m;
    if (PyType_Ready(&MatrixType) < 0)
        return NULL;

    m = PyModule_Create(&module);
    if (m == NULL)
        return NULL;

    Py_INCREF(&MatrixType);
    if (PyModule_AddObject(m, "matrix", (PyObject *)&MatrixType) < 0)
    {
        Py_DECREF(&MatrixType);
        Py_DECREF(m);
        return NULL;
    }

    return m;
}

