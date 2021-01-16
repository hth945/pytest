#include <Python.h>

static PyObject *SpamError;

/********************add_one start**************/
int add_one(int a)
{
    return a + 1;
}

static PyObject *py_add_one(PyObject *self, PyObject *args)
{
    int num;
    if (!PyArg_ParseTuple(args, "i", &num))
        return NULL;
    return PyLong_FromLong(add_one(num));
}
/********************add_one end**************/

/********************spam_system start**************/
static PyObject *spam_system(PyObject *self, PyObject *args)
{
    const char *command;
    int sts;

    if (!PyArg_ParseTuple(args, "s", &command))
        return NULL;
    sts = system(command);
    if (sts < 0)
    {
        PyErr_SetString(SpamError, "System command failed");
        return NULL;
    }
    return PyLong_FromLong(sts);

    // Py_INCREF(Py_None);
    // return Py_None;
}
/********************spam_system end**************/

static PyObject *helloworld(PyObject *self)
{
    return Py_BuildValue("s", "Hello, helloworld Python");
}

static PyMethodDef helloworld_methods[] = {
    {"helloworld", (PyCFunction)helloworld, METH_NOARGS, "helloworld( ): Any message you want to put here!!\n"},
    {"add_one", py_add_one, METH_VARARGS},
    {"system", spam_system, METH_VARARGS, "Execute a shell command."},
    {NULL}};

static struct PyModuleDef helloworld_def = {
    PyModuleDef_HEAD_INIT,
    "helloworld", /* name of module */
    "",           /* module documentation, may be NULL */
    -1,           /* size of per-interpreter state of the module, or -1 if the module keeps state in global variables. */
    helloworld_methods};

PyMODINIT_FUNC PyInit_helloworld(void)
{
    PyObject *m;

    m = PyModule_Create(&helloworld_def);
    if (m == NULL)
        return NULL;

    SpamError = PyErr_NewException("spam.error", NULL, NULL);
    Py_XINCREF(SpamError);
    if (PyModule_AddObject(m, "error", SpamError) < 0)
    {
        Py_XDECREF(SpamError);
        Py_CLEAR(SpamError);
        Py_DECREF(m);
        return NULL;
    }

    return m;
}