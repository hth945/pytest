#ifndef _CYTHON_TEST_H
#define _CYTHON_TEST_H


#include <Python.h>

typedef struct person_info_t
{
    int age;
    char *gender;
}person_info;

void print_person_info(char *name, person_info *info);
int add_number(int a, int b);


#endif