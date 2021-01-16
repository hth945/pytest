#include "cython_test.h"


int add_number(int a, int b)
{
    return a + b;
}

void print_person_info(char *name, person_info *info)
{
    printf("name: %s, age: %d, gender: %s\n",
            name, info->age, info->gender);
}