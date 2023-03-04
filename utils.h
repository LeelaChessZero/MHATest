#pragma once

int divUp(int a, int b);
void compareResults(void *arr1, void *arr2, int size, bool testFp16 = false);
void fillRandomArray(void *out, int size, bool testFp16 = false, float scale = 1.0f);
void dumpContents(void* arr, int size, bool fp16 = false);