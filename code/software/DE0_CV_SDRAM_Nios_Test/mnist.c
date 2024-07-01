/*
  mnist.c

  Usage:
  $ ./mnist train-images train-labels test-images test-labels
*/

#include <assert.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "cnn.h"
//#include <machine/endian.h>
#include "endian.h"
#include "sys/alt_stdio.h"

//SD Card
#include "terasic_includes.h"
#include "system.h"
#include "alt_types.h"


/*  IdxFile
 */
typedef struct _IdxFile
{
    int ndims;
    uint32_t* dims;
    uint8_t* data;
} IdxFile;

#define DEBUG_IDXFILE 1

/* IdxFile_read(fp)
   Reads all the data from given fp.
*/
IdxFile* IdxFile_read(FILE* fp)
{
    /* Read the file header. */
    struct {
        uint16_t magic;
        uint8_t type;
        uint8_t ndims;
        /* big endian */
    } header;
    if (fread(&header, sizeof(header), 1, fp) != 1) return NULL;
#if DEBUG_IDXFILE
    fprintf(stderr, "IdxFile_read: magic=%x, type=%x, ndims=%u\n",
            header.magic, header.type, header.ndims);
#endif
    //if (header.magic != 0) return NULL;
    if (header.type != 0x08) return NULL;
    if (header.ndims < 1) return NULL;

    /* Read the dimensions. */
    IdxFile* self = (IdxFile*)calloc(1, sizeof(IdxFile));
    if (self == NULL) return NULL;
    self->ndims = header.ndims;
    self->dims = (uint32_t*)calloc(self->ndims, sizeof(uint32_t));
    if (self->dims == NULL) return NULL;

    if (fread(self->dims, sizeof(uint32_t), self->ndims, fp) == self->ndims) {
        uint32_t nbytes = sizeof(uint8_t);
        for (int i = 0; i < self->ndims; i++) {
            /* Fix the byte order. */
            uint32_t size = be32toh(self->dims[i]);
#if DEBUG_IDXFILE
            fprintf(stderr, "IdxFile_read: size[%d]=%u\n", i, size);
#endif
            nbytes *= size;
            self->dims[i] = size;
        }
        /* Read the data. */
        self->data = (uint8_t*) malloc(nbytes);
        if (self->data != NULL) {
            fread(self->data, sizeof(uint8_t), nbytes, fp);
#if DEBUG_IDXFILE
            fprintf(stderr, "IdxFile_read: read: %lu bytes\n");
#endif
        }
    }

    return self;
}




void IdxFile_destroy(IdxFile* self)
{
    assert (self != NULL);
    if (self->dims != NULL) {
        free(self->dims);
        self->dims = NULL;
    }
    if (self->data != NULL) {
        free(self->data);
        self->data = NULL;
    }
    free(self);
}

/* IdxFile_get1(self, i)
   Get the i-th record of the Idx1 file. (uint8_t)
 */
uint8_t IdxFile_get1(IdxFile* self, int i)
{
    assert (self != NULL);
    assert (self->ndims == 1);
    assert (i < self->dims[0]);
    return self->data[i];
}

/* IdxFile_get3(self, i, out)
   Get the i-th record of the Idx3 file. (matrix of uint8_t)
 */
void IdxFile_get3(IdxFile* self, int i, uint8_t* out)
{
    assert (self != NULL);
    assert (self->ndims == 3);
    assert (i < self->dims[0]);
    size_t n = self->dims[1] * self->dims[2];
    memcpy(out, &self->data[i*n], n);
}


/* main */
int my_mnist(char *train_images, char *train_labels, char *test_images, char *test_labels)
{

    //if (argc < 4) return 100;
    //while (1);

    /* Use a fixed random seed for debugging. */
    srand(0);
    /* Initialize layers. */
    /* Input layer - 1x28x28. */
    alt_putstr("creating layer! ...\n");
    Layer* linput = Layer_create_input(1, 28, 28);
    alt_putstr("layer created! ...\n");
    assert (linput != NULL);
    alt_putstr("layer created is not null! ...\n");
    Layer* lconv1 = Layer_create_conv(linput, 16, 14, 14, 3, 1, 2, 0.1);
    alt_putstr("creating layer conv ...\n");
    Layer* lconv2 = Layer_create_conv(lconv1, 32, 7, 7, 3, 1, 2, 0.1);
    assert (lconv2 != NULL);
    alt_putstr("layer conv created! ...\n");
    Layer* lfull1 = Layer_create_full(lconv2, 200, 0.1);
    assert (lfull1 != NULL);
    Layer* lfull2 = Layer_create_full(lfull1, 50, 0.1);
    Layer* loutput = Layer_create_full(lfull2, 10, 0.1);


    IdxFile* images_train = NULL;
    {
        FILE* fp = fopen(train_images, "rb");
        alt_putstr("Training...\n");
        if (fp == NULL) {
        	alt_putstr(train_images);
        	return 111;
        }
        alt_putstr("Training...\n");
        images_train = IdxFile_read(fp);
        if (images_train == NULL) return 111;
        fclose(fp);
    }
    alt_putstr("Read images_train...\n");
    IdxFile* labels_train = NULL;
    {
        FILE* fp = fopen(train_labels, "rb");
        if (fp == NULL) return 111;
        labels_train = IdxFile_read(fp);
        if (labels_train == NULL) return 111;
        fclose(fp);
    }
    alt_putstr("Read labels_train...\n");

    //fprintf(stderr, "training...\n");
    double rate = 0.1;
    double etotal = 0;
    int nepoch = 10;
    int batch_size = 32;
    int train_size = images_train->dims[0];
    for (int i = 0; i < nepoch * train_size; i++) {

        uint8_t img[28*28];
        double x[28*28];
        double y[10];
        int index = rand() % train_size;
        IdxFile_get3(images_train, index, img);
        for (int j = 0; j < 28*28; j++) {
            x[j] = img[j]/255.0;
        }
        Layer_setInputs(linput, x);
        Layer_getOutputs(loutput, y);
        int label = IdxFile_get1(labels_train, index);
#if 0
        fprintf(stderr, "label=%u, y=[", label);
        alt_printf("label=%u, y=[", label);
        for (int j = 0; j < 10; j++) {
            fprintf(stderr, " %.3f", y[j]);
        }
        fprintf(stderr, "]\n");
#endif
        for (int j = 0; j < 10; j++) {
            y[j] = (j == label)? 1 : 0;
        }
        Layer_learnOutputs(loutput, y);
        etotal += Layer_getErrorTotal(loutput);
        if ((i % batch_size) == 0) {
            Layer_update(loutput, rate/batch_size);
        }
        if ((i % 1000) == 0) {
            fprintf(stderr, "i=%d, error=%.4f\n", i, etotal/1000);
            etotal = 0;
        }
    }

    IdxFile_destroy(images_train);
    IdxFile_destroy(labels_train);


    //Layer_dump(linput, stdout);
    //Layer_dump(lconv1, stdout);
    //Layer_dump(lconv2, stdout);
    //Layer_dump(lfull1, stdout);
    //Layer_dump(lfull2, stdout);
    //Layer_dump(loutput, stdout);

    alt_putstr("test...\n");
    IdxFile* images_test = NULL;
    {
        FILE* fp = fopen(test_images, "rb");
        if (fp == NULL) return 111;
        images_test = IdxFile_read(fp);
        if (images_test == NULL) return 111;
        fclose(fp);
    }
    alt_putstr("test...\n");
    IdxFile* labels_test = NULL;
    {
        FILE* fp = fopen(test_labels, "rb");
        if (fp == NULL) return 111;
        labels_test = IdxFile_read(fp);
        if (labels_test == NULL) return 111;
        fclose(fp);
    }

    alt_putstr("test !!! ...\n");
    fprintf(stderr, "testing...\n");
    int ntests = images_test->dims[0];
    int ncorrect = 0;
    for (int i = 0; i < ntests; i++) {
        uint8_t img[28*28];
        double x[28*28];
        double y[10];
        IdxFile_get3(images_test, i, img);
        for (int j = 0; j < 28*28; j++) {
            x[j] = img[j]/255.0;
        }
        Layer_setInputs(linput, x);
        Layer_getOutputs(loutput, y);
        int label = IdxFile_get1(labels_test, i);
        /* Pick the most probable label. */
        int mj = -1;
        for (int j = 0; j < 10; j++) {
            if (mj < 0 || y[mj] < y[j]) {
                mj = j;
            }
        }
        if (mj == label) {
            ncorrect++;
        }
        if ((i % 1000) == 0) {
            fprintf(stderr, "i=%d\n", i);
        }
    }

    alt_putstr("test bla !!! ...\n");

    fprintf(stderr, "ntests=%d, ncorrect=%d\n", ntests, ncorrect);

    IdxFile_destroy(images_test);
    IdxFile_destroy(labels_test);

    Layer_destroy(linput);
    Layer_destroy(lconv1);
    Layer_destroy(lconv2);
    Layer_destroy(lfull1);
    Layer_destroy(lfull2);
    Layer_destroy(loutput);
}

int main(){
    const char *test_images_path = "/mnt/host/data/t10k-images-idx3-ubyte";
    const char *test_labels_path = "/mnt/host/data/t10k-labels-idx1-ubyte";
    const char *train_images_path = "/mnt/host/data/train-images-idx3-ubyte";
    const char *train_labels_path = "/mnt/host/data/train-labels-idx1-ubyte";

    my_mnist(train_images_path, train_labels_path, test_images_path, test_labels_path);

    //while(1);
    return 0;
}

